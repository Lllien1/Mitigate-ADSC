"""
Reverse prompt search on MVTec-AD using SAM3.

Given a list of candidate prompts (e.g., 缺陷短语 scratch/crack/...), run all prompts
on each image, keep top-k masks across all prompts, and save visualizations with
prompt + score overlays. Uses image features once and reuses them for all prompts.
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
import torch
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_masks_to_frame

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Class-specific candidate prompts (Primary only from MVTec_AD_prompts)
MVPROMPTS = {
    "bottle": "contamination anomaly,contamination,broken large,broken small,broken area,irregular metallic textures,residue,small crack,dent,small dent",
    "cable": "cable swap,missing wire,cut or tear,missing cable,bent wires,cut inner insulation,cut outer insulation,poke insulation",
    "capsule": "crack,faulty imprint,scratch,deformation,scratches,poke,squeeze,disrupted surface,puncture,indentation,scratch-like imperfections,an irregular rough area,irregular area",
    "carpet": "hole,cut,discoloration,foreign object,thread anomaly,color anomaly,color inconsistency,darker-colored patch,different shade,loose threads,jagged area,metal contamination,pulled thread,knot,rough texture,frayed edge",
    "grid": "bent,bent anomaly,broken,breakage,metal contamination,contamination,distortion,physical distortion,an irregularly shaped spot,scratch,thread anomaly",
    "hazelnut": "crack,hole,cut,discoloration,color inconsistency,a crack,irregular white markings,scratch",
    "leather": "cut,scratch,fold,crease,deformation,puncture,discoloration,poke,color anomaly,color defect,colouration issue,irregular tear,jagged edges,a linear raised distortion,distortion,burned mark",
    "metal nut": "scratch,deformation,warped edge,discoloration,deformity,color anomaly,color variation,material distortion,darker area",
    "pill": "crack,contamination,faulty imprint,imprint issue,misshapen imprint,discoloration,scratch",
    "screw": "scratch,deformation,scratch-like aberration,a linear mark,manipulation or deformation,manipulated or damaged tip,malformed tip,blunted and irregular form,tip damage,surface imperfection,irregular threading,damaged threads",
    "tile": "crack,discoloration,an irregular, translucent strip,translucent material,gray stroke,irregular stroke,stain,glue strip,lighter color,reflective specks",
    "toothbrush": "contamination,contamination and misalignment,misalignment,irregular bristles,visibly empty area,discoloration,foreign material,disarrangement",
    "transistor": "bent lead,cut lead,cut,irregularly cut leg,notch,damaged case,broken area,misalignment,misplaced,bent metal lead,material deformation,chipped area,surface abrasion",
    "wood": "scratch,holes,discoloration,scratches,color anomalies,inconsistent coloration,color variation,a scratch,liquid stains,liquid damage,water stain,rough patch,grain crack",
    "zipper": "broken teeth,split teeth,missing teeth,squeezed teeth,pinched teeth,teeth deformation,a gap,missing area,broken area,fraying fabric,irregularity fabric,border anomaly,loose threads,fraying,frayed texture,fibrous texture,fabric damage,disrupted woven texture,rough texture",
}

# Generic defect prompts used when a class-specific list is missing (adjective+noun)
GENERIC_DEFECTS = (
    "scratched surface,cracked area,dented area,rusted spot,stained area,"
    "discolored patch,missing part,broken piece,contaminated spot,misaligned part"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse prompt search for MVTec-AD with SAM3"
    )
    parser.add_argument("--data-root", type=str, required=True, help="MVTec-AD root")
    parser.add_argument("--weights", type=str, required=True, help="Path to sam3.pt")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/mvtec_prompt_reverse",
        help="Where to save visualizations",
    )
    parser.add_argument(
        "--prompt-list",
        type=str,
        default="",
        help="Comma-separated candidate prompts (adjective+noun). If empty, will fall back to class-specific/default list or template.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{cls}",
        help="Fallback template if prompt-list is empty; {cls} will be replaced by class name",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--conf", type=float, default=0.3, help="Mask score threshold")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Keep top-k masks across all prompts per image (<=0 keeps all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-class image limit for quick runs",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable AUROC computation (pixel and sample level).",
    )
    return parser.parse_args()


def collect_images(class_dir: Path) -> List[Path]:
    test_dir = class_dir / "test"
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[Path] = []
    if test_dir.exists():
        for ext in exts:
            files.extend(test_dir.rglob(ext))
    else:
        for ext in exts:
            files.extend(class_dir.rglob(ext))
    return sorted(files)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gt_mask(image_path: Path, class_dir: Path) -> np.ndarray:
    """
    Load ground-truth mask if available (MVTec format: ground_truth/defect/*_mask.png).
    Returns a float32 binary array in image size.
    """
    # Expect structure: <class>/test/<defect>/<file>.<ext>
    parts = image_path.relative_to(class_dir).parts
    if len(parts) < 2:
        return None
    defect = parts[1]
    if defect == "good":
        return None
    mask_name = image_path.stem + "_mask.png"
    gt_path = class_dir / "ground_truth" / defect / mask_name
    if not gt_path.exists():
        return None
    try:
        gt = Image.open(gt_path).convert("L")
        gt = np.array(gt, dtype=np.uint8)
        gt = (gt > 0).astype(np.float32)
        return gt
    except Exception:
        return None


def get_prompts_for_class(
    cls_name: str, user_prompt_list: List[str], template: str
) -> List[str]:
    """Resolve prompt list priority: user list > class-specific map > template."""
    if user_prompt_list:
        return user_prompt_list

    # Normalize class name for lookup (e.g., metal_nut -> metal nut)
    cls_key = cls_name.replace("_", " ").lower()
    if cls_key in MVPROMPTS:
        prompts = MVPROMPTS[cls_key]
    else:
        prompts = GENERIC_DEFECTS if GENERIC_DEFECTS else template.format(cls=cls_name)
    return [p.strip() for p in prompts.split(",") if p.strip()]


def overlay_masks(
    image: Image.Image,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    prompts: List[str],
    header: str = "",
) -> Image.Image:
    font = ImageFont.load_default()
    palette: List[Tuple[int, int, int]] = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (218, 165, 32),
        (199, 21, 133),
        (70, 130, 180),
    ]

    # deterministic color per prompt (stable hash)
    import hashlib

    color_map = {}

    def get_color(p: str) -> Tuple[int, int, int]:
        if p not in color_map:
            h = hashlib.sha1(p.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % len(palette)
            color_map[p] = palette[idx]
        return color_map[p]

    # Prepare mask colors
    masks_bool = masks.astype(bool)
    colors = np.array([get_color(p) for p in prompts], dtype=np.uint8)

    # Draw masks with contours using official utility
    frame = np.array(image.convert("RGB"), dtype=np.uint8)
    frame = draw_masks_to_frame(frame, masks_bool, colors)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    if header:
        header_bbox = font.getbbox(header)
        header_w = header_bbox[2] - header_bbox[0]
        header_h = header_bbox[3] - header_bbox[1]
        pad = 4
        draw.rectangle(
            [2, 2, 2 + header_w + pad * 2, 2 + header_h + pad * 2],
            fill=(0, 0, 0, 160),
        )
        draw.text((2 + pad, 2 + pad), header, fill=(255, 255, 255), font=font)

    # Legend (color -> prompt) on bottom-left with background, showing scores
    if prompts:
        legend_items = []
        max_w = 0
        for p, s in zip(prompts, scores):
            text = f"{p} ({s:.2f})"
            bbox = font.getbbox(text)
            w = bbox[2] - bbox[0]
            max_w = max(max_w, w)
            legend_items.append((text, w))

        row_h = 12
        box_w = 10
        pad = 4
        legend_h = row_h * len(prompts) + pad * 2
        legend_w = box_w + 4 + max_w + pad * 2
        legend_x = 5
        legend_y = img.height - legend_h - 5
        draw.rectangle(
            [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
            fill=(0, 0, 0, 160),
        )

        for idx, (text, _) in enumerate(legend_items):
            color = get_color(prompts[idx])
            y = legend_y + pad + idx * row_h
            draw.rectangle(
                [legend_x + pad, y, legend_x + pad + box_w, y + box_w],
                fill=color,
            )
            draw.text(
                (legend_x + pad + box_w + 4, y - 2),
                text,
                fill=(255, 255, 255),
                font=font,
            )

    return img


def main() -> None:
    args = parse_args()
    device = args.device
    repo_root = Path(__file__).resolve().parents[1]
    bpe_path = repo_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    # Build model
    print("Loading model...")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        checkpoint_path=args.weights,
        load_from_HF=False,
    )
    processor = Sam3Processor(model=model, device=device, confidence_threshold=args.conf)

    # Prepare prompts
    prompt_list = [p.strip() for p in args.prompt_list.split(",") if p.strip()]

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    ensure_out_dir(out_root)

    class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {data_root}")

    # For AUROC computation
    pixel_preds: List[np.ndarray] = []
    pixel_gts: List[np.ndarray] = []
    sample_scores: List[float] = []
    sample_labels: List[int] = []

    for cls_dir in sorted(class_dirs):
        cls_name = cls_dir.name
        images = collect_images(cls_dir)
        if args.limit > 0:
            images = images[: args.limit]
        if not images:
            print(f"[warn] No images for class {cls_name}, skipping.")
            continue

        prompts_this_class = get_prompts_for_class(
            cls_name=cls_name,
            user_prompt_list=prompt_list,
            template=args.prompt_template,
        )

        print(
            f"Class: {cls_name}, images: {len(images)}, prompts: {len(prompts_this_class)}"
        )
        save_dir = out_root / cls_name
        ensure_out_dir(save_dir)

        class_infer_time = 0.0
        class_imgs_done = 0
        progress = tqdm(images, desc=f"{cls_name}", unit="img")

        for img_path in progress:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:  # corrupt/truncated image
                progress.write(f"[warn] Failed to load {img_path}: {e}")
                continue

            # Compute image features once
            base_state = processor.set_image(image, state={})

            collected_masks = []
            collected_masks_prob = []
            collected_boxes = []
            collected_scores = []
            collected_prompts = []

            start_ts = time.perf_counter()
            for prompt in prompts_this_class:
                # Shallow copy backbone_out + meta to avoid cross-prompt contamination
                state = {
                    "backbone_out": dict(base_state["backbone_out"]),
                    "original_height": base_state["original_height"],
                    "original_width": base_state["original_width"],
                }
                state = processor.set_text_prompt(prompt, state)

                if "masks" not in state or state["masks"].numel() == 0:
                    continue

                masks = state["masks"].squeeze(1).cpu().numpy()
                masks_prob = state["masks_logits"].squeeze(1).cpu().numpy()
                boxes = state["boxes"].cpu().numpy()
                scores = state["scores"].cpu().numpy()

                collected_masks.append(masks)
                collected_masks_prob.append(masks_prob)
                collected_boxes.append(boxes)
                collected_scores.append(scores)
                collected_prompts.extend([prompt] * len(scores))

            if collected_scores:
                masks = np.concatenate(collected_masks, axis=0)
                masks_prob = np.concatenate(collected_masks_prob, axis=0)
                boxes = np.concatenate(collected_boxes, axis=0)
                scores = np.concatenate(collected_scores, axis=0)
            else:
                masks = boxes = scores = masks_prob = None

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_ts
            class_infer_time += elapsed
            class_imgs_done += 1
            inst_fps = 1.0 / elapsed if elapsed > 0 else float("inf")
            avg_fps = (
                class_imgs_done / class_infer_time if class_infer_time > 0 else 0.0
            )
            progress.set_postfix({"fps": f"{inst_fps:.1f}", "avg_fps": f"{avg_fps:.1f}"})

            def make_out_path(img_path: Path) -> Path:
                rel = img_path.relative_to(cls_dir)
                parts = rel.parts
                if len(parts) >= 3:
                    defect = parts[-2]
                    stem = Path(parts[-1]).stem
                    suffix = Path(parts[-1]).suffix
                    name = f"{defect}{stem}{suffix}"
                else:
                    name = img_path.name
                return save_dir / name

            if masks is None or masks.size == 0:
                # No detections; save original for traceability
                out_path = make_out_path(img_path)
                ensure_out_dir(out_path.parent)
                image.save(out_path)
                if not args.no_eval:
                    gt_mask = load_gt_mask(img_path, cls_dir)
                    if gt_mask is None:
                        gt_mask = np.zeros((image.height, image.width), dtype=np.float32)
                    pixel_gts.append(gt_mask.flatten())
                    pixel_preds.append(np.zeros_like(gt_mask, dtype=np.float32).flatten())
                    sample_labels.append(0 if gt_mask.sum() == 0 else 1)
                    sample_scores.append(0.0)
                continue

            # Sort and truncate globally across prompts
            order = np.argsort(-scores)
            if args.top_k > 0:
                order = order[: args.top_k]
            masks = masks[order]
            masks_prob = masks_prob[order]
            boxes = boxes[order]
            scores = scores[order]
            prompts_sorted = [collected_prompts[i] for i in order.tolist()]

            # filter boxes/masks with area > 80% of image (likely non-defect)
            img_area = image.width * image.height
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = areas <= 0.5 * img_area
            if not np.any(keep):
                out_path = make_out_path(img_path)
                ensure_out_dir(out_path.parent)
                image.save(out_path)
                continue
            masks = masks[keep]
            masks_prob = masks_prob[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            prompts_sorted = [p for k, p in zip(keep.tolist(), prompts_sorted) if k]

            top_prompt = prompts_sorted[0] if prompts_sorted else ""

            if not args.no_eval:
                gt_mask = load_gt_mask(img_path, cls_dir)
                if gt_mask is None:
                    gt_mask = np.zeros((image.height, image.width), dtype=np.float32)
                # Combine masks probabilistically: max over score * prob
                combined = np.zeros((image.height, image.width), dtype=np.float32)
                for mprob, s in zip(masks_prob, scores):
                    combined = np.maximum(combined, mprob * s)
                pixel_gts.append(gt_mask.flatten())
                pixel_preds.append(combined.flatten())
                sample_labels.append(0 if gt_mask.sum() == 0 else 1)
                sample_scores.append(float(combined.max()))

            header_text = f"top: {top_prompt}" if top_prompt else ""
            vis = overlay_masks(image, boxes, masks, scores, prompts_sorted, header=header_text)
            out_path = make_out_path(img_path)
            ensure_out_dir(out_path.parent)
            vis.save(out_path)

    # Compute AUROC if enabled
    if not args.no_eval:
        if roc_auc_score is None:
            print("sklearn not available; AUROC not computed.")
        else:
            pixel_labels = np.concatenate(pixel_gts) if pixel_gts else None
            pixel_scores = np.concatenate(pixel_preds) if pixel_preds else None
            sample_labels_arr = np.array(sample_labels)
            sample_scores_arr = np.array(sample_scores)

            if pixel_labels is not None and len(np.unique(pixel_labels)) > 1:
                auroc_px = roc_auc_score(pixel_labels, pixel_scores)
                print(f"Pixel AUROC: {auroc_px:.4f}")
            else:
                auroc_px = None
                print("Pixel AUROC: skipped (only one class present).")

            if sample_labels and len(np.unique(sample_labels_arr)) > 1:
                auroc_sp = roc_auc_score(sample_labels_arr, sample_scores_arr)
                print(f"Sample AUROC: {auroc_sp:.4f}")
            else:
                auroc_sp = None
                print("Sample AUROC: skipped (only one class present).")

    print(f"Done. Visualizations saved to {out_root}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
