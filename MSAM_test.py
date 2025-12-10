# MSAM_test.py (updated)
# encoding: utf-8
import argparse
import os
import sys
import time
from typing import List, Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

# ensure local sam3 package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "sam3"))

from dataset import MVTecMetaDataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official
from sam3.visualization_utils import draw_masks_to_frame

# ---------- Default MVPROMPTS (your desired mapping) ----------
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
    "tile": "crack,discoloration,an irregular,translucent strip,translucent material,gray stroke,irregular stroke,stain,glue strip,lighter color,reflective specks",
    "toothbrush": "contamination,contamination and misalignment,misalignment,irregular bristles,visibly empty area,discoloration,foreign material,disarrangement",
    "transistor": "bent lead,cut lead,cut,irregularly cut leg,notch,damaged case,broken area,misalignment,misplaced,bent metal lead,material deformation,chipped area,surface abrasion",
    "wood": "scratch,holes,discoloration,scratches,color anomalies,inconsistent coloration,color variation,a scratch,liquid stains,liquid damage,water stain,rough patch,grain crack",
    "zipper": "broken teeth,split teeth,missing teeth,squeezed teeth,pinched teeth,teeth deformation,a gap,missing area,broken area,fraying fabric,irregularity fabric,border anomaly,loose threads,fraying,frayed texture,fibrous texture,fabric damage,disrupted woven texture,rough texture",
}

# ---------- helper functions ----------
def get_color_map(palette, prompt: str):
    import hashlib
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(palette)
    return palette[idx]

def safe_binary_metrics(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6):
    """
    pred_bin, gt_bin : tensors of shape (H,W) or (N,H,W) with values 0/1
    Returns per-sample TP, FP, FN, IoU, Dice, Precision, Recall, F1
    """
    if pred_bin.dim() == 2:
        pred_bin = pred_bin.unsqueeze(0)
        gt_bin = gt_bin.unsqueeze(0)
    TP = (pred_bin * gt_bin).sum(dim=(1,2)).float()
    FP = ((pred_bin == 1) & (gt_bin == 0)).sum(dim=(1,2)).float()
    FN = ((pred_bin == 0) & (gt_bin == 1)).sum(dim=(1,2)).float()
    union = TP + FP + FN
    iou = (TP / (union + eps)).cpu().numpy()
    dice = (2 * TP / (2*TP + FP + FN + eps)).cpu().numpy()
    precision = (TP / (TP + FP + eps)).cpu().numpy()
    recall = (TP / (TP + FN + eps)).cpu().numpy()
    f1 = (2 * precision * recall / (precision + recall + 1e-12))
    # ensure shapes
    return {
        "TP": TP.cpu().numpy(),
        "FP": FP.cpu().numpy(),
        "FN": FN.cpu().numpy(),
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ---------- dataset loader for test ----------
def build_loader(
    root: str, meta_path: str, mode: str, batch_size: int,
    include_test_defects: bool = False,
    train_from_test: bool = False,
    specie_split_ratio: float = 0.8,
    specie_split_seed: int = 42,
    save_dir: Optional[str] = None,
):
    ds = MVTecMetaDataset(
        root=root,
        meta_path=meta_path,
        mode=mode,
        k_shot=0,
        aug_rate=0.0,
        include_test_defects=include_test_defects,
        goods_per_class=None,
        train_from_test=train_from_test,
        specie_split_ratio=specie_split_ratio,
        specie_split_seed=specie_split_seed,
        save_dir=save_dir,
    )

    def collate_fn(batch):
        imgs, masks, prompt_lists, is_anomaly, class_names = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        return imgs, masks, list(prompt_lists), list(class_names)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

# ---------- model loading (keep compatible) ----------
import inspect
def _filter_kwargs_for_callable(func, kwargs: dict):
    sig = inspect.signature(func)
    ok = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            ok[k] = v
    return ok

def load_model(args, device):
    common_kwargs = {
        "bpe_path": getattr(args, "bpe_path", None),
        "sam3_ckpt": getattr(args, "sam3_ckpt", None),
        "enable_lora": not getattr(args, "disable_lora", False),
        "lora_rank": getattr(args, "lora_rank", 16),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "freeze_vision": getattr(args, "freeze_vision", False),
        "freeze_text": getattr(args, "freeze_text", False),
        "device": device,
        "enable_parallel_lora": getattr(args, "enable_parallel_lora", False),
        "parallel_lora_rank": getattr(args, "parallel_lora_rank", 16),
        "parallel_lora_alpha": getattr(args, "parallel_lora_alpha", None),
    }

    if args.use_official:
        Constructor = FineTuneSAM3Official
    else:
        Constructor = FineTuneSAM3

    ctor_kwargs = _filter_kwargs_for_callable(Constructor.__init__, common_kwargs)
    model = Constructor(**ctor_kwargs)

    # load base sam3 if needed (non-official)
    if not args.use_official and args.sam3_ckpt and os.path.exists(args.sam3_ckpt):
        state = torch.load(args.sam3_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)

    # load fine-tuned checkpoint if provided
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"[INFO] Loaded fine-tuned weights from {args.ckpt}")

    model.eval()
    return model.to(device)

# ---------- main inference loop ----------
@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = build_loader(
        args.data_root,
        args.meta_path or os.path.join(args.data_root, "meta.json"),
        args.mode,
        args.batch_size,
        include_test_defects=getattr(args, "include_test_defects", False),
        train_from_test=getattr(args, "train_from_test", False),
        specie_split_ratio=getattr(args, "specie_split_ratio", 0.8),
        specie_split_seed=getattr(args, "specie_split_seed", 42),
        save_dir=getattr(args, "save_dir", None),
    )
    model = load_model(args, device)
    os.makedirs(args.output_dir, exist_ok=True)


    to_pil = transforms.ToPILImage()
    palette = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (218, 165, 32),
        (199, 21, 133),
        (70, 130, 180),
    ]
    font = ImageFont.load_default()

    idx = 0
    total_imgs = 0
    total_time = 0.0

    # accumulators
    global_stats = defaultdict(float)
    global_counts = defaultdict(int)
    per_class_stats = defaultdict(lambda: defaultdict(float))
    per_class_counts = defaultdict(int)

    pbar = tqdm(loader, desc="Inference", leave=True)
    for images, masks, prompt_lists, class_names in pbar:
        images = images.to(device)
        # if user provided a single custom prompt -> override
        custom_prompt: List[str] = []
        if args.prompt:
            custom_prompt = [w.strip() for w in args.prompt.split(",") if w.strip()]

        pbar = tqdm(loader, desc="Inference", leave=True)
        for images, masks, prompt_lists, class_names in pbar:
            images = images.to(device)

            # 如果用户提供了自定义 prompt，则覆盖；否则**优先使用 MVPROMPTS 映射**
            if custom_prompt:
                prompt_lists = [custom_prompt for _ in prompt_lists]
            else:
                new_prompt_lists = []
                for i, cls_name in enumerate(class_names):
                    # 规范化类名为小写以便在 MVPROMPTS 中查找
                    cls_key = cls_name.lower().strip()
                    # 优先使用 MVPROMPTS 映射（方案 A 的行为）
                    mapped = MVPROMPTS.get(cls_key, None)
                    if mapped:
                        new_prompt_lists.append([w.strip() for w in mapped.split(",") if w.strip()])
                    else:
                        # 如果 MVPROMPTS 中没有对应项，则回退到 dataset 提供的 prompt（若存在）
                        if prompt_lists and prompt_lists[i]:
                            new_prompt_lists.append(prompt_lists[i])
                        else:
                            new_prompt_lists.append(["anomaly"])
                prompt_lists = new_prompt_lists


        start = time.time()
        out = model(images, prompt_lists)
        infer_time = time.time() - start
        total_time += infer_time
        total_imgs += images.size(0)

        pred_masks = out.get("pred_masks", None)
        if pred_masks is None:
            continue
        # reduce possible aux dimensions
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        if pred_masks.dim() == 4:
            # (B, C, H, W) or (B,1,H,W) -> keep as is
            pass
        pred_masks = torch.sigmoid(pred_masks)
        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        gt = (masks > 0.5).float().to(device)
        valid = gt.flatten(1).sum(dim=1) > 0

        # compute per-image metrics
        for b in range(images.size(0)):
            cls_name = class_names[b]
            # get prediction for sample
            pm = pred_masks[b]  # shape: (C,H,W) or (1,H,W)
            gm = gt[b]  # shape: (1,H,W)

            # combine multi-channel masks (OR across channels)
            if pm.dim() == 3 and pm.shape[0] > 1:
                pm_bin = (pm > 0.5).float()
                pm_comb = pm_bin.max(dim=0).values  # (H,W)
            else:
                # single channel
                if pm.dim() == 3:
                    pm_single = pm.squeeze(0)
                else:
                    pm_single = pm
                pm_comb = (pm_single > 0.5).float()

            gm_single = gm.squeeze(0).float()

            # compute metrics
            metrics = safe_binary_metrics(pm_comb, gm_single)
            iou = metrics["iou"][0]
            dice = metrics["dice"][0]
            prec = metrics["precision"][0]
            rec = metrics["recall"][0]
            f1 = metrics["f1"][0]

            # accumulate global
            global_stats["iou_sum"] += iou
            global_stats["dice_sum"] += dice
            global_stats["prec_sum"] += prec
            global_stats["rec_sum"] += rec
            global_stats["f1_sum"] += f1
            global_stats["img_count"] += 1

            # per-class
            per_class_stats[cls_name]["iou_sum"] += iou
            per_class_stats[cls_name]["dice_sum"] += dice
            per_class_stats[cls_name]["prec_sum"] += prec
            per_class_stats[cls_name]["rec_sum"] += rec
            per_class_stats[cls_name]["f1_sum"] += f1
            per_class_counts[cls_name] += 1

        # visualization / save frames (same as original)
        for b in range(pred_masks.shape[0]):
            cls_name = class_names[b]
            prompts_b = prompt_lists[b] if prompt_lists else []
            prompts_b = prompts_b if isinstance(prompts_b, list) else [prompts_b]
            prompt_text = "_".join(prompts_b) if prompts_b else "prompt"
            sample_dir = os.path.join(args.output_dir, cls_name)
            os.makedirs(sample_dir, exist_ok=True)

            img_pil = to_pil(images[b].cpu())
            mask_np = pred_masks[b].squeeze().cpu().numpy()
            if mask_np.ndim == 2:
                masks_stack = (mask_np > 0.5)[None, ...]
                prompts_for_color = prompts_b if prompts_b else ["prompt"]
            else:
                masks_stack = (mask_np > 0.5)
                prompts_for_color = prompts_b if prompts_b else [f"c{i}" for i in range(mask_np.shape[0])]

            colors = np.array([get_color_map(palette, p) for p in prompts_for_color], dtype=np.uint8)
            frame = np.array(img_pil.convert("RGB"), dtype=np.uint8)
            frame = draw_masks_to_frame(frame, masks_stack.astype(bool), colors)
            overlay_pil = Image.fromarray(frame)

            # legend bottom-left with prompt text
            draw = ImageDraw.Draw(overlay_pil)
            row_h = 12
            box_w = 10
            pad = 4
            legend_items = []
            max_w = 0
            for p in prompts_for_color:
                text = p
                bbox = font.getbbox(text)
                w = bbox[2] - bbox[0]
                max_w = max(max_w, w)
                legend_items.append((text, w))
            legend_h = row_h * len(prompts_for_color) + pad * 2
            legend_w = box_w + 4 + max_w + pad * 2
            legend_x = 5
            legend_y = overlay_pil.height - legend_h - 5
            draw.rectangle(
                [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
                fill=(0, 0, 0),
            )
            for i, (text, _) in enumerate(legend_items):
                color = get_color_map(palette, prompts_for_color[i])
                y = legend_y + pad + i * row_h
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

            filename = f"{cls_name}_{prompt_text}_{idx}.png"
            overlay_path = os.path.join(sample_dir, filename)
            overlay_pil.save(overlay_path)
            idx += 1

        # progress bar update
        speed = total_imgs / total_time if total_time > 0 else 0.0
        avg_dice = global_stats["dice_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0
        avg_iou = global_stats["iou_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0
        pbar.set_postfix(imgs=total_imgs, fps=speed, dice=avg_dice, iou=avg_iou)

    # final summary
    final_speed = total_imgs / total_time if total_time > 0 else 0.0
    final_dice = global_stats["dice_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0
    final_iou = global_stats["iou_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0
    final_prec = global_stats["prec_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0
    final_rec = global_stats["rec_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0
    final_f1 = global_stats["f1_sum"] / global_stats["img_count"] if global_stats["img_count"] > 0 else 0.0

    print(f"[INFO] Inference done. images={total_imgs}, fps={final_speed:.2f}, avg_dice={final_dice:.4f}, avg_iou={final_iou:.4f}, precision={final_prec:.4f}, recall={final_rec:.4f}, f1={final_f1:.4f}")

    # per-class breakdown
    print("\nPer-class summary:")
    for cls, cnt in per_class_counts.items():
        iou = per_class_stats[cls]["iou_sum"] / cnt
        dice = per_class_stats[cls]["dice_sum"] / cnt
        prec = per_class_stats[cls]["prec_sum"] / cnt
        rec = per_class_stats[cls]["rec_sum"] / cnt
        f1 = per_class_stats[cls]["f1_sum"] / cnt
        print(f"  {cls}: n={cnt}, dice={dice:.4f}, iou={iou:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MSAM test")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt", type=str, default=None, help="fine-tuned PEFT checkpoint (sam3_peft_best.pth)")
    parser.add_argument("--sam3_ckpt", type=str, default=None, help="base SAM3 checkpoint for official build")
    parser.add_argument("--use_official", action="store_true")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--bpe_path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt words, comma separated, to override dataset prompts.")
    # dataset split args
    parser.add_argument("--include_test_defects", action="store_true")
    parser.add_argument("--train_from_test", action="store_true")
    parser.add_argument("--specie_split_ratio", type=float, default=0.8)
    parser.add_argument("--specie_split_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None, help="run folder to read splits and/or ckpt")
    # parallel lora args
    parser.add_argument("--enable_parallel_lora", action="store_true")
    parser.add_argument("--parallel_lora_rank", type=int, default=16)
    parser.add_argument("--parallel_lora_alpha", type=float, default=None)

    args = parser.parse_args()
    run_inference(args)
