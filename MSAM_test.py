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
        ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn
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

    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    # meta is expected to have classes keys; adapt to your meta.json structure
    # For MVTec, meta might contain list of classes or mapping; find class example:
    class_list = sorted(list(meta.get('classes', meta.get('class_list', meta.keys()))))
    # Or if meta.json format differs, extract appropriate list of class names
    args.class_list = class_list
    
    # In load_model or where you call Constructor:
    Constructor = FineTuneSAM3Official if args.use_official else FineTuneSAM3
    # pass through class_list to prompt learner init, e.g.,
    model = Constructor(..., class_list=args.class_list, prompt_learner_type='perclass', num_templates=4)

    # load base sam3 if needed (non-official)
    if not args.use_official and args.sam3_ckpt and os.path.exists(args.sam3_ckpt):
        state = torch.load(args.sam3_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)

    # load fine-tuned checkpoint if provided
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[INFO] Loading fine-tuned checkpoint {args.ckpt} ...")
        ckpt = torch.load(args.ckpt, map_location=device)

        # 常见包装：'state_dict' / 'model' / raw state_dict
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
                print("[INFO] checkpoint contains 'state_dict' -> using it")
            elif "model" in ckpt:
                state = ckpt["model"]
                print("[INFO] checkpoint contains 'model' -> using it")
            else:
                state = ckpt
                print("[INFO] using top-level checkpoint dict as state")
        else:
            state = ckpt
            print("[INFO] checkpoint is not a dict, using directly")

        # 去掉 'module.' 前缀（DDP 保存时常见）
        def _strip_module_prefix(sd: dict):
            new = {}
            for k, v in sd.items():
                nk = k
                if k.startswith("module."):
                    nk = k[len("module."):]
                new[nk] = v
            return new

        if isinstance(state, dict):
            state = _strip_module_prefix(state)
        else:
            print("[WARN] state is not a dict; attempting to load directly")

        # 载入并打印 missing/unexpected
        try:
            res = model.load_state_dict(state, strict=False)
            # PyTorch 返回 NamedTuple 或 dict 风格
            missing = getattr(res, "missing_keys", None) or res.get("missing_keys", None)
            unexpected = getattr(res, "unexpected_keys", None) or res.get("unexpected_keys", None)
        except Exception as e:
            print("[ERROR] model.load_state_dict failed:", e)
            # 再试一次在不抛出错误的情况下
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                pass
            missing, unexpected = None, None

        print(f"[INFO] Loaded fine-tuned weights from {args.ckpt}")
        if missing is not None:
            try:
                print(f"[INFO] missing keys: {len(missing)}")
                if len(missing) > 0:
                    print("  sample missing keys:", missing[:50])
            except Exception:
                pass
        if unexpected is not None:
            try:
                print(f"[INFO] unexpected keys: {len(unexpected)}")
                if len(unexpected) > 0:
                    print("  sample unexpected keys:", unexpected[:50])
            except Exception:
                pass

        # 额外检查：prompt_learner / LoRA 状态
        try:
            if hasattr(model, "prompt_learner"):
                ctx = getattr(model.prompt_learner, "ctx", None)
                if ctx is not None:
                    print("[INFO] prompt_learner.ctx found, shape:", tuple(ctx.shape), "norm:", float(ctx.detach().cpu().norm()))
                else:
                    print("[WARN] model.prompt_learner exists but ctx is None")
        except Exception as e:
            print("[WARN] Could not inspect prompt_learner:", e)

        lora_like = [n for n, _ in model.named_parameters() if "lora" in n.lower() or "out_adapter" in n.lower()]
        print(f"[INFO] model has {len(lora_like)} param names containing 'lora'/'out_adapter' (sample):", lora_like[:50])

    model.eval()
    return model.to(device)


# ---------- main inference loop ----------
@torch.no_grad()
def run_inference(args):
    # device / loader / model 构建
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 建议：可以在这里调整 build_loader 内的 num_workers / pin_memory（见后文）
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

    # 绘图/颜色/字体等
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

    # 状态变量
    idx = 0  # 全局文件序号（保持你原来的行为）
    sample_idx = 0  # 用于从 loader.dataset.entries 中读取原始 img_path
    total_imgs = 0
    total_time = 0.0

    # accumulators for final class-level stats
    global_stats = defaultdict(float)   # keys: dice_sum, iou_sum, prec_sum, rec_sum, f1_sum, img_count
    per_class_stats = defaultdict(lambda: defaultdict(float))
    per_class_counts = defaultdict(int)

    # tqdm 以“图片”为单位，总量为 dataset length
    dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else 0
    # 使用 total=dataset_len, unit='img', 那么每处理一张图就 pbar.update(1)
    pbar = tqdm(total=dataset_len, desc="Inference", unit="img", leave=True)

    # 主循环：单层循环，避免重复遍历 loader
    for images, masks, prompt_lists, class_names in loader:
        # images, masks 来自 DataLoader，images.shape[0] = batch_size (最后一个batch可能小于batch_size)
        images = images.to(device)
        # 先处理 prompts（用户自定义 或 MVPROMPTS 映射）
        custom_prompt: List[str] = []
        if getattr(args, "prompt", None):
            custom_prompt = [w.strip() for w in args.prompt.split(",") if w.strip()]

        if custom_prompt:
            prompt_lists = [custom_prompt for _ in prompt_lists]
        else:
            new_prompt_lists = []
            for i, cls_name in enumerate(class_names):
                cls_key = cls_name.lower().strip()
                mapped = MVPROMPTS.get(cls_key, None)
                if mapped:
                    new_prompt_lists.append([w.strip() for w in mapped.split(",") if w.strip()])
                else:
                    if prompt_lists and i < len(prompt_lists) and prompt_lists[i]:
                        new_prompt_lists.append(prompt_lists[i])
                    else:
                        new_prompt_lists.append(["anomaly"])
            prompt_lists = new_prompt_lists

        # 计时（确保测量真实 GPU time）
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        out = model(images, prompt_lists)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        batch_time = t1 - t0
        total_time += batch_time
        batch_imgs = images.size(0)
        total_imgs += batch_imgs

        pred_masks = out.get("pred_masks", None)
        if pred_masks is None:
            # 即使没有预测，也要推进 pbar 与 sample_idx，保持对齐
            pbar.update(batch_imgs)
            sample_idx += batch_imgs
            continue

        # 处理 pred_masks -> 概率
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        pred_masks = torch.sigmoid(pred_masks)
        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        gt = (masks > 0.5).float().to(device)

        # 逐张样本处理（可视化 / 保存 / 统计）
        for b in range(batch_imgs):
            cls_name = class_names[b]
            prompts_b = prompt_lists[b] if prompt_lists else []
            prompts_b = prompts_b if isinstance(prompts_b, list) else [prompts_b]

            pm = pred_masks[b]  # (C,H,W) or (1,H,W)
            gm = gt[b]  # (1,H,W)

            # 将多通道合并到单个布尔 mask (OR across channels)
            if pm.dim() == 3 and pm.shape[0] > 1:
                pm_bin = (pm > 0.5).float()
                pm_comb = pm_bin.max(dim=0).values  # (H,W)
            else:
                if pm.dim() == 3:
                    pm_single = pm.squeeze(0)
                else:
                    pm_single = pm
                pm_comb = (pm_single > 0.5).float()

            gm_single = gm.squeeze(0).float()

            # compute metrics
            metrics = safe_binary_metrics(pm_comb, gm_single)
            iou = float(metrics["iou"][0])
            dice = float(metrics["dice"][0])
            prec = float(metrics["precision"][0])
            rec = float(metrics["recall"][0])
            f1 = float(metrics["f1"][0])

            # accumulate global sums
            global_stats["iou_sum"] += iou
            global_stats["dice_sum"] += dice
            global_stats["prec_sum"] += prec
            global_stats["rec_sum"] += rec
            global_stats["f1_sum"] += f1
            global_stats["img_count"] += 1

            # accumulate per-class
            per_class_stats[cls_name]["iou_sum"] += iou
            per_class_stats[cls_name]["dice_sum"] += dice
            per_class_stats[cls_name]["prec_sum"] += prec
            per_class_stats[cls_name]["rec_sum"] += rec
            per_class_stats[cls_name]["f1_sum"] += f1
            per_class_counts[cls_name] += 1

            # --- 可视化: 排序 top_k + 阈值过滤（保持你之前的展示） ---
            # build candidates (channel-wise or single channel)
            masks_list = []
            prompts_for_candidates = []
            scores = []

            if pm.dim() == 3 and pm.shape[0] > 1:
                C = pm.shape[0]
                for c in range(C):
                    m = pm[c].detach().cpu().numpy()
                    masks_list.append(m)
                if prompts_b and len(prompts_b) == C:
                    prompts_for_candidates = prompts_b
                else:
                    if prompts_b:
                        prompts_for_candidates = (prompts_b * ((C // len(prompts_b)) + 1))[:C]
                    else:
                        prompts_for_candidates = [f"c{i}" for i in range(C)]
                scores = [float(np.mean(m)) for m in masks_list]
            else:
                if pm.dim() == 3:
                    pm_single = pm.squeeze(0)
                else:
                    pm_single = pm
                m = pm_single.detach().cpu().numpy()
                masks_list = [m]
                prompts_for_candidates = [prompts_b[0] if prompts_b else "prompt"]
                scores = [float(np.mean(m))]

            # filter and top_k
            conf_thresh = getattr(args, "conf_thresh", 0.3)
            candidates = []
            for s, m, p in zip(scores, masks_list, prompts_for_candidates):
                if s >= conf_thresh:
                    candidates.append((s, m, p))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_k = getattr(args, "top_k", 5)
                candidates = candidates[:top_k]

                masks_stack = np.stack([(c[1] > 0.5) for c in candidates])
                prompts_for_color = [c[2] for c in candidates]
                scores_for_prompt = [c[0] for c in candidates]

                colors = np.array([get_color_map(palette, p) for p in prompts_for_color], dtype=np.uint8)
                img_pil = to_pil(images[b].cpu())
                frame = np.array(img_pil.convert("RGB"), dtype=np.uint8)
                frame = draw_masks_to_frame(frame, masks_stack.astype(bool), colors)
                overlay_pil = Image.fromarray(frame)

                # legend
                draw = ImageDraw.Draw(overlay_pil)
                row_h = 12
                box_w = 10
                pad = 4
                legend_items = []
                max_w = 0
                for p, s in zip(prompts_for_color, scores_for_prompt):
                    text = f"{p} ({s:.2f})"
                    bbox = font.getbbox(text)
                    w = bbox[2] - bbox[0]
                    max_w = max(max_w, w)
                    legend_items.append((text, w))
                legend_h = row_h * len(prompts_for_color) + pad * 2
                legend_w = box_w + 4 + max_w + pad * 2
                legend_x = 5
                legend_y = overlay_pil.height - legend_h - 5
                draw.rectangle([legend_x, legend_y, legend_x + legend_w, legend_y + legend_h], fill=(0, 0, 0))
                for i_, (text, _) in enumerate(legend_items):
                    color = get_color_map(palette, prompts_for_color[i_])
                    y = legend_y + pad + i_ * row_h
                    draw.rectangle([legend_x + pad, y, legend_x + pad + box_w, y + box_w], fill=color)
                    draw.text((legend_x + pad + box_w + 4, y - 2), text, fill=(255, 255, 255), font=font)
            else:
                # no candidate above threshold: just use the original image as overlay_pil
                overlay_pil = to_pil(images[b].cpu())

            # --- 文件名生成（使用 dataset.entries 对齐） ---
            try:
                entry = loader.dataset.entries[sample_idx + b]
                raw_img_path = entry.img_path
            except Exception:
                raw_img_path = os.path.basename(f"{cls_name}_{idx}.png")

            norm_path = raw_img_path.replace("\\", "/")
            parts = norm_path.split("/")
            if len(parts) >= 2:
                defect = parts[-2]
                base_name = os.path.splitext(parts[-1])[0]
                filename = f"{defect}_{base_name}.png"
            else:
                filename = os.path.basename(norm_path)

            sample_dir = os.path.join(args.output_dir, cls_name)
            os.makedirs(sample_dir, exist_ok=True)
            overlay_path = os.path.join(sample_dir, filename)
            try:
                overlay_pil.save(overlay_path)
            except Exception:
                # fallback: save via BytesIO to avoid PIL blocking on some systems
                import io
                buf = io.BytesIO()
                overlay_pil.save(buf, format="PNG")
                with open(overlay_path, "wb") as f:
                    f.write(buf.getvalue())
            idx += 1

        # end of batch: advance sample_idx and update progress bar + fps
        sample_idx += batch_imgs
        # update pbar by number of images processed
        pbar.update(batch_imgs)
        # show realtime fps (images/sec)
        fps = (total_imgs / total_time) if total_time > 0 else 0.0
        # Use compact postfix to avoid heavy formatting cost
        pbar.set_postfix({"imgs": int(total_imgs), "fps": f"{fps:.2f}"})

    # finished all batches: close pbar
    pbar.close()

    # final summary: per-class averages
    final_img_count = int(global_stats.get("img_count", 0))
    final_speed = (total_imgs / total_time) if total_time > 0 else 0.0
    final_dice = (global_stats["dice_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_iou = (global_stats["iou_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_prec = (global_stats["prec_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_rec = (global_stats["rec_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_f1 = (global_stats["f1_sum"] / final_img_count) if final_img_count > 0 else 0.0

    print(f"[INFO] Inference done. images={total_imgs}, fps={final_speed:.2f}, avg_dice={final_dice:.4f}, avg_iou={final_iou:.4f}, precision={final_prec:.4f}, recall={final_rec:.4f}, f1={final_f1:.4f}")

    # per-class breakdown
    print("\nPer-class summary:")
    for cls, cnt in per_class_counts.items():
        if cnt == 0:
            continue
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

    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold below which masks are ignored")
    parser.add_argument("--top_k", type=int, default=5, help="Keep top-K masks per image by confidence")


    args = parser.parse_args()
    run_inference(args)
