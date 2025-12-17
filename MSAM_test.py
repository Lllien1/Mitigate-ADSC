# MSAM_test.py (fixed, confidence-head ranking)
# encoding: utf-8
import os, sys

PROJECT_ROOT = "/root/autodl-tmp/FiLo_plus/sam3"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import inspect
import json
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from dataset import MVTecMetaDataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official


# ---------- visualization helpers ----------
def get_color_map(palette: List[Tuple[int, int, int]], key: str) -> Tuple[int, int, int]:
    import hashlib
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(palette)
    return palette[idx]


def draw_masks_to_frame(frame: np.ndarray, masks: np.ndarray, colors: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    frame: (H,W,3) uint8
    masks: (N,H,W) bool
    colors: (N,3) uint8
    """
    if masks is None or len(masks) == 0:
        return frame
    out = frame.astype(np.float32)
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.sum() == 0:
            continue
        col = colors[i].astype(np.float32)
        out[m] = out[m] * (1 - alpha) + col * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def safe_binary_metrics(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6):
    """
    pred_bin, gt_bin : tensors of shape (H,W) or (N,H,W) with values 0/1
    Returns per-sample TP, FP, FN, IoU, Dice, Precision, Recall, F1
    """
    if pred_bin.dim() == 2:
        pred_bin = pred_bin.unsqueeze(0)
        gt_bin = gt_bin.unsqueeze(0)
    TP = (pred_bin * gt_bin).sum(dim=(1, 2)).float()
    FP = ((pred_bin == 1) & (gt_bin == 0)).sum(dim=(1, 2)).float()
    FN = ((pred_bin == 0) & (gt_bin == 1)).sum(dim=(1, 2)).float()
    union = TP + FP + FN
    iou = (TP / (union + eps)).cpu().numpy()
    dice = (2 * TP / (2 * TP + FP + FN + eps)).cpu().numpy()
    precision = (TP / (TP + FP + eps)).cpu().numpy()
    recall = (TP / (TP + FN + eps)).cpu().numpy()
    f1 = (2 * precision * recall / (precision + recall + 1e-12))
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


# ---------- model/dataloader helpers ----------
def _filter_kwargs_for_callable(func, kwargs: dict):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _infer_class_list(meta: dict) -> List[str]:
    """
    Support typical meta.json for MVTec:
      meta['train'][cls] / meta['test'][cls] are lists.
    """
    if not isinstance(meta, dict):
        return []
    if "train" in meta or "test" in meta:
        train_keys = list(meta.get("train", {}).keys()) if isinstance(meta.get("train", {}), dict) else []
        test_keys = list(meta.get("test", {}).keys()) if isinstance(meta.get("test", {}), dict) else []
        return sorted(list(set(train_keys + test_keys)))
    if "classes" in meta and isinstance(meta["classes"], list):
        return sorted(list(meta["classes"]))
    if "class_list" in meta and isinstance(meta["class_list"], list):
        return sorted(list(meta["class_list"]))
    return []


def build_loader(
    root: str,
    meta_path: str,
    mode: str,
    batch_size: int,
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
        n = len(batch[0])
        if n == 6:
            imgs, masks, prompt_lists, is_anomaly, class_names, specie_names = zip(*batch)
        elif n == 5:
            imgs, masks, prompt_lists, is_anomaly, class_names = zip(*batch)
            specie_names = [""] * len(batch)
        elif n == 4:
            imgs, masks, prompt_lists, class_names = zip(*batch)
            is_anomaly = [False] * len(batch)
            specie_names = [""] * len(batch)
        else:
            raise ValueError(f"Unexpected dataset item length={n} (expected 4/5/6)")
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        return imgs, masks, list(prompt_lists), list(is_anomaly), list(class_names), list(specie_names)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn
    )


def _normalize_bq(x: Optional[torch.Tensor], B: int, Q: int, device: torch.device) -> Optional[torch.Tensor]:
    """Normalize various SAM3 head outputs to (B,Q) on device."""
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.to(device)

    # common: (L,B,Q,1) or (B,L,Q,1) or (B,Q,1) or (B,Q)
    if x.dim() == 5:
        x = x[-1]
    if x.dim() == 4:
        if x.shape[0] == B:
            x = x[:, -1]
        else:
            x = x[-1]
    if x.dim() == 3:
        x = x.squeeze(-1)
    if x.dim() == 2:
        pass
    elif x.dim() == 1:
        x = x[:, None]
    else:
        x = x.view(B, -1)

    if x.shape[0] != B and x.shape[1] == B:
        x = x.t()
    if x.shape[0] != B:
        x = x[:B]

    if x.shape[1] < Q:
        x = F.pad(x, (0, Q - x.shape[1]), value=0.0)
    elif x.shape[1] > Q:
        x = x[:, :Q]
    return x


def _to_prob(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """If x looks like logits, sigmoid; else clamp."""
    if x is None:
        return None
    if (x.min() < 0.0) or (x.max() > 1.0):
        return torch.sigmoid(x)
    return x.clamp(0.0, 1.0)


def load_model(args, device: torch.device):
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

    with open(args.meta_path, "r") as f:
        meta = json.load(f)
    class_list = _infer_class_list(meta)
    if not class_list:
        raise ValueError("Could not infer class_list from meta.json. Expected keys: train/test -> class dicts.")
    args.class_list = class_list

    Constructor = FineTuneSAM3Official if args.use_official else FineTuneSAM3
    ctor_kwargs = _filter_kwargs_for_callable(Constructor, common_kwargs)
    ctor_kwargs.update(
        {
            "class_list": args.class_list,
            "prompt_learner_type": getattr(args, "prompt_learner_type", "perclass"),
            "num_templates": getattr(args, "num_templates", 4),
            "n_ctx": getattr(args, "n_ctx", 4),
        }
    )
    model = Constructor(**ctor_kwargs).to(device).eval()

    # optionally load fine-tuned weights
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[INFO] Loading fine-tuned checkpoint {args.ckpt} ...")
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded fine-tuned weights. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[INFO] No fine-tuned checkpoint provided (or path not found). Using base SAM3 weights.")

    return model


# ---------- main inference ----------
@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # If you only pass --save_dir (as in the original repo scripts), place visual outputs under that run folder.
    if getattr(args, "save_dir", None) and (not getattr(args, "output_dir", None) or args.output_dir == "./outputs"):
        args.output_dir = os.path.join(args.save_dir, "outputs")

    os.makedirs(args.output_dir, exist_ok=True)

    # palette / font
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    ]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    to_pil = transforms.ToPILImage()

    global_stats = dict(iou_sum=0.0, dice_sum=0.0, prec_sum=0.0, rec_sum=0.0, f1_sum=0.0, img_count=0)
    per_class_stats = {}
    per_class_counts = {}

    dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else 0
    pbar = tqdm(total=dataset_len, desc="Inference", unit="img", leave=True)

    total_time = 0.0
    total_imgs = 0
    idx = 0

    for images, masks, prompt_lists, is_anomaly, class_names, specie_names in loader:
        images = images.to(device)

        # optional global prompt override
        custom_prompt: List[str] = []
        if getattr(args, "prompt", None):
            custom_prompt = [w.strip() for w in args.prompt.split(",") if w.strip()]
        if custom_prompt:
            prompt_lists = [custom_prompt for _ in prompt_lists]

        # measure GPU time
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(images, prompt_lists, class_names)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_time = t1 - t0
        total_time += batch_time

        batch_imgs = images.size(0)
        total_imgs += batch_imgs

        pred_masks = out["pred_masks"]
        # official can return (L,B,Q,H,W)
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        pred_masks = torch.sigmoid(pred_masks)

        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        # --- (C) ranking: confidence heads ---
        B, Q = pred_masks.shape[0], pred_masks.shape[1]
        presence_bq = _normalize_bq(out.get("presence_logit", None), B, Q, device=device)
        iou_bq = _normalize_bq(out.get("iou_predictions", None), B, Q, device=device)
        presence_prob = _to_prob(presence_bq)
        iou_prob = _to_prob(iou_bq)
        if presence_prob is not None and iou_prob is not None:
            query_scores = presence_prob * iou_prob
        elif presence_prob is not None:
            query_scores = presence_prob
        elif iou_prob is not None:
            query_scores = iou_prob
        else:
            query_scores = None  # fallback to mean-mask

        gt = (masks > 0.5).float().to(device)

        for b in range(batch_imgs):
            cls_name = class_names[b]
            if cls_name not in per_class_stats:
                per_class_stats[cls_name] = dict(iou_sum=0.0, dice_sum=0.0, prec_sum=0.0, rec_sum=0.0, f1_sum=0.0)
                per_class_counts[cls_name] = 0

            prompts_b = prompt_lists[b] if prompt_lists else []
            prompts_b = prompts_b if isinstance(prompts_b, list) else [prompts_b]
            base_prompt = prompts_b[0] if prompts_b else cls_name

            pm = pred_masks[b]  # (Q,H,W)
            gm = gt[b].squeeze(0).float()  # (H,W)

            # candidates
            conf_thresh = float(getattr(args, "conf_thresh", 0.3))
            top_k = int(getattr(args, "top_k", 5))

            masks_list = []
            labels = []
            scores = []

            scores_vec = None
            if query_scores is not None:
                scores_vec = query_scores[b].view(-1)

            for q in range(pm.shape[0]):
                m_q = pm[q].detach().cpu().numpy()
                masks_list.append(m_q)
                if scores_vec is not None and q < scores_vec.numel():
                    s_q = float(scores_vec[q].detach().cpu().item())
                else:
                    s_q = float(np.mean(m_q))
                scores.append(s_q)
                labels.append(f"q{q}:{base_prompt}")

            cand = [(s, m, lab) for s, m, lab in zip(scores, masks_list, labels) if s >= conf_thresh]
            cand.sort(key=lambda x: x[0], reverse=True)
            cand = cand[:top_k] if top_k > 0 else cand

            if cand:
                masks_stack = np.stack([(c[1] > 0.5) for c in cand]).astype(bool)
                pm_comb = torch.from_numpy(masks_stack.max(axis=0).astype(np.float32)).to(device)
            else:
                masks_stack = None
                pm_comb = torch.zeros_like(gm).to(device)

            metrics = safe_binary_metrics(pm_comb, gm)
            iou = float(metrics["iou"][0])
            dice = float(metrics["dice"][0])
            prec = float(metrics["precision"][0])
            rec = float(metrics["recall"][0])
            f1 = float(metrics["f1"][0])

            global_stats["iou_sum"] += iou
            global_stats["dice_sum"] += dice
            global_stats["prec_sum"] += prec
            global_stats["rec_sum"] += rec
            global_stats["f1_sum"] += f1
            global_stats["img_count"] += 1

            per_class_stats[cls_name]["iou_sum"] += iou
            per_class_stats[cls_name]["dice_sum"] += dice
            per_class_stats[cls_name]["prec_sum"] += prec
            per_class_stats[cls_name]["rec_sum"] += rec
            per_class_stats[cls_name]["f1_sum"] += f1
            per_class_counts[cls_name] += 1

            # visualization
            img_pil = to_pil(images[b].cpu())
            frame = np.array(img_pil.convert("RGB"), dtype=np.uint8)

            if cand and masks_stack is not None:
                prompts_for_color = [c[2] for c in cand]
                scores_for_prompt = [c[0] for c in cand]
                colors = np.array([get_color_map(palette, p) for p in prompts_for_color], dtype=np.uint8)
                frame = draw_masks_to_frame(frame, masks_stack, colors)
                overlay_pil = Image.fromarray(frame)

                draw = ImageDraw.Draw(overlay_pil)
                row_h, box_w, pad = 12, 10, 4
                legend_items = []
                max_w = 0
                for p, s in zip(prompts_for_color, scores_for_prompt):
                    txt = f"{p} ({s:.2f})"
                    bbox = font.getbbox(txt)
                    w = bbox[2] - bbox[0]
                    max_w = max(max_w, w)
                    legend_items.append(txt)
                legend_h = row_h * len(legend_items) + pad * 2
                legend_w = box_w + 4 + max_w + pad * 2
                x0, y0 = 5, 5
                draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h], fill=(0, 0, 0, 160))
                for i_row, txt in enumerate(legend_items):
                    y = y0 + pad + i_row * row_h
                    col = tuple(colors[i_row].tolist())
                    draw.rectangle([x0 + pad, y, x0 + pad + box_w, y + box_w], fill=col)
                    draw.text((x0 + pad + box_w + 4, y - 1), txt, font=font, fill=(255, 255, 255))
            else:
                overlay_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(overlay_pil)
                draw.text((5, 5), f"no mask >= {conf_thresh:.2f}", font=font, fill=(255, 0, 0))

            # save
            sample_dir = os.path.join(args.output_dir, cls_name)
            os.makedirs(sample_dir, exist_ok=True)
            filename = f"{idx:06d}_dice{dice:.3f}_iou{iou:.3f}.png"
            overlay_path = os.path.join(sample_dir, filename)
            overlay_pil.save(overlay_path)
            idx += 1

        pbar.update(batch_imgs)
        fps = (total_imgs / total_time) if total_time > 0 else 0.0
        pbar.set_postfix({"imgs": int(total_imgs), "fps": f"{fps:.2f}"})

    pbar.close()

    # summary
    final_img_count = int(global_stats.get("img_count", 0))
    final_speed = (total_imgs / total_time) if total_time > 0 else 0.0
    final_dice = (global_stats["dice_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_iou = (global_stats["iou_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_prec = (global_stats["prec_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_rec = (global_stats["rec_sum"] / final_img_count) if final_img_count > 0 else 0.0
    final_f1 = (global_stats["f1_sum"] / final_img_count) if final_img_count > 0 else 0.0

    print("\n========== Global Summary ==========")
    print(f"images={final_img_count}, fps={final_speed:.2f}")
    print(f"dice={final_dice:.4f}, iou={final_iou:.4f}, prec={final_prec:.4f}, rec={final_rec:.4f}, f1={final_f1:.4f}")

    print("\n========== Per-Class Summary ==========")
    for cls, cnt in sorted(per_class_counts.items(), key=lambda x: x[0]):
        if cnt <= 0:
            continue
        iou = per_class_stats[cls]["iou_sum"] / cnt
        dice = per_class_stats[cls]["dice_sum"] / cnt
        prec = per_class_stats[cls]["prec_sum"] / cnt
        rec = per_class_stats[cls]["rec_sum"] / cnt
        f1 = per_class_stats[cls]["f1_sum"] / cnt
        print(f"  {cls}: n={cnt}, dice={dice:.4f}, iou={iou:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MSAM test (confidence-head ranking)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # prompts
    parser.add_argument("--prompt", type=str, default=None, help="override prompt list for ALL samples, comma-separated")

    # model
    parser.add_argument("--use_official", action="store_true")
    parser.add_argument("--sam3_ckpt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="fine-tuned .pth to load")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")

    # prompt learner config (kept compatible)
    parser.add_argument("--prompt_learner_type", type=str, default="perclass")
    parser.add_argument("--num_templates", type=int, default=4)
    parser.add_argument("--n_ctx", type=int, default=4)

    # dataset split from test defects
    parser.add_argument("--train_from_test", action="store_true")
    parser.add_argument("--specie_split_ratio", type=float, default=0.8)
    parser.add_argument("--specie_split_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None, help="run folder to read/write per-specie splits")

    # parallel lora args
    parser.add_argument("--enable_parallel_lora", action="store_true")
    parser.add_argument("--parallel_lora_rank", type=int, default=16)
    parser.add_argument("--parallel_lora_alpha", type=float, default=None)

    # ranking thresholds
    parser.add_argument("--conf_thresh", type=float, default=0.6, help="confidence threshold on (presence*iou)")
    parser.add_argument("--top_k", type=int, default=5, help="keep top-k queries per image")

    args = parser.parse_args()
    run_inference(args)
