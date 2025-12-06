import argparse
import os
import sys
from datetime import datetime
from typing import List

# ensure local sam3 package is importable before importing sam3.*
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sam3.train.matcher import BinaryHungarianMatcher
from sam3.train.loss.loss_fns import sigmoid_focal_loss as sam_sigmoid_focal_loss
from sam3.train.loss.loss_fns import dice_loss as sam_dice_loss
from torch.cuda.amp import autocast, GradScaler

from dataset import MVTecMetaDataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official


def mask_to_box(mask: torch.Tensor):
    """Convert a binary mask (H,W) to cxcywh normalized box; return None if empty."""
    ys, xs = torch.where(mask.bool())
    if ys.numel() == 0:
        return None
    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    H, W = mask.shape
    cx = (x0 + x1) / 2.0 / W
    cy = (y0 + y1) / 2.0 / H
    w = (x1 - x0 + 1) / W
    h = (y1 - y0 + 1) / H
    return torch.tensor([cx, cy, w, h], dtype=torch.float32, device=mask.device)


def build_batched_targets_from_binary_masks(masks: torch.Tensor):
    """
    Treat each binary mask as a single instance.
    masks: (B, H, W) or (B, 1, H, W) or (B, C, H, W)
    """
    if masks.dim() == 4:
        # If channel exists, collapse to single channel (max over C)
        masks = masks.max(dim=1).values
    elif masks.dim() != 3:
        raise ValueError(f"Expected masks shape (B,H,W) or (B,C,H,W), got {masks.shape}")
    B, H, W = masks.shape
    boxes, labels, segments, num_boxes = [], [], [], []
    for b in range(B):
        m = masks[b].bool()
        if m.sum() == 0:
            num_boxes.append(0)
            continue
        box = mask_to_box(m)
        if box is None:
            num_boxes.append(0)
            continue
        boxes.append(box)
        labels.append(torch.tensor(1, dtype=torch.long, device=m.device))
        segments.append(m.to(torch.float32))
        num_boxes.append(1)
    if len(boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=masks.device),
            "labels": torch.zeros((0,), dtype=torch.long, device=masks.device),
            "segments": torch.zeros((0, H, W), dtype=torch.float32, device=masks.device),
            "num_boxes": torch.tensor(num_boxes, dtype=torch.long, device=masks.device),
        }
    return {
        "boxes": torch.stack(boxes, dim=0),
        "labels": torch.stack(labels, dim=0),
        "segments": torch.stack(segments, dim=0),
        "num_boxes": torch.tensor(num_boxes, dtype=torch.long, device=masks.device),
    }


def convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B, device):
    """Convert flat matcher output to list of (src,tgt) per batch."""
    indices = []
    for b in range(B):
        mask = (batch_idx == b)
        if mask.sum() == 0:
            indices.append(
                (
                    torch.zeros((0,), dtype=torch.long, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device),
                )
            )
            continue
        indices.append((src_idx[mask].to(device), tgt_idx[mask].to(device)))
    return indices


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.where(target == 1, prob, 1 - prob)
    loss = ce * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        loss = alpha_t * loss
    return loss.mean()


def pairwise_iou(preds_sigmoid: torch.Tensor, gts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute pairwise IoU between Q preds and G GT masks on CPU."""
    Q = preds_sigmoid.shape[0]
    G = gts.shape[0]
    preds_flat = preds_sigmoid.view(Q, -1)
    gts_flat = gts.view(G, -1)
    inter = torch.einsum("qd,gd->qg", preds_flat, gts_flat)
    sum_preds = preds_flat.sum(dim=1, keepdim=True)
    sum_gts = gts_flat.sum(dim=1, keepdim=True).t()
    union = sum_preds + sum_gts - inter + eps
    return inter / union


def match_one_image(preds_logits: torch.Tensor, gt_masks: torch.Tensor):
    """Hungarian match Q predicted masks to G GT masks (binary)."""
    if gt_masks.numel() == 0 or gt_masks.sum() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    preds_prob = preds_logits.sigmoid().detach().cpu()
    gts = gt_masks.detach().cpu()
    iou = pairwise_iou(preds_prob, gts)  # Q x G
    cost = 1.0 - iou.numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind.astype(np.int64), col_ind.astype(np.int64)


def build_dataloaders(
    root: str,
    meta_path: str,
    mode: str = "test",
    k_shot: int = 0,
    obj_name: str = None,
    aug_rate: float = 0.0,
    batch_size: int = 2,
    balance: bool = False,
):
    ds = MVTecMetaDataset(
        root=root,
        meta_path=meta_path,
        mode=mode,
        k_shot=k_shot,
        obj_name=obj_name,
        aug_rate=aug_rate,
    )

    sampler = None
    if balance:
        labels = [int(is_anomaly) for _, _, _, is_anomaly, _ in ds]
        class_counts = torch.tensor(
            [(1 - torch.tensor(labels)).sum(), torch.tensor(labels).sum()],
            dtype=torch.float,
        )
        class_counts = torch.clamp(class_counts, min=1.0)
        weight = 1.0 / class_counts
        samples_weight = torch.tensor([weight[l] for l in labels])
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, num_samples=len(samples_weight), replacement=True
        )

    def collate_fn(batch):
        imgs, masks, prompt_lists, is_anomaly, class_names = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        is_anomaly_t = torch.tensor(is_anomaly, dtype=torch.bool)
        return imgs, masks, list(prompt_lists), is_anomaly_t, list(class_names)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
    )


def load_sam3_checkpoint(model: torch.nn.Module, ckpt_path: str):
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("Please install safetensors to load .safetensors weights: pip install safetensors") from e
        state = load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    mapped = {}
    for k, v in state.items():
        if k.startswith("detector."):
            nk = k.replace("detector.", "")
            mapped[nk] = v
        elif k.startswith("backbone."):
            mapped[k.replace("backbone.", "")] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"[INFO] Loaded SAM3 ckpt {ckpt_path}, mapped={len(mapped)}, missing={len(missing)}, unexpected={len(unexpected)}")
    return missing, unexpected


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.use_official:
        model = FineTuneSAM3Official(
            bpe_path=args.bpe_path,
            sam3_ckpt=args.sam3_ckpt,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            device=device,
        )
    else:
        model = FineTuneSAM3(
            bpe_path=args.bpe_path,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            device=device,
        )

    if args.sam3_ckpt and os.path.exists(args.sam3_ckpt):
        load_sam3_checkpoint(model, args.sam3_ckpt)

    dataloader = build_dataloaders(
        root=args.data_root,
        meta_path=args.meta_path or os.path.join(args.data_root, "meta.json"),
        mode=args.mode,
        k_shot=args.k_shot,
        obj_name=args.obj_name,
        aug_rate=args.aug_rate,
        batch_size=args.batch_size,
        balance=args.balance,
    )

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, run_name)
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] run_name={run_name}, log_dir={log_dir}, save_dir={save_dir}")

    # Freeze everything except LoRA/prompt params
    for n, p in model.named_parameters():
        if ("lora" in n) or ("prompt_learner" in n) or ("prompt" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    prompt_and_lora: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in n or "prompt_learner" in n:
            prompt_and_lora.append(p)
        else:
            other_params.append(p)
    print(f"[INFO] trainable params: prompt/LoRA={len(prompt_and_lora)}, others={len(other_params)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": prompt_and_lora, "lr": args.lr_prompt},
            {"params": other_params, "lr": args.lr_main},
        ],
        weight_decay=1e-4,
    )
    # 新增：AMP 的 GradScaler（只在 CUDA 下启用）
    scaler = GradScaler(enabled=(device.type == "cuda"))

    matcher = BinaryHungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)

    model.train()
    best_loss = float("inf")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        running_loss = 0.0
        running_steps = 0
        for step, batch in enumerate(pbar):
            images, masks, prompt_lists, is_anomaly, class_names = batch
            images = images.to(device)
            masks = masks.to(device)

            # ===== AMP autocast: forward + loss 计算都放在半精度上下文 =====
            with autocast(enabled=(device.type == "cuda")):
                out = model(images, prompt_lists)
                pred_masks = out["pred_masks"]
                if pred_masks is None:
                    raise RuntimeError("Segmentation head did not return pred_masks.")

                if pred_masks.dim() == 5:
                    pred_masks = pred_masks[-1]
                mask_for_iou = pred_masks
                if pred_masks.shape[-2:] != masks.shape[-2:]:
                    pred_masks = torch.nn.functional.interpolate(
                        pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                if mask_for_iou.shape[-2:] != masks.shape[-2:]:
                    mask_for_iou = torch.nn.functional.interpolate(
                        mask_for_iou, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                pred_masks = pred_masks.clamp(min=-20.0, max=20.0)
                pred_masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=0.0, neginf=0.0)
                mask_for_iou = torch.nan_to_num(mask_for_iou, nan=0.0, posinf=0.0, neginf=0.0)
                masks = torch.nan_to_num(masks, nan=0.0, posinf=0.0, neginf=0.0)
                masks = (masks > 0.5).float()

                # Build targets and run official matcher
                targets = build_batched_targets_from_binary_masks(masks)
                pred_logits = out.get("presence_logit", None)
                if pred_logits is None:
                    pred_logits = torch.zeros((pred_masks.shape[0], pred_masks.shape[1], 1), device=device)
                else:
                    if pred_logits.dim() == 4:
                        pred_logits = pred_logits[-1]
                    if pred_logits.dim() == 1:
                        pred_logits = pred_logits.unsqueeze(-1)
                    if pred_logits.dim() == 2:
                        pred_logits = pred_logits.unsqueeze(-1)
                    if pred_logits.shape[1] != pred_masks.shape[1]:
                        pred_logits = pred_logits.unsqueeze(1).expand(-1, pred_masks.shape[1], -1)
                pred_boxes = out.get("reference_boxes", None)
                if pred_boxes is None:
                    pred_boxes = torch.zeros((pred_masks.shape[0], pred_masks.shape[1], 4), device=device)
                else:
                    if pred_boxes.dim() == 4:
                        pred_boxes = pred_boxes[-1]

                matcher_outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                batch_idx, src_idx, tgt_idx = matcher(matcher_outputs, targets)
                out["indices"] = (
                    convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B=pred_masks.shape[0], device=device)
                    if tgt_idx is not None
                    else []
                )

                if tgt_idx is None or tgt_idx.numel() == 0:
                    loss_focal = torch.tensor(0.0, device=device)
                    loss_dice = torch.tensor(0.0, device=device)
                else:
                    tgt_masks = targets["segments"]
                    pred_matched = pred_masks[batch_idx, src_idx]
                    if pred_matched.shape[-2:] != tgt_masks.shape[-2:]:
                        pred_matched = torch.nn.functional.interpolate(
                            pred_matched.unsqueeze(1),
                            size=tgt_masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    tgt_matched = tgt_masks[tgt_idx]
                    num_boxes = targets["num_boxes"].sum().float().clamp(min=1.0)
                    loss_focal = sam_sigmoid_focal_loss(
                        pred_matched, tgt_matched, num_boxes, alpha=0.25, gamma=2.0, loss_on_multimask=False, triton=False
                    )
                    loss_dice = sam_dice_loss(
                        pred_matched, tgt_matched, num_boxes, loss_on_multimask=False, reduce=True
                    )
                    # Background loss for unmatched queries (encourage background on negatives)
                    all_q = torch.arange(pred_masks.shape[1], device=device)
                    loss_focal_bg = torch.tensor(0.0, device=device)
                    loss_dice_bg = torch.tensor(0.0, device=device)
                    for b in range(pred_masks.shape[0]):
                        mask_b = batch_idx == b
                        matched_q = src_idx[mask_b] if mask_b.any() else torch.zeros(0, device=device, dtype=torch.long)
                        unmatched_q = all_q if matched_q.numel() == 0 else all_q[~torch.isin(all_q, matched_q)]
                        if unmatched_q.numel() == 0:
                            continue
                        preds_bg = pred_masks[b, unmatched_q]
                        zeros = torch.zeros_like(preds_bg)
                        nb = torch.tensor(float(unmatched_q.numel()), device=device).clamp(min=1.0)
                        loss_focal_bg += sam_sigmoid_focal_loss(
                            preds_bg, zeros, nb, alpha=0.25, gamma=2.0, loss_on_multimask=False, triton=False
                        )
                        loss_dice_bg += sam_dice_loss(
                            preds_bg, zeros, nb, loss_on_multimask=False, reduce=True
                        )
                    loss_focal = loss_focal + 0.5 * loss_focal_bg
                    loss_dice = loss_dice + 0.5 * loss_dice_bg

                loss_iou = torch.tensor(0.0, device=device)
                iou_pred = out.get("iou_predictions", None)
                if iou_pred is not None:
                    if iou_pred.dim() == 3:
                        iou_pred = iou_pred[-1]
                    with torch.no_grad():
                        if mask_for_iou.dim() == 3:
                            mask_for_iou = mask_for_iou.unsqueeze(1)
                        target_size = (256, 256)
                        mask_for_iou_down = torch.nn.functional.interpolate(
                            mask_for_iou, size=target_size, mode="bilinear", align_corners=False
                        )
                        gt_masks = masks
                        if gt_masks.dim() == 3:
                            gt_masks = gt_masks.unsqueeze(1)
                        gt_masks_down = torch.nn.functional.interpolate(
                            gt_masks, size=target_size, mode="nearest"
                        )
                        mask_prob = torch.sigmoid(mask_for_iou_down)
                        prob_flat = mask_prob.flatten(2)
                        target_flat = gt_masks_down.flatten(2)
                        intersection = (prob_flat * target_flat).sum(dim=-1)
                        union = prob_flat.sum(dim=-1) + target_flat.sum(dim=-1) - intersection
                        true_iou = torch.where(
                            union > 0, intersection / (union + 1e-6), torch.zeros_like(union)
                        )
                    if iou_pred.shape[1] == 1 and true_iou.shape[1] > 1:
                        iou_pred = iou_pred.expand(true_iou.shape[0], true_iou.shape[1])
                    loss_iou = F.mse_loss(iou_pred, true_iou)

                loss = args.loss_alpha * loss_focal + args.loss_beta * loss_dice + args.loss_gamma * loss_iou
            # ===== AMP autocast 结束 =====

            if not torch.isfinite(loss):
                print(f"[WARN] Skip batch with non-finite loss (loss={loss.item()}, focal={loss_focal.item()}, dice={loss_dice.item()}, iou={loss_iou.item()})")
                continue

            optimizer.zero_grad()
            # 使用 GradScaler 做 backward + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            pbar.set_postfix(
                loss=loss.item(),
                focal=loss_focal.item(),
                dice=loss_dice.item(),
            )

            global_step = epoch * len(dataloader) + step
            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/focal", loss_focal.item(), global_step)
            writer.add_scalar("loss/dice", loss_dice.item(), global_step)
            writer.add_scalar("loss/iou", loss_iou.item(), global_step)

            running_loss += loss.item()
            running_steps += 1

        if running_steps > 0:
            avg_loss = running_loss / running_steps
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(save_dir, "sam3_peft_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[INFO] Epoch {epoch+1}: new best avg_loss {avg_loss:.4f}, saved to {ckpt_path}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root of MVTec-AD dataset.")
    parser.add_argument("--meta_path", type=str, default=None, help="Path to meta.json (defaults to <data_root>/meta.json).")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"], help="Split to load.")
    parser.add_argument("--k_shot", type=int, default=0, help="K-shot for train/train_all.")
    parser.add_argument("--obj_name", type=str, default=None, help="Class name for mode=train.")
    parser.add_argument("--aug_rate", type=float, default=0.0, help="Mosaic augmentation probability.")
    parser.add_argument("--bpe_path", type=str, default=None, help="Path to BPE vocab (defaults to sam3/assets/bpe_simple_vocab_16e6.txt.gz).")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_prompt", type=float, default=5e-4)
    parser.add_argument("--lr_main", type=float, default=5e-5)
    parser.add_argument("--loss_alpha", type=float, default=5.0, help="Weight for focal loss.")
    parser.add_argument("--loss_beta", type=float, default=1.0, help="Weight for dice loss.")
    parser.add_argument("--loss_gamma", type=float, default=1.0, help="Weight for IoU regression loss.")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard base log directory.")
    parser.add_argument("--save_dir", type=str, default="./ckpt", help="Base directory to save checkpoints.")
    parser.add_argument("--sam3_ckpt", type=str, default=None, help="Path to pretrained SAM3 checkpoint to load.")
    parser.add_argument("--use_official", action="store_true", help="Use official builder + checkpoint before PEFT.")
    parser.add_argument("--balance", action="store_true", help="Enable anomaly/non-anomaly weighted sampler.")
    args = parser.parse_args()
    main(args)
