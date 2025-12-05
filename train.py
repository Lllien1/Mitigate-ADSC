import argparse
import os
import sys
from datetime import datetime
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ensure local sam3 package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "sam3"))

from dataset import MVTecMetaDataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary Dice loss with smoothing; robust to empty masks."""
    pred = pred.sigmoid()
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_eff = (2 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1 - dice_eff.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.where(target == 1, prob, 1 - prob)
    loss = ce * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        loss = alpha_t * loss
    return loss.mean()


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

    prompt_and_lora: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in n or "prompt_learner" in n:
            prompt_and_lora.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": prompt_and_lora, "lr": args.lr_prompt},
            {"params": other_params, "lr": args.lr_main},
        ],
        weight_decay=1e-4,
    )

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

            out = model(images, prompt_lists)
            pred_masks = out["pred_masks"]
            if pred_masks is None:
                raise RuntimeError("Segmentation head did not return pred_masks.")

            if pred_masks.dim() == 5:
                pred_masks = pred_masks[-1]
            mask_for_iou = pred_masks
            if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
                pred_masks = pred_masks.max(dim=1, keepdim=True).values
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

            loss_focal = focal_loss(pred_masks, masks)
            loss_dice = dice_loss(pred_masks, masks)
            loss_iou = torch.tensor(0.0, device=device)
            iou_pred = out.get("iou_predictions", None)
            if iou_pred is not None:
                if iou_pred.dim() == 3:
                    iou_pred = iou_pred[-1]
                if mask_for_iou.dim() == 3:
                    mask_for_iou = mask_for_iou.unsqueeze(1)
                mask_prob = torch.sigmoid(mask_for_iou)
                prob_flat = mask_prob.flatten(2)
                target_flat = masks.unsqueeze(1).flatten(2)
                intersection = (prob_flat * target_flat).sum(dim=-1)
                union = prob_flat.sum(dim=-1) + target_flat.sum(dim=-1) - intersection
                # Avoid degenerate all-zero cases: union==0 -> IoU=0
                true_iou = torch.where(
                    union > 0, intersection / (union + 1e-6), torch.zeros_like(union)
                )
                if iou_pred.shape[1] == 1 and true_iou.shape[1] > 1:
                    iou_pred = iou_pred.expand(true_iou.shape[0], true_iou.shape[1])
                loss_iou = F.mse_loss(iou_pred, true_iou)
            loss = args.loss_alpha * loss_focal + args.loss_beta * loss_dice + args.loss_gamma * loss_iou

            if not torch.isfinite(loss):
                print(f"[WARN] Skip batch with non-finite loss (loss={loss.item()}, focal={loss_focal.item()}, dice={loss_dice.item()}, iou={loss_iou.item()})")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
