import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
from datetime import datetime
from typing import List, Optional

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

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import json


def setup_distributed(args):
    """
    初始化分布式（在使用 torch.distributed.run 启动时，环境变量会提供 LOCAL_RANK, RANK, WORLD_SIZE）
    运行前无需手动传 local_rank，torch.distributed.run 会设置。
    """
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.world_size > 1:
        # 使用 NCCL 后端（推荐）
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.distributed = True
    else:
        args.distributed = False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def collate_fn(batch):
    imgs, masks, prompts, anomalies, classes = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack(masks, dim=0)
    return imgs, masks, list(prompts), list(anomalies), list(classes)


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

def build_list_targets_from_binary_masks(masks: torch.Tensor):
    """
    返回 list[dict]，便于直接传给 matcher（每张图一个 dict）。
    每个 dict 的 keys: "boxes" (N,4), "labels" (N,), "segments" (N,H,W)
    如果一张图没有目标，返回 boxes = zeros((0,4)), labels=zeros((0,)), segments=zeros((0,H,W))
    """
    if masks.dim() == 4:
        masks = masks.max(dim=1).values
    elif masks.dim() != 3:
        raise ValueError(f"Expected masks shape (B,H,W) or (B,C,H,W), got {masks.shape}")
    B, H, W = masks.shape
    out = []
    for b in range(B):
        m = masks[b].bool()
        if m.sum() == 0:
            out.append({
                "boxes": torch.zeros((0,4), dtype=torch.float32, device=masks.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=masks.device),
                "segments": torch.zeros((0, H, W), dtype=torch.float32, device=masks.device),
            })
            continue
        box = mask_to_box(m)  # 已经归一化 cxcywh
        if box is None:
            out.append({
                "boxes": torch.zeros((0,4), dtype=torch.float32, device=masks.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=masks.device),
                "segments": torch.zeros((0, H, W), dtype=torch.float32, device=masks.device),
            })
            continue
        out.append({
            "boxes": box.unsqueeze(0),  # (1,4)
            "labels": torch.tensor([1], dtype=torch.long, device=masks.device),
            "segments": m.to(torch.float32).unsqueeze(0),  # (1,H,W)
        })
    return out

def convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B: int, device, targets_num_boxes=None):
    """
    Convert matcher outputs (batch_idx, src_idx, tgt_idx) into per-image lists:
      [(src_q_tensor, tgt_q_tensor), ...] of length B.

    Robust behavior:
      - If tgt_idx is None: reconstruct tgt_q per-image using targets_num_boxes when available.
        * If targets_num_boxes[b] == 1, and some src matched for image b, we set tgt_q to zeros of same length.
        * If targets_num_boxes[b] > 1, we assign tgt_q = 0 repeated (fallback) and print a warning.
      - If tgt_idx is provided and is flattened global indices, convert to local per-image indices
        using cumulative sums of targets_num_boxes.
      - Returns list of tuples (src_q, tgt_q) for each image.
    """
    B = int(B)
    out = []
    # prepare empty default per image
    for _ in range(B):
        out.append((torch.zeros((0,), dtype=torch.long, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device)))

    if batch_idx is None or src_idx is None:
        return out

    # ensure tensors on device/long
    batch_idx = batch_idx.to(device).long()
    src_idx = src_idx.to(device).long()

    # if targets_num_boxes is None, we'll only fill srcs, and leave tgts empty when not present
    if tgt_idx is None:
        # group src_idx by batch index
        for b in range(B):
            mask = (batch_idx == b)
            if mask.any():
                srcs = src_idx[mask]
                # default: no tgt info, try to reconstruct using targets_num_boxes
                if targets_num_boxes is not None:
                    nb = int(targets_num_boxes[b])
                    if nb == 0:
                        tgts = torch.zeros((0,), dtype=torch.long, device=device)
                    elif nb == 1:
                        # assign the only GT (index 0) for all matched srcs
                        tgts = torch.zeros((srcs.numel(),), dtype=torch.long, device=device)
                    else:
                        # multiple GTs for this image but no mapping info -> fallback: assign 0 and warn
                        print(f"[WARN] convert_matcher_output_to_indices: image {b} has {nb} GTs but matcher returned no tgt_idx. Falling back to tgt=0 for all matched srcs.")
                        tgts = torch.zeros((srcs.numel(),), dtype=torch.long, device=device)
                else:
                    # no info about targets; set empty tgt
                    tgts = torch.zeros((0,), dtype=torch.long, device=device)
                out[b] = (srcs, tgts)
            else:
                out[b] = (torch.zeros((0,), dtype=torch.long, device=device),
                          torch.zeros((0,), dtype=torch.long, device=device))
        return out

    # tgt_idx provided: ensure on device and long
    tgt_idx = tgt_idx.to(device).long()

    # If targets_num_boxes provided, compute cumulative offsets for flattened -> local index mapping
    cum = None
    if targets_num_boxes is not None:
        # ensure list/array of ints
        tnb = [int(x) for x in targets_num_boxes]
        cum = [0]
        for nb in tnb:
            cum.append(cum[-1] + nb)
        # cum length B+1
    # Now iterate matches and allocate per-image lists
    # We assume batch_idx, src_idx, tgt_idx are parallel lists of same length
    assert batch_idx.numel() == src_idx.numel() == tgt_idx.numel(), "Matcher outputs lengths mismatch"

    # We'll accumulate in python lists, then convert to tensors
    src_lists = [[] for _ in range(B)]
    tgt_lists = [[] for _ in range(B)]
    for i in range(batch_idx.numel()):
        b = int(batch_idx[i].item())
        s = int(src_idx[i].item())
        tg = int(tgt_idx[i].item())
        if cum is not None:
            # map flattened tg to local index for image b: local = tg - cum[b]
            local = tg - cum[b]
            if local < 0 or local >= (cum[b+1] - cum[b]):
                # Something inconsistent: warn and skip
                print(f"[WARN] convert_matcher_output_to_indices: flattened tgt {tg} maps to local {local} out of range for image {b} (num_boxes={cum[b+1]-cum[b]}). Skipping this match.")
                continue
            tgt_local = int(local)
        else:
            # No targets_num_boxes: we cannot reliably map; keep tgt as is (but will be inconsistent)
            tgt_local = tg
        src_lists[b].append(s)
        tgt_lists[b].append(tgt_local)

    # convert lists to tensors
    for b in range(B):
        if len(src_lists[b]) == 0:
            out[b] = (torch.zeros((0,), dtype=torch.long, device=device),
                      torch.zeros((0,), dtype=torch.long, device=device))
        else:
            out[b] = (torch.tensor(src_lists[b], dtype=torch.long, device=device),
                      torch.tensor(tgt_lists[b], dtype=torch.long, device=device))
    return out




def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.where(target == 1, prob, 1 - prob)
    loss = ce * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        loss = alpha_t * loss
    return loss.mean()

def normalize_presence_logits(pred_logits, B, Q, device):
    """
    Normalize various possible shapes of presence_logit into (B, Q, 1).
    Handles shapes like: (L,B,Q,1), (B,L,Q,1), (B,Q,1), (B,1,1,1), (B,Q), (B,1), (B,)
    """
    if pred_logits is None:
        return torch.zeros((B, Q, 1), device=device, dtype=torch.float32)

    # Move to device
    pred_logits = pred_logits.to(device)

    # If there is a leading layers dimension (4D),
    # try common conventions: (B, L, Q, 1) or (L, B, Q, 1)
    if pred_logits.dim() == 4:
        # case (B, L, Q, 1)
        if pred_logits.shape[0] == B and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits[:, -1, :, :]  # take last layer along dim=1 -> (B,Q,1)
        # case (L, B, Q, 1)
        elif pred_logits.shape[1] == B:
            pred_logits = pred_logits[-1]  # take last layer along dim=0 -> (B,Q,1)
        else:
            pred_logits = pred_logits[-1]

    # Now handle other dims ensuring final shape (B,Q,1)
    if pred_logits.dim() == 3:
        # common case (B, Q, 1) -> fine
        if pred_logits.shape[0] == B and pred_logits.shape[1] == Q:
            return pred_logits
        # if shape is (B,1,1) expand Q
        if pred_logits.shape[0] == B and pred_logits.shape[1] == 1:
            return pred_logits.expand(B, Q, pred_logits.shape[2])
        # if shape swapped (Q, B, 1)
        if pred_logits.shape[0] == Q and pred_logits.shape[1] == B:
            return pred_logits.permute(1, 0, 2)
        # fallback: reshape if possible
        try:
            return pred_logits.reshape(B, Q, pred_logits.shape[-1])
        except Exception:
            return pred_logits.unsqueeze(-1).expand(B, Q, -1)

    if pred_logits.dim() == 2:
        # (B, Q) -> (B,Q,1)
        if pred_logits.shape[0] == B and pred_logits.shape[1] == Q:
            return pred_logits.unsqueeze(-1)
        # (B,1) -> expand to Q
        if pred_logits.shape[0] == B and pred_logits.shape[1] == 1:
            return pred_logits.expand(B, Q).unsqueeze(-1)
        # (Q,B) -> permute then unsqueeze
        if pred_logits.shape[0] == Q and pred_logits.shape[1] == B:
            return pred_logits.permute(1, 0).unsqueeze(-1)
        # fallback
        try:
            return pred_logits.view(B, Q, -1)[:,:Q,:]
        except Exception:
            return pred_logits.unsqueeze(-1).expand(B, Q, -1)

    if pred_logits.dim() == 1:
        if pred_logits.shape[0] == B:
            return pred_logits.unsqueeze(1).expand(B, Q).unsqueeze(-1)
        if pred_logits.shape[0] == Q:
            return pred_logits.unsqueeze(0).expand(B, Q).unsqueeze(-1)
        # final fallback
        return pred_logits.unsqueeze(0).unsqueeze(1).expand(B, Q).unsqueeze(-1)

    # if anything else, try to reduce dims
    while pred_logits.dim() > 3:
        pred_logits = pred_logits[-1]
    try:
        return pred_logits.reshape(B, Q, pred_logits.shape[-1])[:, :Q, :]
    except Exception:
        return torch.zeros((B, Q, 1), device=device, dtype=torch.float32)


def contrastive_loss_from_pooled(v: torch.Tensor, t: torch.Tensor, temp: float = 0.07):
    """
    Symmetric InfoNCE / contrastive loss between v and t (both shape [B, D]).
    Returns scalar loss (mean of v->t and t->v).
    """
    assert v.dim() == 2 and t.dim() == 2 and v.shape[0] == t.shape[0]
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = (v @ t.t()) / temp  # shape (B, B)
    labels = torch.arange(v.shape[0], device=v.device)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_v2t + loss_t2v)



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
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    # new args:
    include_test_defects: bool = False,
    train_from_test: bool = False,
    specie_split_ratio: float = 0.8,
    specie_split_seed: int = 42,
    splits_save_dir: Optional[str] = None,
):
    ds = MVTecMetaDataset(
        root=root,
        meta_path=meta_path,
        mode=mode,
        k_shot=k_shot,
        obj_name=obj_name,
        aug_rate=aug_rate,
        include_test_defects=include_test_defects,
        goods_per_class=None,
        train_from_test=train_from_test,
        specie_split_ratio=specie_split_ratio,
        specie_split_seed=specie_split_seed,
        save_dir=splits_save_dir,  # pass the run-specific folder as dataset.save_dir
    )


    # NOTE: if balance True we try to do weighted sampling.
    sampler = None
    shuffle = True
    if balance and not distributed:
        # existing weighted sampler (only safe in single-process)
        labels = [int(entry.anomaly) for entry in ds.entries]
        labels = torch.tensor(labels, dtype=torch.long)
        anomaly_count = (labels == 1).sum().item()
        normal_count = (labels == 0).sum().item()
        if anomaly_count == 0:
            samples_weight = torch.ones(len(labels), dtype=torch.float)
        else:
            w_anom = 1.0 / anomaly_count
            w_norm = 1.0 / max(normal_count, 1)
            upsample_factor = 5.0
            weights = []
            for l in labels:
                if l == 1:
                    weights.append(w_anom * upsample_factor)
                else:
                    weights.append(w_norm)
            samples_weight = torch.tensor(weights, dtype=torch.float)

        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, num_samples=len(samples_weight), replacement=True
        )
        shuffle = False
    else:
        # when distributed, prefer DistributedSampler to avoid duplicates
        sampler = None
        shuffle = True

    # If distributed: use DistributedSampler
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader



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
    setup_distributed(args)

    if args.distributed:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 用一个方便的标志判断是否为主进程（rank 0）
    is_main_process = (not args.distributed) or (dist.get_rank() == 0)

    # === 新增：从 args 里读出两个超参数 ===
    MASK_DOWNSAMPLE = int(args.mask_downsample)
    NEG_SAMPLES_PER_IMAGE = int(args.neg_samples_per_image)

    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    if isinstance(meta, dict):
        class_list = sorted(list(meta.get('classes', meta.get('class_list', meta.keys()))))
    else:
        class_list = list(meta)
    args.class_list = class_list

    if args.use_official:
        model = FineTuneSAM3Official(
            bpe_path=args.bpe_path,
            sam3_ckpt=args.sam3_ckpt,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            # ----- new: for parallel lora -----
            # enable_parallel_lora=args.enable_parallel_lora,
            # parallel_lora_rank=args.parallel_lora_rank,
            # parallel_lora_alpha=args.parallel_lora_alpha,
            # ----- new: for parallel lora -----
            device=device,
            class_list=args.class_list,
            prompt_learner_type='perclass',
            num_templates=getattr(args, "num_templates", 4),
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

    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # 无论是否分布式，model_core 指向实际的 underlying module（方便后续直接访问 prompt_learner 等属性）
    model_core = model.module if hasattr(model, "module") else model

    if args.sam3_ckpt and os.path.exists(args.sam3_ckpt):
        load_sam3_checkpoint(model, args.sam3_ckpt)

    # --- create run_name and save/log dirs early so dataset can save splits into same folder ---
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, run_name)  # this folder will host ckpt and specie_splits_*.json
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if is_main_process:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[INFO] run_name={run_name}, log_dir={log_dir}, save_dir={save_dir}")
    else:
        writer = None

    dataloader = build_dataloaders(
        root=args.data_root,
        meta_path=args.meta_path or os.path.join(args.data_root, "meta.json"),
        mode=args.mode,
        k_shot=args.k_shot,
        obj_name=args.obj_name,
        aug_rate=args.aug_rate,
        batch_size=args.batch_size,
        balance=args.balance,
        distributed=args.distributed,
        rank=(args.rank if args.distributed else 0),
        world_size=(args.world_size if args.distributed else 1),
        include_test_defects=args.include_test_defects,
        train_from_test=args.train_from_test,
        specie_split_ratio=args.specie_split_ratio,
        specie_split_seed=args.specie_split_seed,
        splits_save_dir=save_dir,   # pass run-specific save_dir to dataset
    )



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
        nl = n.lower()
        if ("lora" in nl) or ("prompt" in nl) or ("template" in nl) or ("out_adapter" in nl):
            prompt_and_lora.append(p)
        else:
            other_params.append(p)
    print(f"[INFO] trainable params: prompt/LoRA={len(prompt_and_lora)}, others={len(other_params)}")
    # Print prompt-related parameters and requires_grad for diagnosis
    print("[INFO] Prompt-related params (name, requires_grad, shape):")
    for n, p in model.named_parameters():
        nl = n.lower()
        if ("prompt" in nl) or ("template" in nl) or ("kweight" in nl) or ("lora" in nl):
            print(f"  {n}: requires_grad={p.requires_grad}, shape={tuple(p.shape)}")


    # ---------------------------
    # (2-a) 可学习的损失权重（Kendall）
    log_var_focal = None
    log_var_dice = None
    log_var_iou = None
    learnable_log_vars = []
    if args.use_learned_loss_weights:
        # create as parameters and ensure they are on right device after device known
        log_var_focal = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        log_var_dice = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        log_var_iou = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        learnable_log_vars = [log_var_focal, log_var_dice, log_var_iou]
        # add to other_params so that optimizer will update them (we'll add a separate param-group to avoid weight decay)
        other_params_for_opt = other_params.copy()
    else:
        learnable_log_vars = []

    # Build optimizer with separate group for learnable log vars to avoid weight decay
    param_groups = [
        {"params": prompt_and_lora, "lr": args.lr_prompt},
        {"params": other_params, "lr": args.lr_main},
    ]
    if len(learnable_log_vars) > 0:
        param_groups.append({"params": learnable_log_vars, "lr": args.lr_main, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # 新增：AMP 的 GradScaler（只在 CUDA 下启用）
    scaler = GradScaler(enabled=(device.type == "cuda"))

    matcher = BinaryHungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)

    model.train()
    best_loss = float("inf")
    for epoch in range(args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        else:
            pbar = dataloader  # 无进度条，仅迭代
        running_loss = 0.0
        running_steps = 0
        for step, batch in enumerate(pbar):
            images, masks, prompt_lists, is_anomaly, class_names = batch
            images = images.to(device)
            masks = masks.to(device)

            # ===== AMP autocast: forward + loss 计算都放在半精度上下文 =====
            with autocast(enabled=(device.type == "cuda")):
                out = model(images, prompt_lists, class_names)
                pred_masks = out["pred_masks"]
                if pred_masks is None:
                    raise RuntimeError("Segmentation head did not return pred_masks.")

                # If model returns multi-layer masks, take last layer
                if pred_masks.dim() == 5:
                    pred_masks = pred_masks[-1]  # (B, Q, H0, W0)

                # --- IMPORTANT: avoid working on full-resolution masks ---
                # Create a downsampled version for IoU / matched / background loss computations.
                # This prevents keeping huge (B,Q,H0,W0) tensors in memory.
                B, Q, H0, W0 = pred_masks.shape
                MD = int(MASK_DOWNSAMPLE)  # from args earlier
                # reshape to (B*Q,1,H0,W0) to interpolate, then reshape back to (B,Q,MD,MD)
                pred_masks_ds = F.interpolate(
                    pred_masks.reshape(B * Q, 1, H0, W0),
                    size=(MD, MD), mode="bilinear", align_corners=False
                ).reshape(B, Q, MD, MD)

                # sanitize the downsampled masks (clamp + nan_to_num) - small memory footprint
                pred_masks_ds = pred_masks_ds.clamp(min=-20.0, max=20.0)
                pred_masks_ds = torch.nan_to_num(pred_masks_ds, nan=0.0, posinf=0.0, neginf=0.0)

                # --- Ensure masks has shape (B, H, W) before downsampling ---
                # Dataset may return masks with channel dim (B, C, H, W) (often C==1).
                # We convert to (B, H, W) by taking max over channel dim (safe for binary masks).
                if masks.dim() == 4:
                    # (B, C, H, W) -> (B, H, W)
                    masks_for_ds = masks.max(dim=1).values
                elif masks.dim() == 3:
                    masks_for_ds = masks
                else:
                    # Unexpected rank: try to squeeze singleton dims until rank==3
                    masks_for_ds = masks
                    while masks_for_ds.dim() > 3:
                        masks_for_ds = masks_for_ds.squeeze(1)

                # Now masks_for_ds is guaranteed to be (B, H, W)
                # Downsample to (MD,MD) with nearest (preserve binary labels)
                masks_ds = F.interpolate(masks_for_ds.unsqueeze(1).float(), size=(MD, MD), mode="nearest").squeeze(1)
                masks_ds = torch.nan_to_num(masks_ds, nan=0.0, posinf=0.0, neginf=0.0)
                masks_ds = (masks_ds > 0.5).float()

                # Replace the original masks variable with the collapsed version if you want
                # so downstream code that expects (B,H,W) works consistently.
                masks = masks_for_ds.float()

                # Keep original pred_masks (full-res) untouched if you ever need it for display.
                # But do NOT perform expensive ops on it; do computations on pred_masks_ds.
                # mask_for_iou used below: use the downsampled version
                mask_for_iou = pred_masks_ds  # (B, Q, MD, MD)

                # Note: we DO NOT interpolate pred_masks to GT size here to avoid huge memory usage.
                # When needed, you can upsample pred_masks_ds for visualization only.
                
                # Also sanitize the original masks if you later use them (kept minimal):
                masks = torch.nan_to_num(masks, nan=0.0, posinf=0.0, neginf=0.0)
                masks = (masks > 0.5).float()


                # === Build list-based targets (per-image) ===
                # --- 1) 保持按图的 targets（便于 debug）
                list_targets = build_list_targets_from_binary_masks(masks)  # returns list of dicts, len=B



                # --- 2) 将 list_targets 展平成一个 batched dict（matcher 在本仓库实现里要求此格式）
                B_cur = len(list_targets)
                device = masks.device
                H, W = masks.shape[1], masks.shape[2]

                boxes_list = []
                labels_list = []
                segments_list = []
                num_boxes_list = []
                for t in list_targets:
                    nb = int(t["boxes"].shape[0])
                    num_boxes_list.append(nb)
                    if nb > 0:
                        boxes_list.append(t["boxes"])
                        labels_list.append(t["labels"])
                        segments_list.append(t["segments"])

                if len(boxes_list) == 0:
                    # no GT in batch
                    targets_flat = {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                        "segments": torch.zeros((0, H, W), dtype=torch.float32, device=device),
                        "num_boxes": torch.tensor(num_boxes_list, dtype=torch.long, device=device),
                    }
                else:
                    targets_flat = {
                        "boxes": torch.cat(boxes_list, dim=0),      # (G,4)
                        "labels": torch.cat(labels_list, dim=0),    # (G,)
                        "segments": torch.cat(segments_list, dim=0),# (G,H,W)
                        "num_boxes": torch.tensor(num_boxes_list, dtype=torch.long, device=device),
                    }

                    # Use targets_flat for downstream code compatibility
                    targets = targets_flat

                    # DEBUG: print summary (use list_targets for per-image detail)
                    print("DBG list_targets summary:")
                    for i, t in enumerate(list_targets):
                        print(f" image {i}: boxes.shape={t['boxes'].shape}, labels.shape={t['labels'].shape}, segments.shape={t['segments'].shape}")

                    # Now we can safely get num_boxes_list from targets (flattened dict)
                    num_boxes_list = targets['num_boxes'].tolist()
                    print("DBG targets_num_boxes:", num_boxes_list)

                    # Batch-level summary: whether each image has GT in this batch
                    batch_has_gt = [1 if n > 0 else 0 for n in num_boxes_list]
                    print("DBG BATCH SAMPLE SUMMARY: batch_size =", images.shape[0], "num_boxes:", num_boxes_list, "has_gt:", batch_has_gt)

                    # Also print mask_sums (per-image) for additional check
                    mask_sums = [int(t['segments'].sum().item()) if t['segments'].numel() > 0 else 0 for t in list_targets]
                    print("DBG mask_sums per image:", mask_sums)




                # ---------- Normalize presence logits robustly ----------
                B = pred_masks.shape[0]
                Q = pred_masks.shape[1]
                pred_logits = out.get("presence_logit", None)
                pred_logits = normalize_presence_logits(pred_logits, B, Q, device)
                # ensure float dtype for BCE and matcher
                pred_logits = pred_logits.float()

                # reference boxes as before (no change)
                pred_boxes = out.get("reference_boxes", None)
                if pred_boxes is None:
                    # fallback，全 0 框
                    pred_boxes = torch.zeros((pred_masks.shape[0], pred_masks.shape[1], 4), device=device)
                else:
                    if pred_boxes.dim() == 4:
                        # 原始 shape: [L, Q, B, 4]（从你打印 [400,2,4] 反推出 L=6）
                        pred_boxes = pred_boxes[-1]  # [Q, B, 4] 或 [B, Q, 4]
                    # 统一成 [B, Q, 4]
                    if pred_boxes.shape[0] == pred_masks.shape[1] and pred_boxes.shape[1] == pred_masks.shape[0]:
                        # 当前是 [Q, B, 4]，需要转成 [B, Q, 4]
                        pred_boxes = pred_boxes.permute(1, 0, 2).contiguous()
                    elif pred_boxes.shape[0] == pred_masks.shape[0] and pred_boxes.shape[1] == pred_masks.shape[1]:
                        # 已经是 [B, Q, 4]，不用动
                        pass
                    else:
                        raise RuntimeError(
                            f"Unexpected pred_boxes shape {pred_boxes.shape} "
                            f"for pred_masks {pred_masks.shape}"
                        )

                # --- matcher and robust index reconstruction ---
                matcher_outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                batch_idx, src_idx, tgt_idx = matcher(matcher_outputs, targets)

                print("DBG matcher raw shapes: batch_idx", None if batch_idx is None else tuple(batch_idx.shape),
                      "src_idx", None if src_idx is None else tuple(src_idx.shape),
                      "tgt_idx", None if tgt_idx is None else tuple(tgt_idx.shape))


                # prepare targets_num_boxes: number of GT boxes per image
                targets_num_boxes = [int(t["boxes"].shape[0]) for t in list_targets]

                # when calling the convert helper, pass targets_num_boxes:
                indices_per_image = convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B=images.shape[0], device=device, targets_num_boxes=targets_num_boxes)
                

                # If matcher returns tgt_idx is None (BinaryHungarianMatcher behavior),
                # we must still convert the flattened batch_idx/src_idx into per-image matched lists.
                # For MVTec (<=1 GT per image) we can map matched src indices to the unique target index per image.
                if tgt_idx is None:
                    # Build per-image lists of matched src indices from batch_idx and src_idx
                    # batch_idx/src_idx are 1D tensors
                    if batch_idx is None or src_idx is None:
                        indices_per_image = [ (torch.zeros((0,), dtype=torch.long, device=device),
                                               torch.zeros((0,), dtype=torch.long, device=device)) for _ in range(pred_masks.shape[0]) ]
                    else:
                        # group src_idx by batch index
                        B = pred_masks.shape[0]
                        src_idx = src_idx.to(device)
                        batch_idx = batch_idx.to(device)
                        indices_per_image = []
                        for b in range(B):
                            mask = (batch_idx == b)
                            if mask.any():
                                srcs = src_idx[mask]
                            else:
                                srcs = torch.zeros((0,), dtype=torch.long, device=device)
                            # tgt list unknown; will be reconstructed below if targets exist
                            indices_per_image.append((srcs, torch.zeros((0,), dtype=torch.long, device=device)))

                    # Now reconstruct global tgt_idx by mapping per-image first target index
                    num_boxes_list = targets["num_boxes"].tolist()
                    # cumulative start indices for flattened targets (same convention as matcher)
                    cum = [0]
                    for nb in num_boxes_list:
                        cum.append(cum[-1] + nb)
                    batch_idx_list = []
                    src_idx_list = []
                    tgt_idx_list = []
                    for b, (srcs, _) in enumerate(indices_per_image):
                        if srcs.numel() == 0:
                            continue
                        if num_boxes_list[b] == 0:
                            # If there is no GT for that image but matcher matched srcs, skip (rare)
                            continue
                        # assign the only GT flattened index for this image (works for MVTec where num_boxes <= 1)
                        tgt_global = cum[b]
                        for s in srcs.cpu().tolist():
                            batch_idx_list.append(b)
                            src_idx_list.append(int(s))
                            tgt_idx_list.append(int(tgt_global))
                    if len(tgt_idx_list) > 0:
                        batch_idx = torch.tensor(batch_idx_list, dtype=torch.long, device=device)
                        src_idx = torch.tensor(src_idx_list, dtype=torch.long, device=device)
                        tgt_idx = torch.tensor(tgt_idx_list, dtype=torch.long, device=device)
                    else:
                        # Keep tgt_idx as None if we couldn't reconstruct any (then no matched pairs effectively)
                        tgt_idx = None

                # assume targets defined earlier
                targets_num_boxes = targets["num_boxes"].tolist()  # list of ints
                out["indices"] = convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B=pred_masks.shape[0], device=device, targets_num_boxes=targets_num_boxes)
                indices = out["indices"]  # local alias for later loops

                # --- end matcher robust handling ---
                

                # ====== DEBUG BLOCK ======
                if step < 20:
                    print("DBG final indices (out['indices']):")
                    for b in range(images.shape[0]):
                        src_q, tgt_q = indices[b]  # indices is alias to out["indices"]
                        print(f"  image {b}: src_q={src_q.cpu().tolist()}, tgt_q={tgt_q.cpu().tolist()}, num_boxes={int(targets['num_boxes'][b].item())}")

                    # 1) targets summary
                    print("DBG targets num_boxes:", targets["num_boxes"].tolist())
                
                    # 2) matcher raw shapes
                    print("DBG matcher raw shapes:", 
                          "batch_idx", None if batch_idx is None else tuple(batch_idx.shape),
                          "src_idx", None if src_idx is None else tuple(src_idx.shape),
                          "tgt_idx", None if tgt_idx is None else tuple(tgt_idx.shape))
                
                    # 3) matched per image (use convert helper)
                    indices_tmp = convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B=images.shape[0], device=device)
                    print("DBG matched per image:", [int(s.shape[0]) for s, _ in indices_tmp])
                
                    # 4) model outputs stats
                    print("DBG pred_masks shape:", pred_masks.shape, "mean/std:", float(pred_masks.mean().item()), float(pred_masks.std().item()))
                    print("DBG mask_for_iou shape:", mask_for_iou.shape, "mean/std:", float(mask_for_iou.mean().item()), float(mask_for_iou.std().item()))
                
                    # presence logits (normalized earlier)
                    try:
                        pl = pred_logits  # we normalized pred_logits earlier in your code
                        print("DBG pred_logits shape:", pl.shape, "mean/std:", float(pl.mean().item()), float(pl.std().item()))
                    except Exception as e:
                        print("DBG pred_logits access error:", e)
                
                    # pred_boxes stats
                    try:
                        pb = pred_boxes
                        print("DBG pred_boxes shape:", pb.shape, "min/max:", float(pb.min().item()), float(pb.max().item()))
                    except Exception as e:
                        print("DBG pred_boxes access error:", e)
                
                    # decoder hidden states if present
                    dec = out.get("decoder_hs", None)
                    print("DBG decoder_hs:", None if dec is None else tuple(dec.shape))
                
                    # prompt prototype and pooled visual feat for align
                    try:
                        prompt_seq, prompt_mask = model_core.prompt_learner(prompt_lists, device=device)
                        # prompt_seq 可有不同 layout：(S,B,D) 或 (B,D) 等，统一成 (B,D)
                        prompt_last = prompt_seq[-1]
                        if prompt_last.dim() == 3:  # (S, B, D)
                            prompt_proto = prompt_last[-1]  # (B, D)
                        elif prompt_last.dim() == 2:
                            if prompt_last.shape[0] == images.shape[0]:
                                prompt_proto = prompt_last  # (B, D)
                            else:
                                prompt_proto = prompt_last.transpose(0, 1) if prompt_last.shape[1] == images.shape[0] else prompt_last.reshape(images.shape[0], -1)
                        else:
                            prompt_proto = prompt_last.reshape(images.shape[0], -1)[:, : (prompt_last.numel() // images.shape[0])]
                        print("DBG prompt_proto shape/mean/std:", prompt_proto.shape, float(prompt_proto.mean().item()), float(prompt_proto.std().item()))
                    except Exception as e:
                        print("DBG prompt_learner error:", e)
                        prompt_proto = None
                
                    if dec is not None:
                        try:
                            if isinstance(dec, torch.Tensor):
                                hs_last = dec[-1] if dec.dim()==4 else dec  # handle (L,B,Q,D) or (B,Q,D)
                                v_pooled = hs_last.mean(dim=1)
                                print("DBG v_pooled shape/mean/std:", v_pooled.shape, float(v_pooled.mean().item()), float(v_pooled.std().item()))
                        except Exception as e:
                            print("DBG decoder_hs -> v_pooled error:", e)
                
                    # 5) quick matched tensors check (if there are matches)  -- use downsampled masks for debug/loss check
                    if tgt_idx is not None and tgt_idx.numel() > 0:
                        try:
                            # pred_masks_ds 已以 (B, Q, MD, MD) 形式存在（我们在前面构建）
                            pm_ds = pred_masks_ds[batch_idx, src_idx]         # (M, MD, MD)
                            # targets["segments"] is (G, H, W) -> downsample to MD
                            tgt_masks = targets["segments"][tgt_idx]          # (M, H, W)
                            if tgt_masks.dim() == 3:
                                tm_ds = F.interpolate(tgt_masks.unsqueeze(1).float(), size=(MD, MD), mode="nearest").squeeze(1)
                            else:
                                # unexpected shape: try collapse channel
                                tmp = tgt_masks
                                while tmp.dim() > 3:
                                    tmp = tmp.squeeze(1)
                                tm_ds = F.interpolate(tmp.unsqueeze(1).float(), size=(MD, MD), mode="nearest").squeeze(1)
                    
                            print("DBG pred_matched (downsampled) shape:", pm_ds.shape, "tgt_matched (downsampled) shape:", tm_ds.shape)
                            print("DBG pred_matched mean/std:", float(pm_ds.mean().item()), float(pm_ds.std().item()))
                            print("DBG tgt_matched mean/std:", float(tm_ds.mean().item()), float(tm_ds.std().item()))
                    
                            # compute provisional focal/dice on downsampled tensors for debug
                            try:
                                debug_focal = sam_sigmoid_focal_loss(pm_ds, tm_ds, max(1.0, float(targets["num_boxes"].sum().float())), alpha=0.25, gamma=2.0, loss_on_multimask=False, triton=False)
                                debug_dice = sam_dice_loss(pm_ds, tm_ds, max(1.0, float(targets["num_boxes"].sum().float())), loss_on_multimask=False, reduce=True)
                                print("DBG provisional focal/dice (downsampled):", float(debug_focal.item()), float(debug_dice.item()))
                            except Exception as e:
                                print("DBG provisional loss compute error:", e)
                        except Exception as e:
                            print("DBG matched tensor error:", e)
                    else:
                        print("DBG no matched pairs in this batch")

                # ====== END DEBUG BLOCK ======


                # matched branch — compute on downsampled masks to save memory
                if tgt_idx is None or tgt_idx.numel() == 0:
                    loss_focal = torch.tensor(0.0, device=device)
                    loss_dice = torch.tensor(0.0, device=device)
                else:
                    tgt_masks = targets["segments"]  # (G, H, W)
                    pred_matched = pred_masks_ds[batch_idx, src_idx]  # (M, MD, MD)
                    tgt_matched = tgt_masks[tgt_idx]              # (M, H, W)

                    # --- downsample matched to MASK_DOWNSAMPLE to save memory ----
                    if pred_matched.dim() == 3:
                        pm_ds = F.interpolate(pred_matched.unsqueeze(1), size=(MASK_DOWNSAMPLE, MASK_DOWNSAMPLE),
                                              mode="bilinear", align_corners=False).squeeze(1)
                    else:
                        pm_ds = pred_matched
                    if tgt_matched.dim() == 3:
                        tm_ds = F.interpolate(tgt_matched.unsqueeze(1), size=(MASK_DOWNSAMPLE, MASK_DOWNSAMPLE),
                                              mode="nearest").squeeze(1)
                    else:
                        tm_ds = tgt_matched

                    num_boxes = float(max(1.0, src_idx.numel()))
                    loss_focal = sam_sigmoid_focal_loss(pm_ds, tm_ds, num_boxes, alpha=0.25, gamma=2.0, loss_on_multimask=False, triton=False)
                    loss_dice = sam_dice_loss(pm_ds, tm_ds, num_boxes, loss_on_multimask=False, reduce=True)

                    # -------------------------
                    # background/unmatched loss (sample K negatives per image), compute on downsampled masks
                    loss_focal_bg = torch.tensor(0.0, device=device)
                    loss_dice_bg = torch.tensor(0.0, device=device)
                    total_bg_samples = 0.0
                    all_q = torch.arange(pred_masks.shape[1], device=device)
                    for b in range(pred_masks.shape[0]):
                        src_q, tgt_q = indices[b]
                        if src_q.numel() == 0:
                            unmatched_q = all_q
                        else:
                            mask_un = torch.ones_like(all_q, dtype=torch.bool)
                            mask_un[src_q] = False
                            unmatched_q = all_q[mask_un]

                        if unmatched_q.numel() == 0:
                            continue
                        
                        k = min(NEG_SAMPLES_PER_IMAGE, int(unmatched_q.numel()))
                        perm = torch.randperm(unmatched_q.numel(), device=device)[:k]
                        sampled_unmatched_q = unmatched_q[perm]

                        preds_bg_ds = pred_masks_ds[b, sampled_unmatched_q]  # already (k, MD, MD)
                        
                        # Downsample to save memory
                        if preds_bg_ds.dim() == 3:
                            preds_bg_ds = F.interpolate(preds_bg_ds.unsqueeze(1), size=(MASK_DOWNSAMPLE, MASK_DOWNSAMPLE),
                                                        mode="bilinear", align_corners=False).squeeze(1)
                        else:
                            preds_bg_ds = preds_bg_ds

                        zeros = torch.zeros_like(preds_bg_ds)

                        nb = float(sampled_unmatched_q.numel())
                        loss_focal_bg += sam_sigmoid_focal_loss(preds_bg_ds, zeros, nb, alpha=0.25, gamma=2.0,
                                                                loss_on_multimask=False, triton=False)
                        loss_dice_bg += sam_dice_loss(preds_bg_ds, zeros, nb, loss_on_multimask=False, reduce=True)
                        total_bg_samples += nb

                    if total_bg_samples > 0:
                        loss_focal_bg = loss_focal_bg / (total_bg_samples / float(pred_masks.shape[0]))
                        loss_dice_bg = loss_dice_bg / (total_bg_samples / float(pred_masks.shape[0]))

                    loss_focal = loss_focal + 0.5 * loss_focal_bg
                    loss_dice = loss_dice + 0.5 * loss_dice_bg


                        
                # -------------------------
                # IoU 回归监督（在 downsample 后计算 true IoU，并用 SmoothL1 回归到模型预测的 iou）
                loss_iou = torch.tensor(0.0, device=device)
                iou_pred = out.get("iou_predictions", None)  # SAM3 的命名可能不同，确认模型输出名

                # only compute when there are matched pairs
                if tgt_idx is not None and tgt_idx.numel() > 0:
                    # 1) prepare predicted iou tensor into (B, Q)
                    if iou_pred is not None:
                        # reuse your robust normalizer: returns (B,Q,1) for many cases
                        iou_pred = normalize_presence_logits(iou_pred, B, Q, device).squeeze(-1)  # (B, Q)
                    else:
                        iou_pred = None

                    # 2) build pred_matched (M, MD, MD) and gt matched masks downsampled to MD
                    pred_matched_ds = pred_masks_ds[batch_idx, src_idx]  # (M, MD, MD)

                    # targets["segments"] is flattened (G, H, W). pick tgt_idx and downsample to MD
                    tgt_masks_flat = targets["segments"][tgt_idx]  # (M, H, W) or (M, 1, H, W)
                    # ensure (M, H, W)
                    if tgt_masks_flat.dim() == 4:
                        # (M, 1, H, W) -> (M, H, W)
                        tgt_masks_flat = tgt_masks_flat.squeeze(1)
                    # Downsample with nearest to preserve binary GT
                    tm_ds = F.interpolate(tgt_masks_flat.unsqueeze(1).float(), size=(MASK_DOWNSAMPLE, MASK_DOWNSAMPLE),
                                          mode="nearest").squeeze(1)  # (M, MD, MD)

                    # 3) compute true IoU per matched pair (use pred sigmoid probability)
                    pred_prob = torch.sigmoid(pred_matched_ds)  # (M, MD, MD)
                    pred_flat = pred_prob.flatten(1)            # (M, N)
                    tgt_flat = tm_ds.flatten(1)                 # (M, N)
                    inter = (pred_flat * tgt_flat).sum(dim=1)   # (M,)
                    sum_p = pred_flat.sum(dim=1)
                    sum_t = tgt_flat.sum(dim=1)
                    union = sum_p + sum_t - inter + 1e-6
                    true_iou = inter / union  # (M,)

                    # 4) if model provides iou_pred, gather matched preds and compute SmoothL1
                    if iou_pred is not None:
                        # iou_pred is (B, Q) -> pick matched entries
                        iou_pred_matched = iou_pred[batch_idx, src_idx]   # (M,)
                        # ensure same device/dtype
                        iou_pred_matched = iou_pred_matched.to(true_iou.device).float()
                        true_iou = true_iou.to(iou_pred_matched.device).float()
                        if iou_pred_matched.numel() > 0:
                            loss_iou = F.smooth_l1_loss(iou_pred_matched, true_iou, reduction="mean")
                        else:
                            loss_iou = torch.tensor(0.0, device=device)
                    else:
                        # If model does not predict IoU, we could optionally add a margin/reg term,
                        # but for now we keep loss_iou = 0 (no regression head)
                        loss_iou = torch.tensor(0.0, device=device)
                else:
                    loss_iou = torch.tensor(0.0, device=device)
                # -------------------------


                # loss_iou = torch.tensor(0.0, device=device)
                # -------------------------
                # Presence BCE loss (requires presence_head=True in model_builder)
                # pred_logits 已被准备成 (B, Q, 1) 形式 earlier; we squeeze trailing dim.
                if pred_logits is not None:
                    # convert to shape (B, Q) for BCEWithLogits (logits)
                    # pred_logits currently is (B, Q, 1) according to normalize_presence_logits
                    presence_logit = pred_logits
                    if presence_logit.dim() == 3 and presence_logit.shape[-1] == 1:
                        presence_logit = presence_logit.squeeze(-1)  # (B, Q)
                    # safety: if still has extra dims, reshape/pad
                    if presence_logit.dim() != 2:
                        # try to force into (B,Q)
                        presence_logit = presence_logit.reshape(B, Q)

                    # build presence target matrix (B, Q)
                    presence_targets = torch.zeros_like(presence_logit, dtype=torch.float32, device=device)

                    # get indices per image using our helper (already computed into 'indices' local alias)
                    indices_per_image = convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B=images.shape[0], device=device)

                    # Defensive assignment: filter out-of-range indices and print diagnostics
                    Q_dim = presence_targets.shape[1]
                    for b in range(images.shape[0]):
                        src_q, _ = indices_per_image[b]
                        if src_q.numel() > 0:
                            # ensure dtype long and on same device
                            src_q = src_q.to(device).long()

                            # detect invalid indices
                            invalid_mask = (src_q < 0) | (src_q >= Q_dim)
                            if invalid_mask.any():
                                print(f"[WARN] presence_targets: found {invalid_mask.sum().item()} invalid src indices for image {b}. "
                                      f"Q={Q_dim}, src_q_invalid={src_q[invalid_mask].cpu().tolist()}")

                                # drop invalid indices before assignment
                                src_q = src_q[~invalid_mask]

                            if src_q.numel() > 0:
                                presence_targets[b, src_q] = 1.0

                    loss_presence = F.binary_cross_entropy_with_logits(presence_logit, presence_targets)
                else:
                    # fallback (shouldn't happen because we normalized earlier), keep zero
                    loss_presence = torch.tensor(0.0, device=device)


                # -------------------------
                # Contrastive alignment (InfoNCE) between visual pooled vector and prompt prototype
                align_loss = torch.tensor(0.0, device=device)
                if args.lambda_align is not None and float(args.lambda_align) > 0.0:
                    # Prefer prompt_seq from model output if available
                    prompt_seq = out.get("prompt_seq", None)
                    if prompt_seq is None:
                        try:
                            prompt_seq, _ = model_core.prompt_learner(prompt_lists, device=device)
                        except Exception as e:
                            print("[WARN] cannot obtain prompt_seq from model_core.prompt_learner:", e)
                            prompt_seq = None

                    # Normalize extraction into (B, D)
                    prompt_proto = None
                    if prompt_seq is not None:
                        # prompt_seq is seq-first (S, B, D)
                        last = prompt_seq[-1]
                        if last.dim() == 2:
                            # last may be (B, D) or (D, B)
                            if last.shape[0] == B:
                                prompt_proto = last
                            elif last.shape[1] == B:
                                prompt_proto = last.transpose(0, 1).contiguous()
                            else:
                                prompt_proto = last.reshape(B, -1)[:, : (last.numel() // B)]
                        elif last.dim() == 3:
                            # (S, B, D) -> we already indexed last as last => usually (B, D)
                            prompt_proto = last if last.shape[0] == B else last[-1]
                        else:
                            prompt_proto = last.reshape(B, -1)[:, : (last.numel() // B)]

                    # Build mask embedding using preferred decoder spatial features, fallback to decoder_hs pooling
                    mask_embed = None
                    decoder_feat = out.get("decoder_features", None)
                    if decoder_feat is not None:
                        # expected decoder_feat shape (B, C, Hf, Wf)
                        try:
                            Bf, C, Hf, Wf = decoder_feat.shape
                            # downsample/resize GT masks to (Hf, Wf)
                            gt_ds = F.interpolate(masks.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1)  # (B,Hf,Wf)
                            feat = decoder_feat.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)  # (B, Hf*Wf, C)
                            mask_flat = gt_ds.view(B, Hf * Wf).unsqueeze(-1)  # (B, Hf*Wf,1)
                            pos_counts = mask_flat.sum(dim=1).clamp(min=1.0).to(device)
                            mask_sum = (feat * mask_flat).sum(dim=1)  # (B,C)
                            mask_embed = mask_sum / pos_counts  # (B,C)
                        except Exception as e:
                            print("[WARN] decoder_features pooling failed:", e)
                            mask_embed = None

                    if mask_embed is None:
                        # fallback to decoder_hs (pooled over queries)
                        decoder_hs = out.get("decoder_hs", None)
                        if decoder_hs is not None:
                            try:
                                hs_last = decoder_hs
                                # if hs_last is (L,B,Q,D) get last L and ensure shape (B,Q,D)
                                if hs_last.dim() == 4:
                                    hs_last = hs_last[-1]  # -> (B,Q,D) or (Q,B,D)
                                if hs_last.dim() == 3 and hs_last.shape[0] == Q and hs_last.shape[1] == B:
                                    hs_last = hs_last.permute(1, 0, 2).contiguous()
                                # pool across queries
                                v_pooled = hs_last.mean(dim=1)  # (B, D)
                                mask_embed = v_pooled
                            except Exception as e:
                                print("[WARN] decoder_hs pooling failed:", e)
                                mask_embed = None

                    # If still None, fallback to zeros to avoid crash
                    if mask_embed is None:
                        mask_embed = torch.zeros((B, prompt_proto.shape[1] if prompt_proto is not None else 128), device=device)

                    # Ensure dim match: project mask_embed to prompt_proto dim if necessary
                    if prompt_proto is not None:
                        Dp = prompt_proto.shape[1]
                        Dm = mask_embed.shape[1]
                        if Dm != Dp:
                            # create or reuse a small linear projector on model_core to map dims
                            if not hasattr(model_core, "_align_proj"):
                                model_core._align_proj = nn.Linear(Dm, Dp).to(device)
                                optimizer.add_param_group({
                                    "params": model_core._align_proj.parameters(),
                                    "lr": args.lr_main,       # 也可以专门设一个 lr，如 args.lr_align（如果需要）
                                    "weight_decay": 0.0
                                })
                            mask_embed = model_core._align_proj(mask_embed)

                    # L2 normalize and compute symmetric InfoNCE
                    if prompt_proto is None:
                        # can't compute align: leave zero
                        align_loss = torch.tensor(0.0, device=device)
                    else:
                        p_norm = F.normalize(prompt_proto, dim=1)
                        m_norm = F.normalize(mask_embed, dim=1)

                        # use helper contrastive_loss_from_pooled (already defined in file)
                        try:
                            align_loss = contrastive_loss_from_pooled(p_norm, m_norm, temp=args.align_temp)
                        except Exception as e:
                            # fallback compute manual symmetric nce
                            logits = (p_norm @ m_norm.t()) / float(args.align_temp)
                            labels_local = torch.arange(B, device=device)
                            align_loss = 0.5 * (F.cross_entropy(logits, labels_local) + F.cross_entropy(logits.t(), labels_local))

                        # Diagnostics: pos/neg stats and norms (log every args.log_freq steps)
                        if (step % getattr(args, "log_freq", 100)) == 0:
                            sim = (p_norm @ m_norm.t()) / float(args.align_temp)
                            labels_local = torch.arange(B, device=device)
                            pos = sim[range(B), labels_local]
                            mask_offdiag = ~torch.eye(B, dtype=torch.bool, device=device)
                            neg = sim.masked_select(mask_offdiag).view(B, B - 1)
                            pos_mean, pos_std = float(pos.mean().item()), float(pos.std().item())
                            neg_mean, neg_std = float(neg.mean().item()), float(neg.std().item())
                            p_mean, p_std = float(p_norm.norm(dim=1).mean().item()), float(p_norm.norm(dim=1).std().item())
                            m_mean, m_std = float(m_norm.norm(dim=1).mean().item()), float(m_norm.norm(dim=1).std().item())
                            print(f"[ALIGN] step={step} align_loss={align_loss.item():.6f} pos_mean={pos_mean:.4f} pos_std={pos_std:.4f} neg_mean={neg_mean:.4f} neg_std={neg_std:.4f}")
                            print(f"[NORM] p_mean={p_mean:.4f} p_std={p_std:.4f} | m_mean={m_mean:.4f} m_std={m_std:.4f}")

                            # TSNE: sample up to tsne_samples per type
                            if (step % getattr(args, "tsne_freq", 500)) == 0:
                                ns = min(getattr(args, "tsne_samples", 64), B)
                                sel = np.random.choice(B, ns, replace=False)
                                p_sample = p_norm[sel].detach().cpu().numpy()
                                m_sample = m_norm[sel].detach().cpu().numpy()
                                X = np.concatenate([p_sample, m_sample], axis=0)
                                try:
                                    Z = TSNE(n_components=2, perplexity=30, init='pca').fit_transform(X)
                                    plt.figure(figsize=(5,5))
                                    plt.scatter(Z[:ns,0], Z[:ns,1], c='C0', label='prompt', alpha=0.8)
                                    plt.scatter(Z[ns:,0], Z[ns:,1], c='C1', label='mask', alpha=0.8)
                                    plt.legend()
                                    plt.title(f"t-SNE step{step}")
                                    tsne_out_dir = os.path.join(args.log_dir, "tsne")
                                    os.makedirs(tsne_out_dir, exist_ok=True)
                                    plt.savefig(os.path.join(tsne_out_dir, f"tsne_step{step}.png"), dpi=150)
                                    plt.close()
                                except Exception as e:
                                    print("[WARN] TSNE failed:", e)
                else:
                    align_loss = torch.tensor(0.0, device=device)
                # -------------------------


                # Combine losses: use learned weights (Kendall) if requested, otherwise use args.loss_alpha/beta/gamma
                if args.use_learned_loss_weights and len(learnable_log_vars) == 3:
                    # log_var_focal/log_var_dice/log_var_iou defined in main scope and added to optimizer
                    loss_main = (torch.exp(-log_var_focal) * loss_focal + log_var_focal) + \
                                (torch.exp(-log_var_dice)  * loss_dice  + log_var_dice) + \
                                (torch.exp(-log_var_iou)   * loss_iou   + log_var_iou)
                    total_loss = loss_main + args.presence_weight * loss_presence + args.lambda_align * align_loss
                else:
                    total_loss = args.loss_alpha * loss_focal + args.loss_beta * loss_dice + args.loss_gamma * loss_iou
                    total_loss = total_loss + args.presence_weight * loss_presence + args.lambda_align * align_loss

                loss = total_loss

            # ===== AMP autocast 结束 =====
        
            if not torch.isfinite(loss):
                print(f"[WARN] Skip batch with non-finite loss (loss={loss.item()}, focal={loss_focal.item()}, dice={loss_dice.item()}, iou={loss_iou.item()})")
                continue

            optimizer.zero_grad()
            # 使用 GradScaler 做 backward + step
            scaler.scale(loss).backward()

            # --- Diagnostic: grad norms for prompt-related params ---
            grad_norms = []
            for n,p in model.named_parameters():
                nl = n.lower()
                if p.grad is not None and (("prompt" in nl) or ("template" in nl) or ("kweight" in nl) or ("lora" in nl)):
                    grad_norms.append((n, float(p.grad.norm().item())))
            if len(grad_norms) > 0 and (step % getattr(args, "log_freq", 100) == 0):
                # print few entries (avoid overwhelming)
                print("[GRADS] sample prompt-related grad norms (top 10):")
                for name, gn in grad_norms[:20]:
                    print(f"  {name}: {gn:.4e}")
                vals = np.array([v for (_, v) in grad_norms])
                print(f"[GRADS] mean={vals.mean():.4e}, std={vals.std():.4e}, max={vals.max():.4e}")
            # --- Diagnostic: grad norms for prompt-related params ---

            scaler.step(optimizer)
            scaler.update()

            if is_main_process:
                pbar.set_postfix(
                    loss=loss.item(),
                    focal=loss_focal.item(),
                    dice=loss_dice.item(),
                )

            global_step = epoch * len(dataloader) + step
            if is_main_process and writer is not None:
                writer.add_scalar("loss/total", loss.item(), global_step)
            if is_main_process and writer is not None:
                writer.add_scalar("loss/focal", loss_focal.item(), global_step)
            if is_main_process and writer is not None:
                writer.add_scalar("loss/dice", loss_dice.item(), global_step)
            if is_main_process and writer is not None:
                writer.add_scalar("loss/iou", loss_iou.item(), global_step)
            if is_main_process and writer is not None:
                writer.add_scalar("loss/presence", loss_presence.item(), global_step)
            if is_main_process and writer is not None:
                writer.add_scalar("loss/align", align_loss.item(), global_step)

            running_loss += loss.item()
            running_steps += 1

        if running_steps > 0:
            avg_loss = running_loss / running_steps
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(save_dir, "sam3_peft_best.pth")
                if is_main_process:
                    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save({"epoch": epoch, "state_dict": sd, "optimizer": optimizer.state_dict()}, ckpt_path)
                    print(f"[INFO] Epoch {epoch+1}: new best avg_loss {avg_loss:.4f}, saved to {ckpt_path}")
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
    parser.add_argument("--neg_samples_per_image", type=int, default=50, help="Max negative (unmatched) queries to sample per image for background loss")
    parser.add_argument("--lambda_align", type=float, default=0.1, help="weight for contrastive alignment loss (InfoNCE)")
    parser.add_argument("--align_temp", type=float, default=0.07, help="temperature for contrastive alignment")
    parser.add_argument("--presence_weight",type=float,default=1.0,help="weight for presence BCE loss")
    parser.add_argument("--use_learned_loss_weights", action="store_true", help="Use learnable log-variance weights for multi-loss balancing (Kendall)")
    parser.add_argument("--mask_downsample", type=int, default=256, help="Downsample masks for background loss calculation to reduce memory")
    parser.add_argument("--enable_parallel_lora", action="store_true", help="Enable parallel LoRA adapters in Attention (official model path)")
    parser.add_argument("--parallel_lora_rank", type=int, default=16, help="Rank for parallel LoRA")
    parser.add_argument("--parallel_lora_alpha", type=float, default=None, help="Alpha scaling for parallel LoRA")
    parser.add_argument("--include_test_defects", action="store_true",help="(legacy) include defects from test split when forming dataset")
    parser.add_argument("--train_from_test", action="store_true",help="When set, build training set from MVTec test split defects only, per-specie split")
    parser.add_argument("--specie_split_ratio", type=float, default=0.8,help="Train ratio per specie (e.g. 0.8 => 80% train, 20% test)")
    parser.add_argument("--specie_split_seed", type=int, default=42,help="Random seed for per-specie split reproducibility")
    parser.add_argument("--splits_save_dir", type=str, default=None,help="If set, write specie_splits_{cls}.json files for reproducibility.")

    #--------------- Diagnostic logging args ---------------
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (steps) for align diagnostics")
    parser.add_argument("--tsne_freq", type=int, default=500, help="TSNE save frequency (steps)")
    parser.add_argument("--tsne_samples", type=int, default=64, help="Number of samples for TSNE projection")

    

    args = parser.parse_args()
    main(args)