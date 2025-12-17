import os, sys

PROJECT_ROOT = "/root/autodl-tmp/FiLo_plus/sam3"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import math
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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


# ==================== 新增：学习率调度器 ====================

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing 学习率调度器
    
    训练过程：
    1. Warmup阶段：LR从0线性增加到base_lr
    2. Cosine阶段：LR从base_lr余弦衰减到min_lr
    """
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr_mult = self._get_lr_mult()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * lr_mult
    
    def _get_lr_mult(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==================== 新增：梯度累积器 ====================

class GradientAccumulator:
    """
    梯度累积，增大有效batch size
    """
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def scale_loss(self, loss):
        """缩放loss以正确累积梯度"""
        return loss / self.accumulation_steps
    
    def should_step(self) -> bool:
        """是否应该执行优化器更新"""
        self.current_step += 1
        return self.current_step >= self.accumulation_steps
    
    def reset(self):
        """重置累积计数"""
        self.current_step = 0


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
    imgs, masks, prompts, anomalies, classes, specie_names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack(masks, dim=0)
    return imgs, masks, list(prompts), list(anomalies), list(classes), list(specie_names)

def chk(name, x):
    if x is None: 
        return True
    ok = torch.isfinite(x).all().item()
    if not ok:
        print(f"[NAN/INF] {name} =", x.detach().float().cpu())
    return ok

def grad_finite_check(model, key="transformer.decoder"):
    bad = []
    for n,p in model.named_parameters():
        if key in n.lower() and p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                bad.append(n)
                if len(bad) >= 5:
                    break
    if bad:
        print("[BAD GRAD] examples:", bad)
        return False
    return True



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


# ==================== 新增：改进的对齐损失相关函数 ====================

# === Decoder LoRA 相关类 ===
class LoRALinear(nn.Module):
    """LoRA adapter for linear layers - 用于给 decoder 添加低秩适配器"""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def apply_lora_to_decoder(model, rank: int = 8, alpha: float = 16.0):
    """给 decoder 的 attention 层添加 LoRA，显存友好的微调方式"""
    decoder = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
        decoder = model.transformer.decoder
    elif hasattr(model, 'module') and hasattr(model.module, 'transformer'):
        decoder = model.module.transformer.decoder
    
    if decoder is None:
        print("[WARN] Cannot find decoder")
        return 0
    
    lora_count = 0
    device = next(decoder.parameters()).device
    lora_modules = nn.ModuleList()  # 用于存储所有 LoRA 模块，确保参数被正确追踪
    
    # 遍历所有 Linear 层，给 attention 相关的添加 LoRA
    for name, module in list(decoder.named_modules()):
        if isinstance(module, nn.Linear):
            if any(k in name for k in ['in_proj', 'out_proj', 'q_proj', 'k_proj', 'v_proj', 
                                        'self_attn', 'cross_attn', 'multihead']):
                in_f, out_f = module.in_features, module.out_features
                lora = LoRALinear(in_f, out_f, rank=rank, alpha=alpha).to(device)
                
                # 存储 lora adapter 到模块列表
                lora_modules.append(lora)
                
                # 包装 forward（使用闭包捕获正确的 lora 引用）
                original_forward = module.forward
                def make_forward(orig_fwd, lora_adapter):
                    def new_fwd(x):
                        return orig_fwd(x) + lora_adapter(x)
                    return new_fwd
                module.forward = make_forward(original_forward, lora)
                
                lora_count += 1
                print(f"  [LoRA] Added to {name}")
    
    # 将 LoRA 模块列表注册到 decoder，确保参数能被追踪
    decoder.decoder_lora_modules = lora_modules
    
    # 解冻 query_embed（非常重要！）
    if hasattr(decoder, 'query_embed'):
        if isinstance(decoder.query_embed, nn.Parameter):
            decoder.query_embed.requires_grad = True
            print("[INFO] Unfroze query_embed (Parameter)")
        elif hasattr(decoder.query_embed, 'weight'):
            decoder.query_embed.weight.requires_grad = True
            print("[INFO] Unfroze query_embed.weight")
    
    print(f"[INFO] Applied LoRA (rank={rank}) to {lora_count} decoder modules")
    
    # 打印 LoRA 参数数量
    lora_params = sum(p.numel() for lora in lora_modules for p in lora.parameters())
    print(f"[INFO] Total LoRA parameters in decoder: {lora_params:,}")
    
    return lora_count


def unfreeze_decoder_selectively(model, mode="last_layer"):
    """选择性解冻 decoder 层"""
    decoder = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
        decoder = model.transformer.decoder
    elif hasattr(model, 'module') and hasattr(model.module, 'transformer'):
        decoder = model.module.transformer.decoder
    
    if decoder is None:
        print("[WARN] Cannot find decoder")
        return
    
    # 首先冻结所有 decoder 参数
    for p in decoder.parameters():
        p.requires_grad = False
    
    if mode == "all":
        for p in decoder.parameters():
            p.requires_grad = True
    elif mode == "last_layer":
        if hasattr(decoder, 'layers') and len(decoder.layers) > 0:
            for p in decoder.layers[-1].parameters():
                p.requires_grad = True
    elif mode == "last_2_layers":
        if hasattr(decoder, 'layers') and len(decoder.layers) >= 2:
            for layer in decoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
    elif mode == "cross_attn":
        for name, param in decoder.named_parameters():
            if 'cross_attn' in name.lower() or 'multihead_attn' in name.lower():
                param.requires_grad = True
    
    # 解冻 query_embed
    if hasattr(decoder, 'query_embed'):
        if isinstance(decoder.query_embed, nn.Parameter):
            decoder.query_embed.requires_grad = True
        elif hasattr(decoder.query_embed, 'weight'):
            decoder.query_embed.weight.requires_grad = True
    
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in decoder.parameters())
    print(f"[INFO] Decoder unfroze ({mode}): {trainable:,} / {total:,} params")


def get_prompt_group_labels(prompt_lists, class_names, device):
    """
    改进的分组策略：只按 prompt 内容分组，忽略 class_name。
    
    原问题: 使用 (class_name, prompt_tuple) 导致每个类别形成独立组
    例如 batch=8 来自 8 个类别 -> 8 个组各 1 个样本 -> 无多正样本对!
    
    改进: 只用 prompt 内容分组
    - "crack" 在 bottle 和 tile 中语义相同，应该是正样本对
    - 相同的缺陷描述形成一组
    """
    group_to_id = {}
    labels = []
    
    for prompts in prompt_lists:
        # 只用 prompt 内容作为 key（忽略 class_name）
        # 这样不同类别的相同缺陷类型可以形成正样本对
        key_prompts = prompts[1:] if (prompts is not None and len(prompts) > 1) else []
        key = tuple(sorted(key_prompts)) if key_prompts else ("__normal__",)
        
        if key not in group_to_id:
            group_to_id[key] = len(group_to_id)
        labels.append(group_to_id[key])
    
    return torch.tensor(labels, dtype=torch.long, device=device)


def get_prompt_group_labels_by_anomaly(prompt_lists, is_anomaly, device):
    """
    更简单的分组策略：按异常/正常二分类分组
    - 所有异常样本 label = 1
    - 所有正常样本 label = 0
    
    优点: 保证批次内总有正样本对（只要有多个异常或多个正常样本）
    """
    labels = [1 if anom else 0 for anom in is_anomaly]
    return torch.tensor(labels, dtype=torch.long, device=device)


def supervised_contrastive_loss(
    v: torch.Tensor,      # (B, D) visual embeddings
    t: torch.Tensor,      # (B, D) text/prompt embeddings  
    labels: torch.Tensor, # (B,) 每个样本的类别/组ID
    temp: float = 0.1     # 增大温度以稳定训练（原 0.07 太尖锐）
):
    """
    改进的 Multi-positive Supervised Contrastive Loss.
    
    关键改进:
    1. 使用更大的温度 (0.1 而非 0.07) - 软化相似度分布
    2. 使用向量化实现替代循环 - 更稳定的梯度
    3. 添加诊断信息打印
    """
    B = v.shape[0]
    device = v.device
    
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    
    # 计算相似度矩阵 (B, B)
    sim_v2t = (v @ t.t()) / temp
    sim_t2v = sim_v2t.t()
    
    # 构建正样本 mask: labels[i] == labels[j] 时为 True
    labels_col = labels.view(-1, 1)
    positive_mask = (labels_col == labels_col.t()).float()  # (B, B)
    
    # 对角线 mask
    eye_mask = torch.eye(B, device=device)
    
    # 排除自身的正样本 mask
    pos_mask_no_self = positive_mask * (1 - eye_mask)
    
    # 检查是否有正样本（排除只有自己一个的情况）
    num_positives_per_sample = pos_mask_no_self.sum(dim=1)
    has_positives = num_positives_per_sample > 0
    
    # 打印诊断信息（每 100 step 打印一次，通过外部控制）
    n_with_pos = has_positives.sum().item()
    n_unique_groups = len(torch.unique(labels))
    
    if has_positives.sum() == 0:
        # 退化到标准 InfoNCE（每个样本只有自己是正样本）
        # 这种情况下 loss ≈ log(B)
        labels_diag = torch.arange(B, device=device)
        loss_v2t = F.cross_entropy(sim_v2t, labels_diag)
        loss_t2v = F.cross_entropy(sim_t2v, labels_diag)
        return 0.5 * (loss_v2t + loss_t2v)
    
    # 向量化的 SupCon loss 实现（比循环更稳定）
    # 对于每个样本 i，loss = -log(sum_j∈P(i) exp(s_ij) / sum_k exp(s_ik))
    
    # exp(sim) for all pairs
    exp_sim_v2t = torch.exp(sim_v2t)  # (B, B)
    
    # 分子：正样本对的 exp(sim) 之和（包含自身）
    pos_exp_sum = (exp_sim_v2t * positive_mask).sum(dim=1)  # (B,)
    
    # 分母：所有样本对的 exp(sim) 之和
    all_exp_sum = exp_sim_v2t.sum(dim=1)  # (B,)
    
    # 避免 log(0) 和数值不稳定
    eps = 1e-8
    loss_v2t = -torch.log(pos_exp_sum / (all_exp_sum + eps) + eps).mean()
    
    # 对称方向 t->v
    exp_sim_t2v = torch.exp(sim_t2v)
    pos_exp_sum_t2v = (exp_sim_t2v * positive_mask).sum(dim=1)
    all_exp_sum_t2v = exp_sim_t2v.sum(dim=1)
    loss_t2v = -torch.log(pos_exp_sum_t2v / (all_exp_sum_t2v + eps) + eps).mean()
    
    return 0.5 * (loss_v2t + loss_t2v)


def query_text_alignment_loss(
    decoder_hs: torch.Tensor,      
    prompt_proto: torch.Tensor,    
    indices: list,                 
    temp: float = 0.2,             # 增大温度！原 0.07 太尖锐
    selection_weight: float = 0.0, # 禁用不稳定的 selection loss
    top_k: int = 64                # 只在 top-k 相似的 query 中竞争
):
    """
    改进的 Query-Text Alignment Loss.
    
    关键改进:
    1. 温度从 0.07 增大到 0.2（软化分布，避免 log(Q) 下界）
    2. 使用 top-k 采样减少负样本数量（Q=900 -> top_k=64）
    3. 移除不稳定的 selection loss
    4. 添加 stop-gradient 稳定训练
    
    原问题: Q=900, temp=0.07 时 loss 下界 ≈ log(900) ≈ 6.8
    改进后: top_k=64, temp=0.2 时 loss 下界 ≈ log(64) ≈ 4.2，且更易收敛
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    # 归一化
    hs_norm = F.normalize(decoder_hs, dim=-1)
    p_norm = F.normalize(prompt_proto, dim=-1)
    
    loss = torch.tensor(0.0, device=device)
    valid_count = 0
    
    for b in range(B):
        src_q, tgt_q = indices[b]
        
        if src_q.numel() == 0:
            continue
        
        # 取第一个匹配的 query
        matched_q_idx = int(src_q[0].item())
        if matched_q_idx >= Q:
            continue
        
        h_matched = hs_norm[b, matched_q_idx]  # (D,)
        p = p_norm[b]  # (D,)
        
        # 计算所有 query 与 prompt 的相似度
        all_sim = (hs_norm[b] @ p) / temp  # (Q,)
        
        # Top-k 负采样：只在最相似的 top_k 个 query 中计算 softmax
        # 这大大减少了负样本数量，使 loss 更容易下降
        effective_k = min(top_k, Q)
        
        if Q > effective_k:
            # 使用 detach 防止 top-k 选择过程产生梯度
            _, top_indices = torch.topk(all_sim.detach(), k=effective_k)
            
            # 确保匹配的 query 在 top_k 中
            if matched_q_idx not in top_indices:
                # 把 matched query 加进去，移除最后一个
                top_indices = torch.cat([
                    torch.tensor([matched_q_idx], device=device),
                    top_indices[:-1]
                ])
            
            # 在 top_k 中找到 matched_q_idx 的位置
            target_in_topk = (top_indices == matched_q_idx).nonzero(as_tuple=True)[0]
            
            # 只取 top_k 的相似度
            sim_topk = all_sim[top_indices]  # (top_k,)
            
            # 计算 cross entropy
            loss = loss + F.cross_entropy(sim_topk.unsqueeze(0), target_in_topk)
        else:
            # Q <= top_k，使用全部
            target = torch.tensor([matched_q_idx], device=device)
            loss = loss + F.cross_entropy(all_sim.unsqueeze(0), target)
        
        valid_count += 1
    
    if valid_count > 0:
        loss = loss / valid_count
    
    return loss

def compute_visual_embedding_with_background(
    decoder_features,    # (B, C, H, W) 或 (B, Q, D)
    masks: torch.Tensor, # (B, H, W) GT masks
    is_anomaly: list,    # 每张图是否是异常
    device
):
    """
    为正常图和异常图分别计算视觉 embedding：
    - 异常图：GT mask 区域的 pooling（缺陷区域）
    - 正常图：全图 global average pooling（背景/正常区域）
    
    这解决了正常样本对 align 零贡献的问题。
    """
    B = masks.shape[0]
    
    if decoder_features is None:
        return None, None
    
    # 处理不同形状的 decoder_features
    if decoder_features.dim() == 4:
        # (B, C, H, W) 空间特征
        _, C, Hf, Wf = decoder_features.shape
        feat = decoder_features.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
        
        # 下采样 mask 到特征图大小
        masks_ds = F.interpolate(
            masks.unsqueeze(1).float(), 
            size=(Hf, Wf), 
            mode='nearest'
        ).squeeze(1)
        masks_flat = masks_ds.view(B, Hf * Wf, 1)
        use_spatial = True
    elif decoder_features.dim() == 3:
        # (B, Q, D) query 特征
        feat = decoder_features
        C = feat.shape[-1]
        masks_flat = None
        use_spatial = False
    else:
        return None, None
    
    embeddings = []
    is_background = []
    
    for b in range(B):
        if is_anomaly[b]:
            # 异常图：mask 区域 pooling
            if use_spatial and masks_flat is not None:
                mask_b = masks_flat[b]
                pos_count = mask_b.sum().clamp(min=1.0)
                emb = (feat[b] * mask_b).sum(dim=0) / pos_count
            else:
                emb = feat[b].mean(dim=0)
            embeddings.append(emb)
            is_background.append(False)
        else:
            # 正常图：全图 global pooling（背景锚点）
            emb = feat[b].mean(dim=0)
            embeddings.append(emb)
            is_background.append(True)
    
    embeddings = torch.stack(embeddings, dim=0)
    is_background_tensor = torch.tensor(is_background, dtype=torch.bool, device=device)
    
    return embeddings, is_background_tensor


def align_loss_with_background_margin(
    prompt_proto: torch.Tensor,     # (B, D)
    visual_embed: torch.Tensor,     # (B, D)
    is_background: torch.Tensor,    # (B,) bool
    group_labels: torch.Tensor,     # (B,) 分组标签
    temp: float = 0.1,              # 增大温度（原 0.07）
    margin: float = 0.3             # 降低 margin（原 0.5）
):
    """
    改进的对齐损失：
    1. 同类对齐：supervised contrastive（使用改进版）
    2. Margin 约束：defect prompt 应该远离 background embedding
    
    关键改进:
    1. 使用更大的温度 0.1
    2. 降低 margin 到 0.3 避免过度惩罚
    3. margin loss 权重从 0.5 降低到 0.2
    """
    B = prompt_proto.shape[0]
    device = prompt_proto.device
    
    p_norm = F.normalize(prompt_proto, dim=-1)
    v_norm = F.normalize(visual_embed, dim=-1)
    
    # 1. Supervised contrastive loss（同类对齐）- 使用改进后的温度
    loss_align = supervised_contrastive_loss(v_norm, p_norm, group_labels, temp)
    
    # 2. Margin loss: defect prompts 应该远离 background embeddings
    anomaly_indices = torch.where(~is_background)[0]
    normal_indices = torch.where(is_background)[0]
    
    loss_margin = torch.tensor(0.0, device=device)
    
    if len(anomaly_indices) > 0 and len(normal_indices) > 0:
        # 批量计算而非循环（更高效）
        p_anom = p_norm[anomaly_indices]  # (Na, D)
        v_bg = v_norm[normal_indices]     # (Nn, D)
        
        # 计算所有异常 prompt 与正常 visual 的相似度
        sim_matrix = p_anom @ v_bg.t()  # (Na, Nn)
        
        # Hinge loss: max(0, sim + margin)
        loss_margin = F.relu(sim_matrix + margin).mean()
    
    # 降低 margin loss 权重（原 0.5，改为 0.2）
    return loss_align + 0.2 * loss_margin

# ==================== 新增函数结束 ====================

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


def contrastive_loss_from_pooled(v: torch.Tensor, t: torch.Tensor, temp: float = 0.07, 
                                  group_labels: torch.Tensor = None):
    """
    Symmetric contrastive loss between v and t (both shape [B, D]).
    如果提供 group_labels，使用 supervised contrastive（同组互为正样本）。
    否则退化为标准 InfoNCE。
    """
    assert v.dim() == 2 and t.dim() == 2 and v.shape[0] == t.shape[0]
    
    if group_labels is not None:
        return supervised_contrastive_loss(v, t, group_labels, temp)
    
    # 原始 InfoNCE fallback
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = (v @ t.t()) / temp
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
    
    # ==================== 修复：正确解析 meta.json 获取类别列表 ====================
    # 问题：旧代码使用 meta.keys() 导致 class_list = ["train", "test"]（2类）
    # 修复：从 meta["train"]/meta["test"] 的子字典中提取真正的类别名
    
    print("=" * 80)
    print("[CLASS LIST DIAGNOSTIC] 类别解析诊断开始")
    print("=" * 80)
    print(f"[DIAG] meta.json 路径: {args.meta_path}")
    print(f"[DIAG] meta 类型: {type(meta)}")
    print(f"[DIAG] meta 顶层 keys: {list(meta.keys()) if isinstance(meta, dict) else 'N/A'}")
    
    # 打印每个顶层key的结构
    if isinstance(meta, dict):
        for top_key in list(meta.keys())[:5]:  # 最多打印5个
            val = meta[top_key]
            if isinstance(val, dict):
                print(f"[DIAG]   meta['{top_key}'] 是 dict，子keys: {list(val.keys())[:10]}... (共{len(val)}个)")
            elif isinstance(val, list):
                print(f"[DIAG]   meta['{top_key}'] 是 list，长度: {len(val)}")
            else:
                print(f"[DIAG]   meta['{top_key}'] 类型: {type(val)}")
    
    def _infer_class_list_from_meta(meta_dict):
        """从 meta.json 正确推断类别列表"""
        if not isinstance(meta_dict, dict):
            print("[DIAG] meta 不是 dict，返回空列表")
            return []
        
        # 情况1: meta 直接有 "classes" 或 "class_list" key
        if "classes" in meta_dict and isinstance(meta_dict["classes"], list):
            result = sorted(list(meta_dict["classes"]))
            print(f"[DIAG] 使用 meta['classes'] 提取类别: {result}")
            return result
        if "class_list" in meta_dict and isinstance(meta_dict["class_list"], list):
            result = sorted(list(meta_dict["class_list"]))
            print(f"[DIAG] 使用 meta['class_list'] 提取类别: {result}")
            return result
        
        # 情况2: meta 结构是 {"train": {cls: [...]}, "test": {cls: [...]}}
        # 需要从 train/test 的子字典中提取类别名
        if "train" in meta_dict or "test" in meta_dict:
            train_keys = []
            test_keys = []
            if isinstance(meta_dict.get("train"), dict):
                train_keys = list(meta_dict["train"].keys())
                print(f"[DIAG]   从 meta['train'] 提取: {train_keys}")
            if isinstance(meta_dict.get("test"), dict):
                test_keys = list(meta_dict["test"].keys())
                print(f"[DIAG]   从 meta['test'] 提取: {test_keys}")
            combined = set(train_keys + test_keys)
            if combined:
                result = sorted(list(combined))
                print(f"[DIAG] 使用 train/test 子keys 合并提取类别")
                return result
        
        # 情况3: fallback - 但要排除 "train", "test" 这种顶层key
        all_keys = list(meta_dict.keys())
        filtered_keys = [k for k in all_keys if k not in ("train", "test", "val", "validation")]
        if filtered_keys:
            print(f"[DIAG] 使用 fallback (过滤后的顶层keys): {filtered_keys}")
            return sorted(filtered_keys)
        
        print(f"[WARN] 使用最终 fallback (所有顶层keys): {all_keys}")
        return sorted(all_keys)
    
    class_list = _infer_class_list_from_meta(meta)
    
    print("-" * 80)
    print(f"[RESULT] 最终 class_list: {class_list}")
    print(f"[RESULT] 类别总数: {len(class_list)}")
    print("-" * 80)
    
    # 安全检查：如果 class_list 看起来像 split 名称而不是类别名称
    if len(class_list) <= 2 and set(class_list).issubset({"train", "test", "val", "validation"}):
        error_msg = (
            f"\n{'='*80}\n"
            f"[CRITICAL ERROR] class_list={class_list} 看起来像 split 名称，不是类别名称!\n\n"
            f"这会导致 PerClassTemplatePromptLearner 只有 {len(class_list)} 个类别模板，\n"
            f"而不是期望的 15 个类别。checkpoint 加载时会报 shape mismatch 错误！\n\n"
            f"请检查 meta.json 结构。期望格式:\n"
            f'  {{"train": {{"bottle": [...], "cable": [...]}}, "test": {{...}}}}\n\n'
            f"实际顶层 keys: {list(meta.keys())}\n"
            f"{'='*80}"
        )
        print(error_msg)
        raise ValueError(error_msg)
    
    # 额外诊断：打印前5个类别的详细信息
    print(f"[DIAG] 前5个类别详情:")
    for i, cls_name in enumerate(class_list[:5]):
        # 尝试获取该类别的样本数量
        train_count = 0
        test_count = 0
        if isinstance(meta.get("train"), dict) and cls_name in meta["train"]:
            train_count = len(meta["train"][cls_name]) if isinstance(meta["train"][cls_name], list) else 0
        if isinstance(meta.get("test"), dict) and cls_name in meta["test"]:
            test_count = len(meta["test"][cls_name]) if isinstance(meta["test"][cls_name], list) else 0
        print(f"[DIAG]   {i+1}. '{cls_name}': train={train_count} samples, test={test_count} samples")
    
    if len(class_list) > 5:
        print(f"[DIAG]   ... 还有 {len(class_list) - 5} 个类别")
    
    print("=" * 80)
    print("[CLASS LIST DIAGNOSTIC] 类别解析诊断结束")
    print("=" * 80)
    
    args.class_list = class_list
    # ==============================================================================

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
            enable_parallel_lora=args.enable_parallel_lora,
            parallel_lora_rank=args.parallel_lora_rank,
            parallel_lora_alpha=args.parallel_lora_alpha,
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
        nl = n.lower()
        if ("lora" in nl) or ("out_adapter" in nl) or ("prompt_learner" in nl) or ("prompt" in nl) \
           or ("segmentation_head" in nl):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # === 新增：Decoder 解冻/LoRA 配置 ===
    # 获取实际的 model_core（处理 DDP 包装的情况）
    model_for_decoder = model.module if hasattr(model, 'module') else model
    
    if getattr(args, 'unfreeze_decoder', 'none') != 'none':
        print(f"[INFO] Unfreezing decoder with mode: {args.unfreeze_decoder}")
        unfreeze_decoder_selectively(model_for_decoder, mode=args.unfreeze_decoder)
    
    if getattr(args, 'decoder_lora', False):
        print(f"[INFO] Applying LoRA to decoder (rank={args.decoder_lora_rank}, alpha={args.decoder_lora_alpha})")
        apply_lora_to_decoder(model_for_decoder, rank=args.decoder_lora_rank, alpha=args.decoder_lora_alpha)

    prompt_and_lora: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    decoder_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        if ("lora" in nl) or ("prompt" in nl) or ("template" in nl) or ("out_adapter" in nl):
            prompt_and_lora.append(p)
        elif "decoder" in nl:
            decoder_params.append(p)
        else:
            other_params.append(p)
    print(f"[INFO] trainable params: prompt/LoRA={len(prompt_and_lora)}, decoder={len(decoder_params)}, others={len(other_params)}")
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

    # Build optimizer with separate groups for different learning rates
    param_groups = [
        {"params": prompt_and_lora, "lr": args.lr_prompt},
        {"params": other_params, "lr": args.lr_main},
    ]
    
    # 新增：为 decoder 参数添加单独的参数组（使用较小的学习率）
    if len(decoder_params) > 0:
        decoder_lr = args.lr_main * 0.5  # decoder 使用更小的学习率以稳定训练
        param_groups.append({"params": decoder_params, "lr": decoder_lr})
        print(f"[INFO] Added {len(decoder_params)} decoder params with lr={decoder_lr}")
    
    if len(learnable_log_vars) > 0:
        param_groups.append({"params": learnable_log_vars, "lr": args.lr_main, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # 新增：AMP 的 GradScaler（只在 CUDA 下启用）
    scaler = GradScaler(
        enabled=(device.type == "cuda"),
        init_scale=2**8,          # 默认 2**16 太激进
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )

    matcher = BinaryHungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)

    # ==================== 新增：学习率调度器和梯度累积器 ====================
    steps_per_epoch = len(dataloader)
    total_steps = args.epochs * steps_per_epoch // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    grad_accum = GradientAccumulator(args.gradient_accumulation)
    
    print("=" * 60)
    print("训练配置:")
    print(f"  总epoch数: {args.epochs}")
    print(f"  每epoch步数: {steps_per_epoch}")
    print(f"  梯度累积: {args.gradient_accumulation}")
    print(f"  有效batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  总优化步数: {total_steps}")
    print(f"  Warmup步数: {warmup_steps} ({args.warmup_ratio*100:.0f}%)")
    print(f"  最终LR比例: {args.min_lr_ratio}")
    print("=" * 60)

    model.train()
    best_loss = float("inf")
    global_optim_step = 0  # 优化器更新计数（用于调度器）
    
    for epoch in range(args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        else:
            pbar = dataloader  # 无进度条，仅迭代
        running_loss = 0.0
        running_steps = 0
        
        # 每个epoch开始时清零梯度
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(pbar):
            images, masks, prompt_lists, is_anomaly, class_names, specie_names = batch
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


                # ---------- matched branch: gather matched preds and GT, then compute losses ----------

                if tgt_idx is None or tgt_idx.numel() == 0:
                    loss_focal = torch.tensor(0.0, device=device)
                    loss_dice = torch.tensor(0.0, device=device)
                else:
                    # gather predictions and GT (as before)
                    pred_matched_ds = pred_masks_ds[batch_idx, src_idx]  # (M, MD, MD)

                    tgt_masks_flat = targets["segments"][tgt_idx]
                    if tgt_masks_flat.dim() == 4:
                        tgt_masks_flat = tgt_masks_flat.squeeze(1)
                    tm_ds = F.interpolate(
                        tgt_masks_flat.unsqueeze(1).float(),
                        size=(MASK_DOWNSAMPLE, MASK_DOWNSAMPLE),
                        mode="nearest"
                    ).squeeze(1)  # (M, MD, MD)

                    num_boxes = float(max(1.0, src_idx.numel()))

                    # --- focal: per-pixel map -> per-mask mean -> normalize ---
                    loss_map = sam_sigmoid_focal_loss(
                        pred_matched_ds, tm_ds, num_boxes,
                        alpha=0.25, gamma=2.0,
                        loss_on_multimask=False, triton=False,
                        reduce=False
                    )  # (M, MD, MD)

                    # per-mask mean over pixels
                    per_mask = loss_map.mean(dim=(1, 2))  # (M,)
                    loss_focal = per_mask.sum() / max(1.0, num_boxes)

                    # --- dice: sam_dice_loss(..., reduce=False) returns per-mask scalars (M,) ---
                    loss_dice_map = sam_dice_loss(
                        pred_matched_ds, tm_ds, num_boxes,
                        loss_on_multimask=False, reduce=False
                    )  # typically shape (M,) for reduce=False

                    # handle both cases robustly:
                    if loss_dice_map.dim() == 3:
                        # (M, H, W) -> collapse to per-mask scalars
                        per_mask_dice = loss_dice_map.mean(dim=(1, 2))
                    elif loss_dice_map.dim() == 1:
                        per_mask_dice = loss_dice_map
                    else:
                        # fallback: flatten trailing dims to compute a per-mask mean
                        per_mask_dice = loss_dice_map.view(loss_dice_map.shape[0], -1).mean(dim=1)

                    loss_dice = per_mask_dice.sum() / max(1.0, num_boxes)

                    # -------------------------
                    # ---------- matched + background losses (FP32 stable) ----------
                    loss_focal = torch.tensor(0.0, device=device)
                    loss_dice  = torch.tensor(0.0, device=device)

                    loss_focal_bg = torch.tensor(0.0, device=device)
                    loss_dice_bg  = torch.tensor(0.0, device=device)
                    num_bg_images = 0

                    # 关键：对 pred_masks_ds 做一次“可反传”的安全化（clamp 会截断极端梯度，nan_to_num 只在出现 nan/inf 时兜底）
                    pred_masks_ds_safe = torch.nan_to_num(pred_masks_ds, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

                    with torch.cuda.amp.autocast(enabled=False):
                        # ===== matched positives =====
                        if tgt_idx is not None and tgt_idx.numel() > 0:
                            pm = pred_masks_ds_safe[batch_idx, src_idx].float()   # (M, MD, MD)

                            tgt_masks_flat = targets["segments"][tgt_idx]
                            if tgt_masks_flat.dim() == 4:
                                tgt_masks_flat = tgt_masks_flat.squeeze(1)
                            tm = F.interpolate(
                                tgt_masks_flat.unsqueeze(1).float(),
                                size=(MASK_DOWNSAMPLE, MASK_DOWNSAMPLE),
                                mode="nearest"
                            ).squeeze(1)  # (M, MD, MD)

                            # 再 clamp 一次，避免 focal 内部极端 logits（尤其是你刚解冻 decoder 时）
                            pm = pm.clamp(-20.0, 20.0)

                            # reduce=False -> per-mask mean -> batch mean
                            fmap = sam_sigmoid_focal_loss(
                                pm, tm, num_boxes=1.0,
                                alpha=0.25, gamma=2.0,
                                loss_on_multimask=False, triton=False,
                                reduce=False
                            )
                            if fmap.dim() == 3:
                                loss_focal = fmap.mean(dim=(1, 2)).mean()
                            else:
                                loss_focal = fmap.view(fmap.shape[0], -1).mean(dim=1).mean()

                            dmap = sam_dice_loss(
                                pm, tm, num_boxes=1.0,
                                loss_on_multimask=False, reduce=False
                            )
                            if dmap.dim() == 3:
                                dmap = dmap.mean(dim=(1, 2))
                            elif dmap.dim() != 1:
                                dmap = dmap.view(dmap.shape[0], -1).mean(dim=1)
                            loss_dice = dmap.mean()

                        # ===== background/unmatched (sample negatives) =====
                        all_q = torch.arange(pred_masks.shape[1], device=device)
                        for b in range(pred_masks.shape[0]):
                            src_q, _ = indices[b]
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
                            sampled_q = unmatched_q[perm]

                            preds_bg = pred_masks_ds_safe[b, sampled_q].float()   # (k, MD, MD)
                            preds_bg = preds_bg.clamp(-20.0, 20.0)
                            zeros = torch.zeros_like(preds_bg)

                            bg_map = sam_sigmoid_focal_loss(
                                preds_bg, zeros, num_boxes=1.0,
                                alpha=0.25, gamma=2.0,
                                loss_on_multimask=False, triton=False,
                                reduce=False
                            )
                            if bg_map.dim() == 3:
                                loss_focal_bg = loss_focal_bg + bg_map.mean(dim=(1, 2)).mean()
                            else:
                                loss_focal_bg = loss_focal_bg + bg_map.view(bg_map.shape[0], -1).mean(dim=1).mean()

                            bg_d = sam_dice_loss(
                                preds_bg, zeros, num_boxes=1.0,
                                loss_on_multimask=False, reduce=False
                            )
                            if bg_d.dim() == 3:
                                bg_d = bg_d.mean(dim=(1, 2))
                            elif bg_d.dim() != 1:
                                bg_d = bg_d.view(bg_d.shape[0], -1).mean(dim=1)
                            loss_dice_bg = loss_dice_bg + bg_d.mean()

                            num_bg_images += 1

                        if num_bg_images > 0:
                            loss_focal_bg = loss_focal_bg / float(num_bg_images)
                            loss_dice_bg  = loss_dice_bg  / float(num_bg_images)

                    # 背景权重：建议先压低，先让 decoder 稳住（你现在 0.1 也可能偏大）
                    bg_w = 0.05
                    loss_focal = loss_focal + bg_w * loss_focal_bg
                    loss_dice  = loss_dice  + bg_w * loss_dice_bg

                    if step % 50 == 0:
                        print(f"[BG LOSS] focal_bg={float(loss_focal_bg):.4f} dice_bg={float(loss_dice_bg):.4f} bg_w={bg_w} num_bg_images={num_bg_images}")


                        
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
                # 改进的 Contrastive Alignment：
                # 1. 使用 supervised contrastive（同类互为正样本）
                # 2. 区分异常/正常样本的 visual embedding
                # 3. 添加 margin loss 推开 defect prompt 和 background
                # 4. 添加 query-level alignment
                # -------------------------
                align_loss = torch.tensor(0.0, device=device)
                query_align_loss = torch.tensor(0.0, device=device)
                
                if args.lambda_align is not None and float(args.lambda_align) > 0.0:
                    # ===== Step 1: 获取 prompt prototype =====
                    prompt_seq = out.get("prompt_seq", None)
                    if prompt_seq is None:
                        try:
                            prompt_seq, _ = model_core.prompt_learner(prompt_lists, device=device)
                        except Exception as e:
                            print("[WARN] cannot obtain prompt_seq from model_core.prompt_learner:", e)
                            prompt_seq = None

                    prompt_proto = None
                    if prompt_seq is not None:
                        last = prompt_seq[-1]
                        if last.dim() == 2:
                            if last.shape[0] == B:
                                prompt_proto = last
                            elif last.shape[1] == B:
                                prompt_proto = last.transpose(0, 1).contiguous()
                            else:
                                prompt_proto = last.reshape(B, -1)[:, : (last.numel() // B)]
                        elif last.dim() == 3:
                            prompt_proto = last if last.shape[0] == B else last[-1]
                        else:
                            prompt_proto = last.reshape(B, -1)[:, : (last.numel() // B)]

                    # ===== Step 2: 获取 visual embedding（区分异常/正常）=====
                    decoder_feat = out.get("decoder_features", None)
                    if decoder_feat is None:
                        decoder_feat = out.get("decoder_hs", None)
                        if decoder_feat is not None and decoder_feat.dim() == 4:
                            decoder_feat = decoder_feat[-1]
                        # 确保 decoder_feat 是 (B, Q, D) 格式
                        if decoder_feat is not None and decoder_feat.dim() == 3:
                            if decoder_feat.shape[0] == Q and decoder_feat.shape[1] == B:
                                decoder_feat = decoder_feat.permute(1, 0, 2).contiguous()
                    
                    # 使用改进的函数：区分异常/正常样本
                    visual_embed, is_background = compute_visual_embedding_with_background(
                        decoder_features=decoder_feat,
                        masks=masks,
                        is_anomaly=is_anomaly,  # 从 batch 数据中获取
                        device=device
                    )
                    
                    if visual_embed is None:
                        visual_embed = torch.zeros((B, prompt_proto.shape[1] if prompt_proto is not None else 128), device=device)
                        is_background = torch.zeros(B, dtype=torch.bool, device=device)

                    # ===== Step 3: 维度对齐 =====
                    if prompt_proto is not None:
                        Dp = prompt_proto.shape[1]
                        Dm = visual_embed.shape[1]
                        if Dm != Dp:
                            if not hasattr(model_core, "_align_proj"):
                                model_core._align_proj = nn.Linear(Dm, Dp).to(device)
                                optimizer.add_param_group({
                                    "params": model_core._align_proj.parameters(),
                                    "lr": args.lr_main,
                                    "weight_decay": 0.0
                                })
                            visual_embed = model_core._align_proj(visual_embed)

                    # ===== Step 4: 计算改进的 align loss =====
                    if prompt_proto is not None:
                        # 选择分组策略
                        if getattr(args, 'use_anomaly_grouping', False):
                            # 简单策略：按异常/正常分组（保证有正样本对）
                            group_labels = get_prompt_group_labels_by_anomaly(prompt_lists, is_anomaly, device)
                            grouping_method = "anomaly"
                        else:
                            # 默认：按 prompt 内容分组
                            group_labels = get_prompt_group_labels(prompt_lists, class_names, device)
                            grouping_method = "prompt"
                        
                        # 诊断信息：检查分组效果
                        n_unique_groups = len(torch.unique(group_labels))
                        n_samples = len(group_labels)
                        if step % 100 == 0:
                            print(f"[ALIGN-GROUP] step={step} method={grouping_method} "
                                  f"n_samples={n_samples} n_unique_groups={n_unique_groups} "
                                  f"has_multi_positive={n_unique_groups < n_samples}")
                        
                        # 使用带 margin 的 supervised contrastive loss
                        align_margin = getattr(args, 'align_margin', 0.3)  # 降低默认 margin
                        align_loss = align_loss_with_background_margin(
                            prompt_proto=prompt_proto,
                            visual_embed=visual_embed,
                            is_background=is_background,
                            group_labels=group_labels,
                            temp=args.align_temp,
                            margin=align_margin
                        )
                        
                        # ===== Step 5: Query-level alignment（关键改进）=====
                        lambda_query_align = getattr(args, 'lambda_query_align', 0.5)
                        if lambda_query_align > 0:
                            decoder_hs = out.get("decoder_hs", None)
                            if decoder_hs is not None:
                                # 确保 decoder_hs 是 (B, Q, D) 格式
                                if decoder_hs.dim() == 4:
                                    decoder_hs = decoder_hs[-1]
                                if decoder_hs.dim() == 3:
                                    if decoder_hs.shape[0] == Q and decoder_hs.shape[1] == B:
                                        decoder_hs = decoder_hs.permute(1, 0, 2).contiguous()
                                
                                # 计算 query-level alignment loss（使用改进的参数）
                                query_align_loss = query_text_alignment_loss(
                                    decoder_hs=decoder_hs,
                                    prompt_proto=prompt_proto,
                                    indices=indices,
                                    temp=getattr(args, 'query_align_temp', 0.2),  # 使用更大的温度
                                    top_k=getattr(args, 'query_align_top_k', 64)   # 使用 top-k 采样
                                )

                        # ===== Diagnostics =====
                        if (step % getattr(args, "log_freq", 100)) == 0:
                            p_norm = F.normalize(prompt_proto, dim=1)
                            m_norm = F.normalize(visual_embed, dim=1)
                            sim = (p_norm @ m_norm.t()) / float(args.align_temp)
                            
                            # 计算同组/异组的相似度统计
                            same_group = (group_labels.unsqueeze(0) == group_labels.unsqueeze(1))
                            pos_sim = sim[same_group].mean().item() if same_group.sum() > 0 else 0
                            neg_sim = sim[~same_group].mean().item() if (~same_group).sum() > 0 else 0
                            
                            print(f"[ALIGN] step={step} align_loss={align_loss.item():.6f} "
                                  f"query_align={query_align_loss.item():.6f} "
                                  f"pos_sim={pos_sim:.4f} neg_sim={neg_sim:.4f}")
                            
                            # 打印 background vs anomaly embedding 统计
                            n_bg = is_background.sum().item()
                            n_anom = (~is_background).sum().item()
                            print(f"[BG/ANOM] n_background={n_bg}, n_anomaly={n_anom}")

                            # TSNE: 改为每个epoch保存一次（在最后一个step时保存）
                            is_last_step = (step == len(dataloader) - 1)
                            if is_last_step and B >= 4:
                                ns = min(getattr(args, "tsne_samples", 64), B)
                                sel = np.random.choice(B, ns, replace=False)
                                p_sample = p_norm[sel].detach().cpu().numpy()
                                m_sample = m_norm[sel].detach().cpu().numpy()
                                labels_sample = group_labels[sel].cpu().numpy()
                                X = np.concatenate([p_sample, m_sample], axis=0)
                                try:
                                    Z = TSNE(n_components=2, perplexity=min(30, ns-1), init='pca').fit_transform(X)
                                    plt.figure(figsize=(8, 8))
                                    # prompt embeddings
                                    scatter1 = plt.scatter(Z[:ns, 0], Z[:ns, 1], c=labels_sample, 
                                                          cmap='tab10', marker='o', alpha=0.8, s=50, label='prompt')
                                    # visual embeddings
                                    scatter2 = plt.scatter(Z[ns:, 0], Z[ns:, 1], c=labels_sample, 
                                                          cmap='tab10', marker='x', alpha=0.8, s=50, label='visual')
                                    plt.legend(fontsize=12)
                                    plt.title(f"t-SNE Epoch {epoch+1} (color=group)", fontsize=14)
                                    plt.xlabel("Dimension 1", fontsize=12)
                                    plt.ylabel("Dimension 2", fontsize=12)
                                    tsne_out_dir = os.path.join(args.log_dir, "tsne")
                                    os.makedirs(tsne_out_dir, exist_ok=True)
                                    plt.savefig(os.path.join(tsne_out_dir, f"tsne_epoch{epoch+1:02d}.png"), dpi=150, bbox_inches='tight')
                                    plt.close()
                                    print(f"[INFO] Saved t-SNE visualization for epoch {epoch+1}")
                                except Exception as e:
                                    print("[WARN] TSNE failed:", e)
                else:
                    align_loss = torch.tensor(0.0, device=device)
                    query_align_loss = torch.tensor(0.0, device=device)
                # -------------------------


                # Combine losses: 包含新的 query_align_loss
                lambda_query_align = getattr(args, 'lambda_query_align', 0.5)
                
                if args.use_learned_loss_weights and len(learnable_log_vars) == 3:
                    loss_main = (torch.exp(-log_var_focal) * loss_focal + log_var_focal) + \
                                (torch.exp(-log_var_dice)  * loss_dice  + log_var_dice) + \
                                (torch.exp(-log_var_iou)   * loss_iou   + log_var_iou)
                    total_loss = loss_main + args.presence_weight * loss_presence + \
                                 args.lambda_align * align_loss + lambda_query_align * query_align_loss
                else:
                    total_loss = args.loss_alpha * loss_focal + args.loss_beta * loss_dice + args.loss_gamma * loss_iou
                    total_loss = total_loss + args.presence_weight * loss_presence + \
                                 args.lambda_align * align_loss + lambda_query_align * query_align_loss

                loss = total_loss

            # ===== AMP autocast 结束 =====
        
            if not torch.isfinite(loss):
                print(f"[WARN] Skip batch with non-finite loss (loss={loss.item()}, focal={loss_focal.item()}, dice={loss_dice.item()}, iou={loss_iou.item()})")
                continue

            # 在 loss.backward() 之前
            chk("loss_total", loss)

            # 你真正的分割loss就是 focal/dice/iou 的组合（用同一套权重）
            loss_seg = args.loss_alpha * loss_focal + args.loss_beta * loss_dice + args.loss_gamma * loss_iou
            chk("loss_seg", loss_seg)

            chk("align_loss", align_loss)
            chk("query_align_loss", query_align_loss)

            # 变量名应为 loss_presence
            chk("presence_loss", loss_presence)
            
            # ==================== 梯度累积逻辑 ====================
            # 缩放loss用于梯度累积
            scaled_loss = grad_accum.scale_loss(loss)
            scaler.scale(scaled_loss).backward()

            # 只有在累积完成后才更新参数
            if grad_accum.should_step():
                # 1) 先 unscale 再检查/clip
                scaler.unscale_(optimizer)

                # 2) clip（先用 1.0 或 0.5 都行，先把 nan 压住）
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                # 3) 再做有限性检查（更可信）
                ok = grad_finite_check(model, "transformer.decoder")
                if not ok:
                    print("[WARN] non-finite grads -> skip step, scaler downscale")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    grad_accum.reset()
                    continue

                # --- Diagnostic: grad norms for prompt-related params ---
                grad_finite_check(model, "transformer.decoder")
                if step == 0:
                    for n, p in model.named_parameters():
                        nl = n.lower()
                        if "transformer.decoder" in nl and p.requires_grad:
                            print("[GRAD CHECK]", n, "grad_is_none=", (p.grad is None),
                                  "grad_norm=", (p.grad.norm().item() if p.grad is not None else None))
                            break
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
                optimizer.zero_grad(set_to_none=True)
                
                # 更新学习率调度器
                scheduler.step()
                global_optim_step += 1
                
                grad_accum.reset()
            # ==================== 梯度累积逻辑结束 ====================

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
            if is_main_process and writer is not None:
                writer.add_scalar("loss/query_align", query_align_loss.item(), global_step)
            # 记录当前学习率
            if is_main_process and writer is not None:
                current_lrs = scheduler.get_lr()
                writer.add_scalar("lr/prompt", current_lrs[0], global_step)
                if len(current_lrs) > 1:
                    writer.add_scalar("lr/main", current_lrs[1], global_step)
            

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
    
    # === Decoder 解冻/LoRA 配置 ===
    parser.add_argument("--unfreeze_decoder", type=str, default="none",
                        choices=["none", "last_layer", "last_2_layers", "cross_attn", "all"],
                        help="Decoder unfreezing mode: none|last_layer|last_2_layers|cross_attn|all")
    parser.add_argument("--decoder_lora", action="store_true",
                        help="Apply LoRA to decoder (memory-efficient alternative to unfreezing)")
    parser.add_argument("--decoder_lora_rank", type=int, default=8,
                        help="LoRA rank for decoder (higher = more capacity, more memory)")
    parser.add_argument("--decoder_lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling for decoder")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard base log directory.")
    parser.add_argument("--save_dir", type=str, default="./ckpt", help="Base directory to save checkpoints.")
    parser.add_argument("--sam3_ckpt", type=str, default=None, help="Path to pretrained SAM3 checkpoint to load.")
    parser.add_argument("--use_official", action="store_true", help="Use official builder + checkpoint before PEFT.")
    parser.add_argument("--balance", action="store_true", help="Enable anomaly/non-anomaly weighted sampler.")
    parser.add_argument("--neg_samples_per_image", type=int, default=50, help="Max negative (unmatched) queries to sample per image for background loss")
    parser.add_argument("--lambda_align", type=float, default=0.1, help="weight for contrastive alignment loss (InfoNCE)")
    parser.add_argument("--align_temp", type=float, default=0.1, help="temperature for contrastive alignment (increased from 0.07 for stability)")
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
    
    # === 新增: align loss 相关参数 ===
    parser.add_argument("--use_anomaly_grouping", action="store_true", 
                        help="Use simpler anomaly/normal grouping instead of prompt-based grouping for align loss")
    parser.add_argument("--query_align_top_k", type=int, default=64, 
                        help="Top-k queries to compete in query alignment loss (reduces from Q=900 to top_k)")
    parser.add_argument("--query_align_temp", type=float, default=0.2,
                        help="Temperature for query alignment loss (higher = softer, easier to learn)")


    #--------------- Diagnostic logging args ---------------
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (steps) for align diagnostics")
    parser.add_argument("--tsne_freq", type=int, default=500, help="TSNE save frequency (steps)")
    parser.add_argument("--tsne_samples", type=int, default=64, help="Number of samples for TSNE projection")
    
    parser.add_argument("--align_margin", type=float, default=0.5,help="Margin for pushing defect prompts away from background embeddings")
    parser.add_argument("--lambda_query_align", type=float, default=0.5,help="Weight for query-level alignment loss (让 prompt 选中正确的 query)")

    # ==================== 新增：训练优化参数 ====================
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup步数占总步数的比例 (默认0.1即10%)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="最终学习率与初始学习率的比值 (默认0.01)")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="梯度累积步数 (默认1即不累积，设为2则有效batch=batch_size*2)")

    args = parser.parse_args()
    main(args)