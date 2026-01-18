import os, sys, random
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
from model_wrapper import FineTuneSAM3Official

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from collections import defaultdict

import json
import io

def few_shot_subsample_entries(entries, shots_per_specie=5, seed=42, verbose=True, balance_good_by_specie: bool = False):
    """Few-Shot 采样"""
    import random
    random.seed(seed)
    
    groups = defaultdict(list)
    for idx, entry in enumerate(entries):
        key = (entry.cls_name, entry.specie_name, int(entry.anomaly))
        groups[key].append(idx)
    
    if verbose:
        print(f"\n[Few-Shot] {shots_per_specie}-shot, 原始: {len(entries)}")
    
    sampled_indices = []
    class_defect_counts = defaultdict(int)
    normal_groups = {}
    
    # 采样缺陷
    for (cls, specie, anom), indices in sorted(groups.items()):
        if anom == 0:
            normal_groups[(cls, specie)] = indices
        else:
            n = min(shots_per_specie, len(indices))
            sampled_indices.extend(random.sample(indices, n))
            class_defect_counts[cls] += n
            if verbose:
                print(f"  {cls}/{specie}: {len(indices)} -> {n}")
    
    # 采样 normal（匹配缺陷数）
    for cls, defect_n in sorted(class_defect_counts.items()):
        if not balance_good_by_specie:
            normal_idx = [i for (c, s), idx in normal_groups.items() if c == cls for i in idx]
            if normal_idx:
                n = min(defect_n, len(normal_idx))
                sampled_indices.extend(random.sample(normal_idx, n))
                if verbose:
                    print(f"  {cls}/good: {len(normal_idx)} -> {n}")
        else:
            per_specie_defect = defaultdict(int)
            for (c, specie, anom), indices in groups.items():
                if c == cls and anom == 1:
                    per_specie_defect[specie] += min(shots_per_specie, len(indices))
            for specie, need in sorted(per_specie_defect.items()):
                normal_idx = normal_groups.get((cls, specie), [])
                if normal_idx:
                    n = min(int(need), len(normal_idx))
                    sampled_indices.extend(random.sample(normal_idx, n))
                    if verbose:
                        print(f"  {cls}/{specie}/good: {len(normal_idx)} -> {n}")
    
    result = [entries[i] for i in sorted(sampled_indices)]
    if verbose:
        n_anom = sum(1 for e in result if e.anomaly == 1)
        print(f"[Few-Shot] 结果: {len(result)} (缺陷:{n_anom}, 正常:{len(result)-n_anom})\n")
    return result

# =============================================================================
# 修改 1: AnomalyFeatureBank 增强版（支持正交化去冗余）
# =============================================================================

class AnomalyFeatureBankV2:
    """
    【v2.2 增强】异常特征 Memory Bank - 支持正交化去冗余
    
    新增功能：
    1. 入库前与 normal prototype 正交化（去除类别特异性）
    2. 支持延迟启用（warm_up_ratio）
    3. 支持聚类去重（可选）
    """
    
    def __init__(
        self, 
        max_size: int = 2048,
        dim: int = 256,
        min_fill_ratio: float = 0.5,
        warm_up_ratio: float = 0.3,      # 【新增】前 30% 不启用
        orthogonalize: bool = True,       # 【新增】是否正交化
    ):
        self.max_size = max_size
        self.dim = dim
        self.min_fill_ratio = min_fill_ratio
        self.warm_up_ratio = warm_up_ratio
        self.orthogonalize = orthogonalize
        
        self.bank = torch.zeros(max_size, dim)
        self.ptr = 0
        self.total_enqueued = 0
        self.is_ready = False
        
        # 用于正交化的 normal prototype（运行时更新）
        self.normal_proto_cache = None
        
    def update_normal_proto(self, proto_normal: torch.Tensor):
        """更新 normal prototype 缓存（每个 step 调用）"""
        # 使用 EMA 更新
        proto = F.normalize(proto_normal.mean(dim=0).detach().cpu().float(), dim=-1)  # (D,)
        if self.normal_proto_cache is None:
            self.normal_proto_cache = proto
        else:
            self.normal_proto_cache = 0.9 * self.normal_proto_cache + 0.1 * proto
    
    def _orthogonalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        将特征与 normal prototype 正交化
        
        公式：f_orth = f - (f · n) * n
        目的：去除与 normal 共享的成分，保留"异常特有"的语义
        """
        if self.normal_proto_cache is None or not self.orthogonalize:
            return features
        
        features = features.float()
        n = self.normal_proto_cache.float().to(features.device)  # (D,)
        
        # 计算投影
        proj = (features @ n).unsqueeze(-1) * n.unsqueeze(0)  # (N, D)
        
        # 正交化
        f_orth = features - proj
        
        # 重新归一化
        f_orth = F.normalize(f_orth, dim=-1)
        
        return f_orth
        
    def enqueue(self, features: torch.Tensor, current_step_ratio: float = 0.0):
        """
        存入新的异常特征
        
        Args:
            features: (N, D) 来自 matched queries 的特征
            current_step_ratio: 当前训练进度 (0~1)
        """
        if features.numel() == 0:
            return
        
        # 【新增】warm-up 阶段不入库
        if current_step_ratio < self.warm_up_ratio:
            return
            
        features = F.normalize(features.detach().cpu().float(), dim=-1)
        
        # 【新增】正交化
        features = self._orthogonalize_features(features)
        
        batch_size = features.shape[0]
        
        if batch_size >= self.max_size:
            self.bank = features[-self.max_size:]
            self.ptr = 0
            self.is_ready = True
            self.total_enqueued += batch_size
            return
        
        end_ptr = self.ptr + batch_size
        
        if end_ptr <= self.max_size:
            self.bank[self.ptr:end_ptr] = features
            self.ptr = end_ptr
        else:
            first_part = self.max_size - self.ptr
            second_part = batch_size - first_part
            self.bank[self.ptr:self.max_size] = features[:first_part]
            self.bank[0:second_part] = features[first_part:]
            self.ptr = second_part
            self.is_ready = True
        
        self.total_enqueued += batch_size
        
        fill_count = min(self.total_enqueued, self.max_size)
        fill_ratio = fill_count / self.max_size
        
        if fill_ratio >= self.min_fill_ratio and not self.is_ready:
            self.is_ready = True
            print(f"[AnomalyBankV2] Ready! Filled {fill_count}/{self.max_size} "
                  f"({fill_ratio:.1%}), orthogonalize={self.orthogonalize}")
    
    def get_anchors(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.total_enqueued == 0:
            return None
        valid_count = min(self.total_enqueued, self.max_size)
        if self.is_ready or valid_count == self.max_size:
            return self.bank.to(device)
        else:
            return self.bank[:valid_count].to(device)
    
    def ready_for_w_learning(self, current_step_ratio: float = 0.0) -> bool:
        """检查是否可以开始 w 学习"""
        # 必须过了 warm-up 阶段 且 bank 已 ready
        return current_step_ratio >= self.warm_up_ratio and self.is_ready
    
    def get_stats(self) -> dict:
        valid_count = min(self.total_enqueued, self.max_size)
        return {
            "total_enqueued": self.total_enqueued,
            "valid_count": valid_count,
            "max_size": self.max_size,
            "fill_ratio": valid_count / self.max_size,
            "is_ready": self.is_ready,
            "orthogonalize": self.orthogonalize,
            "warm_up_ratio": self.warm_up_ratio,
        }


# =============================================================================
# 修改 2: Suspicious Loss 增强版（w 与 abnormal 保持 margin）
# =============================================================================

def compute_suspicious_loss_with_margin(
    decoder_hs: torch.Tensor,         # (B, Q, D)
    proto_suspicious: torch.Tensor,   # (B, D) - w 的 prototype
    proto_abnormal: torch.Tensor,     # (B, D) - W 的 prototype
    anomaly_anchors: torch.Tensor,    # (N, D) - 来自 Memory Bank
    is_anomaly: List[bool],
    temp: float = 0.2,
    top_r: int = 5,
    w_abnormal_margin: float = 0.3,   # 【新增】w 与 abnormal 的 margin
):
    """
    【v2.2 增强】Suspicious Loss + w 与 abnormal 的 margin 约束
    
    核心改进：
    1. 让 w 学习正常图中的 hard negatives
    2. 同时约束 w 不要太像 abnormal（防止 w 变成"异常子类"）
    
    Args:
        w_abnormal_margin: w 与 abnormal 之间应保持的最小 margin
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    if anomaly_anchors is None or anomaly_anchors.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    
    hs_norm = F.normalize(decoder_hs, dim=-1)
    proto_s = F.normalize(proto_suspicious, dim=-1)  # w
    proto_a = F.normalize(proto_abnormal, dim=-1)    # W
    anchors_norm = F.normalize(anomaly_anchors, dim=-1)
    
    loss_align = torch.tensor(0.0, device=device)
    loss_margin = torch.tensor(0.0, device=device)
    count = 0
    
    for b in range(B):
        if not is_anomaly[b]:
            # === Part 1: 让 suspicious queries 对齐到 w ===
            sim_to_anchors = hs_norm[b] @ anchors_norm.t()
            max_sim_to_anomaly, _ = sim_to_anchors.max(dim=1)
            
            r = min(top_r, Q)
            _, suspicious_idx = max_sim_to_anomaly.topk(r)
            
            sim_to_w = (hs_norm[b] @ proto_s[b]) / temp
            sim_to_w = sim_to_w.clamp(min=-10, max=10)  # 添加 clamp
            for idx in suspicious_idx:
                loss_align = loss_align - F.logsigmoid(sim_to_w[idx])
                count += 1
        
        # === Part 2: w 与 abnormal 保持 margin（所有样本都算）===
        # sim(w, W) 应该 < -margin（即 w 和 W 要足够不同）
        sim_w_abnormal = (proto_s[b] @ proto_a[b])  # scalar
        # Hinge loss: max(0, sim + margin)
        loss_margin = loss_margin + F.relu(sim_w_abnormal + w_abnormal_margin)
    
    if count > 0:
        loss_align = loss_align / count
    
    loss_margin = loss_margin / B
    
    # 组合
    return loss_align + 0.5 * loss_margin


def compute_suspicious_loss_hybrid_v2(
    decoder_hs: torch.Tensor,
    proto_suspicious: torch.Tensor,
    proto_abnormal: torch.Tensor,     # 【新增】用于 margin 约束
    anomaly_bank,  # AnomalyFeatureBankV2
    matched_indices: List[tuple],
    is_anomaly: List[bool],
    current_step_ratio: float,        # 【新增】当前训练进度
    temp: float = 0.2,
    top_r: int = 5,
    w_abnormal_margin: float = 0.3,
    fallback_to_statistics: bool = True,
):
    """
    【v2.2 增强】混合版 Suspicious Loss
    
    新增：
    1. 支持 warm-up（训练前 30% 不启用）
    2. w 与 abnormal 的 margin 约束
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device

    if anomaly_bank is not None:
        if current_step_ratio < anomaly_bank.warm_up_ratio:
            return torch.tensor(0.0, device=device)    
    
    # 检查 bank 是否 ready（包含 warm-up 检查）
    if anomaly_bank is not None and anomaly_bank.ready_for_w_learning(current_step_ratio):
        anchors = anomaly_bank.get_anchors(device)
        return compute_suspicious_loss_with_margin(
            decoder_hs, proto_suspicious, proto_abnormal, anchors,
            is_anomaly, temp, top_r, w_abnormal_margin
        )
    
    # 尝试当前 batch
    current_batch_anchors = []
    for b in range(B):
        if is_anomaly[b] and matched_indices is not None:
            src_q, _ = matched_indices[b]
            if src_q.numel() > 0:
                for idx in src_q.cpu().tolist():
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0]
                    idx = int(idx)
                    if idx < Q:
                        current_batch_anchors.append(decoder_hs[b, idx])
    
    if len(current_batch_anchors) > 0:
        anchors = torch.stack(current_batch_anchors, dim=0)
        return compute_suspicious_loss_with_margin(
            decoder_hs, proto_suspicious, proto_abnormal, anchors.detach(),
            is_anomaly, temp, top_r, w_abnormal_margin
        )
    
    # 回退到统计方法
    if fallback_to_statistics:
        return compute_suspicious_loss_by_statistics_v2(
            decoder_hs, proto_suspicious, proto_abnormal,
            is_anomaly, temp, top_r, w_abnormal_margin
        )
    
    return torch.tensor(0.0, device=device)


def compute_suspicious_loss_by_statistics_v2(
    decoder_hs: torch.Tensor,
    proto_suspicious: torch.Tensor,
    proto_abnormal: torch.Tensor,
    is_anomaly: List[bool],
    temp: float = 0.2,
    top_r: int = 5,
    w_abnormal_margin: float = 0.3,
):
    """【v2.2】统计版 + margin 约束"""
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    hs_norm = F.normalize(decoder_hs, dim=-1)
    proto_s = F.normalize(proto_suspicious, dim=-1)
    proto_a = F.normalize(proto_abnormal, dim=-1)
    
    loss_align = torch.tensor(0.0, device=device)
    loss_margin = torch.tensor(0.0, device=device)
    count = 0
    
    for b in range(B):
        if not is_anomaly[b]:
            mean_hs = hs_norm[b].mean(dim=0)
            dist_to_mean = (hs_norm[b] - mean_hs).norm(dim=-1)
            
            r = min(top_r, Q)
            _, suspicious_idx = dist_to_mean.topk(r)
            
            sim_to_w = (hs_norm[b] @ proto_s[b]) / temp
            
            for idx in suspicious_idx:
                loss_align = loss_align - F.logsigmoid(sim_to_w[idx])
                count += 1
        
        # margin 约束
        sim_w_abnormal = (proto_s[b] @ proto_a[b])
        loss_margin = loss_margin + F.relu(sim_w_abnormal + w_abnormal_margin)
    
    if count > 0:
        loss_align = loss_align / count
    loss_margin = loss_margin / B
    
    return loss_align + 0.5 * loss_margin


# =============================================================================
# 修改 3: 收集函数增强（支持正交化）
# =============================================================================

def collect_anomaly_features_to_bank_v2(
    decoder_hs: torch.Tensor,
    matched_indices: List[tuple],
    is_anomaly: List[bool],
    anomaly_bank,  # AnomalyFeatureBankV2
    proto_normal: Optional[torch.Tensor],  # 【新增】用于正交化
    current_step_ratio: float,             # 【新增】当前进度
):
    """
    【v2.2 增强】收集异常特征到 bank
    
    新增：
    1. 更新 normal prototype 缓存
    2. 传递当前进度（用于 warm-up 检查）
    """
    B, Q, D = decoder_hs.shape
    
    # 更新 normal prototype 缓存
    if proto_normal is not None:
        anomaly_bank.update_normal_proto(proto_normal)
    
    features_to_store = []
    for b in range(B):
        if is_anomaly[b] and matched_indices is not None:
            src_q, _ = matched_indices[b]
            if src_q.numel() > 0:
                for idx in src_q.cpu().tolist():
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0]
                    idx = int(idx)
                    if idx < Q:
                        features_to_store.append(decoder_hs[b, idx].detach())
    
    if len(features_to_store) > 0:
        features = torch.stack(features_to_store, dim=0)
        anomaly_bank.enqueue(features, current_step_ratio)

# =============================================================================
# 修复 1: Image-Level Align Loss（二分类形式）
# =============================================================================

def align_loss_binary_classification(
    proto_normal: torch.Tensor,     # (B, D) - 来自 V
    proto_abnormal: torch.Tensor,   # (B, D) - 来自 W（可以是 mean 或选定的）
    visual_embed: torch.Tensor,     # (B, D) - pooled visual embedding
    is_anomaly: List[bool],         # (B,)
    temp: float = 0.25,
    label_smoothing: float = 0.1,   # 标签平滑，增加稳定性
):
    """
    【修复版】Image-Level Align Loss - 二分类形式
    
    核心思想（与 Query-Align 一致）：
    - logits = [sim_to_normal, sim_to_abnormal]
    - 异常图 label=1（应该接近 abnormal）
    - 正常图 label=0（应该接近 normal）
    
    这是最稳定的对齐方式，不会出现"把相似度压扁"的问题。
    
    Args:
        proto_normal: 正常原型 (B, D)
        proto_abnormal: 异常原型 (B, D)  
        visual_embed: 视觉嵌入 (B, D)
        is_anomaly: 是否异常的标签
        temp: 温度参数
        label_smoothing: 标签平滑系数
    
    Returns:
        loss: 标量损失
    """
    B, D = visual_embed.shape
    device = visual_embed.device
    
    # 归一化
    v_norm = F.normalize(visual_embed, dim=-1)  # (B, D)
    p_n = F.normalize(proto_normal, dim=-1)     # (B, D)
    p_a = F.normalize(proto_abnormal, dim=-1)   # (B, D)
    
    # 计算相似度
    sim_to_normal = (v_norm * p_n).sum(dim=-1) / temp    # (B,)
    sim_to_abnormal = (v_norm * p_a).sum(dim=-1) / temp  # (B,)
    
    # 构建 logits: (B, 2) - [normal_score, abnormal_score]
    logits = torch.stack([sim_to_normal, sim_to_abnormal], dim=-1)  # (B, 2)
    
    # 构建 targets: 异常图=1，正常图=0
    targets = torch.tensor(
        [1 if is_anomaly[b] else 0 for b in range(B)],
        dtype=torch.long, device=device
    )
    
    # Cross entropy with label smoothing
    loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    
    return loss


def align_loss_binary_classification_with_margin(
    proto_normal: torch.Tensor,     # (B, D)
    proto_abnormal: torch.Tensor,   # (B, D)
    visual_embed: torch.Tensor,     # (B, D)
    is_anomaly: List[bool],
    temp: float = 0.25,
    margin: float = 0.1,            # 相对 margin
    label_smoothing: float = 0.1,
):
    """
    【增强版】带相对 margin 的二分类对齐损失
    
    额外添加：希望正确类别的相似度比错误类别高出 margin
    即 sim_pos - sim_neg >= margin
    """
    B, D = visual_embed.shape
    device = visual_embed.device
    
    v_norm = F.normalize(visual_embed, dim=-1)
    p_n = F.normalize(proto_normal, dim=-1)
    p_a = F.normalize(proto_abnormal, dim=-1)
    
    sim_to_normal = (v_norm * p_n).sum(dim=-1) / temp
    sim_to_abnormal = (v_norm * p_a).sum(dim=-1) / temp
    
    # 基础二分类 loss
    logits = torch.stack([sim_to_normal, sim_to_abnormal], dim=-1)
    targets = torch.tensor(
        [1 if is_anomaly[b] else 0 for b in range(B)],
        dtype=torch.long, device=device
    )
    loss_ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    
    # 相对 margin loss: max(0, margin - (sim_pos - sim_neg))
    loss_margin = torch.tensor(0.0, device=device)
    for b in range(B):
        if is_anomaly[b]:
            # 异常图: sim_abnormal 应该 > sim_normal + margin
            diff = sim_to_abnormal[b] - sim_to_normal[b]
        else:
            # 正常图: sim_normal 应该 > sim_abnormal + margin
            diff = sim_to_normal[b] - sim_to_abnormal[b]
        
        loss_margin = loss_margin + F.relu(margin - diff)
    
    loss_margin = loss_margin / B
    
    # 组合（margin 权重较小）
    return loss_ce + 0.2 * loss_margin


# =============================================================================
# 修复 2: Anomaly Feature Memory Bank
# =============================================================================

class AnomalyFeatureBank:
    """
    干净的异常特征 Memory Bank
    
    设计原则：
    1. 只存储来自异常图 matched queries 的特征（真实异常区域）
    2. 只有当 bank 填充足够后才开启 w 学习
    3. 使用 FIFO 策略更新，保持特征新鲜度
    
    用途：
    - 为 w 向量的学习提供稳定的异常 anchor
    - 避免"鸡生蛋"问题（不依赖 W 是否学好）
    """
    
    def __init__(
        self, 
        max_size: int = 2048,       # 最大存储数量
        dim: int = 256,             # 特征维度
        min_fill_ratio: float = 0.5, # 最小填充比例才开启 w 学习
    ):
        self.max_size = max_size
        self.dim = dim
        self.min_fill_ratio = min_fill_ratio
        
        # CPU 上存储，避免占用 GPU 显存
        self.bank = torch.zeros(max_size, dim)
        self.ptr = 0
        self.total_enqueued = 0
        self.is_ready = False
        
    def enqueue(self, features: torch.Tensor):
        """
        存入新的异常特征
        
        Args:
            features: (N, D) 来自 matched queries 的特征
        """
        if features.numel() == 0:
            return
            
        features = F.normalize(features.detach().cpu(), dim=-1)
        batch_size = features.shape[0]
        
        # 处理超出容量的情况
        if batch_size >= self.max_size:
            # 直接用最新的填满
            self.bank = features[-self.max_size:]
            self.ptr = 0
            self.is_ready = True
            self.total_enqueued += batch_size
            return
        
        # 正常入队
        end_ptr = self.ptr + batch_size
        
        if end_ptr <= self.max_size:
            self.bank[self.ptr:end_ptr] = features
            self.ptr = end_ptr
        else:
            # 环形覆盖
            first_part = self.max_size - self.ptr
            second_part = batch_size - first_part
            self.bank[self.ptr:self.max_size] = features[:first_part]
            self.bank[0:second_part] = features[first_part:]
            self.ptr = second_part
            self.is_ready = True  # 环绕一次说明已填满
        
        self.total_enqueued += batch_size
        
        # 检查是否达到最小填充比例
        fill_count = min(self.total_enqueued, self.max_size)
        fill_ratio = fill_count / self.max_size
        
        if fill_ratio >= self.min_fill_ratio and not self.is_ready:
            self.is_ready = True
            print(f"[AnomalyBank] Ready! Filled {fill_count}/{self.max_size} "
                  f"({fill_ratio:.1%}), total enqueued: {self.total_enqueued}")
    
    def get_anchors(self, device: torch.device) -> Optional[torch.Tensor]:
        """
        获取所有有效的异常 anchor
        
        Returns:
            (N, D) tensor or None if empty
        """
        if self.total_enqueued == 0:
            return None
        
        valid_count = min(self.total_enqueued, self.max_size)
        
        if self.is_ready or valid_count == self.max_size:
            return self.bank.to(device)
        else:
            # 还没填满，只返回已填充的部分
            return self.bank[:valid_count].to(device)
    
    def ready_for_w_learning(self) -> bool:
        """检查是否可以开始 w 学习"""
        return self.is_ready
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        valid_count = min(self.total_enqueued, self.max_size)
        return {
            "total_enqueued": self.total_enqueued,
            "valid_count": valid_count,
            "max_size": self.max_size,
            "fill_ratio": valid_count / self.max_size,
            "is_ready": self.is_ready,
            "ptr": self.ptr,
        }


# =============================================================================
# 修复 3: Suspicious Loss with Memory Bank
# =============================================================================

def compute_suspicious_loss_with_bank(
    decoder_hs: torch.Tensor,         # (B, Q, D)
    proto_suspicious: torch.Tensor,   # (B, D) - w 的 prototype
    anomaly_anchors: torch.Tensor,    # (N, D) - 来自 Memory Bank
    is_anomaly: List[bool],
    temp: float = 0.2,
    top_r: int = 5,
):
    """
    【Memory Bank 版】w 向量的 Suspicious Alignment Loss
    
    核心改进：
    - 使用 Memory Bank 中的真实异常特征作为 anchor
    - 在正常图中找与这些 anchor 最相似的 queries
    - 让这些 queries 与 w 对齐（吸收 hard negatives）
    
    这样 w 的学习不依赖于 W 是否学好，信号更准确。
    
    Args:
        decoder_hs: decoder 输出的 query 特征 (B, Q, D)
        proto_suspicious: w 的 prototype (B, D)
        anomaly_anchors: 来自 bank 的异常特征 (N, D)
        is_anomaly: batch 中每个样本是否异常
        temp: 温度参数
        top_r: 每张正常图选择多少个疑似异常 query
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    if anomaly_anchors is None or anomaly_anchors.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    
    # 归一化
    hs_norm = F.normalize(decoder_hs, dim=-1)           # (B, Q, D)
    proto_s = F.normalize(proto_suspicious, dim=-1)    # (B, D)
    anchors_norm = F.normalize(anomaly_anchors, dim=-1) # (N, D)
    
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    for b in range(B):
        if not is_anomaly[b]:
            # 只在正常图上学习 w
            
            # Step 1: 计算每个 query 与所有 anomaly anchors 的相似度
            sim_to_anchors = hs_norm[b] @ anchors_norm.t()  # (Q, N)
            
            # Step 2: 取每个 query 的最大相似度（最像某个异常的程度）
            max_sim_to_anomaly, _ = sim_to_anchors.max(dim=1)  # (Q,)
            
            # Step 3: 选择最像异常的 top-r queries
            r = min(top_r, Q)
            _, suspicious_idx = max_sim_to_anomaly.topk(r)
            
            # Step 4: 让这些 queries 与 proto_suspicious (w) 对齐
            sim_to_w = (hs_norm[b] @ proto_s[b]) / temp  # (Q,)
            
            for idx in suspicious_idx:
                # 希望这些"像异常的 query"与 w 相似
                loss = loss - F.logsigmoid(sim_to_w[idx])
                count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


def compute_suspicious_loss_hybrid(
    decoder_hs: torch.Tensor,         # (B, Q, D)
    proto_suspicious: torch.Tensor,   # (B, D)
    anomaly_bank: Optional[AnomalyFeatureBank],
    matched_indices: List[tuple],     # 每张图的 matched indices
    is_anomaly: List[bool],
    temp: float = 0.2,
    top_r: int = 5,
    fallback_to_statistics: bool = True,
):
    """
    【混合版】Suspicious Loss
    
    策略：
    1. 优先使用 Memory Bank（如果 ready）
    2. 如果 bank 未 ready 且 fallback=True，使用当前 batch 的异常特征
    3. 如果都没有，使用统计离群点
    
    Args:
        decoder_hs: (B, Q, D)
        proto_suspicious: (B, D)
        anomaly_bank: Memory Bank 实例（可为 None）
        matched_indices: 当前 batch 的 matched indices
        is_anomaly: batch 标签
        temp: 温度
        top_r: top-r queries
        fallback_to_statistics: 是否在无 anchor 时使用统计方法
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    # 尝试获取 bank anchors
    if anomaly_bank is not None and anomaly_bank.ready_for_w_learning():
        anchors = anomaly_bank.get_anchors(device)
        return compute_suspicious_loss_with_bank(
            decoder_hs, proto_suspicious, anchors, is_anomaly, temp, top_r
        )
    
    # Bank 未 ready，尝试使用当前 batch 的异常特征
    current_batch_anchors = []
    for b in range(B):
        if is_anomaly[b] and matched_indices is not None:
            src_q, _ = matched_indices[b]
            if src_q.numel() > 0:
                for idx in src_q.cpu().tolist():
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0]
                    idx = int(idx)
                    if idx < Q:
                        current_batch_anchors.append(decoder_hs[b, idx])
    
    if len(current_batch_anchors) > 0:
        anchors = torch.stack(current_batch_anchors, dim=0)  # (N, D)
        return compute_suspicious_loss_with_bank(
            decoder_hs, proto_suspicious, anchors.detach(), is_anomaly, temp, top_r
        )
    
    # 回退到统计方法
    if fallback_to_statistics:
        return compute_suspicious_loss_by_statistics(
            decoder_hs, proto_suspicious, is_anomaly, temp, top_r
        )
    
    return torch.tensor(0.0, device=device)


def compute_suspicious_loss_by_statistics(
    decoder_hs: torch.Tensor,       # (B, Q, D)
    proto_suspicious: torch.Tensor, # (B, D)
    is_anomaly: List[bool],
    temp: float = 0.2,
    top_r: int = 5,
):
    """
    【统计版】使用离群点检测找疑似异常
    
    思路：与该图的 query 均值距离最远的 = 最异常
    完全不依赖任何学习的特征，训练初期就能用。
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    hs_norm = F.normalize(decoder_hs, dim=-1)
    proto_s = F.normalize(proto_suspicious, dim=-1)
    
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    for b in range(B):
        if not is_anomaly[b]:
            # 计算均值
            mean_hs = hs_norm[b].mean(dim=0)  # (D,)
            
            # 计算每个 query 与均值的距离
            dist_to_mean = (hs_norm[b] - mean_hs).norm(dim=-1)  # (Q,)
            
            # 距离最远的 = 最"异常"
            r = min(top_r, Q)
            _, suspicious_idx = dist_to_mean.topk(r)
            
            # 对齐到 w
            sim_to_w = (hs_norm[b] @ proto_s[b]) / temp
            
            for idx in suspicious_idx:
                loss = loss - F.logsigmoid(sim_to_w[idx])
                count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


# =============================================================================
# 辅助函数：收集异常特征到 Bank
# =============================================================================

def collect_anomaly_features_to_bank(
    decoder_hs: torch.Tensor,         # (B, Q, D)
    matched_indices: List[tuple],     # 每张图的 (src_q, tgt_q)
    is_anomaly: List[bool],
    anomaly_bank: AnomalyFeatureBank,
):
    """
    将当前 batch 的异常特征存入 Memory Bank
    
    在训练循环中每个 step 调用，不管 bank 是否 ready
    """
    B, Q, D = decoder_hs.shape
    
    features_to_store = []
    
    for b in range(B):
        if is_anomaly[b] and matched_indices is not None:
            src_q, _ = matched_indices[b]
            if src_q.numel() > 0:
                for idx in src_q.cpu().tolist():
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0]
                    idx = int(idx)
                    if idx < Q:
                        features_to_store.append(decoder_hs[b, idx].detach())
    
    if len(features_to_store) > 0:
        features = torch.stack(features_to_store, dim=0)
        anomaly_bank.enqueue(features)

def compute_suspicious_alignment_loss(
    decoder_hs: torch.Tensor,       # (B, Q, D)
    proto_suspicious: torch.Tensor, # (B, D) suspicious prototype (来自 w)
    proto_normal: torch.Tensor,     # (B, D) normal prototype (来自 V)
    proto_abnormal: torch.Tensor,   # (B, D) abnormal prototype (来自 W)
    is_anomaly: list,
    temp: float = 0.2,
    top_r: int = 5,
):
    """
    【新增】w 向量的专属 loss
    
    作用：让 w 学习 normal 图中"疑似异常"的 query 模式
    - 选择 normal 图里 sim_to_abnormal 最大的 top-r queries
    - 把它们对齐到 w（让 w 专门吸收这些"伪异常"模式）
    
    只在 normal 图上更新 w
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    hs_norm = F.normalize(decoder_hs, dim=-1)
    proto_s = F.normalize(proto_suspicious, dim=-1)
    proto_a = F.normalize(proto_abnormal, dim=-1)
    
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    for b in range(B):
        if not is_anomaly[b]:
            # 只在 normal 图上更新 w
            
            # 计算与 abnormal 的相似度
            sim_abnormal = (hs_norm[b] @ proto_a[b]) / temp  # (Q,)
            
            # 选择 sim_to_abnormal 最高的 top-r queries
            r = min(top_r, Q)
            _, top_idx = sim_abnormal.topk(r)
            
            # 计算与 suspicious prototype 的相似度
            sim_suspicious = (hs_norm[b] @ proto_s[b]) / temp  # (Q,)
            
            # 让这些 queries 与 suspicious 更相似（拉近）
            # 使用 InfoNCE 风格的 loss
            for idx in top_idx:
                # 希望 sim_suspicious[idx] 高
                # 用简单的负对数来实现
                loss = loss - F.logsigmoid(sim_suspicious[idx])
            
            count += r
    
    if count > 0:
        loss = loss / count
    
    return loss

def query_text_alignment_loss_with_gradient_gating(
    decoder_hs: torch.Tensor,           # (B, Q, D)
    proto_normal: torch.Tensor,         # (B, D) normal prototype (来自 V)
    proto_abnormal_all: torch.Tensor,   # (B, K, D) K 个独立的 abnormal prototypes (来自 W_k)
    indices: list,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    is_anomaly: list,
    temp: float = 0.2,
    iou_threshold: float = 0.1,
    top_k_normal: int = 5,
    aggregation: str = "max",           # "max" 或 "logsumexp"
    use_gradient_gating: bool = True,   # 是否启用梯度路由
):
    """
    【完整版】支持 K 个独立 abnormal prototypes + 梯度路由 的 Query-Align Loss
    
    关键特性：
    1. K 个 abnormal prototypes 独立使用，不取平均
    2. 使用 max 或 logsumexp 聚合 sim_abnormal
    3. 梯度路由：V 只从 normal 学，W 只从 abnormal 学
    """
    B, Q, D = decoder_hs.shape
    K = proto_abnormal_all.shape[1] if proto_abnormal_all.dim() == 3 else 1
    device = decoder_hs.device
    
    hs_norm = F.normalize(decoder_hs, dim=-1)  # (B, Q, D)
    proto_n = F.normalize(proto_normal, dim=-1)  # (B, D)
    
    # 处理 abnormal prototypes
    if proto_abnormal_all.dim() == 2:
        # (B, D) -> (B, 1, D)
        proto_a_all = F.normalize(proto_abnormal_all.unsqueeze(1), dim=-1)
        K = 1
    else:
        proto_a_all = F.normalize(proto_abnormal_all, dim=-1)  # (B, K, D)
    
    loss = torch.tensor(0.0, device=device)
    total_weight = 0.0
    
    for b in range(B):
        # =====================================================================
        # 梯度路由 (Gradient Gating)
        # =====================================================================
        if use_gradient_gating:
            if is_anomaly[b]:
                # 异常图：W 学习，V 不学习（detach V）
                proto_n_b = proto_n[b].detach()  # V 不回传梯度
                proto_a_all_b = proto_a_all[b]   # W 正常学习
            else:
                # 正常图：V 学习，W 不学习（detach W）
                proto_n_b = proto_n[b]           # V 正常学习
                proto_a_all_b = proto_a_all[b].detach()  # W 不回传梯度
        else:
            proto_n_b = proto_n[b]
            proto_a_all_b = proto_a_all[b]
        
        # 计算与 normal 的相似度
        sim_normal = (hs_norm[b] @ proto_n_b) / temp  # (Q,)
        
        # 计算与每个 abnormal prototype 的相似度
        sim_abnormal_all = (hs_norm[b] @ proto_a_all_b.t()) / temp  # (Q, K)
        
        # =====================================================================
        # 聚合 K 个 abnormal 相似度
        # =====================================================================
        if aggregation == "max":
            sim_abnormal, best_k = sim_abnormal_all.max(dim=1)  # (Q,)
        elif aggregation == "logsumexp":
            sim_abnormal = torch.logsumexp(sim_abnormal_all, dim=1)  # (Q,)
        else:
            sim_abnormal = sim_abnormal_all.mean(dim=1)  # fallback to mean
        
        if is_anomaly[b]:
            # ===== 异常图：matched queries 应该对齐到 abnormal =====
            src_q, _ = indices[b]
            
            if src_q.numel() > 0:
                # 计算 IoU 权重（如果需要）
                ious = None
                if pred_masks is not None and gt_masks is not None:
                    pred_b = pred_masks[b]  # (Q, H, W)
                    gt_b = gt_masks[b]  # (H, W)
                    
                    if pred_b.shape[-2:] != gt_b.shape[-2:]:
                        gt_b = F.interpolate(
                            gt_b.unsqueeze(0).unsqueeze(0).float(),
                            size=pred_b.shape[-2:], mode='nearest'
                        ).squeeze(0).squeeze(0)
                    
                    pred_binary = (pred_b.sigmoid() > 0.5).float()
                    gt_binary = (gt_b > 0.5).float()
                    
                    intersection = (pred_binary * gt_binary.unsqueeze(0)).sum(dim=(1, 2))
                    union = pred_binary.sum(dim=(1, 2)) + gt_binary.sum() - intersection
                    ious = intersection / (union + 1e-6)  # (Q,)
                
                # 使用所有 matched queries
                for idx in src_q.cpu().tolist():
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0]
                    idx = int(idx)
                    
                    if idx < Q:
                        # IoU 权重
                        if ious is not None:
                            iou_val = ious[idx].item()
                            if iou_val < iou_threshold:
                                continue
                            weight = max(iou_val, 0.3)
                        else:
                            weight = 1.0
                        
                        # 二分类: [normal, abnormal_aggregated]
                        logits = torch.stack([sim_normal[idx], sim_abnormal[idx]])
                        target = torch.tensor([1], device=device)  # 应该是 abnormal
                        loss_q = F.cross_entropy(logits.unsqueeze(0), target)
                        
                        loss = loss + weight * loss_q
                        total_weight += weight
        else:
            # ===== 正常图：hard negative queries 应该对齐到 normal =====
            # 选择与 abnormal 最相似的 queries（它们最需要被纠正）
            k = min(top_k_normal, Q)
            _, top_idx = sim_abnormal.topk(k)
            
            for idx in top_idx:
                logits = torch.stack([sim_normal[idx], sim_abnormal[idx]])
                target = torch.tensor([0], device=device)  # 应该是 normal
                loss_q = F.cross_entropy(logits.unsqueeze(0), target)
                loss = loss + loss_q
            
            total_weight += k
    
    if total_weight > 0:
        loss = loss / total_weight
    
    return loss

def query_text_alignment_loss_binary_v4(
    decoder_hs: torch.Tensor,       # (B, Q, D)
    proto_normal: torch.Tensor,     # (B, D)
    proto_abnormal_all: torch.Tensor,  # (B, K, D) K 个独立的 abnormal prototypes
    indices: list,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    is_anomaly: list,
    specie_names: list = None,      # 用于映射 query 到具体的 W_k
    temp: float = 0.2,
    iou_threshold: float = 0.1,
    top_k_normal: int = 5,
):
    """
    【进阶版】支持多个 abnormal prototypes 的 Query-Align Loss
    
    思路：
    - 如果有 specie_names，可以尝试将 query 对齐到对应的 W_k
    - 否则，找与 query 最相似的 W_k
    """
    B, Q, D = decoder_hs.shape
    K = proto_abnormal_all.shape[1]
    device = decoder_hs.device
    
    hs_norm = F.normalize(decoder_hs, dim=-1)
    proto_n = F.normalize(proto_normal, dim=-1)
    proto_a_all = F.normalize(proto_abnormal_all, dim=-1)  # (B, K, D)
    
    loss = torch.tensor(0.0, device=device)
    total_weight = 0.0
    
    for b in range(B):
        # 计算与 normal 的相似度
        sim_normal = (hs_norm[b] @ proto_n[b]) / temp  # (Q,)
        
        # 计算与每个 abnormal prototype 的相似度
        sim_abnormal_all = (hs_norm[b] @ proto_a_all[b].t()) / temp  # (Q, K)
        
        if is_anomaly[b]:
            src_q, _ = indices[b]
            
            if src_q.numel() > 0:
                for idx in src_q.cpu().tolist():
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0]
                    idx = int(idx)
                    
                    if idx < Q:
                        # 方案 A: 使用最相似的 W_k
                        best_k = sim_abnormal_all[idx].argmax().item()
                        sim_abnormal = sim_abnormal_all[idx, best_k]
                        
                        # 二分类: [normal, abnormal_best_k]
                        logits = torch.stack([sim_normal[idx], sim_abnormal])
                        target = torch.tensor([1], device=device)  # 应该是 abnormal
                        loss_q = F.cross_entropy(logits.unsqueeze(0), target)
                        
                        loss = loss + loss_q
                        total_weight += 1.0
        else:
            # 正常图: 选择 hard negatives
            sim_abnormal_max = sim_abnormal_all.max(dim=1)[0]  # (Q,) 每个 query 与最相似的 W_k
            k = min(top_k_normal, Q)
            _, top_idx = sim_abnormal_max.topk(k)
            
            for idx in top_idx:
                logits = torch.stack([sim_normal[idx], sim_abnormal_max[idx]])
                target = torch.tensor([0], device=device)  # 应该是 normal
                loss_q = F.cross_entropy(logits.unsqueeze(0), target)
                loss = loss + loss_q
            
            total_weight += k
    
    if total_weight > 0:
        loss = loss / total_weight
    
    return loss

def query_text_alignment_loss_binary_v3(
    decoder_hs: torch.Tensor,       # (B, Q, D) decoder 输出的 query 特征
    proto_normal: torch.Tensor,     # (B, D) normal prototype
    proto_abnormal: torch.Tensor,   # (B, D) 或 (B, K, D) abnormal prototype(s)
    indices: list,                  # matcher 输出
    gt_masks: torch.Tensor,         # (B, H, W) GT masks
    pred_masks: torch.Tensor,       # (B, Q, H, W) 预测的masks
    is_anomaly: list,               # 每张图是否是异常
    temp: float = 0.2,
    iou_threshold: float = 0.1,     # IoU阈值
    top_k_normal: int = 5,
    use_all_matched: bool = True,   # 使用所有matched queries
    use_iou_weight: bool = True,    # 使用IoU作为权重
):
    """
    【修复版V3】二分类 Query-Text Alignment Loss
    
    核心修复：
    1. 使用所有matched queries，而非只用第一个
    2. 使用IoU作为权重（高IoU的query贡献更大）
    3. 正常图权重改为1.0（随机基线回到log(2)≈0.69）
    4. 支持多个abnormal prototypes
    
    Args:
        decoder_hs: (B, Q, D) query 特征
        proto_normal: (B, D) normal 原型
        proto_abnormal: (B, D) 或 (B, K, D) abnormal 原型
        indices: matcher 输出的匹配索引
        gt_masks: (B, H, W) GT 分割掩码
        pred_masks: (B, Q, H, W) 预测的masks
        is_anomaly: 每张图是否是异常
        temp: 温度参数
        iou_threshold: IoU阈值，低于此值不参与训练
        top_k_normal: 正常图中选择的 hard negative 数量
        use_all_matched: 是否使用所有matched queries
        use_iou_weight: 是否使用IoU作为权重
    
    Returns:
        loss: 标量损失
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    # 归一化
    hs_norm = F.normalize(decoder_hs, dim=-1)  # (B, Q, D)
    proto_n = F.normalize(proto_normal, dim=-1)  # (B, D)
    
    # 处理多个abnormal prototypes
    if proto_abnormal.dim() == 3:
        # (B, K, D) -> (B, D) 取平均
        proto_a = F.normalize(proto_abnormal.mean(dim=1), dim=-1)
    else:
        proto_a = F.normalize(proto_abnormal, dim=-1)
    
    loss = torch.tensor(0.0, device=device)
    total_weight = 0.0
    
    for b in range(B):
        # 计算每个 query 与两个原型的相似度
        sim_normal = (hs_norm[b] @ proto_n[b]) / temp  # (Q,)
        sim_abnormal = (hs_norm[b] @ proto_a[b]) / temp  # (Q,)
        
        # 拼接为二分类 logits: [normal, abnormal]
        logits = torch.stack([sim_normal, sim_abnormal], dim=-1)  # (Q, 2)
        
        if is_anomaly[b]:
            # ===== 异常图：matched queries 应该对齐到 abnormal =====
            src_q, tgt_q = indices[b]
            
            if src_q.numel() > 0:
                # 计算IoU（如果需要）
                ious = None
                if use_iou_weight and pred_masks is not None and gt_masks is not None:
                    pred_b = pred_masks[b]  # (Q, H, W)
                    gt_b = gt_masks[b]  # (H, W)
                    
                    # 确保尺寸匹配
                    if pred_b.shape[-2:] != gt_b.shape[-2:]:
                        gt_b = F.interpolate(
                            gt_b.unsqueeze(0).unsqueeze(0).float(),
                            size=pred_b.shape[-2:], mode='nearest'
                        ).squeeze(0).squeeze(0)
                    
                    # 计算IoU
                    pred_binary = (pred_b.sigmoid() > 0.5).float()
                    gt_binary = (gt_b > 0.5).float()
                    
                    intersection = (pred_binary * gt_binary.unsqueeze(0)).sum(dim=(1, 2))
                    union = pred_binary.sum(dim=(1, 2)) + gt_binary.sum() - intersection
                    ious = intersection / (union + 1e-6)  # (Q,)
                
                # 【修复】使用所有matched queries
                if use_all_matched:
                    matched_indices = src_q.cpu().tolist()
                else:
                    matched_indices = [int(src_q[0].item())]
                
                for idx in matched_indices:
                    if isinstance(idx, (list, tuple)):
                        idx = idx[0] if len(idx) > 0 else 0
                    idx = int(idx)
                    
                    if idx < Q:
                        # 计算权重
                        if ious is not None:
                            iou_val = ious[idx].item()
                            if iou_val < iou_threshold:
                                continue  # 跳过低IoU的query
                            weight = max(iou_val, 0.3)  # 至少0.3权重
                        else:
                            weight = 1.0
                        
                        # Cross-entropy loss
                        target = torch.tensor([1], device=device)
                        loss_q = F.cross_entropy(logits[idx:idx+1], target)
                        
                        loss = loss + weight * loss_q
                        total_weight += weight
        else:
            # ===== 正常图：高响应 query 应该对齐到 normal =====
            k = min(top_k_normal, Q)
            _, top_idx = sim_abnormal.topk(k)
            
            # 这些 query 应该被拉向 normal (class 0)
            targets = torch.zeros(k, dtype=torch.long, device=device)
            loss_b = F.cross_entropy(logits[top_idx], targets)
            
            # 【修复】权重改为1.0，不再是0.5
            loss = loss + loss_b
            total_weight += 1.0
    
    if total_weight > 0:
        loss = loss / total_weight
    
    return loss

def image_level_presence_loss(
    pred_logits: torch.Tensor,      # (B, Q, 1) 或 (B, Q) query级别的logits
    is_anomaly: list,               # 每张图是否是异常
    aggregation: str = "max",        # "max", "mean", "logsumexp"
    use_focal: bool = True,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
):
    """
    【修复版】图像级别的 Presence Loss
    
    核心修复：
    - 从Query级别 (B, Q) 聚合到图像级别 (B,)
    - 问："这张图是否有异常？"
    - 理论下限：log(2) ≈ 0.69
    
    Args:
        pred_logits: (B, Q, 1) 或 (B, Q) query级别的logits
        is_anomaly: 每张图是否是异常
        aggregation: 聚合方法
            - "max": 最异常的query决定整张图
            - "mean": 平均所有query
            - "logsumexp": soft max（可微分）
        use_focal: 是否使用Focal Loss
        focal_alpha: Focal Loss的alpha参数
        focal_gamma: Focal Loss的gamma参数
    
    Returns:
        loss: 标量损失
        acc: 准确率
    """
    device = pred_logits.device
    
    # 确保是 (B, Q) 形状
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.squeeze(-1)
    
    # 聚合到图像级别
    if aggregation == "max":
        image_logits = pred_logits.max(dim=1)[0]  # (B,)
    elif aggregation == "mean":
        image_logits = pred_logits.mean(dim=1)  # (B,)
    elif aggregation == "logsumexp":
        image_logits = torch.logsumexp(pred_logits, dim=1)  # (B,)
    else:
        image_logits = pred_logits.max(dim=1)[0]
    
    # 构建图像级标签
    targets = torch.tensor(is_anomaly, dtype=torch.float32, device=device)
    
    if use_focal:
        # Focal Loss
        p = torch.sigmoid(image_logits)
        ce_loss = F.binary_cross_entropy_with_logits(image_logits, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** focal_gamma
        
        alpha_t = focal_alpha * targets + (1 - focal_alpha) * (1 - targets)
        
        loss = (alpha_t * focal_weight * ce_loss).mean()
    else:
        loss = F.binary_cross_entropy_with_logits(image_logits, targets)
    
    # 计算准确率
    with torch.no_grad():
        preds = (torch.sigmoid(image_logits) > 0.5).float()
        acc = (preds == targets).float().mean()
    
    return loss, acc

def compute_compound_losses(model, outputs, labels, args):
    """计算Compound Prompt Learner的辅助损失"""
    losses = {}
    
    prompt_learner = model.prompt_learner
    if hasattr(model, 'module'):  # DDP情况
        prompt_learner = model.module.prompt_learner
    
    # 检查是否是compound类型
    pl_type = getattr(model, 'prompt_learner_type', None)
    if hasattr(model, 'module'):
        pl_type = getattr(model.module, 'prompt_learner_type', None)
    
    if pl_type != "compound":
        return losses
    
    # 1. 正交约束损失（优先使用 prototype 级别）
    if hasattr(prompt_learner, 'compute_orthogonal_loss_prototype_level'):
        losses['orthogonal_loss'] = prompt_learner.compute_orthogonal_loss_prototype_level()
    elif hasattr(prompt_learner, 'compute_orthogonal_loss'):
        losses['orthogonal_loss'] = prompt_learner.compute_orthogonal_loss()
    
    # 2. 先验损失（L2 正则，V3 不需要参数）
    if hasattr(prompt_learner, 'compute_prior_loss'):
        losses['prior_loss'] = prompt_learner.compute_prior_loss()
    
    # 3. 对比损失（Normal/Abnormal分离，V3 不需要参数）
    if hasattr(prompt_learner, 'compute_contrast_loss'):
        losses['contrast_loss'] = prompt_learner.compute_contrast_loss()
    
    return losses

# ==================== 新增：模型架构打印 ====================
def _print_model_tree(model, name="model", filter_key="", file_handle=None, also_stdout=True):
    """
    Print model structure to stdout and/or file.
    - file_handle: an opened file object (text mode) or None
    - also_stdout: if True, print to terminal as well
    """
    buf = io.StringIO()

    def _w(s=""):
        buf.write(str(s) + "\n")

    _w("\n" + "=" * 80)
    _w(f"[PRINT] {name} repr:\n{model}\n")

    _w(f"[PRINT] {name} named children:")
    for n, m in model.named_children():
        _w(f"  - {n} => {type(m).__name__}")

    _w(f"\n[PRINT] {name} named modules (filtered='{filter_key}'):")
    for n, m in model.named_modules():
        if filter_key and (filter_key not in n):
            continue
        _w(f"{n:60s}  {type(m).__name__}")

    _w(f"\n[PRINT] {name} named parameters (filtered='{filter_key}'):")
    for n, p in model.named_parameters():
        if filter_key and (filter_key not in n):
            continue
        _w(f"{n:80s}  shape={tuple(p.shape)}  trainable={p.requires_grad}")

    _w("=" * 80 + "\n")

    content = buf.getvalue()
    if file_handle is not None:
        file_handle.write(content)
        file_handle.flush()

    if also_stdout:
        print(content, end="")

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


# ==================== 新增：两阶段 Lambda 调度器 ====================

class TwoStageLambdaScheduler:
    """
    两阶段 lambda_query_align 调度器
    
    策略：
    - Stage 1 (前 stage1_ratio 的步数): 使用较低的 lambda 值，让模型先学好 segmentation
    - Stage 2 (剩余步数): 逐渐提升 lambda 值，增强 query-text alignment
    
    过渡方式：
    - 'step': 阶跃切换（在 stage1 结束时突然切换到 stage2 值）
    - 'linear': 线性过渡（在 stage1 结束后线性增加到 stage2 值）
    - 'cosine': 余弦过渡（更平滑的过渡曲线）
    """
    
    def __init__(
        self,
        total_steps: int,
        stage1_ratio: float = 0.35,        # Stage 1 占总步数的比例 (默认35%)
        stage1_lambda: float = 0.08,       # Stage 1 的 lambda 值 (低)
        stage2_lambda: float = 0.20,       # Stage 2 的 lambda 值 (高)
        transition: str = 'linear',        # 过渡方式: 'step', 'linear', 'cosine'
        transition_ratio: float = 0.15,    # 过渡期占 stage2 的比例 (仅 linear/cosine 有效)
    ):
        self.total_steps = total_steps
        self.stage1_ratio = stage1_ratio
        self.stage1_lambda = stage1_lambda
        self.stage2_lambda = stage2_lambda
        self.transition = transition
        self.transition_ratio = transition_ratio
        
        # 计算关键步数点
        self.stage1_end = int(total_steps * stage1_ratio)
        self.transition_steps = int((total_steps - self.stage1_end) * transition_ratio)
        self.transition_end = self.stage1_end + self.transition_steps
        
        self.current_step = 0
        
        print(f"[TwoStageLambdaScheduler] 初始化:")
        print(f"  总步数: {total_steps}")
        print(f"  Stage 1: step 0 ~ {self.stage1_end} ({stage1_ratio*100:.0f}%), lambda={stage1_lambda}")
        print(f"  过渡期: step {self.stage1_end} ~ {self.transition_end} ({transition})")
        print(f"  Stage 2: step {self.transition_end} ~ {total_steps}, lambda={stage2_lambda}")
    
    def step(self):
        """更新当前步数"""
        self.current_step += 1
    
    def get_lambda(self) -> float:
        """获取当前步的 lambda 值"""
        if self.current_step <= self.stage1_end:
            # Stage 1: 使用低 lambda
            return self.stage1_lambda
        
        elif self.current_step <= self.transition_end and self.transition != 'step':
            # 过渡期
            progress = (self.current_step - self.stage1_end) / max(1, self.transition_steps)
            progress = min(1.0, progress)  # 确保不超过1
            
            if self.transition == 'linear':
                return self.stage1_lambda + (self.stage2_lambda - self.stage1_lambda) * progress
            elif self.transition == 'cosine':
                # 使用余弦插值实现更平滑的过渡
                cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
                return self.stage1_lambda + (self.stage2_lambda - self.stage1_lambda) * cosine_factor
            else:
                return self.stage2_lambda
        else:
            # Stage 2: 使用高 lambda
            return self.stage2_lambda
    
    def get_stage(self) -> str:
        """获取当前阶段名称"""
        if self.current_step <= self.stage1_end:
            return "stage1"
        elif self.current_step <= self.transition_end:
            return "transition"
        else:
            return "stage2"


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
    改进的分组策略：基于 prompt_list 结构自动适配不同数据集。
    
    分组逻辑：
    1. MVTec-AD full 模式 (len > 3)：有 specie_name
       - prompt_list = ["anomaly bottle", "damaged", "defect", "broken large", ...]
       - key = (prompt_list[0], prompt_list[3]) = ("anomaly bottle", "broken large")
       - 同类别同 specie 的样本分到同一组
    
    2. 其他情况 (len <= 3)：simple 模式或 VisA
       - prompt_list = ["anomaly bottle"] 或 ["anomaly bottle", "damaged", "defect"]
       - key = (prompt_list[0],) = ("anomaly bottle",)
       - 同类别同状态的样本分到同一组
    
    效果：
    - MVTec full: bottle-broken_large 和 bottle-crack 分到不同组
    - MVTec simple / VisA: 所有 anomaly bottle 一组，所有 normal bottle 一组
    """
    group_to_id = {}
    labels = []
    
    for prompts in prompt_lists:
        if prompts is None or len(prompts) == 0:
            key = ("__unknown__",)
        elif len(prompts) > 3:
            # MVTec-AD full 模式: 有 specie_name 在 [3] 位置
            # key = (cls_template, specie_name)
            key = (prompts[0], prompts[3])
        else:
            # simple 模式或 VisA: 只按 cls_template 分组
            # key = (cls_template,)
            key = (prompts[0],)
        
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


def query_text_alignment_loss_binary(
    decoder_hs: torch.Tensor,       # (B, Q, D) decoder 输出的 query 特征
    proto_normal: torch.Tensor,     # (B, D) normal prototype
    proto_abnormal: torch.Tensor,   # (B, D) abnormal prototype  
    indices: list,                  # matcher 输出
    gt_masks: torch.Tensor,         # (B, H, W) GT masks
    is_anomaly: list,               # 每张图是否是异常
    temp: float = 0.2,
    top_k_normal: int = 5,          # 正常图中选择的 query 数量
):
    """
    二分类版本的 Query-Text Alignment Loss
    
    核心思想：
    - 每个 query 判断是应该对齐到 normal 还是 abnormal
    - 使用 GT mask 生成软标签（query 与异常区域的 IoU）
    - 理论下限是 log(2) ≈ 0.69，远低于原来的 log(400) ≈ 6.0
    
    Args:
        decoder_hs: (B, Q, D) query 特征
        proto_normal: (B, D) normal 原型
        proto_abnormal: (B, D) abnormal 原型
        indices: matcher 输出的匹配索引
        gt_masks: (B, H, W) GT 分割掩码
        is_anomaly: 每张图是否是异常
        temp: 温度参数
        top_k_normal: 正常图中选择的 query 数量
    
    Returns:
        loss: 标量损失
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    # 归一化
    hs_norm = F.normalize(decoder_hs, dim=-1)  # (B, Q, D)
    proto_n = F.normalize(proto_normal, dim=-1)  # (B, D)
    proto_a = F.normalize(proto_abnormal, dim=-1)  # (B, D)
    
    loss = torch.tensor(0.0, device=device)
    valid_count = 0
    
    for b in range(B):
        # 计算每个 query 与两个原型的相似度
        sim_normal = (hs_norm[b] @ proto_n[b]) / temp  # (Q,)
        sim_abnormal = (hs_norm[b] @ proto_a[b]) / temp  # (Q,)
        
        # 拼接为二分类 logits: [normal, abnormal]
        logits = torch.stack([sim_normal, sim_abnormal], dim=-1)  # (Q, 2)
        
        if is_anomaly[b]:
            # ===== 异常图：matched query 应该对齐到 abnormal =====
            src_q, tgt_q = indices[b]
            
            if src_q.numel() > 0:
                matched_idx = int(src_q[0].item())
                if matched_idx < Q:
                    # matched query 应该对齐到 abnormal (class 1)
                    target = torch.tensor([1], device=device)
                    loss_b = F.cross_entropy(logits[matched_idx:matched_idx+1], target)
                    loss = loss + loss_b
                    valid_count += 1
        else:
            # ===== 正常图：高响应 query 应该对齐到 normal =====
            # 选择与 abnormal 相似度最高的 query（它们最需要被纠正）
            k = min(top_k_normal, Q)
            _, top_idx = sim_abnormal.topk(k)
            
            # 这些 query 应该被拉向 normal (class 0)
            targets = torch.zeros(k, dtype=torch.long, device=device)
            loss_b = F.cross_entropy(logits[top_idx], targets)
            loss = loss + 0.5 * loss_b  # 正常图权重降低
            valid_count += 1
    
    if valid_count > 0:
        loss = loss / valid_count
    
    return loss


def query_text_alignment_loss_v2(
    decoder_hs: torch.Tensor,       # (B, Q, D)
    prompt_proto: torch.Tensor,     # (B, D) - 异常prompt
    prompt_proto_normal: torch.Tensor,  # (B, D) - 正常prompt（可选，可为None）
    indices: list,                  # matcher输出
    pred_masks: torch.Tensor,       # (B, Q, H, W) - 预测的masks
    gt_masks: torch.Tensor,         # (B, H, W) 或 (B, 1, H, W) - GT masks
    is_anomaly: list,               # 每张图是否是异常
    temp: float = 0.2,
    top_k: int = 64,
    use_soft_target: bool = True,   # 修复1：使用软标签
    include_normal: bool = True,    # 修复2：normal图参与
    normal_margin: float = 0.3,     # normal图的margin
    use_full_softmax_ratio: float = 0.2,  # 修复3：前20%步用full softmax
    current_step_ratio: float = 1.0,      # 当前步数占总步数的比例
):
    """
    改进版 Query-Text Alignment Loss (v2) - 修复版
    
    关键修复：soft_target 使用 "matcher硬标签 + IoU软修正" 混合方式
    - 保证 matched query 至少有 base_weight (70%) 的权重
    - IoU 只作为额外的加权，不会导致均匀分布
    """
    B, Q, D = decoder_hs.shape
    device = decoder_hs.device
    
    # 归一化
    hs_norm = F.normalize(decoder_hs, dim=-1)
    p_norm = F.normalize(prompt_proto, dim=-1)
    
    # 如果有normal prompt
    if prompt_proto_normal is not None:
        p_norm_normal = F.normalize(prompt_proto_normal, dim=-1)
    else:
        p_norm_normal = None
    
    # 修复3：根据当前步数决定是否用full softmax
    use_full_softmax = (current_step_ratio < use_full_softmax_ratio)
    
    loss_anomaly = torch.tensor(0.0, device=device)
    loss_normal = torch.tensor(0.0, device=device)
    valid_anomaly = 0
    valid_normal = 0
    
    # 处理gt_masks形状
    if gt_masks is not None and gt_masks.dim() == 4:
        gt_masks = gt_masks.squeeze(1)  # (B, H, W)
    
    for b in range(B):
        src_q, tgt_q = indices[b]
        
        # 计算所有query与异常prompt的相似度
        all_sim = (hs_norm[b] @ p_norm[b]) / temp  # (Q,)
        
        if is_anomaly[b] and src_q.numel() > 0:
            # ===== 异常图：获取matched query index =====
            matched_q_idx = int(src_q[0].item())
            if matched_q_idx >= Q:
                continue
            
            # 决定用哪些query参与竞争
            if use_full_softmax or Q <= top_k:
                # 全量softmax
                effective_indices = torch.arange(Q, device=device)
                sim_selected = all_sim
                target_idx_in_selected = torch.tensor([matched_q_idx], device=device)
            else:
                # top-k采样
                effective_k = min(top_k, Q)
                _, top_indices = torch.topk(all_sim.detach(), k=effective_k)
                
                # 确保 matched query 在 top_k 中
                if matched_q_idx not in top_indices:
                    top_indices = torch.cat([
                        torch.tensor([matched_q_idx], device=device),
                        top_indices[:-1]
                    ])
                
                effective_indices = top_indices
                sim_selected = all_sim[top_indices]
                target_idx_in_selected = (top_indices == matched_q_idx).nonzero(as_tuple=True)[0]
            
            num_selected = len(effective_indices)
            
            # ===== 计算 loss =====
            if use_soft_target and pred_masks is not None and gt_masks is not None:
                # 混合方式：matcher硬标签 + IoU软修正
                pm_b = pred_masks[b]  # (Q, H, W)
                gt_b = gt_masks[b]    # (H, W)
                
                # 确保尺寸匹配
                if pm_b.shape[-2:] != gt_b.shape[-2:]:
                    gt_b_resized = F.interpolate(
                        gt_b.unsqueeze(0).unsqueeze(0).float(),
                        size=pm_b.shape[-2:],
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                else:
                    gt_b_resized = gt_b.float()
                
                # 计算所有query的IoU
                pm_b_prob = torch.sigmoid(pm_b) if pm_b.min() < 0 else pm_b
                pm_b_binary = (pm_b_prob > 0.5).float()
                intersection = (pm_b_binary * gt_b_resized).sum(dim=(1, 2))
                union = pm_b_binary.sum(dim=(1, 2)) + gt_b_resized.sum() - intersection
                iou_scores = intersection / (union + 1e-6)  # (Q,)
                
                # 只取 selected indices 的 IoU
                iou_selected = iou_scores[effective_indices]  # (K,) or (Q,)
                
                # 关键修复：混合 soft target
                # 基础权重：matched query 至少占 70%
                base_weight = 0.7
                
                # 初始化 soft_target：所有 query 给一点点基础权重
                soft_target = torch.full((num_selected,), (1 - base_weight) / num_selected, device=device)
                # matched query 给 base_weight
                soft_target[target_idx_in_selected] = base_weight
                
                # IoU 修正：如果有其他高 IoU 的 query，给它们一些额外权重
                max_iou = iou_selected.max()
                if max_iou > 0.1:  # 只有当有query的IoU > 0.1时才修正
                    # 找到IoU > 0.1的queries
                    high_iou_mask = iou_selected > 0.1
                    if high_iou_mask.sum() > 0:
                        # 重新分配 (1-base_weight) 的权重给高IoU queries
                        iou_weights = iou_selected * high_iou_mask.float()
                        iou_weights = iou_weights / (iou_weights.sum() + 1e-6) * (1 - base_weight)
                        soft_target = torch.zeros_like(soft_target)
                        soft_target[target_idx_in_selected] = base_weight
                        soft_target = soft_target + iou_weights
                
                # 归一化确保和为1
                soft_target = soft_target / (soft_target.sum() + 1e-6)
                
                # 计算 KL 散度 loss
                log_probs = F.log_softmax(sim_selected, dim=0)
                loss_b = -(soft_target * log_probs).sum()
            else:
                # 纯硬标签方式
                loss_b = F.cross_entropy(sim_selected.unsqueeze(0), target_idx_in_selected.view(1))
            
            loss_anomaly = loss_anomaly + loss_b
            valid_anomaly += 1
        
        elif include_normal and not is_anomaly[b]:
            # ===== 修复2：Normal图也参与（直接惩罚高相似度）=====
            # 目标：normal图中的所有query不应该与异常prompt太相似
            max_sim_to_abnormal = all_sim.max()
            
            if p_norm_normal is not None:
                # 有normal prompt时：query应更接近normal prompt
                sim_to_normal = (hs_norm[b] @ p_norm_normal[b]) / temp
                max_sim_to_normal = sim_to_normal.max()
                margin_loss = F.relu(max_sim_to_abnormal - max_sim_to_normal + normal_margin)
            else:
                # 没有normal prompt时：直接惩罚与异常prompt的高相似度
                # 目标：让 max_sim_to_abnormal 低于某个阈值（比如 0）
                # 使用 soft hinge: 当 max_sim > -margin 时产生惩罚
                margin_loss = F.relu(max_sim_to_abnormal + normal_margin)
                # 或者更aggressive: 让整体相似度分布更平坦
                # sim_var = all_sim.var()  # 方差越小越均匀
                # margin_loss = F.relu(max_sim_to_abnormal) + 0.1 * sim_var
            
            loss_normal = loss_normal + margin_loss
            valid_normal += 1
    
    # 合并损失
    total_loss = torch.tensor(0.0, device=device)
    
    if valid_anomaly > 0:
        total_loss = total_loss + loss_anomaly / valid_anomaly
    
    if valid_normal > 0:
        total_loss = total_loss + 0.5 * loss_normal / valid_normal
    
    return total_loss

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


def pool_prompt_features_sowa_style(
    prompt_seq: torch.Tensor,
    prompt_lists: list,
    is_anomaly: list,
    device: torch.device
) -> torch.Tensor:
    """
    SOWA风格的prompt特征池化
    
    对于每个样本，根据其is_anomaly状态选择对应的prompt feature：
    - 异常样本：使用abnormal类型的prompt（或所有prompts的mean）
    - 正常样本：使用normal类型的prompt（或所有prompts的mean）
    
    Args:
        prompt_seq: prompt特征，可能的格式:
            - (L, B, D): L个层，每层B个样本，D维特征
            - (B, L, D): B个样本，L个token，D维特征
            - (B, D): 直接是每个样本的prompt特征
        prompt_lists: List[str]，每个样本的prompt文本
        is_anomaly: List[bool]，每个样本是否是异常
        device: 设备
    
    Returns:
        prompt_proto: (B, D) 每个样本对应的prompt prototype
    """
    if prompt_seq is None:
        return None
    
    # 获取最后一层的特征（通常是最有用的）
    if prompt_seq.dim() == 3:
        # 可能是 (L, B, D) 或 (B, L, D)
        if prompt_seq.shape[1] == len(prompt_lists):
            # (L, B, D) 格式
            last = prompt_seq[-1]  # (B, D)
        elif prompt_seq.shape[0] == len(prompt_lists):
            # (B, L, D) 格式 - 对L维度做mean pooling
            last = prompt_seq.mean(dim=1)  # (B, D)
        else:
            # 不确定格式，取最后一个
            last = prompt_seq[-1]
    elif prompt_seq.dim() == 2:
        last = prompt_seq  # 已经是 (B, D)
    else:
        # 其他情况，reshape
        B = len(prompt_lists)
        last = prompt_seq.reshape(B, -1)[:, : (prompt_seq.numel() // B)]
    
    # 确保是 (B, D) 格式
    B = len(prompt_lists)
    if last.shape[0] != B:
        if last.shape[1] == B:
            last = last.transpose(0, 1).contiguous()
        else:
            last = last.reshape(B, -1)[:, : (last.numel() // B)]
    
    return last.to(device)


def compute_filo_aligned_features(
    patch_tokens: torch.Tensor,  # (B, N, D) FiLo的patch tokens
    masks: torch.Tensor,         # (B, H, W) GT masks
    is_anomaly: list,            # 每张图是否是异常
    device
):
    """
    使用FiLo的patch_tokens计算aligned_features，用于align_loss
    
    这个函数让FiLo的梯度流入align_loss，使FiLo能够替代LoRA的作用。
    
    对于异常图：使用mask区域的patch tokens的加权pooling
    对于正常图：使用全局average pooling作为背景锚点
    """
    B, N, D = patch_tokens.shape
    H_feat = W_feat = int(N ** 0.5)  # 假设是方形
    
    if masks is None:
        # 没有mask，使用全局pooling
        embeddings = patch_tokens.mean(dim=1)  # (B, D)
        is_background = torch.zeros(B, dtype=torch.bool, device=device)
        return embeddings, is_background
    
    # 下采样mask到patch token的空间尺寸
    masks_ds = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(H_feat, W_feat),
        mode='nearest'
    ).squeeze(1)  # (B, H_feat, W_feat)
    masks_flat = masks_ds.view(B, N, 1)  # (B, N, 1)
    
    embeddings = []
    is_background_list = []
    
    for b in range(B):
        if is_anomaly is not None and is_anomaly[b]:
            # 异常图：mask区域的加权pooling
            mask_b = masks_flat[b]  # (N, 1)
            pos_count = mask_b.sum().clamp(min=1.0)
            emb = (patch_tokens[b] * mask_b).sum(dim=0) / pos_count  # (D,)
            embeddings.append(emb)
            is_background_list.append(False)
        else:
            # 正常图：全局pooling（背景锚点）
            emb = patch_tokens[b].mean(dim=0)  # (D,)
            embeddings.append(emb)
            is_background_list.append(True)
    
    embeddings = torch.stack(embeddings, dim=0)  # (B, D)
    is_background_tensor = torch.tensor(is_background_list, dtype=torch.bool, device=device)
    
    return embeddings, is_background_tensor


def align_loss_with_background_margin(
    prompt_proto: torch.Tensor,     # (B, D)
    visual_embed: torch.Tensor,     # (B, D)
    is_background: torch.Tensor,    # (B,) bool
    group_labels: torch.Tensor,     # (B,) 分组标签
    temp: float = 0.15,              # 增大温度（原 0.07）
    margin: float = 0.2             # 降低 margin（原 0.5）
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



class MinNormalBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that enforces at least `min_normals` normal samples per batch.

    - Draws `min_normals` indices from normal pool each batch (with cycling).
    - Fills the remaining slots with anomaly indices (with cycling).
    - If a pool is empty, it falls back to sampling from the other pool.
    """
    def __init__(self, dataset, batch_size: int, min_normals: int = 2, drop_last: bool = False, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.min_normals = int(min_normals)
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0

        # Build index pools (dataset is expected to expose .entries or yield objects with .anomaly)
        entries = getattr(dataset, "entries", None)
        if entries is None:
            # fallback: materialize anomalies by iterating (may be slow but ok for MVTEC)
            entries = list(dataset)

        self.norm_idx = []
        self.anom_idx = []
        for i, e in enumerate(entries):
            a = int(getattr(e, "anomaly", 0))
            (self.anom_idx if a == 1 else self.norm_idx).append(i)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch)
    
        norm = self.norm_idx.copy()
        anom = self.anom_idx.copy()
        rng.shuffle(norm)
        rng.shuffle(anom)
    
        ni, ai = 0, 0
    
        def _next_from(pool, idx_ptr):
            if not pool:
                return None, idx_ptr
            if idx_ptr >= len(pool):
                rng.shuffle(pool)
                idx_ptr = 0
            out = pool[idx_ptr]
            return out, idx_ptr + 1
    
        n_batches = len(self)
    
        for _ in range(n_batches):
            batch = []
    
            # -------- decide how many normals in THIS batch (randomized) --------
            if norm and anom:
                # you要求：每个batch负样本数量在 [2, batch_size-2]，保证至少2个anomaly
                min_norm = max(1, int(self.min_normals))              # 你一般会设为2
                min_anom = 2 if self.batch_size >= 4 else 1
                max_norm = max(0, self.batch_size - min_anom)
    
                # 保护：batch_size太小/参数不合法时退化
                if max_norm < min_norm:
                    n_norm = max_norm
                else:
                    # n_norm = rng.randint(min_norm, max_norm)
                    n_norm = min_norm
            elif norm:
                # 没有anomaly时只能全normal（不建议出现，但保证不崩）
                n_norm = min(self.batch_size, max(1, int(self.min_normals)))
            else:
                # 没有normal
                n_norm = 0
    
            # -------- take normals --------
            for _k in range(n_norm):
                v, ni = _next_from(norm, ni)
                if v is not None:
                    batch.append(v)
    
            # -------- fill the rest with anomalies (fallback to normals if needed) --------
            while len(batch) < self.batch_size:
                if anom:
                    v, ai = _next_from(anom, ai)
                    if v is not None:
                        batch.append(v)
                        continue
                    
                # fallback: no anomalies, try normals
                v, ni = _next_from(norm, ni)
                if v is not None:
                    batch.append(v)
                else:
                    break
                
            # ★关键：打乱batch内部顺序，避免负样本永远在前面
            rng.shuffle(batch)
    
            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                yield batch



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
    min_normals_per_batch: int = 0,
    # CoOp/CoCoOp prompt mode
    prompt_mode: str = "simple",
    few_shot_per_specie: int = 0,
    few_shot_balance_good_by_specie: bool = False,
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
        prompt_mode=prompt_mode,
    )

    # --- optional few-shot subsampling per specie ---
    if few_shot_per_specie and few_shot_per_specie > 0 and mode in ("train", "train_all"):
        # 这里使用你上面定义好的 few_shot_subsample_entries
        orig_n = len(ds.entries)
        ds.entries = few_shot_subsample_entries(
            entries=ds.entries,
            shots_per_specie=few_shot_per_specie,
            seed=specie_split_seed,
            verbose=True,
            balance_good_by_specie=bool(few_shot_balance_good_by_specie),
        )
        print(
            f"[INFO] Few-shot subsampling enabled: {few_shot_per_specie} samples/specie, "
            f"{orig_n} -> {len(ds.entries)} entries"
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

    # Optional: enforce at least N normal samples per batch (single-process only).
    batch_sampler = None
    if (not distributed) and (int(min_normals_per_batch) > 0) and (mode in ("train", "train_all")):
        try:
            labels = [int(e.anomaly) for e in ds.entries]
        except Exception:
            labels = [int(getattr(e, "anomaly", 0)) for e in ds]
        has_norm = any(l == 0 for l in labels)
        has_anom = any(l == 1 for l in labels)
        if has_norm and has_anom:
            batch_sampler = MinNormalBatchSampler(
                ds,
                batch_size=batch_size,
                min_normals=int(min_normals_per_batch),
                drop_last=False,
                seed=int(specie_split_seed),
            )

    if batch_sampler is not None:
        dataloader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
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


def _summarize_entries_for_debug(entries):
    stats = {
        "total": int(len(entries)),
        "anomaly": int(sum(1 for e in entries if int(getattr(e, "anomaly", 0)) == 1)),
        "normal": int(sum(1 for e in entries if int(getattr(e, "anomaly", 0)) == 0)),
        "by_class": {},
    }
    by = defaultdict(lambda: defaultdict(lambda: {"anomaly": 0, "normal": 0, "total": 0}))
    for e in entries:
        cls = str(getattr(e, "cls_name", ""))
        sp = str(getattr(e, "specie_name", ""))
        an = int(getattr(e, "anomaly", 0))
        rec = by[cls][sp]
        rec["total"] += 1
        if an == 1:
            rec["anomaly"] += 1
        else:
            rec["normal"] += 1
    for cls in sorted(by.keys()):
        stats["by_class"][cls] = {sp: dict(by[cls][sp]) for sp in sorted(by[cls].keys())}
    return stats



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

    # 解析 selected_levels 参数
    def _parse_selected_levels(args):
        """解析stages消融实验配置"""
        # 预定义的消融配置
        ABLATION_CONFIGS = {
            'single_level_0': [0],
            'single_level_1': [1],
            'single_level_2': [2],
            'single_level_3': [3],
            'levels_0_1': [0, 1],
            'levels_1_2': [1, 2],
            'levels_2_3': [2, 3],
            'levels_0_2': [0, 2],
            'levels_0_1_2': [0, 1, 2],
            'levels_1_2_3': [1, 2, 3],
            'all_levels': [0, 1, 2, 3],
            'sowa_style': [1, 2, 3],  # SOWA推荐：跳过最高分辨率层
        }
        
        # 优先使用预定义配置
        if hasattr(args, 'ablation_config') and args.ablation_config:
            config = ABLATION_CONFIGS.get(args.ablation_config)
            if config:
                print(f"[INFO] 使用消融实验配置: {args.ablation_config} -> levels {config}")
                return config
        
        # 其次使用手动指定的层
        if hasattr(args, 'selected_levels') and args.selected_levels:
            try:
                levels = [int(x.strip()) for x in args.selected_levels.split(',')]
                print(f"[INFO] 手动指定的层级: {levels}")
                return levels
            except:
                pass
        
        return None  # 使用默认配置

    if args.use_official:
        model = FineTuneSAM3Official(
            bpe_path=args.bpe_path,
            sam3_ckpt=args.sam3_ckpt,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layer_ids=args.lora_layer_ids,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            enable_parallel_lora=args.enable_parallel_lora,
            parallel_lora_rank=args.parallel_lora_rank,
            parallel_lora_alpha=args.parallel_lora_alpha,
            parallel_lora_target=args.parallel_lora_target,
            parallel_lora_layer_ids=args.parallel_lora_layer_ids,
            enable_out_adapter_lora=args.enable_out_adapter_lora,
            device=device,
            class_list=args.class_list,
            num_templates=getattr(args, "num_templates", 4),
            # ===== CoOp/CoCoOp参数 =====
            prompt_learner_type=args.prompt_learner_type,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            class_token_position=args.class_token_position,
            use_keywords=args.use_keywords,
            cocoop_vis_dim=args.cocoop_vis_dim,
            cocoop_reduction=args.cocoop_reduction,
            # ===== Compound Prompt Learning =====
            compound_mode=getattr(args, "compound_mode", "cocoop"),
            compound_n_ctx=getattr(args, "compound_n_ctx", 4),
            compound_n_ctx_offset=getattr(args, "compound_n_ctx_offset", 4),
            compound_num_abnormal=getattr(args, "compound_num_abnormal", 10),
            compound_enable_dap=getattr(args, "compound_enable_dap", False),
            compound_dap_top_k=getattr(args, "compound_dap_top_k", 10),
            compound_meta_reduction=getattr(args, "compound_meta_reduction", 16),
            compound_dap_use_multilevel=getattr(args, "compound_dap_use_multilevel", False),
            compound_dap_num_levels=getattr(args, "compound_dap_num_levels", 0),
            compound_use_text_encoder=getattr(args, "compound_use_text_encoder", False),
            compound_abnormal_word=getattr(args, "compound_abnormal_word", "anomaly"),
            compound_pooling=getattr(args, "compound_pooling", "ctx_only"),
            compound_abnormal_order=getattr(args, "compound_abnormal_order", "v_then_wk"),
            compound_dap_spurious_filter=getattr(args, "compound_dap_spurious_filter", False),
            compound_dap_spurious_alpha=getattr(args, "compound_dap_spurious_alpha", 1.0),
            compound_disable_w=getattr(args, "compound_disable_w", False),
            # ===== 多尺度特征 & Stages消融 =====
            num_feature_levels=getattr(args, "num_feature_levels", 1),
            selected_levels=_parse_selected_levels(args),
            enable_vv_attention=getattr(args, "enable_vv_attention", False),
            vv_num_heads=getattr(args, "vv_num_heads", 8),
            vv_dropout=getattr(args, "vv_dropout", 0.1),
            # ===== FiLo模块 (6路卷积MMCI) =====
            enable_filo=getattr(args, "enable_filo", False),
            filo_dim_out=getattr(args, "filo_dim_out", 768),
            filo_k_linear=getattr(args, "filo_k_linear", 4),
            filo_k_cov=getattr(args, "filo_k_cov", 4),
            filo_image_size=getattr(args, "filo_image_size", 518),
            filo_use_alternating=getattr(args, "filo_use_alternating", True),
            # # ===== 方案B: FiLo到Decoder回灌 =====
            filo_to_decoder=getattr(args, "filo_to_decoder", False),
            filo_decoder_mode=getattr(args, "filo_decoder_mode", "memory"),
            filo_decoder_tokens=getattr(args, "filo_decoder_tokens", 64),
            # ===== MSAD模块 =====
            enable_msad=getattr(args, "enable_msad", False),
            msad_use_shape_attention=getattr(args, "msad_use_shape_attention", True),
            msad_learnable_level_weights=getattr(args, "msad_learnable_level_weights", True),
            msad_learnable_temperature=getattr(args, "msad_learnable_temperature", True),
            msad_temperature=getattr(args, "msad_temperature", 100.0),
            msad_output_size=getattr(args, "msad_output_size", 518),
            msad_num_levels=getattr(args, "msad_num_levels", None),
            msad_return_similarity_logits=getattr(args, "msad_return_similarity_logits", False),
            msad_use_vision_adapter=getattr(args, "msad_use_vision_adapter", False),
            msad_vision_adapter_reduction=getattr(args, "msad_vision_adapter_reduction", 2),
            msad_vision_adapter_shared=(getattr(args, "msad_vision_adapter_shared", True) and (not getattr(args, "msad_vision_adapter_not_shared", False))),
            # ===== Spurious Gating =====
            enable_spurious_gating=(getattr(args, "enable_spurious_gating", True) and (not getattr(args, "disable_spurious_gating", False))),
            spurious_sim_temp=getattr(args, "spurious_sim_temp", 0.07),
            spurious_topk_ratio=getattr(args, "spurious_topk_ratio", 0.02),
            spurious_score_threshold=getattr(args, "spurious_score_threshold", 0.20),
            spurious_kappa=getattr(args, "spurious_kappa", 8.0),
            spurious_quality_threshold=getattr(args, "spurious_quality_threshold", 0.03),
            # ===== 方案C: 置信度融合头 =====
            enable_conf_fusion_head=getattr(args, "enable_conf_fusion_head", False),
            conf_fusion_hidden_dim=getattr(args, "conf_fusion_hidden_dim", 64),
            enable_multiscale_output=bool(getattr(args, "enable_multiscale_vis", False) or getattr(args, "align_multilevel", False)),
        )
    else:
        model = FineTuneSAM3(
            bpe_path=args.bpe_path,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layer_ids=args.lora_layer_ids,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            device=device,
        )

    model.to(device)

    if args.print_backbone:
        dump_f = None
        if getattr(args, "print_backbone_to_txt", False):
            txt_path = getattr(args, "print_backbone_txt_path", "backbone_dump.txt")
            # 建议放到当前 run 的 save_dir 下（更好管理）
            # 但此时 save_dir 还没生成的话，就先直接写到工作目录
            dump_f = open(txt_path, "w", encoding="utf-8")
            print(f"[INFO] Saving backbone dump to: {txt_path}")
    
        also_stdout = not getattr(args, "print_backbone_no_stdout", False)
    
        try:
            _print_model_tree(model, name="FineTuneSAM3Official",
                              filter_key=args.print_modules_filter,
                              file_handle=dump_f, also_stdout=also_stdout)
    
            if hasattr(model, "backbone"):
                _print_model_tree(model.backbone, name="model.backbone",
                                  filter_key=args.print_modules_filter,
                                  file_handle=dump_f, also_stdout=also_stdout)
    
                if hasattr(model.backbone, "vision_backbone"):
                    _print_model_tree(model.backbone.vision_backbone, name="model.backbone.vision_backbone",
                                      filter_key=args.print_modules_filter,
                                      file_handle=dump_f, also_stdout=also_stdout)
    
                    if hasattr(model.backbone.vision_backbone, "trunk"):
                        _print_model_tree(model.backbone.vision_backbone.trunk, name="model.backbone.vision_backbone.trunk",
                                          filter_key=args.print_modules_filter,
                                          file_handle=dump_f, also_stdout=also_stdout)
    
            if hasattr(model, "transformer"):
                _print_model_tree(model.transformer, name="model.transformer",
                                  filter_key=args.print_modules_filter,
                                  file_handle=dump_f, also_stdout=also_stdout)
    
        finally:
            if dump_f is not None:
                dump_f.close()
    
        raise SystemExit(0)


    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # 无论是否分布式，model_core 指向实际的 underlying module（方便后续直接访问 prompt_learner 等属性）
    model_core = model.module if hasattr(model, "module") else model

    if args.sam3_ckpt and os.path.exists(args.sam3_ckpt) and (not bool(getattr(args, "use_official", False)) or bool(getattr(args, "force_secondary_sam3_load", False))):
        load_sam3_checkpoint(model, args.sam3_ckpt)
    elif args.sam3_ckpt and os.path.exists(args.sam3_ckpt) and bool(getattr(args, "use_official", False)):
        print("[INFO] Skip secondary SAM3 ckpt load (official model already loaded checkpoint_path). Use --force_secondary_sam3_load to override.")

    # --- create run_name and save/log dirs early so dataset can save splits ---
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

    r2_corr_state = {"n": 0, "mean_x": 0.0, "mean_y": 0.0, "C": 0.0, "M2x": 0.0, "M2y": 0.0}

    meta_path_for_dataset = args.meta_path or os.path.join(args.data_root, "meta.json")
    splits_dir = getattr(args, "splits_save_dir", None)
    if splits_dir is not None:
        os.makedirs(splits_dir, exist_ok=True)

    dataloader = build_dataloaders(
        root=args.data_root,
        meta_path=meta_path_for_dataset,
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
        splits_save_dir=splits_dir or save_dir,
        min_normals_per_batch=getattr(args, 'min_normals_per_batch', 0),
        prompt_mode=args.prompt_mode,
        few_shot_per_specie=args.few_shot_per_specie,
        few_shot_balance_good_by_specie=bool(getattr(args, "few_shot_balance_good_by_specie", False)),
    )

    if is_main_process and hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "entries"):
        try:
            stats = _summarize_entries_for_debug(dataloader.dataset.entries)
            with open(os.path.join(save_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Wrote dataset_stats.json: total={stats['total']} anomaly={stats['anomaly']} normal={stats['normal']}")
        except Exception as e:
            print(f"[WARN] Failed to write dataset_stats.json: {e}")



    # Freeze everything except LoRA/prompt params
    for n, p in model.named_parameters():
        nl = n.lower()
        if ("lora" in nl) or ("out_adapter" in nl) or ("prompt_learner" in nl) \
           or ("meta_net" in nl) or ("msad" in nl):
            p.requires_grad = True
        elif getattr(args, "train_seg_head", False) and nl.startswith("segmentation_head"):
            p.requires_grad = True
        else:
            p.requires_grad = False

    if bool(getattr(args, "compound_disable_w", False)) and hasattr(model, "prompt_learner"):
        pl = getattr(model, "prompt_learner", None)
        if hasattr(pl, "w") and isinstance(getattr(pl, "w"), torch.nn.Parameter):
            pl.w.requires_grad = False

    if str(getattr(args, "train_objective", "seg")).lower() == "rank":
        for n, p in model.named_parameters():
            nl = n.lower()
            if ("prompt_learner" in nl) or ("msad" in nl) or ("meta_net" in nl):
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

    if getattr(args, "enable_msad", False):
        print("[INFO] MSAD params (name, requires_grad, shape):")
        for n, p in model.named_parameters():
            nl = n.lower()
            if "msad" in nl:
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

    if getattr(args, "enable_msad", False):
        msad_ids = {id(p) for n, p in model.named_parameters() if ("msad" in n.lower()) and p.requires_grad}
        opt_ids = set()
        for pg in optimizer.param_groups:
            for p in pg.get("params", []):
                opt_ids.add(id(p))
        msad_in_opt = len(msad_ids.intersection(opt_ids))
        print(f"[INFO] MSAD params in optimizer: {msad_in_opt}/{len(msad_ids)}")

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
    
    # ==================== 新增：两阶段 lambda_query_align 调度器 ====================
    lambda_scheduler = None
    if getattr(args, 'enable_two_stage', False):
        lambda_scheduler = TwoStageLambdaScheduler(
            total_steps=total_steps,
            stage1_ratio=getattr(args, 'stage1_ratio', 0.30),
            stage1_lambda=getattr(args, 'stage1_lambda', 0.10),
            stage2_lambda=getattr(args, 'stage2_lambda', 0.20),
            transition=getattr(args, 'lambda_transition', 'linear'),
            transition_ratio=getattr(args, 'transition_ratio', 0.25),
        )
    
    print("=" * 60)
    print("训练配置:")
    print(f"  总epoch数: {args.epochs}")
    print(f"  每epoch步数: {steps_per_epoch}")
    print(f"  梯度累积: {args.gradient_accumulation}")
    print(f"  有效batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  总优化步数: {total_steps}")
    print(f"  Warmup步数: {warmup_steps} ({args.warmup_ratio*100:.0f}%)")
    print(f"  最终LR比例: {args.min_lr_ratio}")
    if lambda_scheduler:
        print(f"  两阶段调度: 已启用")
        print(f"    Stage 1 (0~{int(total_steps*args.stage1_ratio)}步): lambda={args.stage1_lambda}")
        print(f"    Stage 2 ({int(total_steps*args.stage1_ratio)}~{total_steps}步): lambda={args.stage2_lambda}")
    else:
        print(f"  两阶段调度: 未启用，使用固定 lambda_query_align={args.lambda_query_align}")
    print("=" * 60)

    model.train()
    best_loss = float("inf")
    global_optim_step = 0  # 优化器更新计数（用于调度器）
    stop_training = False

    anomaly_bank = AnomalyFeatureBankV2(
        max_size=2048,
        dim=256,
        min_fill_ratio=0.5,
        warm_up_ratio=getattr(args, 'bank_warm_up_ratio', 0.3),
        orthogonalize=getattr(args, 'bank_orthogonalize', True),
    )
    print(f"[INFO] Initialized AnomalyFeatureBankV2: "
          f"max_size=2048, warm_up={anomaly_bank.warm_up_ratio}, "
          f"orthogonalize={anomaly_bank.orthogonalize}")
    print(f"[INFO] Initialized AnomalyFeatureBank: max_size=2048, min_fill=50%")
    
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

            # 【新增】类别无关设计
            if getattr(args, 'class_agnostic', False):
                agnostic_name = getattr(args, 'agnostic_name', 'object')
                class_names = [agnostic_name] * len(class_names)

            images = images.to(device)
            masks = masks.to(device)
            
            # 获取当前的 lambda_query_align 值（两阶段调度或固定值）
            if lambda_scheduler is not None:
                current_lambda_query_align = lambda_scheduler.get_lambda()
            else:
                current_lambda_query_align = getattr(args, 'lambda_query_align', 0.5)

            # ===== AMP autocast: forward + loss 计算都放在半精度上下文 =====
            with autocast(enabled=(device.type == "cuda")):
                out = model(images, prompt_lists, class_names)
                pred_masks = out["pred_masks"]
                if pred_masks is None:
                    raise RuntimeError("Segmentation head did not return pred_masks.")

                if bool(getattr(args, "debug_dump_features", False)) and step == 0:
                    dump = {}
                    tfs = out.get("text_features_structured", None)
                    if isinstance(tfs, dict):
                        for k, v in tfs.items():
                            if isinstance(v, torch.Tensor):
                                dump[f"tfs.{k}"] = v.detach().float().cpu().numpy()
                    for k in ("eta_spurious", "msad_anomaly_score"):
                        v = out.get(k, None)
                        if isinstance(v, torch.Tensor):
                            dump[k] = v.detach().float().cpu().numpy()
                    dump["is_anomaly"] = is_anomaly.detach().cpu().numpy() if isinstance(is_anomaly, torch.Tensor) else np.array(is_anomaly)
                    dump["class_names"] = np.array(list(class_names))
                    dump["prompt_lists0"] = np.array(prompt_lists[0] if isinstance(prompt_lists, list) and len(prompt_lists) > 0 else [])
                    np.savez_compressed(os.path.join(save_dir, f"debug_features_step{step}.npz"), **dump)

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

                # ---- log len(src_q) distribution ----
                if is_main_process:
                    # indices: List[(src_q, tgt_q)] length=B
                    src_q_lens = [int(src_q.numel()) for (src_q, _) in indices]  # 每张图匹配到的query数量
                    nonempty_ratio = sum(l > 0 for l in src_q_lens) / max(1, len(src_q_lens))
                    mean_len = sum(src_q_lens) / max(1, len(src_q_lens))
                    max_len = max(src_q_lens) if len(src_q_lens) > 0 else 0
                

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
                        tfs = out.get("text_features_structured", None)
                        if isinstance(tfs, dict) and ("normal" in tfs):
                            prompt_proto = tfs["normal"]
                        else:
                            res = model_core.prompt_learner(prompt_lists, device=device)
                            if isinstance(res, (tuple, list)) and len(res) >= 2 and torch.is_tensor(res[0]) and res[0].dim() == 4:
                                prefixes = res[0]
                                proto = prefixes.mean(dim=2)
                                prompt_proto = proto[:, 0]
                            elif torch.is_tensor(res):
                                prompt_proto = res.mean(dim=0) if res.dim() == 3 else res
                            else:
                                prompt_proto = None
                        if prompt_proto is not None:
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
                        print(f"[BG LOSS] focal_bg={loss_focal_bg.detach().item():.4f} dice_bg={loss_dice_bg.detach().item():.4f} bg_w={bg_w} num_bg_images={num_bg_images}")


                        
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

                # -------------------------
                # FiLo Anomaly Map Loss (如果启用FiLo)
                # 使用Focal + Dice Loss监督FiLo的anomaly_map拟合GT mask
                # -------------------------
                loss_filo = torch.tensor(0.0, device=device)
                lambda_filo = getattr(args, 'lambda_filo', 0.0)
                
                if lambda_filo > 0.0:
                    filo_anomaly_maps = out.get("filo_anomaly_maps", [])
                    filo_agg_map = out.get("filo_aggregated_map", None)
                    
                    if filo_agg_map is not None or len(filo_anomaly_maps) > 0:
                        # 使用聚合的map或最后一层的map
                        if filo_agg_map is not None:
                            filo_map = filo_agg_map  # (B, 2, H, W)
                        else:
                            # 多层时取最后一层
                            filo_map = filo_anomaly_maps[-1]  # (B, 2, H, W)
                        
                        # filo_map[:, 1] 是abnormal通道，应该拟合GT mask
                        filo_abnormal = filo_map[:, 1]  # (B, H, W)
                        
                        # ===== 风险2修复：检查FiLo输出是否是概率 =====
                        filo_min = filo_abnormal.min().item()
                        filo_max = filo_abnormal.max().item()
                        
                        if step < 5:
                            # 前几步打印检查
                            print(f"[FiLo Check] min={filo_min:.4f}, max={filo_max:.4f}, "
                                  f"is_prob={(filo_min >= -0.01 and filo_max <= 1.01)}")
                        
                        # 如果不是概率（0-1范围），需要sigmoid
                        if filo_min < -0.01 or filo_max > 1.01:
                            filo_abnormal = torch.sigmoid(filo_abnormal)
                            if step < 5:
                                print(f"[FiLo] Applied sigmoid: now min={filo_abnormal.min():.4f}, max={filo_abnormal.max():.4f}")
                        
                        # GT mask
                        gt_mask = masks  # (B, H, W)
                        if gt_mask.dim() == 4:
                            gt_mask = gt_mask.squeeze(1)
                        
                        # 确保尺寸匹配
                        if filo_abnormal.shape[-2:] != gt_mask.shape[-2:]:
                            filo_abnormal = F.interpolate(
                                filo_abnormal.unsqueeze(1),
                                size=gt_mask.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(1)
                        
                        # 在计算BCE时禁用autocast（BCE在autocast下不安全）
                        with torch.amp.autocast('cuda', enabled=False):
                            # 转为float32
                            filo_abnormal_f32 = filo_abnormal.float()
                            gt_mask_f32 = gt_mask.float()
                            
                            # 现在filo_abnormal应该在0-1范围内
                            filo_abnormal_clamped = filo_abnormal_f32.clamp(1e-6, 1 - 1e-6)
                            
                            # Focal-like loss (binary)
                            pt = torch.where(gt_mask_f32 > 0.5, filo_abnormal_clamped, 1 - filo_abnormal_clamped)
                            focal_weight = (1 - pt).pow(2.0)  # gamma=2
                            bce = F.binary_cross_entropy(filo_abnormal_clamped, gt_mask_f32, reduction='none')
                            loss_filo_focal = (focal_weight * bce).mean()
                            
                            # Dice loss
                            intersection = (filo_abnormal_clamped * gt_mask_f32).sum(dim=(1, 2))
                            union = filo_abnormal_clamped.sum(dim=(1, 2)) + gt_mask_f32.sum(dim=(1, 2))
                            dice = (2 * intersection + 1e-6) / (union + 1e-6)
                            loss_filo_dice = (1 - dice).mean()
                        
                        # 组合
                        loss_filo = 0.5 * loss_filo_focal + 0.5 * loss_filo_dice
                        
                        if step % 50 == 0:
                            print(f"[FiLo Loss] focal={float(loss_filo_focal):.4f} dice={float(loss_filo_dice):.4f} total={float(loss_filo):.4f}")

                # -------------------------
                # MSAD Anomaly Map Loss (如果启用MSAD)
                # 使用Focal + Dice Loss监督MSAD的anomaly_score拟合GT mask
                # -------------------------
                loss_msad = torch.tensor(0.0, device=device)
                lambda_msad = getattr(args, 'lambda_msad', 0.0)
                loss_msad_img = torch.tensor(0.0, device=device)
                lambda_msad_img = float(getattr(args, "lambda_msad_img", 0.0) or 0.0)
                
                if lambda_msad > 0.0:
                    msad_map = out.get("msad_aggregated_map", None)
                    msad_score = out.get("msad_anomaly_score", None)

                    use_2ch = str(getattr(args, "train_objective", "seg")).lower() == "rank"
                    if use_2ch and isinstance(msad_map, torch.Tensor):
                        gt_mask = masks
                        if gt_mask.dim() == 4:
                            gt_mask = gt_mask.squeeze(1)
                        if msad_map.shape[-2:] != gt_mask.shape[-2:]:
                            gt_mask_resized = F.interpolate(
                                gt_mask.unsqueeze(1).float(),
                                size=msad_map.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze(1)
                        else:
                            gt_mask_resized = gt_mask.float()

                        with torch.amp.autocast('cuda', enabled=False):
                            gt_f32 = gt_mask_resized.float()
                            p_norm = msad_map[:, 0].float().clamp(1e-6, 1.0 - 1e-6)
                            p_abn = msad_map[:, 1].float().clamp(1e-6, 1.0 - 1e-6)

                            tgt_norm = (1.0 - gt_f32).clamp(0.0, 1.0)
                            tgt_abn = gt_f32.clamp(0.0, 1.0)

                            pt_abn = torch.where(tgt_abn > 0.5, p_abn, 1.0 - p_abn)
                            pt_norm = torch.where(tgt_norm > 0.5, p_norm, 1.0 - p_norm)
                            fw_abn = (1.0 - pt_abn).pow(2.0)
                            fw_norm = (1.0 - pt_norm).pow(2.0)

                            bce_abn = F.binary_cross_entropy(p_abn, tgt_abn, reduction="none")
                            bce_norm = F.binary_cross_entropy(p_norm, tgt_norm, reduction="none")
                            loss_msad_focal = (fw_abn * bce_abn).mean() + (fw_norm * bce_norm).mean()

                            inter_abn = (p_abn * tgt_abn).sum(dim=(1, 2))
                            union_abn = p_abn.sum(dim=(1, 2)) + tgt_abn.sum(dim=(1, 2))
                            dice_abn = (2.0 * inter_abn + 1e-6) / (union_abn + 1e-6)

                            inter_norm = (p_norm * tgt_norm).sum(dim=(1, 2))
                            union_norm = p_norm.sum(dim=(1, 2)) + tgt_norm.sum(dim=(1, 2))
                            dice_norm = (2.0 * inter_norm + 1e-6) / (union_norm + 1e-6)

                            loss_msad_dice = (1.0 - dice_abn).mean() + (1.0 - dice_norm).mean()

                        loss_msad = 0.5 * loss_msad_focal + 0.5 * loss_msad_dice
                        if step % 50 == 0:
                            print(f"[MSAD Loss-2ch] focal={loss_msad_focal.detach().item():.4f} dice={loss_msad_dice.detach().item():.4f} total={loss_msad.detach().item():.4f}")

                    elif msad_score is not None:
                        gt_mask = masks
                        if gt_mask.dim() == 4:
                            gt_mask = gt_mask.squeeze(1)
                        if msad_score.shape[-2:] != gt_mask.shape[-2:]:
                            gt_mask_resized = F.interpolate(
                                gt_mask.unsqueeze(1).float(),
                                size=msad_score.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze(1)
                        else:
                            gt_mask_resized = gt_mask.float()

                        with torch.amp.autocast('cuda', enabled=False):
                            msad_score_f32 = msad_score.float()
                            gt_mask_f32 = gt_mask_resized.float()
                            msad_score_clamped = msad_score_f32.clamp(1e-6, 1 - 1e-6)
                            pt = torch.where(gt_mask_f32 > 0.5, msad_score_clamped, 1 - msad_score_clamped)
                            focal_weight = (1 - pt).pow(2.0)
                            bce = F.binary_cross_entropy(msad_score_clamped, gt_mask_f32, reduction='none')
                            loss_msad_focal = (focal_weight * bce).mean()

                            intersection = (msad_score_clamped * gt_mask_f32).sum(dim=(1, 2))
                            union = msad_score_clamped.sum(dim=(1, 2)) + gt_mask_f32.sum(dim=(1, 2))
                            dice = (2 * intersection + 1e-6) / (union + 1e-6)
                            loss_msad_dice = (1 - dice).mean()

                        loss_msad = 0.5 * loss_msad_focal + 0.5 * loss_msad_dice
                        if step % 50 == 0:
                            print(f"[MSAD Loss] focal={loss_msad_focal.detach().item():.4f} dice={loss_msad_dice.detach().item():.4f} total={loss_msad.detach().item():.4f}")

                if lambda_msad_img > 0.0:
                    msad_score = out.get("msad_anomaly_score", None)
                    if msad_score is not None:
                        with torch.amp.autocast('cuda', enabled=False):
                            s = msad_score.float().clamp(1e-6, 1.0 - 1e-6)
                            pool = str(getattr(args, "msad_img_pool", "q95")).lower()
                            if pool == "max":
                                img_score = s.flatten(1).max(dim=1).values
                            elif pool == "mean":
                                img_score = s.flatten(1).mean(dim=1)
                            elif pool == "topk_mean":
                                flat = s.flatten(1)
                                k = max(1, int(round(float(getattr(args, "msad_img_topk_ratio", 0.02)) * flat.shape[1])))
                                img_score = flat.topk(k=k, dim=1).values.mean(dim=1)
                            else:
                                img_score = torch.quantile(s.flatten(1), 0.95, dim=1)

                            targets = torch.tensor(is_anomaly, dtype=torch.float32, device=device)
                            loss_msad_img = F.binary_cross_entropy(img_score, targets)
                            if step % 50 == 0:
                                print(f"[MSAD ImgLoss] pool={pool} loss={float(loss_msad_img):.4f} score_mean={float(img_score.mean()):.4f}")

                # -------------------------
                # MSAD Pixel-wise Margin Constraints (如果启用)
                # defect像素: logit(abn-norm) >= m_defect
                # spurious像素(GT=0且spurious_map top-p): logit(abn-norm) <= -m_spurious
                # -------------------------
                loss_msad_margin = torch.tensor(0.0, device=device)
                lambda_msad_margin = float(getattr(args, "lambda_msad_margin", 0.0) or 0.0)
                if lambda_msad_margin > 0.0:
                    msad_map = out.get("msad_aggregated_map", None)
                    msad_score = out.get("msad_anomaly_score", None)
                    sp_map = out.get("spurious_map", None)
                    if (msad_map is not None or msad_score is not None):
                        gt_mask = masks
                        if gt_mask.dim() == 4:
                            gt_mask = gt_mask.squeeze(1)
                        with torch.amp.autocast('cuda', enabled=False):
                            if msad_map is not None:
                                p_norm = msad_map[:, 0].float()
                                p_abn = msad_map[:, 1].float()
                            else:
                                p_abn = msad_score.float()
                                p_norm = (1.0 - p_abn).float()

                            if p_abn.shape[-2:] != gt_mask.shape[-2:]:
                                gt_mask_f32 = F.interpolate(
                                    gt_mask.unsqueeze(1).float(),
                                    size=p_abn.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False,
                                ).squeeze(1)
                            else:
                                gt_mask_f32 = gt_mask.float()

                            sp_map_f32 = None
                            if isinstance(sp_map, torch.Tensor):
                                if sp_map.shape[-2:] != p_abn.shape[-2:]:
                                    sp_map_f32 = F.interpolate(
                                        sp_map.unsqueeze(1).float(),
                                        size=p_abn.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False,
                                    ).squeeze(1)
                                else:
                                    sp_map_f32 = sp_map.float()

                            eps = 1e-6
                            p_norm = p_norm.clamp(eps, 1.0 - eps)
                            p_abn = p_abn.clamp(eps, 1.0 - eps)
                            logit = torch.log(p_abn) - torch.log(p_norm)

                            defect_mask = gt_mask_f32 > 0.5
                            progress = float(epoch * len(dataloader) + step) / float(max(1, args.epochs * len(dataloader) - 1))
                            warm = float(getattr(args, "r2_warmup_ratio", 0.1))
                            ramp = float(getattr(args, "r2_spurious_ramp_ratio", 0.2))
                            w_spu = 0.0 if progress <= warm else min(1.0, (progress - warm) / max(1e-6, ramp))
                            top_p = float(getattr(args, "spurious_top_p", 0.02) or 0.0)
                            spurious_mask = torch.zeros_like(defect_mask)
                            if top_p > 0.0 and sp_map_f32 is not None:
                                B, H, W = sp_map_f32.shape
                                k = int(max(1, min(H * W, round(top_p * H * W))))
                                flat = sp_map_f32.view(B, -1)
                                topk_vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)
                                thr = topk_vals[:, -1].view(B, 1, 1)
                                spurious_mask = (sp_map_f32 >= thr) & (gt_mask_f32 <= 0.5)
                            if bool(getattr(args, "spurious_margin_require_quality", True)) and top_p > 0.0 and sp_map_f32 is not None:
                                qthr = float(getattr(args, "spurious_quality_threshold", 0.03))
                                peak = float((topk_vals.mean() - flat.mean()).detach().item())
                                if peak < qthr:
                                    spurious_mask = torch.zeros_like(spurious_mask)
                                    w_spu = 0.0
                            if sp_map_f32 is None:
                                w_spu = 0.0

                            m_def = float(getattr(args, "msad_margin_defect", 0.3))
                            m_spu = float(getattr(args, "msad_margin_spurious", 0.3))

                            l_def = torch.relu(m_def - logit)
                            l_spu = torch.relu(m_spu + logit)

                            loss_def = l_def[defect_mask].mean() if defect_mask.any() else torch.tensor(0.0, device=device)
                            loss_spu = l_spu[spurious_mask].mean() if spurious_mask.any() else torch.tensor(0.0, device=device)
                            loss_msad_margin = loss_def + float(w_spu) * loss_spu

                            if step % 50 == 0:
                                print(f"[MSAD Margin] defect={float(loss_def):.4f} spurious={float(loss_spu):.4f} w_spu={float(w_spu):.2f} total={float(loss_msad_margin):.4f}")

                # -------------------------
                # R2-2: MSAD Similarity(Logits) Level Margin Constraints (可选)
                # 直接约束softmax前 similarity logits: z = logits_abn - logits_norm
                # -------------------------
                loss_msad_sim_margin = torch.tensor(0.0, device=device)
                lambda_msad_sim_margin = float(getattr(args, "lambda_msad_sim_margin", 0.0) or 0.0)
                if lambda_msad_sim_margin > 0.0:
                    sim_logits = out.get("msad_aggregated_logits_map", None)
                    if str(getattr(args, "msad_sim_margin_source", "agg")).lower() != "agg":
                        maps = out.get("msad_similarity_logits_maps", None)
                        if isinstance(maps, list) and len(maps) > 0:
                            sim_logits = maps[0]
                    sp_map = out.get("spurious_map", None)
                    if isinstance(sim_logits, torch.Tensor):
                        gt_mask = masks
                        if gt_mask.dim() == 4:
                            gt_mask = gt_mask.squeeze(1)
                        with torch.amp.autocast('cuda', enabled=False):
                            logits_norm = sim_logits[:, 0].float()
                            logits_abn = sim_logits[:, 1].float()
                            if logits_abn.shape[-2:] != gt_mask.shape[-2:]:
                                gt_mask_f32 = F.interpolate(
                                    gt_mask.unsqueeze(1).float(),
                                    size=logits_abn.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False,
                                ).squeeze(1)
                            else:
                                gt_mask_f32 = gt_mask.float()

                            sp_map_f32 = None
                            if isinstance(sp_map, torch.Tensor):
                                if sp_map.shape[-2:] != logits_abn.shape[-2:]:
                                    sp_map_f32 = F.interpolate(
                                        sp_map.unsqueeze(1).float(),
                                        size=logits_abn.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False,
                                    ).squeeze(1)
                                else:
                                    sp_map_f32 = sp_map.float()

                            z = logits_abn - logits_norm
                            defect_mask = gt_mask_f32 > 0.5

                            progress = float(epoch * len(dataloader) + step) / float(max(1, args.epochs * len(dataloader) - 1))
                            warm = float(getattr(args, "r2_warmup_ratio", 0.1))
                            ramp = float(getattr(args, "r2_spurious_ramp_ratio", 0.2))
                            w_spu = 0.0 if progress <= warm else min(1.0, (progress - warm) / max(1e-6, ramp))

                            top_p = float(getattr(args, "spurious_top_p", 0.02) or 0.0)
                            spurious_mask = torch.zeros_like(defect_mask)
                            if top_p > 0.0 and sp_map_f32 is not None:
                                B, H, W = sp_map_f32.shape
                                k = int(max(1, min(H * W, round(top_p * H * W))))
                                flat = sp_map_f32.view(B, -1)
                                topk_vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)
                                thr = topk_vals[:, -1].view(B, 1, 1)
                                spurious_mask = (sp_map_f32 >= thr) & (gt_mask_f32 <= 0.5)

                            if bool(getattr(args, "spurious_margin_require_quality", True)) and sp_map_f32 is not None:
                                qthr = float(getattr(args, "spurious_quality_threshold", 0.03))
                                peak = float((topk_vals.mean() - flat.mean()).detach().item()) if top_p > 0.0 else 0.0
                                if peak < qthr:
                                    spurious_mask = torch.zeros_like(spurious_mask)
                                    w_spu = 0.0
                            if sp_map_f32 is None:
                                w_spu = 0.0

                            m_def = float(getattr(args, "msad_sim_margin_defect", 0.3))
                            m_spu = float(getattr(args, "msad_sim_margin_spurious", 0.3))
                            l_def = torch.relu(m_def - z)
                            l_spu = torch.relu(m_spu + z)
                            loss_def = l_def[defect_mask].mean() if defect_mask.any() else torch.tensor(0.0, device=device)
                            loss_spu = l_spu[spurious_mask].mean() if spurious_mask.any() else torch.tensor(0.0, device=device)
                            loss_msad_sim_margin = loss_def + float(w_spu) * loss_spu

                            if step % 50 == 0:
                                print(f"[MSAD SimMargin] defect={float(loss_def):.4f} spurious={float(loss_spu):.4f} w_spu={float(w_spu):.2f} total={float(loss_msad_sim_margin):.4f}")

                # loss_iou = torch.tensor(0.0, device=device)
                # -------------------------
                # ===== 【修复】图像级别的 Presence Loss =====
                if pred_logits is not None:
                    # pred_logits: (B, Q, 1) 或 (B, Q)
                    presence_logit = pred_logits
                    if presence_logit.dim() == 3 and presence_logit.shape[-1] == 1:
                        presence_logit = presence_logit.squeeze(-1)  # (B, Q)
                    
                    # 【修复】聚合到图像级别
                    image_presence_logit = presence_logit.max(dim=1)[0]  # (B,)
                    
                    # 图像级标签
                    presence_targets_image = torch.tensor(is_anomaly, dtype=torch.float32, device=device)
                    
                    # Focal Loss (更好处理类别不平衡)
                    p = torch.sigmoid(image_presence_logit)
                    ce_loss = F.binary_cross_entropy_with_logits(
                        image_presence_logit, presence_targets_image, reduction='none'
                    )
                    p_t = p * presence_targets_image + (1 - p) * (1 - presence_targets_image)
                    focal_weight = (1 - p_t) ** 2.0  # gamma=2.0
                    alpha_t = 0.5 * presence_targets_image + 0.5 * (1 - presence_targets_image)
                    
                    loss_presence = (alpha_t * focal_weight * ce_loss).mean()
                    
                    # 计算准确率用于logging
                    with torch.no_grad():
                        presence_preds = (p > 0.5).float()
                        presence_acc = (presence_preds == presence_targets_image).float().mean()
                        if step % 100 == 0:
                            print(f"[Presence] Image-level: loss={loss_presence.item():.4f}, "
                                  f"acc={presence_acc.item():.2%}, "
                                  f"n_anomaly={sum(is_anomaly)}/{len(is_anomaly)}")
                else:
                    loss_presence = torch.tensor(0.0, device=device)

                # -------------------------
                # 方案C: 置信度融合头损失
                # 使用与presence相同的GT监督fused_conf
                # -------------------------
                loss_conf_fusion = torch.tensor(0.0, device=device)
                lambda_conf_fusion = getattr(args, 'lambda_conf_fusion', 0.0)
                
                if lambda_conf_fusion > 0.0:
                    fused_conf = out.get("fused_conf", None)
                    if fused_conf is not None and pred_logits is not None:
                        # fused_conf是融合后的置信度 (B, Q)
                        # 使用与presence相同的target
                        if presence_targets is not None:
                            loss_conf_fusion = F.binary_cross_entropy_with_logits(
                                fused_conf, presence_targets
                            )
                        else:
                            # 构建target
                            conf_targets = torch.zeros_like(fused_conf, dtype=torch.float32, device=device)
                            indices_per_image = convert_matcher_output_to_indices(batch_idx, src_idx, tgt_idx, B=images.shape[0], device=device)
                            for b in range(images.shape[0]):
                                src_q, _ = indices_per_image[b]
                                if src_q.numel() > 0:
                                    src_q = src_q.to(device).long()
                                    valid_mask = (src_q >= 0) & (src_q < fused_conf.shape[1])
                                    src_q = src_q[valid_mask]
                                    if src_q.numel() > 0:
                                        conf_targets[b, src_q] = 1.0
                            loss_conf_fusion = F.binary_cross_entropy_with_logits(fused_conf, conf_targets)
                        
                        if step % 50 == 0:
                            print(f"[ConfFusion Loss] loss={float(loss_conf_fusion):.4f}")

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
                    # 【修复】优先使用结构化原型 (compound prompt 输出)
                    text_features_structured = out.get("text_features_structured", None)
                    
                    if text_features_structured is not None:
                        # 【修复】处理字典格式
                        if isinstance(text_features_structured, dict):
                            prompt_proto_normal = text_features_structured['normal']       # (B, D)
                            prompt_proto_abnormal_all = text_features_structured['abnormal_all']  # (B, K, D)
                            prompt_proto_abnormal = text_features_structured['abnormal_mean']     # (B, D)
                            num_abnormal = text_features_structured['num_abnormal']
                            
                            # 【v2.1 新增】获取 w 的原型
                            proto_suspicious = text_features_structured.get('proto_suspicious', None)  # (B, D)
                            
                            prompt_proto = prompt_proto_abnormal  # 保持 align_loss 兼容
                            
                            if step % 500 == 0:
                                print(f"[ALIGN] Using structured prototypes: "
                                      f"normal={prompt_proto_normal.shape}, "
                                      f"abnormal_all={prompt_proto_abnormal_all.shape}, "
                                      f"K={num_abnormal}, "
                                      f"proto_suspicious={'Yes' if proto_suspicious is not None else 'No'}")
                        else:
                            # 兼容旧格式 (B, 2, D)
                            prompt_proto_normal = text_features_structured[:, 0, :]
                            prompt_proto_abnormal = text_features_structured[:, 1, :]
                            prompt_proto_abnormal_all = None
                            prompt_proto = prompt_proto_abnormal
                            proto_suspicious = None  # 【v2.1】旧格式不支持
                    else:
                        # 回退到原来的方式
                        prompt_seq = out.get("prompt_seq", None)
                        if prompt_seq is None:
                            try:
                                prompt_seq, _ = model_core.prompt_learner(prompt_lists, device=device)
                            except Exception as e:
                                print("[WARN] cannot obtain prompt_seq from model_core.prompt_learner:", e)
                                prompt_seq = None

                        # 使用SOWA风格的pooling获取prompt_proto
                        prompt_proto = pool_prompt_features_sowa_style(
                            prompt_seq=prompt_seq,
                            prompt_lists=prompt_lists,
                            is_anomaly=is_anomaly,
                            device=device
                        )
                        prompt_proto_normal = None
                        prompt_proto_abnormal = prompt_proto
                        proto_suspicious = None  # 【v2.1】回退模式不支持

                    # ===== Step 2: 获取 visual embedding（区分异常/正常）=====
                    # 优先级：FiLo patch_tokens > decoder_features
                    
                    # 尝试使用FiLo的patch_tokens（最优先）
                    filo_qkv = out.get("filo_patch_tokens_qkv", [])
                    filo_vv = out.get("filo_patch_tokens_vv", [])
                    use_multilevel_align = bool(getattr(args, "align_multilevel", False))
                    max_levels = int(getattr(args, "align_multilevel_max_levels", 0) or 0)
                    visual_embed_levels = []
                    is_background = None
                    
                    if len(filo_qkv) > 0 or len(filo_vv) > 0:
                        # 合并所有FiLo patch tokens
                        all_filo_tokens = filo_qkv + filo_vv
                        if max_levels > 0:
                            all_filo_tokens = all_filo_tokens[:max_levels]

                        if use_multilevel_align and len(all_filo_tokens) > 1:
                            for tok in all_filo_tokens:
                                ve, ib = compute_filo_aligned_features(
                                    patch_tokens=tok,
                                    masks=masks,
                                    is_anomaly=is_anomaly,
                                    device=device
                                )
                                if ve is not None:
                                    visual_embed_levels.append(ve)
                                    if is_background is None:
                                        is_background = ib
                        else:
                            filo_feat = all_filo_tokens[0]
                            visual_embed, is_background = compute_filo_aligned_features(
                                patch_tokens=filo_feat,
                                masks=masks,
                                is_anomaly=is_anomaly,
                                device=device
                            )
                            if visual_embed is not None:
                                visual_embed_levels = [visual_embed]
                        
                        if step % 100 == 0 and len(visual_embed_levels) > 0:
                            print(f"[ALIGN] Using FiLo patch_tokens: n_levels={len(visual_embed_levels)} "
                                  f"shape0={tuple((filo_qkv + filo_vv)[0].shape)} "
                                  f"embed0={tuple(visual_embed_levels[0].shape)}")
                    
                    else:
                        # 回退到 decoder 特征
                        ms = out.get("multiscale_features", None)
                        used_feats = None
                        if isinstance(ms, dict):
                            used_feats = ms.get("used_features", None)
                        if use_multilevel_align and isinstance(used_feats, list) and len(used_feats) > 0:
                            feats_lv = used_feats
                            if max_levels > 0:
                                feats_lv = feats_lv[:max_levels]
                            for f in feats_lv:
                                ve, ib = compute_visual_embedding_with_background(
                                    decoder_features=f,
                                    masks=masks,
                                    is_anomaly=is_anomaly,
                                    device=device
                                )
                                if ve is not None:
                                    visual_embed_levels.append(ve)
                                    if is_background is None:
                                        is_background = ib
                        else:
                            decoder_feat = out.get("decoder_features", None)
                            if decoder_feat is None:
                                decoder_feat = out.get("decoder_hs", None)
                                if decoder_feat is not None and decoder_feat.dim() == 4:
                                    decoder_feat = decoder_feat[-1]
                                if decoder_feat is not None and decoder_feat.dim() == 3:
                                    if decoder_feat.shape[0] == Q and decoder_feat.shape[1] == B:
                                        decoder_feat = decoder_feat.permute(1, 0, 2).contiguous()
                            
                            visual_embed, is_background = compute_visual_embedding_with_background(
                                decoder_features=decoder_feat,
                                masks=masks,
                                is_anomaly=is_anomaly,
                                device=device
                            )
                            if visual_embed is not None:
                                visual_embed_levels = [visual_embed]
                    
                    if len(visual_embed_levels) == 0:
                        visual_embed_levels = [torch.zeros((B, prompt_proto.shape[1] if prompt_proto is not None else 128), device=device)]
                        is_background = torch.zeros(B, dtype=torch.bool, device=device)
                    visual_embed = visual_embed_levels[0]

                    # ===== Step 3: 维度对齐 =====
                    if prompt_proto is not None:
                        Dp = prompt_proto.shape[1]
                        aligned_levels = []
                        for ve in visual_embed_levels:
                            Dm = ve.shape[1]
                            if Dm != Dp:
                                if not hasattr(model_core, "_align_proj"):
                                    model_core._align_proj = nn.Linear(Dm, Dp).to(device)
                                    optimizer.add_param_group({
                                        "params": model_core._align_proj.parameters(),
                                        "lr": args.lr_main,
                                        "weight_decay": 0.0
                                    })
                                ve = model_core._align_proj(ve)
                            aligned_levels.append(ve)
                        visual_embed_levels = aligned_levels
                        visual_embed = visual_embed_levels[0]

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
                        
                        # ===== 【v2.1 修复】使用二分类形式的 align loss =====
                        # 确保有 normal 和 abnormal 两个原型
                        if prompt_proto_normal is not None and prompt_proto_abnormal is not None:
                            per_level_losses = [
                                align_loss_binary_classification(
                                    proto_normal=prompt_proto_normal,
                                    proto_abnormal=prompt_proto_abnormal,
                                    visual_embed=ve,
                                    is_anomaly=is_anomaly,
                                    temp=args.align_temp,
                                    label_smoothing=0.1,
                                )
                                for ve in (visual_embed_levels if use_multilevel_align else [visual_embed])
                            ]
                            if len(per_level_losses) > 1:
                                w = None
                                if str(getattr(args, "align_multilevel_weight_source", "uniform")).lower() == "msad":
                                    msad_mod = getattr(model_core, "msad", None)
                                    lw = getattr(getattr(msad_mod, "aggregator", None), "level_weights", None) if msad_mod is not None else None
                                    if isinstance(lw, torch.Tensor) and lw.numel() == len(per_level_losses):
                                        w = torch.softmax(lw.detach().float(), dim=0)
                                if w is None:
                                    w = torch.ones(len(per_level_losses), device=device, dtype=torch.float32) / float(len(per_level_losses))
                                align_loss = sum(per_level_losses[i] * w[i] for i in range(len(per_level_losses)))
                            else:
                                align_loss = per_level_losses[0]
                        else:
                            # 回退到原来的方法（不应该发生）
                            align_loss = align_loss_with_background_margin(
                                prompt_proto=prompt_proto,
                                visual_embed=visual_embed,
                                is_background=is_background,
                                group_labels=group_labels,
                                temp=args.align_temp,
                                margin=align_margin
                            )
                        
                        if step % 100 == 0:
                            print(f"[ALIGN-v2.1] binary_classification loss={align_loss.item():.4f}")
                        
                        # ===== Step 5: Query-level alignment（v2改进版）=====
                        # current_lambda_query_align 已在 batch 循环开始时初始化（两阶段调度或固定值）
                        use_query_align_v2 = getattr(args, 'use_query_align_v2', True)
                        
                        if current_lambda_query_align > 0:
                            decoder_hs = out.get("decoder_hs", None)
                            if decoder_hs is not None:
                                # 确保 decoder_hs 是 (B, Q, D) 格式
                                if decoder_hs.dim() == 4:
                                    decoder_hs = decoder_hs[-1]
                                if decoder_hs.dim() == 3:
                                    if decoder_hs.shape[0] == Q and decoder_hs.shape[1] == B:
                                        decoder_hs = decoder_hs.permute(1, 0, 2).contiguous()
                                
                                if use_query_align_v2:
                                    # ===== 【修复】使用二分类版本（如果有结构化原型）=====
                                    
                                    # 获取pred_masks
                                    pred_masks_for_qa = out.get("pred_masks", None)
                                    if pred_masks_for_qa is not None and pred_masks_for_qa.dim() == 5:
                                        pred_masks_for_qa = pred_masks_for_qa[-1]  # 取最后一层
                                    
                                    # 构建gt_masks (B, H, W) - 统一使用pred_masks的尺寸
                                    target_size = pred_masks_for_qa.shape[-2:] if pred_masks_for_qa is not None else (72, 72)
                                    gt_masks_list = []
                                    for t in list_targets:
                                        if t["segments"].numel() > 0:
                                            seg = t["segments"].sum(dim=0).clamp(0, 1)  # (H, W)
                                            # 如果尺寸不匹配，进行resize
                                            if seg.shape[-2:] != target_size:
                                                seg = F.interpolate(
                                                    seg.unsqueeze(0).unsqueeze(0).float(),
                                                    size=target_size, mode='nearest'
                                                ).squeeze(0).squeeze(0)
                                            gt_masks_list.append(seg)
                                        else:
                                            gt_masks_list.append(torch.zeros(target_size, device=device))
                                    gt_masks_for_qa = torch.stack(gt_masks_list, dim=0)  # (B, H, W)
                                    
                                    # 【修复】根据是否有结构化原型选择不同的 loss 函数
                                    if text_features_structured is not None and prompt_proto_normal is not None:
                                        # 【完整版】使用 K 个独立 abnormal prototypes + 梯度路由
                                        # 优先使用 abnormal_all（K 个独立的）
                                        proto_abnormal_for_qa = prompt_proto_abnormal_all if prompt_proto_abnormal_all is not None else prompt_proto_abnormal
                                        
                                        query_align_loss = query_text_alignment_loss_with_gradient_gating(
                                            decoder_hs=decoder_hs,
                                            proto_normal=prompt_proto_normal,
                                            proto_abnormal_all=proto_abnormal_for_qa,
                                            indices=indices,
                                            gt_masks=gt_masks_for_qa,
                                            pred_masks=pred_masks_for_qa,
                                            is_anomaly=is_anomaly,
                                            temp=getattr(args, 'query_align_temp', 0.2),
                                            iou_threshold=getattr(args, 'query_align_iou_threshold', 0.1),
                                            top_k_normal=getattr(args, 'query_align_top_k_normal', 5),
                                            aggregation="max",          # 使用 max 聚合
                                            use_gradient_gating=True,   # 启用梯度路由
                                        )
                                    else:
                                        # 回退到v2版本
                                        total_steps = len(dataloader) * args.epochs
                                        global_step_for_qa = epoch * len(dataloader) + step
                                        current_step_ratio = global_step_for_qa / total_steps
                                        
                                        query_align_loss = query_text_alignment_loss_v2(
                                            decoder_hs=decoder_hs,
                                            prompt_proto=prompt_proto,
                                            prompt_proto_normal=None,
                                            indices=indices,
                                            pred_masks=pred_masks_for_qa,
                                            gt_masks=gt_masks_for_qa,
                                            is_anomaly=is_anomaly,
                                            temp=getattr(args, 'query_align_temp', 0.2),
                                            top_k=getattr(args, 'query_align_top_k', 64),
                                            use_soft_target=getattr(args, 'query_align_soft_target', True),
                                            include_normal=getattr(args, 'query_align_include_normal', True),
                                            normal_margin=getattr(args, 'query_align_normal_margin', 0.3),
                                            use_full_softmax_ratio=getattr(args, 'query_align_full_softmax_ratio', 0.2),
                                            current_step_ratio=current_step_ratio,
                                        )
                                else:
                                    # 使用原始v1版本
                                    query_align_loss = query_text_alignment_loss(
                                        decoder_hs=decoder_hs,
                                        prompt_proto=prompt_proto,
                                        indices=indices,
                                        temp=getattr(args, 'query_align_temp', 0.2),
                                        top_k=getattr(args, 'query_align_top_k', 64)
                                    )

                                # ===== 【v2.1 新增】收集异常特征到 Memory Bank =====
                                indices_for_bank = out.get("indices", None)
                                if indices_for_bank is not None and decoder_hs is not None:
                                    # 计算当前进度
                                    total_steps = len(dataloader) * args.epochs
                                    current_step = epoch * len(dataloader) + step
                                    current_step_ratio = current_step / total_steps

                                    collect_anomaly_features_to_bank_v2(
                                        decoder_hs=decoder_hs,
                                        matched_indices=indices_for_bank,
                                        is_anomaly=is_anomaly,
                                        anomaly_bank=anomaly_bank,
                                        proto_normal=prompt_proto_normal,
                                        current_step_ratio=current_step_ratio,
                                    )                                  
                        else:
                             query_align_loss = torch.tensor(0.0, device=device)

                        # 【新增】w 向量的专属 loss（只在 normal 图上更新）
                        loss_suspicious = torch.tensor(0.0, device=device)
                        lambda_suspicious = getattr(args, 'lambda_suspicious', 0.1)

                        # ===== 【v2.1 修复】使用 Memory Bank 的 suspicious loss =====
                        if lambda_suspicious > 0 and proto_suspicious is not None:
                            indices_for_suspicious = out.get("indices", None)
                            
                            # 计算当前进度（如果上面没有计算）
                            total_steps = len(dataloader) * args.epochs
                            current_step = epoch * len(dataloader) + step
                            current_step_ratio = current_step / total_steps
                            
                            # 检查是否禁用 w 学习
                            if not getattr(args, 'disable_w_learning', False):
                                loss_suspicious = compute_suspicious_loss_hybrid_v2(
                                    decoder_hs=decoder_hs,
                                    proto_suspicious=proto_suspicious,
                                    proto_abnormal=prompt_proto_abnormal,
                                    anomaly_bank=None if getattr(args, 'disable_bank', False) else anomaly_bank,
                                    matched_indices=indices_for_suspicious,
                                    is_anomaly=is_anomaly,
                                    current_step_ratio=current_step_ratio,
                                    temp=getattr(args, 'query_align_temp', 0.2),
                                    top_r=5,
                                    w_abnormal_margin=getattr(args, 'w_abnormal_margin', 0.3),
                                )
                                
                                if step % 100 == 0:
                                    bank_stats = anomaly_bank.get_stats()
                                    print(f"[SUSPICIOUS-v2.2] loss={loss_suspicious.item():.4f}, "
                                          f"bank_ready={bank_stats['is_ready']}, "
                                          f"bank_fill={bank_stats['fill_ratio']:.1%}, "
                                          f"warm_up_passed={current_step_ratio >= bank_stats['warm_up_ratio']}")
                            else:
                                loss_suspicious = torch.tensor(0.0, device=device)
                            
                            if step % 100 == 0:
                                bank_stats = anomaly_bank.get_stats()
                                print(f"[SUSPICIOUS-v2.1] loss={loss_suspicious.item():.4f}, "
                                      f"bank_ready={bank_stats['is_ready']}, "
                                      f"bank_fill={bank_stats['fill_ratio']:.1%}")
                        else:
                            loss_suspicious = torch.tensor(0.0, device=device)
                        
                            
                        # ===== Diagnostics (每 log_freq 步打印一次) =====
                        if (step % getattr(args, "log_freq", 100)) == 0:
                            p_norm = F.normalize(prompt_proto, dim=1)
                            m_norm = F.normalize(visual_embed, dim=1)
                            sim = (p_norm @ m_norm.t()) / float(args.align_temp)
                            
                            # 计算同组/异组的相似度统计
                            same_group = (group_labels.unsqueeze(0) == group_labels.unsqueeze(1))
                            pos_sim = sim[same_group].mean().item() if same_group.sum() > 0 else 0
                            neg_sim = sim[~same_group].mean().item() if (~same_group).sum() > 0 else 0
                            
                            # 计算当前是否使用 full softmax
                            total_steps = len(dataloader) * args.epochs
                            global_step_diag = epoch * len(dataloader) + step
                            current_ratio_diag = global_step_diag / total_steps
                            use_full_sm = current_ratio_diag < getattr(args, 'query_align_full_softmax_ratio', 0.2)
                            
                            print(f"[ALIGN] step={step} align_loss={align_loss.item():.6f} "
                                  f"query_align={query_align_loss.item():.6f} "
                                  f"pos_sim={pos_sim:.4f} neg_sim={neg_sim:.4f}")
                            if prompt_proto_normal is not None and prompt_proto_abnormal is not None:
                                v0 = F.normalize(visual_embed, dim=1)
                                pn0 = F.normalize(prompt_proto_normal, dim=1)
                                pa0 = F.normalize(prompt_proto_abnormal, dim=1)
                                sim_vn = (v0 * pn0).sum(dim=1) / float(args.align_temp)
                                sim_va = (v0 * pa0).sum(dim=1) / float(args.align_temp)
                                anom_mask = torch.as_tensor(is_anomaly, device=device, dtype=torch.bool)
                                norm_mask = ~anom_mask
                                normal_margin = (sim_vn[norm_mask] - sim_va[norm_mask]).mean().item() if norm_mask.any() else 0.0
                                anomaly_margin = (sim_va[anom_mask] - sim_vn[anom_mask]).mean().item() if anom_mask.any() else 0.0
                                proto_gap = (pn0 * pa0).sum(dim=1).mean().item()
                                print(f"[ALIGN-MARGIN] proto_gap={proto_gap:.4f} normal_margin={normal_margin:.4f} anomaly_margin={anomaly_margin:.4f}")
                                if bool(getattr(args, "align_multilevel", False)) and "visual_embed_levels" in locals() and len(visual_embed_levels) > 1:
                                    nm_list = []
                                    am_list = []
                                    for ve in visual_embed_levels:
                                        v = F.normalize(ve, dim=1)
                                        sim_vn_l = (v * pn0).sum(dim=1) / float(args.align_temp)
                                        sim_va_l = (v * pa0).sum(dim=1) / float(args.align_temp)
                                        nm = (sim_vn_l[norm_mask] - sim_va_l[norm_mask]).mean().item() if norm_mask.any() else 0.0
                                        am = (sim_va_l[anom_mask] - sim_vn_l[anom_mask]).mean().item() if anom_mask.any() else 0.0
                                        nm_list.append(round(float(nm), 4))
                                        am_list.append(round(float(am), 4))
                                    print(f"[ALIGN-LEVEL-MARGIN] normal_margin={nm_list} anomaly_margin={am_list}")
                            eta_dbg = out.get("eta_spurious", None)
                            if isinstance(eta_dbg, torch.Tensor):
                                print(f"[SPURIOUS] eta mean={eta_dbg.mean().detach().item():.4f} std={eta_dbg.std().detach().item():.4f} "
                                      f"min={eta_dbg.min().detach().item():.4f} max={eta_dbg.max().detach().item():.4f}")
                            dap_patches = out.get("compound_dap_weights", None)
                            if isinstance(dap_patches, torch.Tensor) and dap_patches.dim() == 3:
                                pmn = getattr(model_core, "prompt_learner", None)
                                pnet = getattr(pmn, "patch_meta_net", None) if pmn is not None else None
                                if pnet is not None:
                                    selected_flat = dap_patches.reshape(dap_patches.shape[0], -1)
                                    bias = pnet(selected_flat)
                                    bnorm = bias.norm(dim=1)
                                    print(f"[DAP] bias_norm mean={bnorm.mean().detach().item():.4f} std={bnorm.std().detach().item():.4f}")
                            msad_mod = getattr(model_core, "msad", None)
                            if msad_mod is not None:
                                temp_p = getattr(getattr(msad_mod, "scorer", None), "temperature", None)
                                lw_p = getattr(getattr(msad_mod, "aggregator", None), "level_weights", None)
                                if isinstance(temp_p, torch.Tensor):
                                    print(f"[MSAD] temperature={temp_p.detach().item():.4f}")
                                if isinstance(lw_p, torch.Tensor) and lw_p.numel() > 0:
                                    w = torch.softmax(lw_p.detach().float(), dim=0)
                                    print(f"[MSAD] level_weights_softmax={w.cpu().numpy().round(4).tolist()}")
                            print(f"[QA-DBG] step_ratio={current_ratio_diag:.3f} use_full_softmax={use_full_sm} "
                                  f"n_anomaly={sum(is_anomaly)} n_normal={len(is_anomaly)-sum(is_anomaly)}")
                            
                            # 打印 background vs anomaly embedding 统计
                            n_bg = is_background.sum().item()
                            n_anom = (~is_background).sum().item()
                            print(f"[BG/ANOM] n_background={n_bg}, n_anomaly={n_anom}")

                        # ===== TSNE: 每个epoch结束时保存可视化 (独立于 log_freq) =====
                        is_last_step = (step == len(dataloader) - 1)
                        if is_last_step and B >= 4 and is_main_process:
                            try:
                                # 重新计算 TSNE 所需的变量（确保变量存在）
                                p_norm_tsne = F.normalize(prompt_proto, dim=1)
                                m_norm_tsne = F.normalize(visual_embed, dim=1)
                                
                                ns = min(getattr(args, "tsne_samples", 64), B)
                                sel = np.random.choice(B, ns, replace=False)
                                p_sample = p_norm_tsne[sel].detach().cpu().numpy()
                                m_sample = m_norm_tsne[sel].detach().cpu().numpy()
                                labels_sample = group_labels[sel].cpu().numpy()
                                is_bg_sample = is_background[sel].cpu().numpy()  # True=normal, False=anomaly
                                
                                # 获取采样样本的 class_names
                                class_names_sample = [class_names[i] for i in sel]
                                is_anomaly_sample = [is_anomaly[i] for i in sel]  # True=anomaly, False=normal
                                
                                # 为每个类别分配颜色
                                unique_classes = sorted(list(set(class_names_sample)))
                                # 使用更多颜色的colormap
                                n_classes = len(unique_classes)
                                if n_classes <= 10:
                                    cmap = plt.cm.tab10
                                elif n_classes <= 20:
                                    cmap = plt.cm.tab20
                                else:
                                    cmap = plt.cm.turbo
                                class_to_color = {cls: cmap(i / max(n_classes - 1, 1)) for i, cls in enumerate(unique_classes)}
                                colors_sample = [class_to_color[cls] for cls in class_names_sample]
                                
                                extra_vecs = []
                                extra_names = []
                                if prompt_proto_normal is not None and prompt_proto_abnormal is not None:
                                    pn_mean = F.normalize(prompt_proto_normal, dim=1)[sel].mean(dim=0)
                                    pa_mean = F.normalize(prompt_proto_abnormal, dim=1)[sel].mean(dim=0)
                                    extra_vecs.extend([pn_mean.detach().cpu().numpy(), pa_mean.detach().cpu().numpy()])
                                    extra_names.extend(["proto_normal", "proto_abnormal"])
                                if proto_suspicious is not None:
                                    ps_mean = F.normalize(proto_suspicious, dim=1)[sel].mean(dim=0)
                                    extra_vecs.append(ps_mean.detach().cpu().numpy())
                                    extra_names.append("proto_suspicious")
                                
                                if extra_vecs:
                                    X = np.concatenate([p_sample, m_sample, np.stack(extra_vecs, axis=0)], axis=0)
                                else:
                                    X = np.concatenate([p_sample, m_sample], axis=0)
                                Z = TSNE(n_components=2, perplexity=min(30, ns-1), init='pca', random_state=42).fit_transform(X)
                                
                                # 创建多面板图
                                fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                                
                                # ===== Panel 1: Visual Embeddings (按类别着色，按Normal/Anomaly区分形状) =====
                                ax1 = axes[0, 0]
                                # 只绘制 visual embeddings (Z[ns:])
                                for i in range(ns):
                                    marker = 'o' if is_bg_sample[i] else '*'  # Normal=圆, Anomaly=星
                                    size = 80 if is_bg_sample[i] else 150  # 星型稍大一点
                                    ax1.scatter(Z[ns + i, 0], Z[ns + i, 1], 
                                               c=[colors_sample[i]], marker=marker, s=size, 
                                               edgecolors='black', linewidths=0.5, alpha=0.8)
                                
                                # 创建图例
                                from matplotlib.lines import Line2D
                                # 类别图例
                                class_legend = [Line2D([0], [0], marker='s', color='w', markerfacecolor=class_to_color[cls], 
                                                       markersize=10, label=cls) for cls in unique_classes]
                                # 形状图例
                                shape_legend = [
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                           markersize=10, label='Normal', markeredgecolor='black'),
                                    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                                           markersize=14, label='Anomaly', markeredgecolor='black')
                                ]
                                legend1 = ax1.legend(handles=class_legend, loc='upper left', fontsize=8, title='Class')
                                ax1.add_artist(legend1)
                                ax1.legend(handles=shape_legend, loc='upper right', fontsize=9, title='Type')
                                ax1.set_title(f"Visual Embeddings (color=class, shape=type)", fontsize=12)
                                ax1.set_xlabel("Dim 1")
                                ax1.set_ylabel("Dim 2")
                                
                                # ===== Panel 2: Prompt vs Visual (按类别着色) =====
                                ax2 = axes[0, 1]
                                # 绘制 prompt embeddings (Z[:ns]) 用 x 标记
                                for i in range(ns):
                                    ax2.scatter(Z[i, 0], Z[i, 1], c=[colors_sample[i]], marker='x', s=60, alpha=0.8, linewidths=2)
                                # 绘制 visual embeddings (Z[ns:]) 用圆/星标记
                                for i in range(ns):
                                    marker = 'o' if is_bg_sample[i] else '*'
                                    size = 60 if is_bg_sample[i] else 120
                                    ax2.scatter(Z[ns + i, 0], Z[ns + i, 1], 
                                               c=[colors_sample[i]], marker=marker, s=size, 
                                               edgecolors='black', linewidths=0.5, alpha=0.8)
                                
                                if extra_vecs:
                                    base = 2 * ns
                                    for j, name in enumerate(extra_names):
                                        if name == "proto_normal":
                                            c = "cyan"
                                        elif name == "proto_abnormal":
                                            c = "magenta"
                                        else:
                                            c = "yellow"
                                        ax2.scatter(Z[base + j, 0], Z[base + j, 1],
                                                   c=c, marker='D', s=180, alpha=0.9,
                                                   edgecolors='black', linewidths=0.8)
                                
                                # 图例
                                pv_legend = [
                                    Line2D([0], [0], marker='x', color='gray', markersize=10, label='Prompt', linestyle='None'),
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
                                           label='Visual (Normal)', markeredgecolor='black'),
                                    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=14, 
                                           label='Visual (Anomaly)', markeredgecolor='black')
                                ]
                                if extra_vecs:
                                    pv_legend.extend([
                                        Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan', markersize=10,
                                               label='Proto (Normal)', markeredgecolor='black'),
                                        Line2D([0], [0], marker='D', color='w', markerfacecolor='magenta', markersize=10,
                                               label='Proto (Abnormal)', markeredgecolor='black'),
                                    ])
                                    if "proto_suspicious" in extra_names:
                                        pv_legend.append(
                                            Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow', markersize=10,
                                                   label='Proto (Suspicious)', markeredgecolor='black')
                                        )
                                ax2.legend(handles=pv_legend, loc='upper right', fontsize=9)
                                ax2.set_title(f"Prompt-Visual Alignment (color=class)", fontsize=12)
                                ax2.set_xlabel("Dim 1")
                                ax2.set_ylabel("Dim 2")
                                
                                # ===== Panel 3: 相似度分布 =====
                                ax3 = axes[1, 0]
                                # 计算 prompt-visual 相似度
                                p_norm_sel = p_norm_tsne[sel]
                                m_norm_sel = m_norm_tsne[sel]
                                sim_matrix = (p_norm_sel @ m_norm_sel.t()).detach().cpu().numpy()
                                diag_sim = np.diag(sim_matrix)  # 同一样本的相似度
                                
                                normal_idx = np.where(is_bg_sample)[0]
                                anomaly_idx = np.where(~is_bg_sample)[0]
                                
                                if len(normal_idx) > 0:
                                    ax3.hist(diag_sim[normal_idx], bins=20, alpha=0.6, label=f'Normal (n={len(normal_idx)})', color='green')
                                if len(anomaly_idx) > 0:
                                    ax3.hist(diag_sim[anomaly_idx], bins=20, alpha=0.6, label=f'Anomaly (n={len(anomaly_idx)})', color='red')
                                ax3.axvline(x=diag_sim.mean(), color='black', linestyle='--', label=f'Mean={diag_sim.mean():.3f}')
                                ax3.legend(fontsize=9)
                                ax3.set_title("Prompt-Visual Similarity Distribution", fontsize=12)
                                ax3.set_xlabel("Cosine Similarity")
                                ax3.set_ylabel("Count")
                                
                                # ===== Panel 4: 统计信息 =====
                                ax4 = axes[1, 1]
                                # 计算统计量
                                sim_for_stats = (p_norm_sel @ m_norm_sel.t()) / float(args.align_temp)
                                group_sel = group_labels[sel]
                                same_grp = (group_sel.unsqueeze(0) == group_sel.unsqueeze(1))
                                pos_sim_val = sim_for_stats[same_grp].mean().item() if same_grp.sum() > 0 else 0
                                neg_sim_val = sim_for_stats[~same_grp].mean().item() if (~same_grp).sum() > 0 else 0
                                n_bg_val = is_background.sum().item()
                                n_anom_val = (~is_background).sum().item()
                                proto_gap_val = 0.0
                                normal_margin_val = 0.0
                                anomaly_margin_val = 0.0
                                if prompt_proto_normal is not None and prompt_proto_abnormal is not None:
                                    v0 = F.normalize(visual_embed, dim=1)[sel]
                                    pn0 = F.normalize(prompt_proto_normal, dim=1)[sel]
                                    pa0 = F.normalize(prompt_proto_abnormal, dim=1)[sel]
                                    sim_vn = (v0 * pn0).sum(dim=1) / float(args.align_temp)
                                    sim_va = (v0 * pa0).sum(dim=1) / float(args.align_temp)
                                    anom_mask_s = torch.as_tensor([is_anomaly[i] for i in sel], device=device, dtype=torch.bool)
                                    norm_mask_s = ~anom_mask_s
                                    normal_margin_val = (sim_vn[norm_mask_s] - sim_va[norm_mask_s]).mean().item() if norm_mask_s.any() else 0.0
                                    anomaly_margin_val = (sim_va[anom_mask_s] - sim_vn[anom_mask_s]).mean().item() if anom_mask_s.any() else 0.0
                                    proto_gap_val = (pn0 * pa0).sum(dim=1).mean().item()
                                
                                # 计算类别分布
                                class_dist = {}
                                for cls, anom in zip(class_names_sample, is_anomaly_sample):
                                    if cls not in class_dist:
                                        class_dist[cls] = {'normal': 0, 'anomaly': 0}
                                    if anom:
                                        class_dist[cls]['anomaly'] += 1
                                    else:
                                        class_dist[cls]['normal'] += 1
                                
                                class_dist_str = "\n".join([f"    {cls}: N={v['normal']}, A={v['anomaly']}" 
                                                           for cls, v in sorted(class_dist.items())])
                                
                                stats_text = f"""Epoch {epoch+1} Statistics
                                
                                    Alignment Loss:
                                      align_loss: {align_loss.item():.4f}
                                      query_align: {query_align_loss.item():.4f}

                                    Similarity Stats:
                                      pos_sim (same group): {pos_sim_val:.4f}
                                      neg_sim (diff group): {neg_sim_val:.4f}
                                      separation: {pos_sim_val - neg_sim_val:.4f}
                                      proto_gap (cos n/a): {proto_gap_val:.4f}
                                      normal_margin: {normal_margin_val:.4f}
                                      anomaly_margin: {anomaly_margin_val:.4f}

                                    Sample Distribution (batch):
                                      Normal: {n_bg_val}
                                      Anomaly: {n_anom_val}
                                      Total: {B}

                                    Class Distribution (sampled):
                                    {class_dist_str}

                                    Prompt-Visual Sim:
                                      Mean: {diag_sim.mean():.4f}
                                      Std: {diag_sim.std():.4f}"""
                                
                                ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
                                        verticalalignment='top', fontfamily='monospace',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                                ax4.axis('off')
                                ax4.set_title("Training Statistics", fontsize=12)
                                
                                plt.suptitle(f"t-SNE Visualization - Epoch {epoch+1} (●=Normal, ★=Anomaly)", fontsize=14, fontweight='bold')
                                plt.tight_layout()
                                
                                out_dirs = []
                                save_to = str(getattr(args, "tsne_save_to", "log_dir")).lower()
                                if save_to in ("log_dir", "both"):
                                    out_dirs.append(os.path.join(log_dir, "tsne"))
                                if save_to in ("save_dir", "both"):
                                    out_dirs.append(os.path.join(save_dir, "tsne"))
                                out_dirs = [d for d in dict.fromkeys(out_dirs) if d]
                                for d in out_dirs:
                                    os.makedirs(d, exist_ok=True)

                                tsne_save_paths = [os.path.join(d, f"tsne_epoch{epoch+1:02d}.png") for d in out_dirs]
                                for p in tsne_save_paths:
                                    plt.savefig(p, dpi=150, bbox_inches="tight")
                                plt.close()
                                if tsne_save_paths:
                                    print(f"[INFO] Saved enhanced t-SNE visualization to {tsne_save_paths[0]}")

                                try:
                                    Z_visual = Z[ns:2 * ns]
                                    y = (~is_bg_sample).astype(np.int64)
                                    k = int(min(5, max(ns - 1, 1)))

                                    sil = float("nan")
                                    if len(np.unique(y)) >= 2 and ns >= 4 and np.min(np.bincount(y)) >= 2:
                                        sil = float(silhouette_score(Z_visual, y, metric="euclidean"))

                                    correct = 0
                                    boundary = 0
                                    for i in range(ns):
                                        d = np.sum((Z_visual - Z_visual[i:i+1]) ** 2, axis=1)
                                        d[i] = np.inf
                                        nn = np.argsort(d)[:k]
                                        pred = int(np.round(np.mean(y[nn])))
                                        if pred == int(y[i]):
                                            correct += 1
                                        frac_diff = float(np.mean(y[nn] != y[i]))
                                        if frac_diff >= 0.34:
                                            boundary += 1
                                    knn_acc = float(correct) / float(ns) if ns > 0 else float("nan")
                                    boundary_ratio = float(boundary) / float(ns) if ns > 0 else float("nan")

                                    metrics = {
                                        "epoch": int(epoch + 1),
                                        "ns": int(ns),
                                        "k": int(k),
                                        "silhouette_tsne_visual": sil,
                                        "knn_acc_tsne_visual": knn_acc,
                                        "boundary_ratio_tsne_visual": boundary_ratio,
                                    }
                                except Exception as e:
                                    metrics = {"epoch": int(epoch + 1), "error": str(e)}

                                dump = {
                                    "epoch": int(epoch + 1),
                                    "ns": int(ns),
                                    "sel": sel.astype(np.int64),
                                    "Z": Z.astype(np.float32),
                                    "Z_visual": Z[ns:2 * ns].astype(np.float32),
                                    "Z_prompt": Z[:ns].astype(np.float32),
                                    "is_bg": is_bg_sample.astype(np.bool_),
                                    "is_anomaly": np.asarray(is_anomaly_sample, dtype=np.bool_),
                                    "class_names": np.asarray(class_names_sample),
                                    "extra_names": np.asarray(extra_names),
                                    "diag_sim": diag_sim.astype(np.float32),
                                    "stats": np.asarray(
                                        {
                                            "align_loss": float(align_loss.item()),
                                            "query_align": float(query_align_loss.item()),
                                            "pos_sim": float(pos_sim_val),
                                            "neg_sim": float(neg_sim_val),
                                            "proto_gap": float(proto_gap_val),
                                            "normal_margin": float(normal_margin_val),
                                            "anomaly_margin": float(anomaly_margin_val),
                                            "prompt_visual_mean": float(diag_sim.mean()),
                                            "prompt_visual_std": float(diag_sim.std()),
                                        },
                                        dtype=object,
                                    ),
                                    "metrics": np.asarray(metrics, dtype=object),
                                }
                                npz_paths = [os.path.join(d, f"tsne_epoch{epoch+1:02d}.npz") for d in out_dirs]
                                for p in npz_paths:
                                    np.savez_compressed(p, **dump)
                                if npz_paths:
                                    print(f"[INFO] Saved t-SNE data to {npz_paths[0]}")
                            except Exception as e:
                                import traceback
                                print(f"[WARN] TSNE visualization failed: {e}")
                                traceback.print_exc()
                else:
                    align_loss = torch.tensor(0.0, device=device)
                    query_align_loss = torch.tensor(0.0, device=device)
                # -------------------------


                # Combine losses: 包含新的 query_align_loss
                # current_lambda_query_align 已在 batch 循环开始时初始化
                
                # 获取lambda_filo和lambda_conf_fusion
                lambda_filo = getattr(args, 'lambda_filo', 0.0)
                lambda_conf_fusion = getattr(args, 'lambda_conf_fusion', 0.0)
                lambda_msad = getattr(args, 'lambda_msad', 0.0)  # 新增
                lambda_msad_img = getattr(args, 'lambda_msad_img', 0.0)
                lambda_msad_margin = getattr(args, 'lambda_msad_margin', 0.0)
                lambda_msad_sim_margin = getattr(args, 'lambda_msad_sim_margin', 0.0)
                lambda_suspicious = float(getattr(args, "lambda_suspicious", 0.0) or 0.0)

                if str(getattr(args, "train_objective", "seg")).lower() == "rank":
                    loss_focal = torch.tensor(0.0, device=device)
                    loss_dice = torch.tensor(0.0, device=device)
                    loss_iou = torch.tensor(0.0, device=device)
                    loss_presence = torch.tensor(0.0, device=device)
                    align_loss = torch.tensor(0.0, device=device)
                    query_align_loss = torch.tensor(0.0, device=device)
                    loss_filo = torch.tensor(0.0, device=device)
                    loss_conf_fusion = torch.tensor(0.0, device=device)
                    loss_suspicious = torch.tensor(0.0, device=device)
                    loss_main = torch.tensor(0.0, device=device)
                
                if args.use_learned_loss_weights and len(learnable_log_vars) == 3:
                    loss_main = (torch.exp(-log_var_focal) * loss_focal + log_var_focal) + \
                                (torch.exp(-log_var_dice)  * loss_dice  + log_var_dice) + \
                                (torch.exp(-log_var_iou)   * loss_iou   + log_var_iou)
                    total_loss = loss_main + args.presence_weight * loss_presence + \
                                 args.lambda_align * align_loss + current_lambda_query_align * query_align_loss + \
                                 lambda_filo * loss_filo + lambda_msad * loss_msad + lambda_msad_img * loss_msad_img + lambda_msad_margin * loss_msad_margin + lambda_msad_sim_margin * loss_msad_sim_margin + lambda_conf_fusion * loss_conf_fusion + \
                                 lambda_suspicious * loss_suspicious  # 【新增】
                else:
                    total_loss = args.loss_alpha * loss_focal + args.loss_beta * loss_dice + args.loss_gamma * loss_iou
                    total_loss = total_loss + args.presence_weight * loss_presence + \
                                 args.lambda_align * align_loss + current_lambda_query_align * query_align_loss + \
                                 lambda_filo * loss_filo + lambda_msad * loss_msad + lambda_msad_img * loss_msad_img + lambda_msad_margin * loss_msad_margin + lambda_msad_sim_margin * loss_msad_sim_margin + lambda_conf_fusion * loss_conf_fusion + \
                                 lambda_suspicious * loss_suspicious  # 【新增】

                # ===== Compound Prompt Learner 损失 =====
                loss_orthogonal = torch.tensor(0.0, device=device)
                loss_prior = torch.tensor(0.0, device=device)
                loss_contrast = torch.tensor(0.0, device=device)
                
                if args.prompt_learner_type == "compound":
                    # 获取labels tensor
                    labels_tensor = torch.tensor(
                        [1 if a else 0 for a in is_anomaly], 
                        device=device
                    )
                    compound_losses = compute_compound_losses(model, out, labels_tensor, args)
                    
                    lambda_orth = getattr(args, 'lambda_orthogonal', 0.1)
                    lambda_prior_val = getattr(args, 'lambda_prior', 0.1)
                    lambda_contrast_val = getattr(args, 'lambda_contrast', 0.05)
                    
                    if 'orthogonal_loss' in compound_losses:
                        loss_orthogonal = compound_losses['orthogonal_loss']
                        total_loss = total_loss + lambda_orth * loss_orthogonal
                    
                    if 'prior_loss' in compound_losses:
                        loss_prior = compound_losses['prior_loss']
                        total_loss = total_loss + lambda_prior_val * loss_prior
                    
                    if 'contrast_loss' in compound_losses:
                        loss_contrast = compound_losses['contrast_loss']
                        total_loss = total_loss + lambda_contrast_val * loss_contrast

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

                if bool(getattr(args, "debug_prompt_grads", False)) and (step % getattr(args, "log_freq", 100) == 0):
                    targets = []
                    for n, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        nl = n.lower()
                        if ("prompt_learner." in nl) or ("patch_meta_net" in nl):
                            targets.append((n, float(p.grad.norm().detach().item())))
                    if len(targets) > 0:
                        targets.sort(key=lambda x: x[1], reverse=True)
                        print("[PROMPT-GRADS] top norms:")
                        for name, gn in targets[:20]:
                            print(f"  {name}: {gn:.4e}")
                    else:
                        print("[PROMPT-GRADS] no prompt-related grads found (all None)")

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
                
                # 更新两阶段 lambda 调度器
                if lambda_scheduler is not None:
                    lambda_scheduler.step()
                
                global_optim_step += 1
                if int(getattr(args, "max_train_steps", 0) or 0) > 0 and global_optim_step >= int(getattr(args, "max_train_steps", 0) or 0):
                    if is_main_process:
                        print(f"[INFO] Reached max_train_steps={int(getattr(args, 'max_train_steps', 0) or 0)}, stopping.")
                    stop_training = True
                
                grad_accum.reset()
            # ==================== 梯度累积逻辑结束 ====================

            if is_main_process:
                # 显示当前 lambda 值（如果使用两阶段调度）
                if lambda_scheduler is not None:
                    pbar.set_postfix(
                        loss=loss.item(),
                        focal=loss_focal.item(),
                        dice=loss_dice.item(),
                        lam_qa=f"{current_lambda_query_align:.3f}",
                        stage=lambda_scheduler.get_stage()[:2],  # s1/tr/s2
                    )
                else:
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
            if is_main_process and writer is not None:
                writer.add_scalar("loss/suspicious alignment", loss_suspicious.item(), global_step)            

            if is_main_process and writer is not None:
                diag_freq = int(getattr(args, "r2_diag_freq", 50) or 50)
                if diag_freq > 0 and (global_step % diag_freq) == 0:
                    eta_dbg = out.get("eta_spurious", None)
                    msad_score_dbg = out.get("msad_anomaly_score", None)
                    eta_mean = None
                    msad_mean = None
                    if isinstance(eta_dbg, torch.Tensor):
                        eta_mean = float(eta_dbg.detach().mean().item())
                        writer.add_scalar("r2/eta_mean", eta_mean, global_step)
                    if isinstance(msad_score_dbg, torch.Tensor):
                        msad_mean = float(msad_score_dbg.detach().mean().item())
                        writer.add_scalar("r2/msad_mean", msad_mean, global_step)
                    if (eta_mean is not None) and (msad_mean is not None):
                        s = r2_corr_state
                        s["n"] += 1
                        dx = eta_mean - s["mean_x"]
                        s["mean_x"] += dx / s["n"]
                        dy = msad_mean - s["mean_y"]
                        s["mean_y"] += dy / s["n"]
                        s["C"] += dx * (msad_mean - s["mean_y"])
                        s["M2x"] += dx * (eta_mean - s["mean_x"])
                        s["M2y"] += dy * (msad_mean - s["mean_y"])
                        denom = (s["M2x"] * s["M2y"]) ** 0.5
                        corr = float(s["C"] / denom) if (s["n"] > 2 and denom > 1e-12) else float("nan")
                        writer.add_scalar("r2/corr_eta_msad_mean", corr, global_step)

                    msad_map_dbg = out.get("msad_aggregated_map", None)
                    sp_map_dbg = out.get("spurious_map", None)
                    if isinstance(msad_map_dbg, torch.Tensor) and isinstance(sp_map_dbg, torch.Tensor):
                        gt_mask = masks
                        if gt_mask.dim() == 4:
                            gt_mask = gt_mask.squeeze(1)
                        with torch.no_grad():
                            p_norm = msad_map_dbg[:, 0].float()
                            p_abn = msad_map_dbg[:, 1].float()
                            if p_abn.shape[-2:] != gt_mask.shape[-2:]:
                                gt_mask_f32 = F.interpolate(gt_mask.unsqueeze(1).float(), size=p_abn.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                            else:
                                gt_mask_f32 = gt_mask.float()
                            if sp_map_dbg.shape[-2:] != p_abn.shape[-2:]:
                                sp_map_f32 = F.interpolate(sp_map_dbg.unsqueeze(1).float(), size=p_abn.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                            else:
                                sp_map_f32 = sp_map_dbg.float()
                            eps = 1e-6
                            logit = (torch.log(p_abn.clamp(eps, 1 - eps)) - torch.log(p_norm.clamp(eps, 1 - eps))).detach()
                            defect_mask = gt_mask_f32 > 0.5
                            top_p = float(getattr(args, "spurious_top_p", 0.02) or 0.0)
                            spurious_mask = torch.zeros_like(defect_mask)
                            if top_p > 0.0:
                                B, H, W = sp_map_f32.shape
                                k = int(max(1, min(H * W, round(top_p * H * W))))
                                flat = sp_map_f32.view(B, -1)
                                topk_vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)
                                thr = topk_vals[:, -1].view(B, 1, 1)
                                spurious_mask = (sp_map_f32 >= thr) & (gt_mask_f32 <= 0.5)

                            def _log_q(prefix, vals):
                                if vals.numel() == 0:
                                    return
                                for q in (0.5, 0.9, 0.99):
                                    writer.add_scalar(f"r2/{prefix}_p{int(q*100):02d}", float(torch.quantile(vals, q).item()), global_step)

                            if defect_mask.any():
                                _log_q("logit_defect", logit[defect_mask])
                            if spurious_mask.any():
                                _log_q("logit_spurious", logit[spurious_mask])

            if is_main_process and writer is not None:
                writer.add_scalar("debug/src_q_nonempty_ratio", nonempty_ratio, global_step)
                writer.add_scalar("debug/src_q_len_mean", mean_len, global_step)
                writer.add_scalar("debug/src_q_len_max", max_len, global_step)
                # 可选：直方图（看分布最直观）
                writer.add_histogram("debug/src_q_len_hist", torch.tensor(src_q_lens), global_step)
            # 记录当前学习率
            if is_main_process and writer is not None:
                current_lrs = scheduler.get_lr()
                writer.add_scalar("lr/prompt", current_lrs[0], global_step)
                if len(current_lrs) > 1:
                    writer.add_scalar("lr/main", current_lrs[1], global_step)
            # 记录当前 lambda_query_align（如果使用两阶段调度）
            if is_main_process and writer is not None and lambda_scheduler is not None:
                writer.add_scalar("lambda/query_align", current_lambda_query_align, global_step)
            
            # ===== 多尺度特征 / 注意力可视化落盘 =====
            if getattr(args, 'enable_multiscale_vis', False) and is_main_process:
                vis_freq = getattr(args, 'multiscale_vis_freq', 500)
                if global_step > 0 and (global_step % vis_freq) == 0:
                    try:
                        from multiscale_modules import FeatureVisualizer
                        vis_root = os.path.join(save_dir, getattr(args, 'multiscale_vis_dir', 'feature_vis'))
                        os.makedirs(vis_root, exist_ok=True)
                        visualizer = FeatureVisualizer(save_dir=vis_root)
                        
                        ms = out.get("multiscale_features", None)
                        if ms is not None and "used_features" in ms:
                            feats = ms["used_features"]  # List[(B,C,H,W)]
                            
                            # 1) 多尺度特征图（均值/方差 + 原图）
                            visualizer.visualize_fpn_levels(
                                fpn_features=feats,
                                image=images,
                                save_name=f"step_{global_step}_features"
                            )
                            
                            # 2) 特征统计
                            visualizer.visualize_feature_statistics(
                                features=feats,
                                save_name=f"step_{global_step}_stats"
                            )
                            
                            # 3) 文本-视觉相似度（用最高分辨率层）
                            if "prompt_seq" in out and len(feats) > 0:
                                visualizer.visualize_text_visual_similarity(
                                    visual_feat=feats[0],
                                    text_embed=out["prompt_seq"],
                                    save_name=f"step_{global_step}_sim"
                                )
                            
                            # 4) V-V 注意力权重
                            if "vv_attention_weights" in out:
                                for i, attn in enumerate(out["vv_attention_weights"]):
                                    if i < len(feats):
                                        H, W = feats[i].shape[-2], feats[i].shape[-1]
                                        visualizer.visualize_attention_weights(
                                            attn_weights=attn,
                                            feature_shape=(H, W),
                                            save_name=f"step_{global_step}_vv_attn_l{i}"
                                        )
                            
                            # 5) MMCI 注意力权重
                            print(f"[VIS] Saved multiscale visualization at step {global_step} to {vis_root}")
                            
                    except Exception as e:
                        print(f"[WARN] multiscale visualization failed: {e}")
            # ===== 多尺度可视化结束 =====
            

            running_loss += loss.item()
            running_steps += 1
            if stop_training:
                break

        if stop_training:
            break

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

def apply_run_profile(args: argparse.Namespace) -> argparse.Namespace:
    def _set_if_default(name: str, default, value) -> None:
        if hasattr(args, name) and getattr(args, name) == default:
            setattr(args, name, value)

    prof = str(getattr(args, "run_profile", "custom")).lower()
    if getattr(args, "disable_spurious_gating", False):
        args.enable_spurious_gating = False
    if prof == "zero_shot":
        args.train_objective = "rank"
        args.disable_lora = True
        args.freeze_vision = True
        args.freeze_text = True
        args.unfreeze_decoder = "none"
        args.train_seg_head = False
        args.lambda_align = 0.0
        args.lambda_query_align = 0.0
        args.enable_two_stage = False
        args.lambda_suspicious = 0.0
        if hasattr(args, "disable_bank"):
            args.disable_bank = True
        if hasattr(args, "disable_w_learning"):
            args.disable_w_learning = True
        args.lambda_msad_margin = 0.0
        args.lambda_msad_sim_margin = 0.0
        if hasattr(args, "msad_return_similarity_logits"):
            args.msad_return_similarity_logits = False
        args.prompt_learner_type = "compound"
        args.compound_disable_w = True
        args.compound_use_text_encoder = True
        args.compound_abnormal_word = str(getattr(args, "compound_abnormal_word", "anomaly") or "anomaly")
        args.compound_pooling = str(getattr(args, "compound_pooling", "ctx_only") or "ctx_only")
        args.enable_msad = True
        args.msad_use_vision_adapter = True
        args.enable_spurious_gating = False
        args.lambda_msad = float(getattr(args, "lambda_msad", 0.3) or 0.3)
        args.lambda_msad_img = float(getattr(args, "lambda_msad_img", 0.1) or 0.1)
        if hasattr(args, "msad_img_pool"):
            args.msad_img_pool = str(getattr(args, "msad_img_pool", "q95") or "q95")
    elif prof in ("few_shot", "few_shot_full", "few_shot_no_w"):
        args.train_objective = "seg"
        args.disable_lora = True
        args.freeze_vision = True
        args.freeze_text = True
        args.unfreeze_decoder = "all" if prof in ("few_shot_full", "few_shot_no_w") else "none"
        args.train_seg_head = True
        args.prompt_learner_type = "compound"
        args.compound_disable_w = True if prof == "few_shot_no_w" else False
        args.compound_use_text_encoder = True
        args.compound_abnormal_word = str(getattr(args, "compound_abnormal_word", "anomaly") or "anomaly")
        args.compound_pooling = str(getattr(args, "compound_pooling", "ctx_only") or "ctx_only")
        if prof in ("few_shot_full", "few_shot_no_w"):
            if hasattr(args, "enable_parallel_lora"):
                args.enable_parallel_lora = True
            if hasattr(args, "parallel_lora_rank"):
                args.parallel_lora_rank = int(getattr(args, "parallel_lora_rank", 16) or 16)
            if hasattr(args, "parallel_lora_alpha") and getattr(args, "parallel_lora_alpha", None) is None:
                args.parallel_lora_alpha = 64.0
            if hasattr(args, "parallel_lora_target"):
                args.parallel_lora_target = str(getattr(args, "parallel_lora_target", "qv_only") or "qv_only")
            if hasattr(args, "compound_enable_dap"):
                args.compound_enable_dap = True
            _set_if_default("lambda_msad_margin", 0.0, 0.3)
            _set_if_default("lambda_msad_sim_margin", 0.0, 0.3)
            if hasattr(args, "msad_return_similarity_logits"):
                args.msad_return_similarity_logits = True
        _set_if_default("loss_alpha", 5.0, 5.0)
        _set_if_default("loss_beta", 1.0, 1.0)
        _set_if_default("loss_gamma", 1.0, 0.5)
        _set_if_default("presence_weight", 1.0, 0.6)
        _set_if_default("neg_samples_per_image", 50, 10)
        _set_if_default("min_normals_per_batch", 2, 4)
        _set_if_default("lambda_align", 0.1, 0.50)
        if hasattr(args, "align_multilevel") and bool(getattr(args, "align_multilevel", False)) is False:
            args.align_multilevel = True
        _set_if_default("align_multilevel_weight_source", "uniform", "uniform")
        _set_if_default("align_multilevel_max_levels", 0, 0)
        _set_if_default("align_temp", 0.1, 0.25)
        _set_if_default("align_margin", 0.5, 0.25)
        if hasattr(args, "enable_two_stage") and bool(getattr(args, "enable_two_stage", False)) is False:
            args.enable_two_stage = True
        _set_if_default("stage1_ratio", 0.35, 0.4)
        _set_if_default("stage1_lambda", 0.08, 0.2)
        _set_if_default("stage2_lambda", 0.20, 0.3)
        _set_if_default("lambda_transition", "linear", "linear")
        _set_if_default("transition_ratio", 0.15, 0.4)
        _set_if_default("query_align_top_k", 64, 128)
        _set_if_default("query_align_temp", 0.2, 0.2)
        _set_if_default("lambda_query_align", 0.5, 0.5)
        _set_if_default("bank_warm_up_ratio", 0.3, 0.6)
        if hasattr(args, "bank_orthogonalize") and bool(getattr(args, "bank_orthogonalize", False)) is False:
            args.bank_orthogonalize = True
        _set_if_default("w_abnormal_margin", 0.3, 0.3)
        if prof == "few_shot_no_w":
            args.lambda_suspicious = 0.0
        else:
            _set_if_default("lambda_suspicious", 0.1, 0.3)
        _set_if_default("lambda_msad", 0.0, 0.3)
        _set_if_default("msad_num_levels", None, 3)
        if hasattr(args, "msad_use_shape_attention") and bool(getattr(args, "msad_use_shape_attention", True)) is False:
            args.msad_use_shape_attention = True
        if (not getattr(args, "disable_spurious_gating", False)) and hasattr(args, "enable_spurious_gating") and bool(getattr(args, "enable_spurious_gating", True)) is False:
            args.enable_spurious_gating = True
        _set_if_default("spurious_score_threshold", 0.20, 0.20)
        if prof == "few_shot_no_w":
            if hasattr(args, "disable_w_learning"):
                args.disable_w_learning = True
            if hasattr(args, "disable_bank"):
                args.disable_bank = True
        else:
            if hasattr(args, "disable_w_learning"):
                args.disable_w_learning = False
            if hasattr(args, "disable_bank"):
                args.disable_bank = False
        args.enable_msad = True
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_profile", type=str, default="custom",
                        choices=["custom", "zero_shot", "few_shot", "few_shot_full", "few_shot_no_w"])
    parser.add_argument("--data_root", type=str, required=True, help="Root of MVTec-AD dataset.")
    parser.add_argument("--meta_path", type=str, default=None, help="Path to meta.json (defaults to <data_root>/meta.json).")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"], help="Split to load.")
    parser.add_argument("--k_shot", type=int, default=0, help="K-shot for train/train_all.")
    parser.add_argument("--few_shot_per_specie", type=int, default=0,
                        help="Few-shot 每个 specie 采样数 (0=禁用, 5=5-shot)")
    parser.add_argument("--few_shot_balance_good_by_specie", action="store_true", default=False,
                        help="Few-shot: normal(good) 采样按 specie 对齐 defect 分布")
    parser.add_argument("--obj_name", type=str, default=None, help="Class name for mode=train.")
    parser.add_argument("--aug_rate", type=float, default=0.0, help="Mosaic augmentation probability.")
    parser.add_argument("--bpe_path", type=str, default=None, help="Path to BPE vocab (defaults to sam3/assets/bpe_simple_vocab_16e6.txt.gz).")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="If >0, stop after N optimizer steps (smoke test / quick debug).")
    parser.add_argument("--lr_prompt", type=float, default=5e-4)
    parser.add_argument("--lr_main", type=float, default=5e-5)
    parser.add_argument("--loss_alpha", type=float, default=5.0, help="Weight for focal loss.")
    parser.add_argument("--loss_beta", type=float, default=1.0, help="Weight for dice loss.")
    parser.add_argument("--loss_gamma", type=float, default=1.0, help="Weight for IoU regression loss.")
    parser.add_argument("--lambda_suspicious", type=float, default=0.1,
                    help="Weight for suspicious alignment loss")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_layer_ids", nargs="*", type=int, default=None,
                        help="Which SAM3 encoder blocks to apply LoRA to (e.g., --lora_layer_ids 0 2 4). Default: all blocks.")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--force_secondary_sam3_load", action="store_true", default=False,
                        help="Force running load_sam3_checkpoint even when --use_official.")
    
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
    parser.add_argument("--lambda_filo", type=float, default=0.0, 
                        help="Weight for FiLo anomaly map loss (Focal+Dice). Set >0 to enable FiLo supervision.")
    parser.add_argument("--presence_weight",type=float,default=1.0,help="weight for presence BCE loss")
    parser.add_argument("--use_learned_loss_weights", action="store_true", help="Use learnable log-variance weights for multi-loss balancing (Kendall)")
    parser.add_argument("--mask_downsample", type=int, default=256, help="Downsample masks for background loss calculation to reduce memory")
    parser.add_argument("--enable_parallel_lora", action="store_true", help="Enable parallel LoRA adapters in Attention (official model path)")
    parser.add_argument("--parallel_lora_rank", type=int, default=16, help="Rank for parallel LoRA")
    parser.add_argument("--parallel_lora_alpha", type=float, default=None, help="Alpha scaling for parallel LoRA")
    parser.add_argument("--parallel_lora_target", type=str, default="qv_only",
                        choices=["qv_only", "qkv_all"],
                        help="Parallel LoRA target on attention.qkv")
    parser.add_argument("--parallel_lora_layer_ids", nargs="*", type=int, default=None,
                        help="Which SAM3 encoder blocks to apply parallel LoRA to. Default: all blocks.")
    parser.add_argument("--enable_out_adapter_lora", action="store_true", default=False,
                        help="(legacy) Enable out_adapter side-branch LoRA in Attention")
    parser.add_argument("--include_test_defects", action="store_true",help="(legacy) include defects from test split when forming dataset")
    parser.add_argument("--train_from_test", action="store_true",help="When set, build training set from MVTec test split defects only, per-specie split")
    parser.add_argument("--specie_split_ratio", type=float, default=0.8,help="Train ratio per specie (e.g. 0.8 => 80% train, 20% test)")
    parser.add_argument("--specie_split_seed", type=int, default=42,help="Random seed for per-specie split reproducibility")
    parser.add_argument("--splits_save_dir", type=str, default=None,help="If set, write specie_splits_{cls}.json files for reproducibility.")
    
    # === 新增: align loss 相关参数 ===
    parser.add_argument("--use_anomaly_grouping", action="store_true", 
                        help="Use simpler anomaly/normal grouping instead of prompt-based grouping for align loss")
    parser.add_argument("--align_multilevel", action="store_true",
                        help="对齐损失使用多层FPN/多层token的逐层聚合")
    parser.add_argument("--align_multilevel_weight_source", type=str, default="uniform",
                        choices=["uniform", "msad"],
                        help="多层对齐加权策略：uniform或使用MSAD的level_weights")
    parser.add_argument("--align_multilevel_max_levels", type=int, default=0,
                        help="多层对齐最大层数(0=全部可用层)")
    parser.add_argument("--query_align_top_k", type=int, default=64, 
                        help="Top-k queries to compete in query alignment loss (reduces from Q=900 to top_k)")
    parser.add_argument("--query_align_temp", type=float, default=0.2,
                        help="Temperature for query alignment loss (higher = softer, easier to learn)")
    
    # ===== Query Align V2 改进参数 =====
    parser.add_argument("--use_query_align_v2", action="store_true", default=True,
                        help="使用改进的query_align_v2版本（三个关键修复）")
    parser.add_argument("--query_align_soft_target", action="store_true", default=True,
                        help="[修复1] 使用IoU软标签代替硬标签，减少matcher漂移噪声")
    parser.add_argument("--query_align_include_normal", action="store_true", default=True,
                        help="[修复2] Normal图也参与query_align（margin loss）")
    parser.add_argument("--query_align_normal_margin", type=float, default=0.3,
                        help="[修复2] Normal图margin loss的margin值")
    parser.add_argument("--query_align_full_softmax_ratio", type=float, default=0.2,
                        help="[修复3] 前N%%步使用full softmax（更稳定），之后切回top-k")


    #--------------- Diagnostic logging args ---------------
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (steps) for align diagnostics")
    parser.add_argument("--tsne_freq", type=int, default=500, help="TSNE save frequency (steps)")
    parser.add_argument("--tsne_samples", type=int, default=64, help="Number of samples for TSNE projection")
    parser.add_argument("--tsne_save_to", type=str, default="log_dir",
                        choices=["log_dir", "save_dir", "both"],
                        help="t-SNE产物保存位置：log_dir 或 save_dir 或两者都保存")
    
    parser.add_argument("--align_margin", type=float, default=0.5,help="Margin for pushing defect prompts away from background embeddings")
    parser.add_argument("--lambda_query_align", type=float, default=0.5,help="Weight for query-level alignment loss (会被两阶段调度器覆盖)")

    # ==================== 新增：两阶段 query_align 调度参数 ====================
    parser.add_argument("--enable_two_stage", action="store_true",
                        help="启用两阶段 lambda_query_align 调度 (推荐开启)")
    parser.add_argument("--stage1_ratio", type=float, default=0.35,
                        help="Stage 1 占总步数的比例 (默认0.35即35%)")
    parser.add_argument("--stage1_lambda", type=float, default=0.08,
                        help="Stage 1 的 lambda_query_align 值 (低，让模型先学 segmentation)")
    parser.add_argument("--stage2_lambda", type=float, default=0.20,
                        help="Stage 2 的 lambda_query_align 值 (高，增强 query-text alignment)")
    parser.add_argument("--lambda_transition", type=str, default="linear",
                        choices=["step", "linear", "cosine"],
                        help="Stage 1 到 Stage 2 的过渡方式 (默认 linear)")
    parser.add_argument("--transition_ratio", type=float, default=0.15,
                        help="过渡期占 Stage 2 的比例 (默认0.15，仅 linear/cosine 有效)")

    # ==================== 新增：训练优化参数 ====================
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup步数占总步数的比例 (默认0.1即10%)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="最终学习率与初始学习率的比值 (默认0.01)")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="梯度累积步数 (默认1即不累积，设为2则有效batch=batch_size*2)")
    parser.add_argument("--min_normals_per_batch", type=int, default=2,
                        help="在训练 batch 中至少包含多少个 normal/good 样本（用于压制假阳性；单卡时生效）")
    
    # ==================== CoOp/CoCoOp 提示学习参数 ====================
    parser.add_argument("--prompt_learner_type", type=str, default="perclass",
                        choices=["averaged", "static", "perclass", "coop", "cocoop", "compound"],
                        help="提示学习器类型")
    parser.add_argument("--n_ctx", type=int, default=4,
                        help="可学习上下文向量数量")
    parser.add_argument("--ctx_init", type=str, default="",
                        help="上下文初始化文本，如'a photo of a'")
    parser.add_argument("--class_token_position", type=str, default="end",
                        choices=["end", "middle", "front"],
                        help="类别token位置")
    parser.add_argument("--use_keywords", action="store_true",
                        help="是否使用关键词聚合（默认关闭，推荐不开启）")
    parser.add_argument("--cocoop_vis_dim", type=int, default=256,
                        help="CoCoOp Meta-Net输入维度")
    parser.add_argument("--cocoop_reduction", type=int, default=16,
                        help="CoCoOp Meta-Net瓶颈缩减因子")
    # ==================== Compound Prompt Learning ====================
    parser.add_argument("--compound_mode", type=str, default="cocoop",
                        choices=["coop", "cocoop"],
                        help="Compound: 模式选择 (coop=静态, cocoop=Meta-Net条件化)")
    parser.add_argument("--compound_n_ctx", type=int, default=4,
                        help="Compound: 共享上下文向量数量")
    parser.add_argument("--compound_n_ctx_offset", type=int, default=4,
                        help="Compound: 正常/异常偏移向量数量")
    parser.add_argument("--compound_num_abnormal", type=int, default=10,
                        help="Compound: 异常prompt数量")
    parser.add_argument("--compound_enable_dap", action="store_true",
                        help="Compound: 启用数据依赖异常先验")
    parser.add_argument("--compound_dap_top_k", type=int, default=10,
                        help="Compound: DAP top-k")
    parser.add_argument("--compound_meta_reduction", type=int, default=16,
                        help="Compound: Meta-Net瓶颈缩减因子（仅cocoop模式）")
    parser.add_argument("--compound_dap_use_multilevel", action="store_true",
                        help="Compound: DAP使用多层FPN特征拼接后的patch集合")
    parser.add_argument("--compound_dap_num_levels", type=int, default=0,
                        help="Compound: DAP使用的FPN层数(0=使用全部可用层)")
    parser.add_argument("--compound_use_text_encoder", action="store_true",
                        help="Compound: 将learnable ctx注入token embedding并经过SAM3 text encoder编码")
    parser.add_argument("--compound_abnormal_word", type=str, default="anomaly",
                        choices=["anomaly", "damaged"],
                        help="Compound: abnormal模板关键词(训练/测试需保持一致)")
    parser.add_argument("--compound_pooling", type=str, default="ctx_only",
                        choices=["ctx_only", "all_tokens"],
                        help="Compound(use_text_encoder): prompt向量聚合方式(只聚合ctx段或全token均值)")
    parser.add_argument("--compound_abnormal_order", type=str, default="v_then_wk",
                        choices=["v_then_wk", "wk_then_v"],
                        help="Compound: abnormal 前缀token顺序 (V+W_k 或 W_k+V)")
    parser.add_argument("--debug_prompt_grads", action="store_true",
                        help="Debug: 打印compound prompt相关参数的梯度范数")
    parser.add_argument("--debug_dump_features", action="store_true",
                        help="Debug: 保存一小份text/msad特征到npz用于新旧版本对比")
    parser.add_argument("--lambda_orthogonal", type=float, default=0.1,
                        help="Compound: 正交约束损失权重")
    parser.add_argument("--lambda_prior", type=float, default=0.1,
                        help="Compound: DAP先验损失权重")
    parser.add_argument("--lambda_contrast", type=float, default=0.05,
                        help="Compound: Normal/Abnormal对比损失权重")
    # 数据集prompt模式
    parser.add_argument("--prompt_mode", type=str, default="simple",
                        choices=["simple", "full"],
                        help="数据集prompt模式: simple(推荐)只有类别描述, full包含关键词")
    # 在 argparse 部分添加
    parser.add_argument("--class_agnostic", action="store_true", default=False,
                        help="【零样本】使用类别无关设计，将所有类别名替换为通用名")
    parser.add_argument("--agnostic_name", type=str, default="object",
                        help="类别无关模式下使用的通用名称 (默认: object)")

    parser.add_argument("--train_seg_head", action="store_true", default=False,
                        help="仅在显式指定时训练 segmentation_head（默认冻结以保持开放分割能力）")

    # ==================== 多尺度特征 & V-V注意力 & FiLo ====================
    parser.add_argument("--num_feature_levels", type=int, default=1,
                        help="使用多少层FPN特征 (1-4, SAM3通常支持4层)")
    parser.add_argument("--enable_vv_attention", action="store_true",
                        help="启用V-V自注意力增强视觉特征")
    parser.add_argument("--vv_num_heads", type=int, default=8,
                        help="V-V注意力的头数")
    parser.add_argument("--vv_dropout", type=float, default=0.1,
                        help="V-V注意力的dropout率")
    # FiLo模块 (6路卷积MMCI)
    parser.add_argument("--enable_filo", action="store_true",
                        help="启用FiLo模块 (LinearLayer + CovLayer双分支解码)")
    parser.add_argument("--filo_dim_out", type=int, default=768,
                        help="FiLo输出维度 (对齐文本特征)")
    parser.add_argument("--filo_k_linear", type=int, default=4,
                        help="FiLo LinearLayer层数")
    parser.add_argument("--filo_k_cov", type=int, default=4,
                        help="FiLo CovLayer层数")
    parser.add_argument("--filo_image_size", type=int, default=518,
                        help="FiLo异常图输出尺寸")
    parser.add_argument("--filo_use_alternating", action="store_true", default=True,
                        help="FiLo是否交替分配FPN层 (偶数→Linear, 奇数→Cov)")
    
    
    # ==================== 方案B: FiLo到Decoder回灌 ====================
    parser.add_argument("--filo_to_decoder", action="store_true",
                        help="启用FiLo特征到Decoder的回灌（结构回灌）")
    parser.add_argument("--filo_decoder_mode", type=str, default="memory",
                        choices=["memory", "query_bias", "cross_attn"],
                        help="FiLo到Decoder的回灌模式: memory(扩展memory), query_bias(query偏置), cross_attn(交叉注意力)")
    parser.add_argument("--filo_decoder_tokens", type=int, default=64,
                        help="FiLo压缩后的token数量（用于memory模式）")
    
    # ==================== 方案C: 置信度融合头 ====================
    parser.add_argument("--enable_conf_fusion_head", action="store_true",
                        help="启用置信度融合头（学习融合presence/iou/filo）")
    parser.add_argument("--conf_fusion_hidden_dim", type=int, default=64,
                        help="置信度融合头的隐藏维度")
    parser.add_argument("--lambda_conf_fusion", type=float, default=0.0,
                        help="置信度融合头损失权重 (设置>0启用训练)")
    
    parser.add_argument("--enable_multiscale_vis", action="store_true",
                        help="启用多尺度特征可视化输出")
    parser.add_argument("--multiscale_vis_freq", type=int, default=500,
                        help="多尺度特征可视化保存频率(steps)")
    parser.add_argument("--multiscale_vis_dir", type=str, default="feature_vis",
                        help="多尺度可视化输出子目录（相对 save_dir）")
    
    # Stages消融实验
    parser.add_argument("--selected_levels", type=str, default=None,
                        help="指定使用的FPN层级，逗号分隔，如'0,1,2'用于消融实验")
    parser.add_argument("--ablation_config", type=str, default=None,
                        choices=['single_level_0', 'single_level_1', 'single_level_2', 'single_level_3',
                                 'levels_0_1', 'levels_1_2', 'levels_2_3', 'levels_0_2',
                                 'levels_0_1_2', 'levels_1_2_3', 'all_levels',
                                 'sowa_style'],
                        help="预定义的stages消融实验配置")
    
    parser.add_argument("--print_backbone", action="store_true",
                    help="Print SAM3 backbone / trunk / transformer structure then exit.")
    parser.add_argument("--print_modules_filter", type=str, default="",
                        help="Optional substring filter to print only matching module names (e.g., 'vision_backbone.trunk').")
    
    parser.add_argument("--print_backbone_to_txt", action="store_true",
                        help="Save printed backbone structure to a txt file.")
    parser.add_argument("--print_backbone_txt_path", type=str, default="backbone_dump.txt",
                        help="Path to save backbone dump txt (relative ok).")
    parser.add_argument("--print_backbone_no_stdout", action="store_true",
                        help="Do not print to stdout when dumping to txt.")
    
    # v2.2 零样本优化参数
    parser.add_argument('--bank_warm_up_ratio', type=float, default=0.3,
                        help='Memory Bank warm-up ratio (default: 0.3, 前30%不启用)')
    parser.add_argument('--bank_orthogonalize', action='store_true', default=True,
                        help='是否对 bank 中的特征进行正交化')
    parser.add_argument('--w_abnormal_margin', type=float, default=0.3,
                        help='w 与 abnormal 之间的 margin (default: 0.3)')
    parser.add_argument('--disable_bank', action='store_true', default=False,
                        help='【消融】禁用 bank，只使用统计 fallback')
    parser.add_argument('--disable_w_learning', action='store_true', default=False,
                        help='【消融】禁用 w 学习 (lambda_suspicious=0)')

    parser.add_argument("--train_objective", type=str, default="seg",
                        choices=["seg", "rank"],
                        help="训练目标：seg=分割训练；rank=MSAD排序/异常图训练(更接近FiLo/SOWA)")
    
    # ==================== MSAD模块参数 (Multi-Shape Anomaly Detection) ====================
    parser.add_argument("--enable_msad", action="store_true",
                        help="启用MSAD (Multi-Shape Anomaly Detection) 模块")
    parser.add_argument("--msad_use_shape_attention", action="store_true", default=True,
                        help="使用可学习形状注意力 (Learnable Shape Attention)")
    parser.add_argument("--msad_learnable_level_weights", action="store_true", default=True,
                        help="使用可学习的层级权重")
    parser.add_argument("--msad_learnable_temperature", action="store_true", default=True,
                        help="使用可学习的温度参数")
    parser.add_argument("--msad_temperature", type=float, default=100.0,
                        help="异常评分温度参数初始值")
    parser.add_argument("--msad_output_size", type=int, default=518,
                        help="MSAD输出异常图尺寸")
    parser.add_argument("--msad_num_levels", type=int, default=None,
                        help="MSAD使用的FPN层数（默认4；不足则重复最低层特征）")
    parser.add_argument("--lambda_msad", type=float, default=0.0,
                        help="MSAD异常图监督损失权重 (>0启用)")
    parser.add_argument("--lambda_msad_img", type=float, default=0.0,
                        help="MSAD图像级排序损失权重 (>0启用)")
    parser.add_argument("--msad_img_pool", type=str, default="q95",
                        choices=["max", "q95", "mean", "topk_mean"],
                        help="MSAD图像级分数池化方式")
    parser.add_argument("--msad_img_topk_ratio", type=float, default=0.02,
                        help="msad_img_pool=topk_mean时的topk比例")
    parser.add_argument("--lambda_msad_margin", type=float, default=0.0,
                        help="MSAD像素级margin约束损失权重 (>0启用)")
    parser.add_argument("--lambda_msad_sim_margin", type=float, default=0.0,
                        help="R2-2: MSAD similarity(logits)层margin损失权重 (>0启用)")
    parser.add_argument("--msad_sim_margin_defect", type=float, default=0.3,
                        help="R2-2: 缺陷像素(sim_logits)的最小margin")
    parser.add_argument("--msad_sim_margin_spurious", type=float, default=0.3,
                        help="R2-2: spurious像素(sim_logits)的最小margin")
    parser.add_argument("--msad_sim_margin_source", type=str, default="agg",
                        choices=["agg", "level0"],
                        help="R2-2: similarity logits来源(agg=聚合后, level0=最高分辨率层)")
    parser.add_argument("--msad_return_similarity_logits", action="store_true",
                        help="训练: 让MSAD返回softmax前similarity logits以支持R2-2")
    parser.add_argument("--msad_use_vision_adapter", action="store_true", default=False,
                        help="MSAD: 在FPN特征进入MSAD前加轻量Conv Adapter(残差)以增强跨域对齐")
    parser.add_argument("--msad_vision_adapter_reduction", type=int, default=2,
                        help="MSAD vision adapter: reduction ratio")
    parser.add_argument("--msad_vision_adapter_shared", action="store_true", default=True,
                        help="MSAD vision adapter: 是否各层共享同一个adapter")
    parser.add_argument("--msad_vision_adapter_not_shared", action="store_true", default=False,
                        help="MSAD vision adapter: 各层使用独立adapter（覆盖 msad_vision_adapter_shared）")
    parser.add_argument("--spurious_margin_require_quality", action="store_true", default=True,
                        help="R2: 仅当spurious_map足够尖锐时启用spurious侧margin")
    parser.add_argument("--r2_warmup_ratio", type=float, default=0.1,
                        help="R2: warmup比例（期间不启用spurious侧margin）")
    parser.add_argument("--r2_spurious_ramp_ratio", type=float, default=0.2,
                        help="R2: spurious侧margin线性ramp比例")
    parser.add_argument("--r2_diag_freq", type=int, default=50,
                        help="R2: 诊断曲线写入频率(step)")
    parser.add_argument("--msad_margin_defect", type=float, default=0.3,
                        help="缺陷像素logit(abn-norm)的最小margin")
    parser.add_argument("--msad_margin_spurious", type=float, default=0.3,
                        help="spurious像素logit(norm-abn)的最小margin")
    parser.add_argument("--spurious_top_p", type=float, default=0.02,
                        help="从spurious_map选取top-p像素作为spurious集合（仅margin约束用）")

    parser.add_argument("--compound_dap_spurious_filter", action="store_true",
                        help="DAP: 用spurious_map过滤top-k patch，降低伪异常污染W_k")
    parser.add_argument("--compound_dap_spurious_alpha", type=float, default=1.0,
                        help="DAP: spurious加权系数α（score*=max(0,1-α*spurious)）")
    parser.add_argument("--compound_disable_w", action="store_true",
                        help="调试：禁用w向量(使normal_ctx仅V，abnormal_ctx为V+W)，更接近FAPrompt结构")
    
    # ==================== Spurious Prompt Gating 参数 ====================
    parser.add_argument("--enable_spurious_gating", action="store_true", default=True,
                        help="启用 Spurious Prompt Gating (eta调制w向量)")
    parser.add_argument("--disable_spurious_gating", action="store_true", default=False,
                        help="显式禁用 Spurious Prompt Gating（优先级高于 enable_spurious_gating）")
    parser.add_argument("--spurious_sim_temp", type=float, default=0.07,
                        help="Spurious prompt 相似度温度")
    parser.add_argument("--spurious_topk_ratio", type=float, default=0.02,
                        help="Top-k pooling 比例")
    parser.add_argument("--spurious_score_threshold", type=float, default=0.20,
                        help="激活阈值")
    parser.add_argument("--spurious_kappa", type=float, default=8.0,
                        help="Sigmoid 斜率")
    parser.add_argument("--spurious_quality_threshold", type=float, default=0.03,
                        help="Map peakiness 阈值")
    

    args = parser.parse_args()
    args = apply_run_profile(args)
    main(args)
