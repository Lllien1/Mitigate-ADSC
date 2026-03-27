# MSAM_test.py (v4.0 - MSAD + Spurious Gating)
# encoding: utf-8
# 
# v4.0 更新：
# 1. 移除 FiLo 相关代码，替换为 MSAD (Multi-Shape Anomaly Detection)
# 2. 添加 Spurious Gating 支持
# 3. 更新模型构建参数以匹配 v4.0 接口
# 4. 添加 MSAD 输出可视化

import os, sys

PROJECT_ROOT = "/root/autodl-tmp/FiLo_plus/sam3"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import inspect
import json
import time
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_erosion, binary_dilation
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from dataset import MVTecMetaDataset, VisADataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official

# Optional: sklearn for AUC computation
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not found, AUC metrics will be skipped")


# ---------- visualization helpers ----------
def get_color_map(palette: List[Tuple[int, int, int]], key: str) -> Tuple[int, int, int]:
    import hashlib
    k = str(key)
    kl = k.lower()
    if k == "MSAD":
        return (255, 165, 0)
    if k == "FUSED":
        return (0, 128, 255)
    if any(w in kl for w in ("anomaly", "damaged", "defect", "flaw")):
        return (255, 165, 0)
    if any(w in kl for w in ("normal", "clean", "intact")):
        return (0, 128, 255)
    if kl.startswith("mask:"):
        if "msad" in kl:
            return (255, 165, 0)
        if "fused" in kl:
            return (0, 128, 255)
        if any(w in kl for w in ("anomaly", "damaged", "defect", "flaw")):
            return (255, 165, 0)
        if any(w in kl for w in ("normal", "clean", "intact")):
            return (0, 128, 255)
    h = hashlib.sha1(k.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(palette)
    return palette[idx]


def _alpha_map_from_mask(mask: np.ndarray, alpha_center: float, alpha_edge: float, power: float = 1.0, blur_sigma: float = 0.8) -> np.ndarray:
    """Create an alpha map where the interior is more transparent and edges are more opaque."""
    if mask.sum() == 0:
        return np.zeros(mask.shape, dtype=np.float32)

    dist = distance_transform_edt(mask.astype(np.uint8)).astype(np.float32)
    dmax = float(dist.max())
    if dmax <= 1e-6:
        alpha = np.zeros_like(dist, dtype=np.float32)
        alpha[mask] = float(alpha_edge)
        return alpha

    dist_n = dist / dmax
    alpha_in = float(alpha_center) + (float(alpha_edge) - float(alpha_center)) * ((1.0 - dist_n) ** float(power))
    alpha = np.zeros_like(dist, dtype=np.float32)
    alpha[mask] = alpha_in[mask]

    if blur_sigma and blur_sigma > 0:
        alpha = gaussian_filter(alpha, sigma=float(blur_sigma))
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha.astype(np.float32)


def draw_masks_to_frame(
    frame: np.ndarray,
    masks: np.ndarray,
    colors: np.ndarray,
    alpha_center: float = 0.25,
    alpha_edge: float = 0.80,
    power: float = 1.0,
    blur_sigma: float = 0.8,
) -> np.ndarray:
    """Draw masks with center-light, edge-dark overlay."""
    if masks is None or len(masks) == 0:
        return frame
    out = frame.astype(np.float32)

    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        if m.sum() == 0:
            continue
        col = colors[i].astype(np.float32)
        a = _alpha_map_from_mask(m, alpha_center=alpha_center, alpha_edge=alpha_edge, power=power, blur_sigma=blur_sigma)
        if a.max() <= 0:
            continue
        out = out * (1.0 - a[..., None]) + col[None, None, :] * a[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- Metrics Computation ----------
def safe_binary_metrics(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6):
    """
    Compute per-sample binary metrics.
    Returns TP, FP, FN, TN and derived metrics.
    """
    if pred_bin.dim() == 2:
        pred_bin = pred_bin.unsqueeze(0)
        gt_bin = gt_bin.unsqueeze(0)
    
    pred_bin = pred_bin.float()
    gt_bin = gt_bin.float()
    
    TP = (pred_bin * gt_bin).sum(dim=(1, 2)).float()
    FP = ((pred_bin == 1) & (gt_bin == 0)).sum(dim=(1, 2)).float()
    FN = ((pred_bin == 0) & (gt_bin == 1)).sum(dim=(1, 2)).float()
    TN = ((pred_bin == 0) & (gt_bin == 0)).sum(dim=(1, 2)).float()
    
    union = TP + FP + FN
    iou = (TP / (union + eps)).cpu().numpy()
    dice = (2 * TP / (2 * TP + FP + FN + eps)).cpu().numpy()
    precision = (TP / (TP + FP + eps)).cpu().numpy()
    recall = (TP / (TP + FN + eps)).cpu().numpy()
    f1 = (2 * precision * recall / (precision + recall + 1e-12))
    
    # Total pixels
    total_pixels = (pred_bin.shape[1] * pred_bin.shape[2])
    pred_area_ratio = ((TP + FP) / total_pixels).cpu().numpy()  # predicted mask area ratio
    
    return {
        "TP": TP.cpu().numpy(),
        "FP": FP.cpu().numpy(),
        "FN": FN.cpu().numpy(),
        "TN": TN.cpu().numpy(),
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_area_ratio": pred_area_ratio,
    }


def compute_biou(pred, gt, dilation_ratio=0.02, eps=1e-6):
    """
    Boundary IoU (mBIoU).
    pred, gt: torch.Tensor or np.ndarray, shape (H, W)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    pred = pred > 0
    gt = gt > 0

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    h, w = pred.shape
    diag = np.sqrt(h * h + w * w)
    r = max(1, int(round(dilation_ratio * diag)))

    # boundary = mask XOR eroded(mask)
    struct = np.ones((3, 3), dtype=bool)
    pred_er = binary_erosion(pred, structure=struct, iterations=r)
    gt_er = binary_erosion(gt, structure=struct, iterations=r)

    pred_b = pred ^ pred_er
    gt_b = gt ^ gt_er

    # tolerant boundary band
    pred_band = binary_dilation(pred_b, structure=struct, iterations=r)
    gt_band = binary_dilation(gt_b, structure=struct, iterations=r)

    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()

    return float(inter) / (float(union) + eps)


class MetricsAccumulator:
    """
    Accumulator for computing metrics with proper separation of anomaly/normal samples.
    
    指标说明:
    =========
    1. 分割指标 (仅 anomaly 样本):
       - Dice, IoU, Precision, Recall, F1
       - mIoU: mean IoU across classes (macro average)
       - mBIoU: mean Boundary IoU
       - Micro average: 累加 TP/FP/FN 后计算
       - Macro average: 每个样本算指标后平均
    
    2. 假阳性指标 (仅 normal 样本):
       - FPR: 预测为异常的 normal 样本比例 (image-level)
       - Pixel FPR: normal 样本中被预测为异常的像素比例
       - Mean pred area: normal 样本上预测 mask 的平均面积
    
    3. 图像级指标 (全集):
       - Image-AUC: 基于 anomaly score 的图像级 AUC
       - Pixel-AUC: 基于像素级预测的 AUC
    """
    
    def __init__(self):
        # === Anomaly-only segmentation metrics ===
        self.anomaly_stats = {
            "TP_sum": 0.0, "FP_sum": 0.0, "FN_sum": 0.0,
            "iou_sum": 0.0, "dice_sum": 0.0, 
            "prec_sum": 0.0, "rec_sum": 0.0, "f1_sum": 0.0,
            "biou_sum": 0.0,
            "count": 0,
        }
        
        # Per-class stats for anomaly samples (for mIoU)
        self.anomaly_per_class = defaultdict(lambda: {
            "TP_sum": 0.0, "FP_sum": 0.0, "FN_sum": 0.0,
            "iou_sum": 0.0, "dice_sum": 0.0, 
            "prec_sum": 0.0, "rec_sum": 0.0, "f1_sum": 0.0,
            "biou_sum": 0.0,
            "count": 0,
        })
        
        # === Normal-only FPR metrics ===
        self.normal_stats = {
            "total_pixels": 0,
            "fp_pixels": 0,  # pixels incorrectly predicted as defect
            "images_with_fp": 0,  # images with any FP prediction
            "pred_area_sum": 0.0,  # sum of predicted mask area ratios
            "count": 0,
        }
        
        self.normal_per_class = defaultdict(lambda: {
            "total_pixels": 0,
            "fp_pixels": 0,
            "images_with_fp": 0,
            "pred_area_sum": 0.0,
            "count": 0,
        })
        
        # === Image-level AUC data ===
        self.image_labels = []  # 1 for anomaly, 0 for normal
        self.image_scores = []  # anomaly scores
        
        # === Pixel-level AUC data ===
        self.pixel_preds = []  # flattened predictions (subsample for memory)
        self.pixel_labels = []  # flattened GT labels
        self.pixel_subsample_rate = 0.01  # subsample to avoid OOM
        
    def update_anomaly(self, metrics: dict, biou: float, cls_name: str, anomaly_score: float = None):
        """Update stats for an anomaly sample."""
        n = len(metrics["iou"])
        for i in range(n):
            # Global stats
            self.anomaly_stats["TP_sum"] += metrics["TP"][i]
            self.anomaly_stats["FP_sum"] += metrics["FP"][i]
            self.anomaly_stats["FN_sum"] += metrics["FN"][i]
            self.anomaly_stats["iou_sum"] += metrics["iou"][i]
            self.anomaly_stats["dice_sum"] += metrics["dice"][i]
            self.anomaly_stats["prec_sum"] += metrics["precision"][i]
            self.anomaly_stats["rec_sum"] += metrics["recall"][i]
            self.anomaly_stats["f1_sum"] += metrics["f1"][i]
            self.anomaly_stats["biou_sum"] += float(biou)
            self.anomaly_stats["count"] += 1
            
            # Per-class stats
            cls_stats = self.anomaly_per_class[cls_name]
            cls_stats["TP_sum"] += metrics["TP"][i]
            cls_stats["FP_sum"] += metrics["FP"][i]
            cls_stats["FN_sum"] += metrics["FN"][i]
            cls_stats["iou_sum"] += metrics["iou"][i]
            cls_stats["dice_sum"] += metrics["dice"][i]
            cls_stats["prec_sum"] += metrics["precision"][i]
            cls_stats["rec_sum"] += metrics["recall"][i]
            cls_stats["f1_sum"] += metrics["f1"][i]
            cls_stats["biou_sum"] += float(biou)
            cls_stats["count"] += 1
        
        # Image-level AUC
        if anomaly_score is not None:
            self.image_labels.append(1)
            self.image_scores.append(anomaly_score)
    
    def update_normal(self, metrics: dict, cls_name: str, total_pixels: int,
                      anomaly_score: float = None):
        """Update stats for a normal/good sample."""
        n = len(metrics["FP"])
        for i in range(n):
            fp = metrics["FP"][i]
            pred_area = metrics["pred_area_ratio"][i]
            
            # Global stats
            self.normal_stats["total_pixels"] += total_pixels
            self.normal_stats["fp_pixels"] += int(fp)
            if fp > 0:
                self.normal_stats["images_with_fp"] += 1
            self.normal_stats["pred_area_sum"] += pred_area
            self.normal_stats["count"] += 1
            
            # Per-class stats
            cls_stats = self.normal_per_class[cls_name]
            cls_stats["total_pixels"] += total_pixels
            cls_stats["fp_pixels"] += int(fp)
            if fp > 0:
                cls_stats["images_with_fp"] += 1
            cls_stats["pred_area_sum"] += pred_area
            cls_stats["count"] += 1
        
        # Image-level AUC
        if anomaly_score is not None:
            self.image_labels.append(0)
            self.image_scores.append(anomaly_score)
    
    def update_pixel_auc(self, pred_prob_map: np.ndarray, gt_map: np.ndarray):
        """Update pixel-level AUC data (subsampled)."""
        pred_flat = pred_prob_map.flatten()
        gt_flat = gt_map.flatten()
        
        # Subsample
        n = len(pred_flat)
        k = max(1, int(n * self.pixel_subsample_rate))
        indices = np.random.choice(n, k, replace=False)
        
        self.pixel_preds.extend(pred_flat[indices].tolist())
        self.pixel_labels.extend(gt_flat[indices].tolist())
    
    def compute_summary(self) -> dict:
        """Compute final summary metrics."""
        eps = 1e-6
        results = {}
        
        # === Anomaly metrics ===
        a = self.anomaly_stats
        count = max(1, a["count"])
        
        # Macro average
        results["anomaly_macro"] = {
            "count": a["count"],
            "dice": a["dice_sum"] / count,
            "iou": a["iou_sum"] / count,
            "precision": a["prec_sum"] / count,
            "recall": a["rec_sum"] / count,
            "f1": a["f1_sum"] / count,
            "biou": a["biou_sum"] / count,
        }
        
        # mIoU (per-class then average)
        class_ious = []
        class_bious = []
        for cls_name, cls_stats in self.anomaly_per_class.items():
            if cls_stats["count"] > 0:
                class_ious.append(cls_stats["iou_sum"] / cls_stats["count"])
                class_bious.append(cls_stats["biou_sum"] / cls_stats["count"])
        results["anomaly_macro"]["mIoU"] = np.mean(class_ious) if class_ious else 0.0
        results["anomaly_macro"]["mBIoU"] = np.mean(class_bious) if class_bious else 0.0
        
        # Micro average
        TP, FP, FN = a["TP_sum"], a["FP_sum"], a["FN_sum"]
        results["anomaly_micro"] = {
            "dice": 2 * TP / (2 * TP + FP + FN + eps),
            "iou": TP / (TP + FP + FN + eps),
            "precision": TP / (TP + FP + eps),
            "recall": TP / (TP + FN + eps),
            "f1": 2 * TP / (2 * TP + FP + FN + eps),
        }
        
        # Per-class anomaly metrics
        results["anomaly_per_class"] = {}
        for cls_name, cls_stats in self.anomaly_per_class.items():
            c = max(1, cls_stats["count"])
            results["anomaly_per_class"][cls_name] = {
                "count": cls_stats["count"],
                "dice": cls_stats["dice_sum"] / c,
                "iou": cls_stats["iou_sum"] / c,
                "precision": cls_stats["prec_sum"] / c,
                "recall": cls_stats["rec_sum"] / c,
            }
        
        # === Normal FPR metrics ===
        n = self.normal_stats
        n_count = max(1, n["count"])
        results["normal_fpr"] = {
            "count": n["count"],
            "image_fpr": n["images_with_fp"] / n_count,
            "pixel_fpr": n["fp_pixels"] / max(1, n["total_pixels"]),
            "mean_pred_area": n["pred_area_sum"] / n_count,
        }
        
        # Per-class normal FPR
        results["normal_per_class"] = {}
        for cls_name, cls_stats in self.normal_per_class.items():
            c = max(1, cls_stats["count"])
            results["normal_per_class"][cls_name] = {
                "count": cls_stats["count"],
                "image_fpr": cls_stats["images_with_fp"] / c,
                "pixel_fpr": cls_stats["fp_pixels"] / max(1, cls_stats["total_pixels"]),
                "mean_pred_area": cls_stats["pred_area_sum"] / c,
            }
        
        # === AUC metrics ===
        results["image_auc"] = np.nan
        results["pixel_auc"] = np.nan
        
        if HAS_SKLEARN and len(self.image_labels) > 0:
            labels = np.array(self.image_labels)
            scores = np.array(self.image_scores)
            if len(np.unique(labels)) > 1:
                try:
                    results["image_auc"] = roc_auc_score(labels, scores)
                except:
                    pass
        
        if HAS_SKLEARN and len(self.pixel_labels) > 0:
            labels = np.array(self.pixel_labels)
            preds = np.array(self.pixel_preds)
            if len(np.unique(labels)) > 1:
                try:
                    results["pixel_auc"] = roc_auc_score(labels, preds)
                except:
                    pass
        
        return results


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _infer_class_list(meta: dict) -> List[str]:
    """从 meta.json 推断类名列表，兼容多种结构"""
    cls_set = set()
    samples = meta.get("samples")
    data_list = meta.get("data")
    if isinstance(samples, list):
        for entry in samples:
            if isinstance(entry, dict):
                cn = entry.get("cls_name")
                if cn:
                    cls_set.add(cn)
    if isinstance(data_list, list):
        for entry in data_list:
            if isinstance(entry, dict):
                cn = entry.get("cls_name")
                if cn:
                    cls_set.add(cn)
    if not cls_set:
        for split_name in ("train", "test", "val", "validation"):
            split_meta = meta.get(split_name, {})
            if isinstance(split_meta, dict):
                for cls_name in split_meta.keys():
                    if cls_name:
                        cls_set.add(cls_name)
    return sorted(list(cls_set))


def build_loader(data_root: str, meta_path: str, mode: str, batch_size: int,
                 include_test_defects: bool = False,
                 train_from_test: bool = False, specie_split_ratio: float = 0.8,
                 specie_split_seed: int = 42, save_dir: Optional[str] = None,
                 dataset_type: str = "mvtec", obj_name: Optional[str] = None,
                 prompt_mode: str = "simple",
                 visa_missing_mask_behavior: str = "error"):

    if dataset_type.lower() == "visa":
        visa_kwargs = dict(
            root=data_root,
            csv_path=meta_path,
            mode=mode,
            obj_name=obj_name,
            prompt_mode=prompt_mode,
            missing_mask_behavior=visa_missing_mask_behavior,
            # 注意：不要再传 train_from_test/specie_split_ratio/save_dir 等 VisADataset 不支持的参数
        )
        ds = VisADataset(**_filter_kwargs_for_callable(VisADataset.__init__, visa_kwargs))
    else:
        mvtec_kwargs = dict(
            root=data_root,
            meta_path=meta_path,
            mode=mode,
            include_test_defects=include_test_defects,
            train_from_test=train_from_test,
            specie_split_ratio=specie_split_ratio,
            specie_split_seed=specie_split_seed,
            save_dir=save_dir,
            obj_name=obj_name,
            prompt_mode=prompt_mode,
        )
        ds = MVTecMetaDataset(**_filter_kwargs_for_callable(MVTecMetaDataset.__init__, mvtec_kwargs))

    ...


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


def _forward_once(model, images: torch.Tensor, prompt_lists: List[List[str]], class_names: List[str],
                  masks_size: Tuple[int, int], device: torch.device, upsample: bool = True,
                  use_msad_output: bool = False,
                  msad_mask_alpha: float = 0.0,
                  msad_score_thresh: Optional[float] = None):
    """Run one forward pass and return (pred_masks_prob[B,Q,H,W], query_scores[B,Q], msad_candidate, raw_out).
    
    Args:
        use_msad_output: 是否使用MSAD的anomaly_score输出
        msad_mask_alpha: MSAD与SAM3 mask的融合比例 (0=纯SAM3, 1=纯MSAD)
    """
    out = model(images, prompt_lists, class_names)

    pred_masks = out["pred_masks"]
    if pred_masks.dim() == 5:
        pred_masks = pred_masks[-1]
    pred_masks = torch.sigmoid(pred_masks)

    if upsample and pred_masks.shape[-2:] != masks_size:
        pred_masks = F.interpolate(pred_masks, size=masks_size, mode="bilinear", align_corners=False)

    B, Q = pred_masks.shape[0], pred_masks.shape[1]
    
    # 先计算原始query_scores（用于选择最可信的query）
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
        query_scores = None
    
    # ===== MSAD候选（不修改任何query mask；作为额外候选参与top-k） =====
    msad_map_single = None  # (B, 1, H, W) MSAD单通道异常图
    msad_score_b = None     # (B,) 用于候选排序的图像级分数
    
    if use_msad_output and msad_mask_alpha > 0:
        msad_score = out.get("msad_anomaly_score", None)  # (B, H, W)
        msad_agg = out.get("msad_aggregated_map", None)  # (B, 2, H, W)
        
        if msad_score is not None:
            # 使用anomaly_score直接
            msad_map_single = msad_score.unsqueeze(1)  # (B, 1, H, W)
        elif msad_agg is not None:
            # 使用aggregated_map的abnormal通道
            msad_map_single = msad_agg[:, 1:2, :, :]  # (B, 1, H, W)
        
        if msad_map_single is not None:
            pred_mask_size = pred_masks.shape[-2:]
            if msad_map_single.shape[-2:] != pred_mask_size:
                msad_map_single = F.interpolate(msad_map_single, size=pred_mask_size, mode="bilinear", align_corners=False)

            if msad_score_thresh is not None:
                t = float(msad_score_thresh)
                t = max(0.0, min(t, 0.999999))
                msad_map_single = ((msad_map_single - t) / max(1e-6, 1.0 - t)).clamp(0.0, 1.0)

            with torch.no_grad():
                msad_score_b = torch.quantile(msad_map_single.flatten(2), 0.95, dim=2).squeeze(1)

            if query_scores is None:
                query_scores = torch.zeros((B, Q), device=pred_masks.device, dtype=pred_masks.dtype)
            with torch.no_grad():
                qs_msad = (pred_masks * msad_map_single).flatten(2).mean(dim=2)
                query_scores = torch.maximum(query_scores, qs_msad)

    msad_candidate = {"map": msad_map_single, "score": msad_score_b, "alpha": float(msad_mask_alpha)}
    return pred_masks, query_scores, msad_candidate, out


def _select_candidates(pm_b: torch.Tensor, scores_vec: Optional[torch.Tensor], prompt_text: str,
                      conf_thresh: float, top_k: int, mask_thresh: float,
                      extra_candidates: Optional[List[Tuple[float, np.ndarray, str]]] = None,
                      msad_mask_thresh: Optional[float] = None,
                      msad_mask_top_p: float = 0.0):
    """Select top-k candidate masks for ONE image under ONE prompt."""
    Q = pm_b.shape[0]
    masks_list = []
    scores = []

    for q in range(Q):
        m_q = pm_b[q].detach().cpu().numpy().astype(np.float32)
        masks_list.append(m_q)
        if scores_vec is not None and q < scores_vec.numel():
            s_q = float(scores_vec[q].detach().cpu().item())
        else:
            s_q = float(np.mean(m_q))
        scores.append(s_q)

    cand = [(s, m, prompt_text) for s, m in zip(scores, masks_list) if s >= conf_thresh]
    if extra_candidates:
        cand.extend(extra_candidates)
    cand.sort(key=lambda x: x[0], reverse=True)
    if top_k > 0:
        cand = cand[:top_k]

    if cand:
        msad_thr = float(mask_thresh if msad_mask_thresh is None else msad_mask_thresh)
        msad_top_p = float(msad_mask_top_p or 0.0)
        bin_list = []
        for s, m, name in cand:
            if name == "MSAD" and msad_top_p > 0.0:
                thr = float(np.quantile(m.reshape(-1), max(0.0, min(0.999999, 1.0 - msad_top_p))))
                bin_list.append(m > thr)
            elif name == "MSAD":
                bin_list.append(m > msad_thr)
            else:
                bin_list.append(m > mask_thresh)
        masks_stack = np.stack(bin_list).astype(bool)
        pm_comb = torch.from_numpy(masks_stack.max(axis=0).astype(np.float32)).to(pm_b.device)
        pred_prob_map = np.maximum.reduce([c[1] for c in cand]).astype(np.float32)
        max_score = float(cand[0][0])
    else:
        H, W = pm_b.shape[-2], pm_b.shape[-1]
        pm_comb = torch.zeros((H, W), dtype=torch.float32, device=pm_b.device)
        pred_prob_map = np.zeros((H, W), dtype=np.float32)
        max_score = 0.0

    return cand, pm_comb, pred_prob_map, max_score


def load_model(args, device: torch.device):
    # 根据数据集类型推断 class_list
    dataset_type = getattr(args, "dataset", "mvtec").lower()
    
    if dataset_type == "visa":
        # VisA: 从 CSV 文件中提取类名
        import csv
        meta_path = args.meta_path or os.path.join(args.data_root, "split_csv", "1cls.csv")
        class_set = set()
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = row.get("object", "").strip()
                if obj:
                    class_set.add(obj)
        class_list = sorted(list(class_set))
        if not class_list:
            raise ValueError(f"Could not infer class_list from CSV file: {meta_path}")
        print(f"[INFO] VisA classes: {class_list}")
    else:
        # MVTec-AD: 从 meta.json 中提取类名
        meta_path = args.meta_path or os.path.join(args.data_root, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        class_list = _infer_class_list(meta)
        if not class_list:
            raise ValueError(f"Could not infer class_list from meta.json: {meta_path}")
    
    if getattr(args, "class_agnostic", False):
        agnostic_name = getattr(args, "agnostic_name", None) or getattr(args, "obj_name", None) or "object"
        class_list = [agnostic_name]
    
    args.class_list = class_list
    args.meta_path = meta_path  # 确保 args.meta_path 被设置

    prof = str(getattr(args, "run_profile", "custom")).lower()
    if prof == "zero_shot":
        args.use_msad_output = True
        args.msad_mask_alpha = 1.0
        args.msad_score_thresh = None
        args.msad_use_vision_adapter = True
        args.conf_thresh = float(getattr(args, "conf_thresh", 0.1) or 0.1)
        args.top_k = int(getattr(args, "top_k", 1) or 1)
        args.mask_thresh = float(getattr(args, "mask_thresh", 0.5) or 0.5)
        args.disable_spurious_gating = True
        args.disable_dap = True
        args.disable_dap_spurious_filter = True
        args.prompt_learner_type = "compound"
        args.compound_use_text_encoder = True
        args.prompt_mode = "simple"
    elif prof in ("few_shot", "few_shot_full", "few_shot_no_w"):
        args.use_msad_output = bool(getattr(args, "use_msad_output", False))
        if getattr(args, "prompt_learner_type", "perclass") == "perclass":
            args.prompt_learner_type = "compound"
        if not getattr(args, "disable_lora", False):
            args.disable_lora = True
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
        if prof == "few_shot_no_w":
            args.compound_disable_w = True
        if getattr(args, "prompt_learner_type", "") == "compound":
            args.compound_use_text_encoder = True
            args.compound_abnormal_word = str(getattr(args, "compound_abnormal_word", "anomaly") or "anomaly")
            args.compound_pooling = str(getattr(args, "compound_pooling", "ctx_only") or "ctx_only")
        if not hasattr(args, "msad_mask_alpha"):
            args.msad_mask_alpha = 0.0

    ckpt_path = getattr(args, "ckpt", None)
    if ckpt_path and os.path.exists(str(ckpt_path)):
        try:
            raw = torch.load(str(ckpt_path), map_location="cpu")
            state = raw.get("model", raw) if isinstance(raw, dict) else raw
            keys = list(state.keys()) if isinstance(state, dict) else []
            has_out_adapter = any(".out_adapter." in k for k in keys)
            if has_out_adapter and (not bool(getattr(args, "enable_out_adapter_lora", False))):
                args.enable_out_adapter_lora = True
            if has_out_adapter and bool(getattr(args, "enable_parallel_lora", False)):
                args.enable_parallel_lora = False
        except Exception:
            pass

    spurious_prompt_set = None
    if getattr(args, "spurious_prompt_set_file", None):
        path = str(args.spurious_prompt_set_file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                spurious_prompt_set = [ln.strip() for ln in f.readlines() if ln.strip()]
    if spurious_prompt_set is None and getattr(args, "spurious_prompt_set", None):
        spurious_prompt_set = [s.strip() for s in str(args.spurious_prompt_set).split(",") if s.strip()]

    common_kwargs = {
        "bpe_path": getattr(args, "bpe_path", None),
        "sam3_ckpt": getattr(args, "sam3_ckpt", None),
        "enable_lora": not getattr(args, "disable_lora", False),
        "lora_rank": getattr(args, "lora_rank", 16),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "enable_parallel_lora": getattr(args, "enable_parallel_lora", False),
        "parallel_lora_rank": getattr(args, "parallel_lora_rank", 16),
        "parallel_lora_alpha": getattr(args, "parallel_lora_alpha", None),
        "parallel_lora_target": getattr(args, "parallel_lora_target", "qv_only"),
        "parallel_lora_layer_ids": getattr(args, "parallel_lora_layer_ids", None),
        "enable_out_adapter_lora": getattr(args, "enable_out_adapter_lora", False),
        "lora_layer_ids": getattr(args, "lora_layer_ids", None),
        "freeze_vision": getattr(args, "freeze_vision", False),
        "freeze_text": getattr(args, "freeze_text", False),
        "device": device,
    }

    Constructor = FineTuneSAM3Official if args.use_official else FineTuneSAM3
    ctor_kwargs = _filter_kwargs_for_callable(Constructor, common_kwargs)
    ctor_kwargs.update({
        "class_list": args.class_list,
        "prompt_learner_type": getattr(args, "prompt_learner_type", "perclass"),
        "num_templates": getattr(args, "num_templates", 4),
        "n_ctx": getattr(args, "n_ctx", 4),
        # CoOp/CoCoOp 参数
        "ctx_init": getattr(args, "ctx_init", ""),
        "class_token_position": getattr(args, "class_token_position", "end"),
        "use_keywords": getattr(args, "use_keywords", False),
        "cocoop_vis_dim": getattr(args, "cocoop_vis_dim", 256),
        "cocoop_reduction": getattr(args, "cocoop_reduction", 16),
        # Compound Prompt Learning 参数
        "compound_mode": getattr(args, "compound_mode", "cocoop"),
        "compound_n_ctx": getattr(args, "compound_n_ctx", 4),
        "compound_n_ctx_offset": getattr(args, "compound_n_ctx_offset", 4),
        "compound_num_abnormal": getattr(args, "compound_num_abnormal", 10),
        "compound_enable_dap": bool(getattr(args, "compound_enable_dap", False)) and (not bool(getattr(args, "disable_dap", False))),
        "compound_dap_top_k": getattr(args, "compound_dap_top_k", 10),
        "compound_meta_reduction": getattr(args, "compound_meta_reduction", 16),
        "compound_dap_use_multilevel": getattr(args, "compound_dap_use_multilevel", False),
        "compound_dap_num_levels": getattr(args, "compound_dap_num_levels", 0),
        "compound_use_text_encoder": getattr(args, "compound_use_text_encoder", False),
        "compound_abnormal_word": getattr(args, "compound_abnormal_word", "anomaly"),
        "compound_pooling": getattr(args, "compound_pooling", "ctx_only"),
        "compound_abnormal_order": getattr(args, "compound_abnormal_order", "v_then_wk"),
        "compound_dap_spurious_filter": bool(getattr(args, "compound_dap_spurious_filter", False)) and (not bool(getattr(args, "disable_dap_spurious_filter", False))),
        "compound_dap_spurious_alpha": getattr(args, "compound_dap_spurious_alpha", 1.0),
        "compound_disable_w": getattr(args, "compound_disable_w", False),
        # 多尺度特征
        "num_feature_levels": getattr(args, "num_feature_levels", 4),
        # MSAD 参数
        "enable_msad": getattr(args, "enable_msad", False),
        "msad_use_shape_attention": getattr(args, "msad_use_shape_attention", True),
        "msad_learnable_level_weights": getattr(args, "msad_learnable_level_weights", True),
        "msad_learnable_temperature": getattr(args, "msad_learnable_temperature", True),
        "msad_temperature": getattr(args, "msad_temperature", 100.0),
        "msad_output_size": getattr(args, "msad_output_size", 518),
        "msad_num_levels": getattr(args, "msad_num_levels", None),
        "msad_return_similarity_logits": getattr(args, "msad_return_similarity_logits", False),
        "msad_use_vision_adapter": getattr(args, "msad_use_vision_adapter", False),
        "msad_vision_adapter_reduction": getattr(args, "msad_vision_adapter_reduction", 2),
        "msad_vision_adapter_shared": (getattr(args, "msad_vision_adapter_shared", True) and (not getattr(args, "msad_vision_adapter_not_shared", False))),
        # Spurious Gating 参数
        "spurious_score_threshold": getattr(args, "spurious_score_threshold", 0.20),
        "spurious_topk_ratio": getattr(args, "spurious_topk_ratio", 0.02),
        "spurious_prompt_set": spurious_prompt_set,
    })
    model = Constructor(**ctor_kwargs).to(device).eval()

    # 加载 checkpoint
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[INFO] Loading fine-tuned checkpoint {args.ckpt} ...")
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            # 打印checkpoint中保存的配置（如果有）
            if "config" in ckpt:
                print(f"[INFO] Checkpoint config: {ckpt['config']}")
            if "args" in ckpt:
                saved_args = ckpt["args"]
                print(f"[INFO] Checkpoint was trained with:")
                for key in ["enable_msad", "msad_use_shape_attention", 
                           "enable_spurious_gating", "num_feature_levels"]:
                    if hasattr(saved_args, key):
                        print(f"       --{key} = {getattr(saved_args, key)}")
        else:
            state = ckpt
        
        # 过滤形状不匹配的权重
        model_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        
        for k, v in state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    skipped_keys.append(f"{k}: ckpt={v.shape} vs model={model_state[k].shape}")
            else:
                # 允许额外的键
                filtered_state[k] = v

        if skipped_keys and getattr(args, "prompt_learner_type", "") == "compound":
            inferred = {}
            if isinstance(state, dict):
                if "prompt_learner.V" in state:
                    inferred["compound_n_ctx"] = int(state["prompt_learner.V"].shape[0])
                if "prompt_learner.w" in state:
                    inferred["compound_n_ctx_offset"] = int(state["prompt_learner.w"].shape[0])
                if "prompt_learner.W" in state:
                    inferred["compound_num_abnormal"] = int(state["prompt_learner.W"].shape[0])
                    if "compound_n_ctx_offset" not in inferred:
                        inferred["compound_n_ctx_offset"] = int(state["prompt_learner.W"].shape[1])

            need_rebuild = False
            for kk, vv in inferred.items():
                if kk in ctor_kwargs and ctor_kwargs[kk] != vv:
                    need_rebuild = True
                    break

            if need_rebuild:
                print("[WARN] Detected prompt_learner shape mismatch; rebuilding model to match checkpoint prompt dims:")
                for kk, vv in inferred.items():
                    if kk in ctor_kwargs and ctor_kwargs[kk] != vv:
                        print(f"       {kk}: test={ctor_kwargs[kk]} -> ckpt={vv}")
                        ctor_kwargs[kk] = vv
                        if hasattr(args, kk):
                            setattr(args, kk, vv)
                model = Constructor(**ctor_kwargs).to(device).eval()
                model_state = model.state_dict()
                filtered_state = {}
                skipped_keys = []
                for k, v in state.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state[k] = v
                        else:
                            skipped_keys.append(f"{k}: ckpt={v.shape} vs model={model_state[k].shape}")
                    else:
                        filtered_state[k] = v
        
        if skipped_keys:
            print(f"[WARN] Skipped {len(skipped_keys)} mismatched weights (cross-dataset inference):")
            for sk in skipped_keys[:5]:
                print(f"       {sk}")
            if len(skipped_keys) > 5:
                print(f"       ... and {len(skipped_keys) - 5} more")
        
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"[INFO] Loaded fine-tuned weights. missing={len(missing)} unexpected={len(unexpected)}")
        
        # 打印详细的 missing/unexpected keys
        if len(missing) > 0:
            print(f"[DEBUG] Missing keys (first 10):")
            for mk in missing[:10]:
                print(f"       - {mk}")
            if len(missing) > 10:
                print(f"       ... and {len(missing) - 10} more")
        
        if len(unexpected) > 0:
            print(f"[DEBUG] Unexpected keys (first 10):")
            for uk in unexpected[:10]:
                print(f"       - {uk}")
            if len(unexpected) > 10:
                print(f"       ... and {len(unexpected) - 10} more")
        
        # 配置验证
        if "config" in ckpt:
            train_config = ckpt["config"]
            test_config = {
                "enable_msad": getattr(args, "enable_msad", False),
                "enable_spurious_gating": getattr(args, "enable_spurious_gating", False),
            }
            
            mismatches = []
            for key in test_config:
                if key in train_config and train_config[key] != test_config[key]:
                    mismatches.append(f"  {key}: train={train_config[key]}, test={test_config[key]}")
            
            if mismatches:
                print("\n" + "="*60)
                print("[WARNING] Configuration MISMATCH between training and testing!")
                print("="*60)
                for m in mismatches:
                    print(m)
                print("="*60)
                print("This may cause missing/unexpected keys and poor performance!")
                print("Please ensure test config matches training config.")
                print("="*60 + "\n")
    else:
        print("[INFO] No fine-tuned checkpoint provided. Using base SAM3 weights.")

    return model


def _get_sample_id_from_meta(ds, global_idx: int, fallback_cls: str, fallback_specie: str):
    """
    Try to recover the meta.json entry for naming.
    """
    meta_entry = None
    for attr in ("entries", "items", "meta_list", "data_list", "samples", "records", "metas", "data"):
        v = getattr(ds, attr, None)
        if isinstance(v, list) and global_idx < len(v):
            meta_entry = v[global_idx]
            break

    cls_name = fallback_cls
    specie_name = fallback_specie
    stem = f"{global_idx:03d}"
    ext = ".png"

    if meta_entry is not None:
        if isinstance(meta_entry, dict):
            cls_name = meta_entry.get("cls_name", cls_name)
            specie_name = meta_entry.get("specie_name", specie_name)
            img_path = meta_entry.get("img_path") or meta_entry.get("image_path") or meta_entry.get("img")
        else:
            cls_name = getattr(meta_entry, "cls_name", cls_name)
            specie_name = getattr(meta_entry, "specie_name", specie_name)
            img_path = getattr(meta_entry, "img_path", None)
        
        if isinstance(img_path, str) and img_path:
            base = os.path.basename(img_path)
            stem2, ext2 = os.path.splitext(base)
            if stem2:
                stem = stem2
            if ext2:
                ext = ext2

    if not specie_name:
        specie_name = "unknown"
    if not cls_name:
        cls_name = "unknown"
    return specie_name, cls_name, stem, ext


def visualize_msad_output(out: dict, images: torch.Tensor, save_dir: str, dataset, dataset_pos: int,
                          fallback_class_names: List[str], fallback_specie_names: List[str]):
    """可视化MSAD输出"""
    msad_score = out.get("msad_anomaly_score", None)
    msad_agg = out.get("msad_aggregated_map", None)
    spurious_map = out.get("spurious_map", None)
    eta_spurious = out.get("eta_spurious", None)
    
    if msad_score is None and msad_agg is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    B = images.shape[0]
    for b in range(B):
        global_i = dataset_pos + b
        specie_f = fallback_specie_names[b] if b < len(fallback_specie_names) else ""
        cls_f = fallback_class_names[b] if b < len(fallback_class_names) else "unknown"
        specie_n, cls_n, stem, ext = _get_sample_id_from_meta(dataset, global_i, cls_f, specie_f)
        fig_dir = os.path.join(save_dir, cls_n)
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"{specie_n}_{cls_n}_{stem}{ext}")
        
        # 计算子图数量
        n_plots = 2  # 原图 + MSAD异常图
        if spurious_map is not None:
            n_plots += 1
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        
        # 1. 原图
        img_np = images[b].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        # 2. MSAD异常图
        if msad_score is not None:
            score_map = msad_score[b].cpu().numpy()
        elif msad_agg is not None:
            score_map = msad_agg[b, 1].cpu().numpy()  # abnormal channel
        else:
            score_map = np.zeros((256, 256))
        
        im = axes[1].imshow(score_map, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("MSAD Anomaly Score")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Spurious响应图（如果有）
        if spurious_map is not None and n_plots > 2:
            sp_map = spurious_map[b].cpu().numpy()
            eta_val = eta_spurious[b].item() if eta_spurious is not None else 0.0
            im = axes[2].imshow(sp_map, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f"Spurious Map (η={eta_val:.2f})")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)


# ---------- main inference ----------
@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确定数据集类型和元数据路径
    dataset_type = getattr(args, "dataset", "mvtec").lower()
    if dataset_type == "visa":
        meta_path = args.meta_path or os.path.join(args.data_root, "split_csv", "1cls.csv")
    else:
        meta_path = args.meta_path or os.path.join(args.data_root, "meta.json")
    
    splits_dir = getattr(args, "splits_dir", None) or getattr(args, "save_dir", None)
    loader = build_loader(
        args.data_root,
        meta_path,
        args.mode,
        args.batch_size,
        include_test_defects=getattr(args, "include_test_defects", False),
        train_from_test=getattr(args, "train_from_test", False),
        specie_split_ratio=getattr(args, "specie_split_ratio", 0.8),
        specie_split_seed=getattr(args, "specie_split_seed", 42),
        save_dir=splits_dir,
        dataset_type=dataset_type,
        obj_name=getattr(args, "obj_name", None),
        prompt_mode=getattr(args, "prompt_mode", "simple"),
        visa_missing_mask_behavior=getattr(args, "visa_missing_mask_behavior", "error"),
    )
    model = load_model(args, device)
    
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

    msad_mask_auto_q = getattr(args, "msad_mask_auto_q", None)
    if (
        msad_mask_auto_q is not None
        and float(msad_mask_auto_q) > 0.0
        and getattr(args, "msad_mask_thresh", None) is None
        and float(getattr(args, "msad_mask_top_p", 0.0) or 0.0) <= 0.0
        and bool(getattr(args, "use_msad_output", False))
    ):
        q = float(msad_mask_auto_q)
        q = max(0.0, min(q, 0.999999))
        tmp_loader = build_loader(
            args.data_root,
            meta_path,
            args.mode,
            args.batch_size,
            include_test_defects=getattr(args, "include_test_defects", False),
            train_from_test=getattr(args, "train_from_test", False),
            specie_split_ratio=getattr(args, "specie_split_ratio", 0.8),
            specie_split_seed=getattr(args, "specie_split_seed", 42),
            save_dir=splits_dir,
            dataset_type=dataset_type,
            obj_name=getattr(args, "obj_name", None),
            prompt_mode=getattr(args, "prompt_mode", "simple"),
            visa_missing_mask_behavior=getattr(args, "visa_missing_mask_behavior", "error"),
        )
        sampled = []
        max_samples = 200000
        per_img = 2048
        for images, masks, prompt_lists, is_anomaly, class_names, specie_names in tmp_loader:
            if len(sampled) >= max_samples:
                break
            images = images.to(device)
            class_names = list(class_names)
            if getattr(args, "class_agnostic", False):
                agnostic_name = getattr(args, "agnostic_name", None) or getattr(args, "obj_name", None) or "object"
                class_names = [agnostic_name] * len(class_names)
            prompt_mode = getattr(args, "prompt_mode", "simple").lower()
            dataset_type = getattr(args, "dataset", "mvtec").lower()
            if prompt_mode == "simple":
                anomaly_lists = [[f"anomaly {class_names[i]}"] for i in range(len(class_names))]
            elif dataset_type == "visa":
                anomaly_lists = [[f"damaged {class_names[i]}"] for i in range(len(class_names))]
            else:
                anomaly_lists = [[f"anomaly {class_names[i]}"] for i in range(len(class_names))]
            _, _, msad_cand_1, _ = _forward_once(
                model,
                images,
                anomaly_lists,
                class_names,
                masks_size=masks.shape[-2:],
                device=device,
                upsample=False,
                use_msad_output=True,
                msad_mask_alpha=float(getattr(args, "msad_mask_alpha", 0.0) or 0.0),
                msad_score_thresh=getattr(args, "msad_score_thresh", None),
            )
            msad_map = msad_cand_1.get("map", None) if isinstance(msad_cand_1, dict) else None
            if not isinstance(msad_map, torch.Tensor):
                continue
            for b in range(msad_map.shape[0]):
                if bool(is_anomaly[b]):
                    continue
                flat = msad_map[b, 0].detach().flatten()
                if flat.numel() == 0:
                    continue
                k = int(min(per_img, flat.numel()))
                idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
                vals = flat[idx].detach().float().cpu().tolist()
                sampled.extend(vals)
                if len(sampled) >= max_samples:
                    break
        if sampled:
            thr = float(torch.quantile(torch.tensor(sampled, dtype=torch.float32), q).item())
            args.msad_mask_thresh = thr
            print(f"[INFO] Auto msad_mask_thresh from normal-q{q:.6f}: {thr:.6f}")

    # Initialize metrics accumulator
    metrics_acc = MetricsAccumulator()

    dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else 0
    pbar = tqdm(
        total=dataset_len,
        desc="Inference",
        unit="img",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    total_time = 0.0
    total_imgs = 0
    dataset_pos = 0
    eta_all = []
    
    # MSAD可视化目录
    msad_vis_dir = os.path.join(args.output_dir, "msad_vis") if getattr(args, "save_msad_vis", False) else None

    for images, masks, prompt_lists, is_anomaly, class_names, specie_names in loader:
        images = images.to(device)

        original_class_names = list(class_names)
        if getattr(args, "class_agnostic", False):
            agnostic_name = getattr(args, "agnostic_name", None) or getattr(args, "obj_name", None) or "object"
            class_names = [agnostic_name] * len(class_names)

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

        # MSAD输出配置
        use_msad_output = getattr(args, "use_msad_output", False)
        msad_mask_alpha = getattr(args, "msad_mask_alpha", 0.0)
        msad_score_thresh = getattr(args, "msad_score_thresh", None)
        
        if custom_prompt:
            pred_masks_1, query_scores_1, msad_cand_1, out_1 = _forward_once(
                model, images, prompt_lists, class_names, masks_size=masks.shape[-2:], device=device,
                use_msad_output=use_msad_output, 
                msad_mask_alpha=msad_mask_alpha,
                msad_score_thresh=msad_score_thresh,
            )
            pred_masks_2, query_scores_2, msad_cand_2, out_2 = None, None, None, None
        else:
            # 根据 prompt_mode 选择推理时的 prompt 格式
            prompt_mode = getattr(args, "prompt_mode", "simple").lower()
            dataset_type = getattr(args, "dataset", "mvtec").lower()
            
            if prompt_mode == "simple":
                anomaly_lists = [[f"anomaly {class_names[i]}"] for i in range(len(class_names))]
                normal_lists = [[f"normal {class_names[i]}"] for i in range(len(class_names))]
            elif dataset_type == "visa":
                anomaly_lists = [[f"damaged {class_names[i]}"] for i in range(len(class_names))]
                normal_lists = [[f"normal {class_names[i]}"] for i in range(len(class_names))]
            else:
                anomaly_lists = [[f"anomaly {class_names[i]}"] for i in range(len(class_names))]
                normal_lists = [[f"normal {class_names[i]}"] for i in range(len(class_names))]

            pred_masks_1, query_scores_1, msad_cand_1, out_1 = _forward_once(
                model, images, anomaly_lists, class_names, masks_size=masks.shape[-2:], device=device, upsample=False,
                use_msad_output=use_msad_output, 
                msad_mask_alpha=msad_mask_alpha,
                msad_score_thresh=msad_score_thresh,
            )
            pred_masks_2, query_scores_2, msad_cand_2, out_2 = _forward_once(
                model, images, normal_lists, class_names, masks_size=masks.shape[-2:], device=device, upsample=False,
                use_msad_output=use_msad_output, 
                msad_mask_alpha=msad_mask_alpha,
                msad_score_thresh=msad_score_thresh,
            )

        if bool(getattr(args, "debug_dump_features", False)) and dataset_pos == 0 and out_1 is not None:
            dump = {}
            tfs = out_1.get("text_features_structured", None)
            if isinstance(tfs, dict):
                for k, v in tfs.items():
                    if isinstance(v, torch.Tensor):
                        dump[f"tfs.{k}"] = v.detach().float().cpu().numpy()
            for k in ("eta_spurious", "msad_anomaly_score"):
                v = out_1.get(k, None)
                if isinstance(v, torch.Tensor):
                    dump[k] = v.detach().float().cpu().numpy()
            dump["class_names"] = np.array(list(original_class_names))
            dump["prompt_mode"] = np.array([getattr(args, "prompt_mode", "simple")])
            dump["compound_use_text_encoder"] = np.array([bool(getattr(args, "compound_use_text_encoder", False))])
            dump["compound_abnormal_word"] = np.array([str(getattr(args, "compound_abnormal_word", "anomaly"))])
            dump["compound_pooling"] = np.array([str(getattr(args, "compound_pooling", "ctx_only"))])
            os.makedirs(args.output_dir, exist_ok=True)
            np.savez_compressed(os.path.join(args.output_dir, "debug_features_test.npz"), **dump)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_time = t1 - t0
        total_time += batch_time

        batch_imgs = images.size(0)
        total_imgs += batch_imgs

        gt = (masks > 0.5).float().to(device)

        if bool(getattr(args, "report_eta_stats", False)) and out_1 is not None:
            v = out_1.get("eta_spurious", None)
            if isinstance(v, torch.Tensor):
                eta_all.append(v.detach().float().cpu().numpy())
        
        # 可视化MSAD输出
        if msad_vis_dir is not None and out_1 is not None:
            visualize_msad_output(
                out_1,
                images,
                msad_vis_dir,
                dataset=loader.dataset,
                dataset_pos=dataset_pos,
                fallback_class_names=original_class_names,
                fallback_specie_names=list(specie_names),
            )

        for b in range(batch_imgs):
            cls_name = class_names[b]
            is_anom = is_anomaly[b]

            gm = gt[b].squeeze(0)

            conf_thresh = float(getattr(args, "conf_thresh", 0.3))
            top_k = int(getattr(args, "top_k", 5))
            
            masks_size = tuple(gm.shape[-2:])
            
            def _max_prompt_score(pm_low: torch.Tensor, scores_vec: Optional[torch.Tensor]) -> float:
                if scores_vec is not None:
                    return float(scores_vec.view(-1).max().item())
                return float(pm_low.mean(dim=(1, 2)).max().item())
            
            if custom_prompt:
                chosen_prompt = ",".join(custom_prompt)
            
                pm_low = pred_masks_1[b]
                sv = query_scores_1[b].view(-1) if query_scores_1 is not None else None
            
                max_score = _max_prompt_score(pm_low, sv)
                alt_prompt, alt_score = "", 0.0
            
            else:
                prompt_mode = getattr(args, "prompt_mode", "simple").lower()
                dataset_type = getattr(args, "dataset", "mvtec").lower()
                
                if prompt_mode == "simple":
                    p1 = f"anomaly {cls_name}"
                    p2 = f"normal {cls_name}"
                elif dataset_type == "visa":
                    p1 = f"damaged {cls_name}"
                    p2 = f"normal {cls_name}"
                else:
                    p1 = f"anomaly {cls_name}"
                    p2 = f"normal {cls_name}"
            
                pm1_low = pred_masks_1[b]
                pm2_low = pred_masks_2[b]
            
                sv1 = query_scores_1[b].view(-1) if query_scores_1 is not None else None
                sv2 = query_scores_2[b].view(-1) if query_scores_2 is not None else None
            
                max_score1 = _max_prompt_score(pm1_low, sv1)
                max_score2 = _max_prompt_score(pm2_low, sv2)

                mask_prompt_source = str(getattr(args, "mask_prompt_source", "anomaly")).lower()
                if mask_prompt_source == "best":
                    if max_score2 > max_score1:
                        chosen_prompt, alt_prompt = p2, p1
                        max_score, alt_score = max_score2, max_score1
                        pm_low, sv = pm2_low, sv2
                        msad_cand = msad_cand_2
                    else:
                        chosen_prompt, alt_prompt = p1, p2
                        max_score, alt_score = max_score1, max_score2
                        pm_low, sv = pm1_low, sv1
                        msad_cand = msad_cand_1
                else:
                    chosen_prompt, alt_prompt = p1, p2
                    max_score, alt_score = max_score1, max_score2
                    pm_low, sv = pm1_low, sv1
                    msad_cand = msad_cand_1
            if custom_prompt:
                msad_cand = msad_cand_1
            
            pm = pm_low
            if pm.shape[-2:] != masks_size:
                pm = F.interpolate(pm.unsqueeze(0), size=masks_size, mode="bilinear", align_corners=False).squeeze(0)
            
            mask_thresh = float(getattr(args, "mask_thresh", 0.5))
            msad_mask_thresh = getattr(args, "msad_mask_thresh", None)
            msad_mask_top_p = float(getattr(args, "msad_mask_top_p", 0.0) or 0.0)
            extra = []
            if bool(use_msad_output) and isinstance(msad_cand, dict):
                msad_map = msad_cand.get("map", None)
                if isinstance(msad_map, torch.Tensor):
                    msad_map_b = msad_map[b, 0]
                    if msad_map_b.shape[-2:] != masks_size:
                        msad_map_b = F.interpolate(msad_map_b.unsqueeze(0).unsqueeze(0), size=masks_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                    msad_np = msad_map_b.detach().cpu().numpy().astype(np.float32)
                    msad_score_b = msad_cand.get("score", None)
                    if isinstance(msad_score_b, torch.Tensor):
                        msad_s = float(msad_score_b[b].detach().cpu().item())
                    else:
                        msad_s = float(np.quantile(msad_np.reshape(-1), 0.95))
                    extra.append((msad_s, msad_np, "MSAD"))

                    alpha = float(msad_cand.get("alpha", 0.0))
                    if 0.0 < alpha < 1.0:
                        if sv is not None:
                            q_best = int(sv.argmax().item())
                            best = pm[q_best].detach().cpu().numpy().astype(np.float32)
                        else:
                            best = pm.detach().cpu().numpy().astype(np.float32).mean(axis=0)
                        fused_np = ((1.0 - alpha) * best + alpha * msad_np).astype(np.float32)
                        fused_s = float(np.quantile(fused_np.reshape(-1), 0.95))
                        extra.append((fused_s, fused_np, "FUSED"))

            cand, pm_comb, pred_prob_map, max_score = _select_candidates(
                pm, sv, chosen_prompt, conf_thresh, top_k, mask_thresh,
                extra_candidates=extra,
                msad_mask_thresh=msad_mask_thresh,
                msad_mask_top_p=msad_mask_top_p,
            )

            # Compute metrics
            metrics = safe_binary_metrics(pm_comb, gm)
            biou = compute_biou(pm_comb, gm)
            
            total_pixels = int(gm.numel())
            
            if is_anom:
                metrics_acc.update_anomaly(metrics, biou, cls_name, anomaly_score=max_score)
            else:
                metrics_acc.update_normal(metrics, cls_name, total_pixels, anomaly_score=max_score)
            
            gt_map = gm.cpu().numpy()
            metrics_acc.update_pixel_auc(pred_prob_map, gt_map)

            # Visualization
            img_pil = to_pil(images[b].cpu())
            frame = np.array(img_pil.convert("RGB"), dtype=np.uint8)

            if cand:
                msad_thr = float(mask_thresh if msad_mask_thresh is None else msad_mask_thresh)
                bin_list = []
                prompts_for_color = []
                scores_for_prompt = []
                for s, m, name in cand:
                    prompts_for_color.append(name)
                    scores_for_prompt.append(s)
                    if name == "MSAD" and msad_mask_top_p > 0.0:
                        thr = float(np.quantile(m.reshape(-1), max(0.0, min(0.999999, 1.0 - msad_mask_top_p))))
                        bin_list.append(m > thr)
                    elif name == "MSAD":
                        bin_list.append(m > msad_thr)
                    else:
                        bin_list.append(m > mask_thresh)
                masks_stack = np.stack(bin_list).astype(bool)
                colors = np.array([get_color_map(palette, p) for p in prompts_for_color], dtype=np.uint8)
                frame = draw_masks_to_frame(
                    frame, masks_stack, colors,
                    alpha_center=float(getattr(args, 'mask_alpha_center', 0.25)),
                    alpha_edge=float(getattr(args, 'mask_alpha_edge', 0.80)),
                    power=float(getattr(args, 'mask_alpha_power', 1.0)),
                    blur_sigma=float(getattr(args, 'mask_alpha_blur', 0.8)),
                )
                overlay_pil = Image.fromarray(frame)

                draw = ImageDraw.Draw(overlay_pil)
                sample_type = "ANOMALY" if is_anom else "NORMAL"
                label_color = (255, 255, 255) if is_anom else (255, 165, 0)
                header_text = f"[{sample_type}] top: {chosen_prompt} ({max_score:.2f})" + (f" | alt: {alt_prompt} ({alt_score:.2f})" if alt_prompt else "")
                draw.text((5, 5), header_text, font=font, fill=label_color)
                row_h, box_w, pad = 12, 10, 4
                legend_mode = str(getattr(args, "vis_legend_mode", "auto")).lower()
                if legend_mode == "auto":
                    legend_mode = "both" if str(getattr(args, "run_profile", "custom")).lower() == "zero_shot" else "candidates"

                legend_entries = []
                if legend_mode in ("prompts", "both"):
                    legend_entries.append(("PROMPT: " + chosen_prompt, float(max_score)))
                    if alt_prompt:
                        legend_entries.append(("PROMPT: " + alt_prompt, float(alt_score)))
                if legend_mode in ("candidates", "both"):
                    for p, s in zip(prompts_for_color, scores_for_prompt):
                        legend_entries.append(("MASK: " + str(p), float(s)))

                legend_items = []
                legend_colors = []
                max_w = 0
                for p, s in legend_entries:
                    txt = f"{p} ({s:.2f})"
                    bbox = font.getbbox(txt)
                    w = bbox[2] - bbox[0]
                    max_w = max(max_w, w)
                    legend_items.append(txt)
                    legend_colors.append(get_color_map(palette, p))
                colors = np.array(legend_colors, dtype=np.uint8)
                legend_h = row_h * len(legend_items) + pad * 2
                legend_w = box_w + 4 + max_w + pad * 2
                x0, y0 = 5, 20
                draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h], fill=(0, 0, 0, 160))
                for i_row, txt in enumerate(legend_items):
                    y = y0 + pad + i_row * row_h
                    col = tuple(colors[i_row].tolist())
                    draw.rectangle([x0 + pad, y, x0 + pad + box_w, y + box_w], fill=col)
                    draw.text((x0 + pad + box_w + 4, y - 1), txt, font=font, fill=(255, 255, 255))
            else:
                overlay_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(overlay_pil)
                sample_type = "ANOMALY" if is_anom else "NORMAL"
                label_color = (255, 0, 0) if is_anom else (0, 255, 0)
                no_mask_text = f"[{sample_type}] no mask >= {conf_thresh:.2f}, top={chosen_prompt}:{max_score:.2f}" + (f" alt={alt_prompt}:{alt_score:.2f}" if alt_prompt else "")
                draw.text((5, 5), no_mask_text, font=font, fill=label_color)

            # save
            global_i = dataset_pos + b
            specie_n = specie_names[b] if specie_names else ""
            cls_n = original_class_names[b] if b < len(original_class_names) else cls_name
            specie_n, cls_n, stem, ext = _get_sample_id_from_meta(loader.dataset, global_i, cls_n, specie_n)

            sample_dir = os.path.join(args.output_dir, cls_n)
            os.makedirs(sample_dir, exist_ok=True)

            filename = f"{specie_n}_{cls_n}_{stem}{ext}"
            overlay_path = os.path.join(sample_dir, filename)
            overlay_pil.save(overlay_path)

        dataset_pos += batch_imgs
        pbar.update(batch_imgs)
        fps = (total_imgs / total_time) if total_time > 0 else 0.0
        pbar.set_postfix(
            {"imgs": int(total_imgs), "fps": f"{fps:.2f}"},
            refresh=False
        )

    pbar.close()

    # Compute and print summary
    results = metrics_acc.compute_summary()
    
    print("\n" + "=" * 70)
    print("                         EVALUATION RESULTS")
    print("=" * 70)
    
    # --- Anomaly-only metrics ---
    print("\n[1] ANOMALY SAMPLES - Segmentation Metrics")
    print("-" * 50)
    anom_macro = results["anomaly_macro"]
    anom_micro = results["anomaly_micro"]
    
    print(f"  Sample count: {anom_macro['count']}")
    print(f"\n  Macro Average (mean of per-sample metrics):")
    print(f"    Dice:      {anom_macro['dice']:.4f}")
    print(f"    IoU:       {anom_macro['iou']:.4f}")
    print(f"    Precision: {anom_macro['precision']:.4f}")
    print(f"    Recall:    {anom_macro['recall']:.4f}")
    print(f"    F1:        {anom_macro['f1']:.4f}")
    print(f"    BIoU:      {anom_macro['biou']:.4f}")
    print(f"    mIoU:      {anom_macro['mIoU']:.4f}")
    print(f"    mBIoU:     {anom_macro['mBIoU']:.4f}")
    
    print(f"\n  Micro Average (from accumulated TP/FP/FN):")
    print(f"    Dice:      {anom_micro['dice']:.4f}")
    print(f"    IoU:       {anom_micro['iou']:.4f}")
    print(f"    Precision: {anom_micro['precision']:.4f}")
    print(f"    Recall:    {anom_micro['recall']:.4f}")
    print(f"    F1:        {anom_micro['f1']:.4f}")
    
    # Per-class anomaly metrics
    print("\n  Per-Class Anomaly Metrics:")
    for cls_name in sorted(results["anomaly_per_class"].keys()):
        cls_res = results["anomaly_per_class"][cls_name]
        if cls_res["count"] > 0:
            print(f"    {cls_name}: n={cls_res['count']}, "
                  f"Dice={cls_res['dice']:.4f}, IoU={cls_res['iou']:.4f}, "
                  f"P={cls_res['precision']:.4f}, R={cls_res['recall']:.4f}")
    
    # --- Normal-only FPR metrics ---
    print("\n[2] NORMAL SAMPLES - False Positive Metrics")
    print("-" * 50)
    norm_fpr = results["normal_fpr"]
    print(f"  Sample count: {norm_fpr['count']}")
    print(f"  Image-level FPR: {norm_fpr['image_fpr']:.4f} ({norm_fpr['image_fpr']*100:.2f}% have false positives)")
    print(f"  Pixel-level FPR: {norm_fpr['pixel_fpr']:.6f} ({norm_fpr['pixel_fpr']*100:.4f}% pixels)")
    print(f"  Mean pred area:  {norm_fpr['mean_pred_area']:.6f} ({norm_fpr['mean_pred_area']*100:.4f}% of image)")
    
    # Per-class normal FPR
    print("\n  Per-Class Normal FPR:")
    for cls_name in sorted(results["normal_per_class"].keys()):
        cls_res = results["normal_per_class"][cls_name]
        if cls_res["count"] > 0:
            print(f"    {cls_name}: n={cls_res['count']}, "
                  f"img_fpr={cls_res['image_fpr']:.4f}, "
                  f"px_fpr={cls_res['pixel_fpr']:.6f}, "
                  f"area={cls_res['mean_pred_area']:.6f}")
    
    # --- AUC metrics ---
    print("\n[3] IMAGE & PIXEL LEVEL AUC")
    print("-" * 50)
    print(f"  Image-AUC: {results['image_auc']:.4f}" if not np.isnan(results['image_auc']) else "  Image-AUC: N/A (need both pos/neg samples)")
    print(f"  Pixel-AUC: {results['pixel_auc']:.4f}" if not np.isnan(results['pixel_auc']) else "  Pixel-AUC: N/A (need both pos/neg samples)")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    final_speed = (total_imgs / total_time) if total_time > 0 else 0.0
    if bool(getattr(args, "report_eta_stats", False)) and len(eta_all) > 0:
        eta_np = np.concatenate([e.reshape(-1) for e in eta_all], axis=0)
        near0 = float((eta_np < 1e-3).mean())
        near1 = float((eta_np > 1.0 - 1e-3).mean())
        print(f"[ETA] N={eta_np.size} mean={float(eta_np.mean()):.6f} std={float(eta_np.std()):.6f} min={float(eta_np.min()):.6f} max={float(eta_np.max()):.6f} near0={near0:.3f} near1={near1:.3f}")
    print(f"  Total images: {total_imgs}, Speed: {final_speed:.2f} fps")
    print(f"  Anomaly samples: {anom_macro['count']}, Normal samples: {norm_fpr['count']}")
    print(f"\n  Key Metrics (anomaly-only):")
    print(f"    mIoU:  {anom_macro['mIoU']:.4f}")
    print(f"    mBIoU: {anom_macro['mBIoU']:.4f}")
    print(f"    Dice:  {anom_macro['dice']:.4f}")
    print(f"\n  Key Metrics (normal-only):")
    print(f"    Image FPR: {norm_fpr['image_fpr']:.4f}")
    print(f"    Pixel FPR: {norm_fpr['pixel_fpr']:.6f}")
    print("=" * 70)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MSAM test v4.0 (MSAD + Spurious Gating)")
    parser.add_argument("--run_profile", type=str, default="custom",
                        choices=["custom", "zero_shot", "few_shot", "few_shot_full", "few_shot_no_w"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None,
                        help="Path to meta.json (MVTec) or CSV file (VisA). If None, uses default for dataset type.")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    # 数据集选择
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"],
                        help="Dataset type: mvtec (MVTec-AD) or visa (VisA)")
    parser.add_argument("--obj_name", type=str, default=None,
                        help="Filter by class/object name (e.g., 'bottle' for MVTec, 'candle' for VisA)")
    parser.add_argument("--visa_missing_mask_behavior", type=str, default="error",
                        choices=["error", "skip", "keep_label_zero_mask", "flip_to_normal"],
                        help="VisA: anomaly样本mask缺失时的处理策略")

    # prompts
    parser.add_argument("--prompt", type=str, default=None, help="override prompt list for ALL samples, comma-separated")

    # model
    parser.add_argument("--use_official", action="store_true")
    parser.add_argument("--sam3_ckpt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="fine-tuned .pth to load")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--enable_parallel_lora", action="store_true")
    parser.add_argument("--parallel_lora_rank", type=int, default=16)
    parser.add_argument("--parallel_lora_alpha", type=float, default=None)
    parser.add_argument("--parallel_lora_target", type=str, default="qv_only",
                        choices=["qv_only", "qkv_all"])
    parser.add_argument("--parallel_lora_layer_ids", nargs="*", type=int, default=None)
    parser.add_argument("--enable_out_adapter_lora", action="store_true", default=False)
    parser.add_argument("--lora_layer_ids", nargs="*", type=int, default=None,
                        help="Which SAM3 encoder blocks to apply LoRA to (e.g., --lora_layer_ids 0 2 4). Default: all blocks.")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")

    # prompt learner config
    parser.add_argument("--prompt_learner_type", type=str, default="perclass",
                        choices=["averaged", "static", "perclass", "coop", "cocoop", "compound"],
                        help="提示学习器类型")
    parser.add_argument("--num_templates", type=int, default=4)
    parser.add_argument("--n_ctx", type=int, default=4,
                        help="可学习上下文向量数量")
    parser.add_argument("--ctx_init", type=str, default="",
                        help="上下文初始化文本")
    parser.add_argument("--class_token_position", type=str, default="end",
                        choices=["end", "middle", "front"],
                        help="类别token位置")
    parser.add_argument("--use_keywords", action="store_true",
                        help="是否使用关键词聚合")
    parser.add_argument("--cocoop_vis_dim", type=int, default=256,
                        help="CoCoOp Meta-Net输入维度")
    parser.add_argument("--cocoop_reduction", type=int, default=16,
                        help="CoCoOp Meta-Net瓶颈缩减因子")
    
    # Compound Prompt Learning
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
    parser.add_argument("--disable_dap", action="store_true",
                        help="推理消融：强制禁用DAP（即使传了--compound_enable_dap）")
    parser.add_argument("--compound_dap_top_k", type=int, default=10,
                        help="Compound: DAP top-k")
    parser.add_argument("--compound_meta_reduction", type=int, default=16,
                        help="Compound: Meta-Net瓶颈缩减因子")
    parser.add_argument("--compound_dap_use_multilevel", action="store_true",
                        help="Compound: DAP使用多层FPN特征拼接后的patch集合")
    parser.add_argument("--compound_dap_num_levels", type=int, default=0,
                        help="Compound: DAP使用的FPN层数(0=使用全部可用层)")
    parser.add_argument("--compound_use_text_encoder", action="store_true",
                        help="Compound: 将learnable ctx注入token embedding并经过SAM3 text encoder编码")
    parser.add_argument("--compound_abnormal_word", type=str, default="anomaly",
                        choices=["anomaly", "damaged"],
                        help="Compound: abnormal模板关键词(需与训练一致)")
    parser.add_argument("--compound_pooling", type=str, default="ctx_only",
                        choices=["ctx_only", "all_tokens"],
                        help="Compound(use_text_encoder): prompt向量聚合方式(只聚合ctx段或全token均值)")
    parser.add_argument("--compound_abnormal_order", type=str, default="v_then_wk",
                        choices=["v_then_wk", "wk_then_v"],
                        help="Compound: abnormal 前缀token顺序 (V+W_k 或 W_k+V)")
    parser.add_argument("--compound_dap_spurious_filter", action="store_true",
                        help="DAP: 用spurious_map过滤top-k patch，降低伪异常污染W_k")
    parser.add_argument("--disable_dap_spurious_filter", action="store_true",
                        help="推理消融：强制禁用DAP spurious过滤（即使传了--compound_dap_spurious_filter）")
    parser.add_argument("--compound_dap_spurious_alpha", type=float, default=1.0,
                        help="DAP: spurious加权系数α（score*=max(0,1-α*spurious)）")
    parser.add_argument("--compound_disable_w", action="store_true",
                        help="调试：禁用w向量(使normal_ctx仅V，abnormal_ctx为V+W)，更接近FAPrompt结构")
    parser.add_argument("--debug_dump_features", action="store_true",
                        help="Debug: 保存一小份text/msad特征到npz用于新旧版本对比")
    
    parser.add_argument("--prompt_mode", type=str, default="simple",
                        choices=["simple", "full"],
                        help="数据集prompt模式")
    
    parser.add_argument("--class_agnostic", action="store_true",
                        help="Use agnostic_name (or 'object') instead of specific class name in prompts.")
    parser.add_argument("--agnostic_name", type=str, default=None,
                        help="Class-agnostic name used when --class_agnostic (falls back to obj_name or 'object').")

    # dataset split from test defects
    parser.add_argument("--train_from_test", action="store_true")
    parser.add_argument("--specie_split_ratio", type=float, default=0.8)
    parser.add_argument("--specie_split_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Directory containing specie_splits_{cls}.json (alias of --save_dir, preferred).")

    # ranking thresholds
    parser.add_argument("--conf_thresh", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mask_thresh", type=float, default=0.5,
                        help="最终mask二值化阈值（同样作用于MSAD-only输出）")
    parser.add_argument("--mask_prompt_source", type=str, default="anomaly",
                        choices=["anomaly", "best"],
                        help="当同时跑anomaly/normal两个prompt时，最终mask来源：anomaly(仅用异常prompt)或best(二者取最大分)")

    # visualization
    parser.add_argument("--mask_alpha_center", type=float, default=0.1)
    parser.add_argument("--mask_alpha_edge", type=float, default=0.9)
    parser.add_argument("--mask_alpha_power", type=float, default=1.0)
    parser.add_argument("--mask_alpha_blur", type=float, default=0.8)
    parser.add_argument("--vis_legend_mode", type=str, default="auto",
                        choices=["auto", "candidates", "prompts", "both"],
                        help="可视化左上角legend内容：候选mask(candidates)、prompt(prompts)、两者(both)。auto在zero_shot显示both，其余显示candidates。")
    
    # 多尺度特征
    parser.add_argument("--num_feature_levels", type=int, default=4,
                        help="使用的 FPN 特征层数")
    
    # ==================== MSAD 参数 ====================
    parser.add_argument("--enable_msad", action="store_true",
                        help="启用 MSAD (Multi-Shape Anomaly Detection) 模块")
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
                        help="MSAD使用的FPN层数 (默认使用全部)")
    parser.add_argument("--msad_return_similarity_logits", action="store_true",
                        help="Debug/Train: 让MSAD返回softmax前similarity logits（推理默认不需要）")
    parser.add_argument("--msad_use_vision_adapter", action="store_true", default=False,
                        help="MSAD: 在FPN特征进入MSAD前加轻量Conv Adapter(残差)以增强跨域对齐")
    parser.add_argument("--msad_vision_adapter_reduction", type=int, default=2,
                        help="MSAD vision adapter: reduction ratio")
    parser.add_argument("--msad_vision_adapter_shared", action="store_true", default=True,
                        help="MSAD vision adapter: 是否各层共享同一个adapter")
    parser.add_argument("--msad_vision_adapter_not_shared", action="store_true", default=False,
                        help="MSAD vision adapter: 各层使用独立adapter（覆盖 msad_vision_adapter_shared）")
    
    # MSAD输出融合参数
    parser.add_argument("--use_msad_output", action="store_true",
                        help="使用MSAD的anomaly_score输出进行推理")
    parser.add_argument("--msad_mask_alpha", type=float, default=0.0,
                        help="MSAD mask与SAM3 mask的融合比例 (0=纯SAM3, 1=纯MSAD)")
    parser.add_argument("--msad_score_thresh", type=float, default=None,
                        help="可选：对MSAD anomaly_score做阈值重标定后再融合：score'=clamp((score-th)/(1-th),0..1)")
    parser.add_argument("--msad_mask_thresh", type=float, default=None,
                        help="MSAD候选的二值化阈值(默认沿用--mask_thresh)")
    parser.add_argument("--msad_mask_top_p", type=float, default=0.0,
                        help="若>0，则MSAD按每图top-p像素生成mask(覆盖msad_mask_thresh)")
    parser.add_argument("--msad_mask_auto_q", type=float, default=None,
                        help="若设置(如0.995)，先用正常样本的MSAD分数分位数自动估计msad_mask_thresh")
    
    # MSAD可视化
    parser.add_argument("--save_msad_vis", action="store_true",
                        help="保存MSAD异常图可视化结果")
    
    # ==================== Spurious Gating 参数 ====================
    parser.add_argument("--enable_spurious_gating", action="store_true",
                        help="启用 Spurious Prompt Gating (eta调制w向量)")
    parser.add_argument("--disable_spurious_gating", action="store_true",
                        help="推理消融：强制禁用spurious gating（即使传了--enable_spurious_gating）")
    parser.add_argument("--spurious_score_threshold", type=float, default=0.20,
                        help="激活阈值")
    parser.add_argument("--spurious_topk_ratio", type=float, default=0.02,
                        help="Top-k pooling 比例")
    parser.add_argument("--spurious_prompt_set_file", type=str, default=None,
                        help="可选：自定义spurious prompt集合文件(每行一个prompt)")
    parser.add_argument("--spurious_prompt_set", type=str, default=None,
                        help="可选：自定义spurious prompt集合(逗号分隔)")
    parser.add_argument("--report_eta_stats", action="store_true",
                        help="打印整套测试集eta统计(mean/std/min/max与近0/近1比例)")

    args = parser.parse_args()
    run_inference(args)
