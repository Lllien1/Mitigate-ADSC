# MSAM_test.py (fixed, confidence-head ranking)
# encoding: utf-8
# 
# 改进版本：
# 1. 分割指标（Dice/IoU/Precision/Recall/F1）仅在 anomaly 样本上计算
# 2. 假阳性指标（FPR/平均面积）仅在 normal/good 样本上计算
# 3. 添加 Image-AUC, Pixel-AUC, mIoU, mBIoU 等指标
# 4. 支持宏平均（macro）和微平均（micro）
# 5. 支持 TIE 模块（Text-guided Image Embedding Translation）进行推理

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
from scipy.ndimage import distance_transform_edt, gaussian_filter,binary_erosion, binary_dilation
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from dataset import MVTecMetaDataset, VisADataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official

# ==================== TIE 模块导入（可选）====================
try:
    from spurious_mitigation import TIEModule, TIELoss
    TIE_AVAILABLE = True
except ImportError:
    TIE_AVAILABLE = False
    print("[WARN] TIE module not available. To enable, ensure spurious_mitigation.py is in the same directory.")

# Optional: sklearn for AUC computation
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not found, AUC metrics will be skipped")


# ==================== TIE 推理支持 ====================

class SegmentationHeadWithTIE(nn.Module):
    """Wrap SAM3 segmentation head: translate obj_queries (decoder queries) with TIE before mask prediction.
    
    For inference, set `self._is_anomaly` via `set_is_anomaly` before each forward pass.
    If is_anomaly is None, TIE translation is applied uniformly (assuming potential anomaly).
    """

    def __init__(self, seg_head: nn.Module, tie_module: nn.Module, apply_to_features: bool = False, infer_scale: float = 1.0):
        super().__init__()
        self.seg_head = seg_head
        self.tie_module = tie_module
        self.apply_to_features = bool(apply_to_features)
        self.infer_scale = float(infer_scale)
        self._is_anomaly: Optional[torch.Tensor] = None

    def set_is_anomaly(self, is_anomaly: Optional[torch.Tensor]):
        self._is_anomaly = is_anomaly

    def forward(self, backbone_feats, obj_queries, **kwargs):
        is_anomaly = self._is_anomaly
        oq = obj_queries
        bf = backbone_feats

        # Optional: translate backbone features
        if self.apply_to_features and getattr(self.tie_module, "feature_tie", None) is not None:
            try:
                if isinstance(backbone_feats, (list, tuple)) and len(backbone_feats) > 0:
                    vf = backbone_feats[0]
                    tie_out_feat = self.tie_module(visual_features=vf, is_anomaly=is_anomaly, return_diagnostics=False)
                    if tie_out_feat.get("translated_features", None) is not None:
                        bf = list(backbone_feats)
                        tf = tie_out_feat["translated_features"]
                        s = float(getattr(self, "infer_scale", 1.0))
                        # scaled replace: vf + s*(tf - vf)
                        bf[0] = vf + s * (tf - vf)
            except Exception:
                bf = backbone_feats

        # Translate queries
        if getattr(self.tie_module, "query_tie", None) is not None and oq is not None:
            try:
                if oq.dim() == 4:
                    oq2 = oq.clone()
                    last = oq2[-1]
                    tie_out = self.tie_module(decoder_hs=last, is_anomaly=is_anomaly, return_diagnostics=False)
                    if tie_out.get("translated_queries", None) is not None:
                        tq = tie_out["translated_queries"]
                        s = float(getattr(self, "infer_scale", 1.0))
                        # scaled replace: last + s*(tq - last)
                        oq2[-1] = last + s * (tq - last)
                    oq = oq2
                elif oq.dim() == 3:
                    tie_out = self.tie_module(decoder_hs=oq, is_anomaly=is_anomaly, return_diagnostics=False)
                    if tie_out.get("translated_queries", None) is not None:
                        tq = tie_out["translated_queries"]
                        s = float(getattr(self, "infer_scale", 1.0))
                        oq = oq + s * (tq - oq)
            except Exception:
                pass

        return self.seg_head(backbone_feats=bf, obj_queries=oq, **kwargs)


def _build_tie_module_for_inference(args, model_core: nn.Module, device: torch.device) -> Optional[nn.Module]:
    """Build TIE module for inference. Similar to train_tie.py but simplified."""
    if not TIE_AVAILABLE:
        return None
    
    tie_mode = getattr(args, "tie_mode", "none")
    if tie_mode == "none":
        return None

    embed_dim = int(getattr(getattr(model_core, "transformer", None), "d_model", 256))
    num_vec = int(getattr(args, "tie_num_vectors", 4))
    spurious_source = getattr(args, "tie_source", "learnable")
    
    # 默认启用 queries，除非显式禁用
    apply_to_queries = not getattr(args, "no_tie_apply_to_queries", False)
    adaptive_scale = not getattr(args, "no_tie_adaptive_scale", False)
    apply_to_features = bool(getattr(args, "tie_apply_to_features", False))

    text_encoder = getattr(model_core, "text_encoder", None)
    if spurious_source in ("text", "hybrid") and text_encoder is None:
        spurious_source = "learnable"

    default_spurious_prompts = [
        "a photo with normal texture",
        "a photo of regular surface",
        "a photo without defects",
        "an image of undamaged material",
    ]
    spurious_prompts = getattr(args, "tie_spurious_prompts", None) or default_spurious_prompts

    tie_module = nn.Module()
    if apply_to_queries:
        tie_module.query_tie = TIEModule(
            embed_dim=embed_dim,
            spurious_source=spurious_source,
            num_spurious_vectors=num_vec,
            text_encoder=text_encoder,
            spurious_prompts=spurious_prompts,
            adaptive_scale=adaptive_scale,
        )
    else:
        tie_module.query_tie = None

    if apply_to_features:
        tie_module.feature_tie = TIEModule(
            embed_dim=embed_dim,
            spurious_source=spurious_source,
            num_spurious_vectors=num_vec,
            text_encoder=text_encoder,
            spurious_prompts=spurious_prompts,
            adaptive_scale=adaptive_scale,
        )
    else:
        tie_module.feature_tie = None

    # Define forward method
    def _tie_forward(decoder_hs=None, visual_features=None, is_anomaly=None, return_diagnostics=False):
        outputs = {}
        if decoder_hs is not None and tie_module.query_tie is not None:
            hs = decoder_hs[-1] if (isinstance(decoder_hs, torch.Tensor) and decoder_hs.dim() == 4) else decoder_hs
            if return_diagnostics:
                translated, diag = tie_module.query_tie(hs, is_anomaly=is_anomaly, return_diagnostics=True)
                outputs["diagnostics_query"] = diag
            else:
                translated = tie_module.query_tie(hs, is_anomaly=is_anomaly, return_diagnostics=False)
            outputs["translated_queries"] = translated
        if visual_features is not None and tie_module.feature_tie is not None:
            if return_diagnostics:
                translated, diag = tie_module.feature_tie(visual_features, is_anomaly=is_anomaly, return_diagnostics=True)
                outputs["diagnostics_feat"] = diag
            else:
                translated = tie_module.feature_tie(visual_features, is_anomaly=is_anomaly, return_diagnostics=False)
            outputs["translated_features"] = translated
        return outputs

    tie_module.forward = _tie_forward
    tie_module.to(device)

    print(f"[INFO] TIE module built for inference: mode={tie_mode}, queries={apply_to_queries}, features={apply_to_features}")
    return tie_module

# ==================== End TIE 推理支持 ====================


# ---------- visualization helpers ----------
def get_color_map(palette: List[Tuple[int, int, int]], key: str) -> Tuple[int, int, int]:
    import hashlib
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
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
            self.normal_stats["fp_pixels"] += fp
            self.normal_stats["pred_area_sum"] += pred_area
            if fp > 0:
                self.normal_stats["images_with_fp"] += 1
            self.normal_stats["count"] += 1
            
            # Per-class stats
            cls_stats = self.normal_per_class[cls_name]
            cls_stats["total_pixels"] += total_pixels
            cls_stats["fp_pixels"] += fp
            cls_stats["pred_area_sum"] += pred_area
            if fp > 0:
                cls_stats["images_with_fp"] += 1
            cls_stats["count"] += 1
        
        # Image-level AUC
        if anomaly_score is not None:
            self.image_labels.append(0)
            self.image_scores.append(anomaly_score)
    
    def update_pixel_auc(self, pred_probs: np.ndarray, gt_mask: np.ndarray):
        """Update pixel-level AUC data with subsampling."""
        pred_flat = pred_probs.flatten()
        gt_flat = gt_mask.flatten()
        
        # Subsample to avoid memory issues
        n = len(pred_flat)
        n_sample = max(1, int(n * self.pixel_subsample_rate))
        indices = np.random.choice(n, n_sample, replace=False)
        
        self.pixel_preds.extend(pred_flat[indices].tolist())
        self.pixel_labels.extend(gt_flat[indices].tolist())
    
    def compute_summary(self) -> dict:
        """Compute final summary metrics."""
        results = {}
        eps = 1e-6
        
        # ========== Anomaly-only metrics ==========
        anom = self.anomaly_stats
        anom_count = max(anom["count"], 1)
        
        # Macro average (mean of per-sample metrics)
        results["anomaly_macro"] = {
            "dice": anom["dice_sum"] / anom_count,
            "iou": anom["iou_sum"] / anom_count,
            "precision": anom["prec_sum"] / anom_count,
            "recall": anom["rec_sum"] / anom_count,
            "f1": anom["f1_sum"] / anom_count,
            "biou": anom["biou_sum"] / anom_count,
            "count": anom["count"],
        }
        
        # Micro average (compute from accumulated TP/FP/FN)
        TP, FP, FN = anom["TP_sum"], anom["FP_sum"], anom["FN_sum"]
        results["anomaly_micro"] = {
            "dice": (2 * TP) / (2 * TP + FP + FN + eps),
            "iou": TP / (TP + FP + FN + eps),
            "precision": TP / (TP + FP + eps),
            "recall": TP / (TP + FN + eps),
        }
        results["anomaly_micro"]["f1"] = (
            2 * results["anomaly_micro"]["precision"] * results["anomaly_micro"]["recall"] /
            (results["anomaly_micro"]["precision"] + results["anomaly_micro"]["recall"] + eps)
        )
        
        # mIoU: mean IoU across classes
        class_ious = []
        class_bious = []
        for cls_name, cls_stats in self.anomaly_per_class.items():
            if cls_stats["count"] > 0:
                class_ious.append(cls_stats["iou_sum"] / cls_stats["count"])
                class_bious.append(cls_stats["biou_sum"] / cls_stats["count"])
        
        results["anomaly_macro"]["mIoU"] = np.mean(class_ious) if class_ious else 0.0
        results["anomaly_macro"]["mBIoU"] = np.mean(class_bious) if class_bious else 0.0
        
        # Per-class anomaly results
        results["anomaly_per_class"] = {}
        for cls_name, cls_stats in self.anomaly_per_class.items():
            cnt = max(cls_stats["count"], 1)
            results["anomaly_per_class"][cls_name] = {
                "dice": cls_stats["dice_sum"] / cnt,
                "iou": cls_stats["iou_sum"] / cnt,
                "precision": cls_stats["prec_sum"] / cnt,
                "recall": cls_stats["rec_sum"] / cnt,
                "f1": cls_stats["f1_sum"] / cnt,
                "biou": cls_stats["biou_sum"] / cnt,
                "count": cls_stats["count"],
            }
        
        # ========== Normal-only FPR metrics ==========
        norm = self.normal_stats
        norm_count = max(norm["count"], 1)
        
        results["normal_fpr"] = {
            "image_fpr": norm["images_with_fp"] / norm_count,  # % of normal images with any FP
            "pixel_fpr": norm["fp_pixels"] / max(norm["total_pixels"], 1),  # % of FP pixels
            "mean_pred_area": norm["pred_area_sum"] / norm_count,  # mean predicted area ratio
            "count": norm["count"],
        }
        
        # Per-class normal FPR
        results["normal_per_class"] = {}
        for cls_name, cls_stats in self.normal_per_class.items():
            cnt = max(cls_stats["count"], 1)
            results["normal_per_class"][cls_name] = {
                "image_fpr": cls_stats["images_with_fp"] / cnt,
                "pixel_fpr": cls_stats["fp_pixels"] / max(cls_stats["total_pixels"], 1),
                "mean_pred_area": cls_stats["pred_area_sum"] / cnt,
                "count": cls_stats["count"],
            }
        
        # ========== AUC metrics ==========
        if HAS_SKLEARN and len(self.image_labels) > 1:
            labels = np.array(self.image_labels)
            scores = np.array(self.image_scores)
            # Need both positive and negative samples for AUC
            if len(np.unique(labels)) == 2:
                results["image_auc"] = roc_auc_score(labels, scores)
            else:
                results["image_auc"] = float("nan")
        else:
            results["image_auc"] = float("nan")
        
        if HAS_SKLEARN and len(self.pixel_labels) > 1:
            labels = np.array(self.pixel_labels)
            preds = np.array(self.pixel_preds)
            if len(np.unique(labels)) == 2:
                results["pixel_auc"] = roc_auc_score(labels, preds)
            else:
                results["pixel_auc"] = float("nan")
        else:
            results["pixel_auc"] = float("nan")
        
        return results


# ---------- model/dataloader helpers ----------
def _filter_kwargs_for_callable(func, kwargs: dict):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _infer_class_list(meta: dict) -> List[str]:
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
    dataset_type: str = "mvtec",  # "mvtec" or "visa"
    obj_name: Optional[str] = None,  # filter by class name
    prompt_mode: str = "simple",  # CoOp/CoCoOp prompt mode
):
    """Build dataloader for MVTec-AD or VisA dataset."""
    
    if dataset_type.lower() == "visa":
        # VisA dataset: meta_path should be the CSV file (e.g., 1cls.csv)
        ds = VisADataset(
            root=root,
            csv_path=meta_path,
            mode=mode,
            obj_name=obj_name,
            prompt_mode=prompt_mode,
        )
    else:
        # MVTec-AD dataset: meta_path should be meta.json
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
            obj_name=obj_name,
            prompt_mode=prompt_mode,
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
                  masks_size: Tuple[int,int], device: torch.device, upsample: bool = True,
                  is_anomaly: Optional[torch.Tensor] = None,
                  use_filo_output: bool = False, 
                  filo_mask_alpha: float = 0.0,
                  filo_conf_alpha: float = 0.0):
    """Run one forward pass and return (pred_masks_prob[B,Q,H,W], query_scores[B,Q], raw_out).
    
    Args:
        is_anomaly: Optional tensor of shape (B,) indicating which samples are anomalies.
                   If None and TIE in_forward mode is used, TIE translation is applied uniformly.
        use_filo_output: 是否使用FiLo的anomaly_map输出
        filo_mask_alpha: FiLo mask与SAM3 mask的融合比例 (0=纯SAM3, 1=纯FiLo)
        filo_conf_alpha: FiLo置信度与原始置信度的融合比例 (0=纯原始, 1=纯FiLo)
    """
    # 如果模型使用 TIE in_forward 模式，设置 is_anomaly
    # 检查是否有 SegmentationHeadWithTIE 包装
    seg_head = getattr(model, "segmentation_head", None)
    if seg_head is not None and hasattr(seg_head, "set_is_anomaly"):
        # 对于推理，我们通常不知道 ground truth，设置为 None 让 TIE 均匀应用
        # 或者可以传入 is_anomaly 参数
        seg_head.set_is_anomaly(is_anomaly)
    
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
    
    # ===== 方案C: 使用fused_conf替代query_scores =====
    fused_conf = out.get("fused_conf", None)
    if fused_conf is not None:
        # fused_conf是融合后的置信度 (B, Q)，已经是logits
        fused_prob = torch.sigmoid(fused_conf)
        query_scores = fused_prob
    
    # ===== FiLo输出融合（修复版：只融合最可信的query）=====
    filo_map_single = None  # (B, 1, H, W) FiLo的单通道异常图
    
    if use_filo_output and (filo_mask_alpha > 0 or filo_conf_alpha > 0):
        filo_maps = out.get("filo_anomaly_maps", [])
        filo_agg = out.get("filo_aggregated_map", None)
        
        if filo_agg is not None:
            # filo_agg: (B, 2, H, W) - [normal, abnormal]
            filo_map_single = filo_agg[:, 1:2, :, :]  # (B, 1, H, W)
        elif len(filo_maps) > 0:
            filo_map_single = filo_maps[-1][:, 1:2, :, :]  # (B, 1, H, W)
        
        if filo_map_single is not None:
            # 检查FiLo输出是否在0-1范围（softmax概率）
            filo_min = filo_map_single.min().item()
            filo_max = filo_map_single.max().item()
            if filo_min < -0.01 or filo_max > 1.01:
                # 不是概率，需要sigmoid
                filo_map_single = torch.sigmoid(filo_map_single)
            
            # 上采样到pred_masks的尺寸（不是masks_size）
            # 关键修复：确保filo_map和pred_masks尺寸一致
            pred_mask_size = pred_masks.shape[-2:]  # 当前pred_masks的尺寸
            if filo_map_single.shape[-2:] != pred_mask_size:
                filo_map_single = F.interpolate(
                    filo_map_single, size=pred_mask_size, 
                    mode="bilinear", align_corners=False
                )
            
            # ===== 风险1修复：只对最可信的query做融合 =====
            if filo_mask_alpha > 0:
                for b in range(B):
                    # 找到该图最可信的query
                    if query_scores is not None:
                        q_star = query_scores[b].argmax().item()
                    else:
                        # 没有置信度时，用mask面积最大的query
                        q_star = pred_masks[b].sum(dim=(1, 2)).argmax().item()
                    
                    # 只对q_star做融合
                    filo_b = filo_map_single[b, 0]  # (H, W)
                    sam3_b = pred_masks[b, q_star]  # (H, W)
                    
                    if filo_mask_alpha >= 1.0:
                        pred_masks[b, q_star] = filo_b
                    else:
                        pred_masks[b, q_star] = (1 - filo_mask_alpha) * sam3_b + filo_mask_alpha * filo_b
            
            # ===== 风险3修复：置信度融合独立控制 =====
            if filo_conf_alpha > 0 and query_scores is not None:
                # FiLo置信度：每张图的abnormal响应最大值
                filo_conf_per_image = filo_map_single.view(B, -1).max(dim=-1)[0]  # (B,)
                
                for b in range(B):
                    # 找到最可信的query，只调整它的置信度
                    q_star = query_scores[b].argmax().item()
                    orig_conf = query_scores[b, q_star]
                    filo_conf = filo_conf_per_image[b]
                    
                    # 融合置信度
                    query_scores[b, q_star] = (1 - filo_conf_alpha) * orig_conf + filo_conf_alpha * filo_conf

    return pred_masks, query_scores, out


def _select_candidates(pm_b: torch.Tensor, scores_vec: Optional[torch.Tensor], prompt_text: str,
                      conf_thresh: float, top_k: int):
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
    cand.sort(key=lambda x: x[0], reverse=True)
    if top_k > 0:
        cand = cand[:top_k]

    if cand:
        masks_stack = np.stack([(c[1] > 0.5) for c in cand]).astype(bool)
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
    
    args.class_list = class_list
    args.meta_path = meta_path  # 确保 args.meta_path 被设置

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
        # VV Attention 参数
        "enable_vv_attention": getattr(args, "enable_vv_attention", False),
        "vv_num_heads": getattr(args, "vv_num_heads", 8),
        "vv_dropout": getattr(args, "vv_dropout", 0.1),
        # FiLo 参数
        "enable_filo": getattr(args, "enable_filo", False),
        "filo_dim_out": getattr(args, "filo_dim_out", 768),
        "filo_k_linear": getattr(args, "filo_k_linear", 4),
        "filo_k_cov": getattr(args, "filo_k_cov", 4),
        "filo_image_size": getattr(args, "filo_image_size", 518),
        "filo_use_alternating": getattr(args, "filo_use_alternating", True),
        "num_feature_levels": getattr(args, "num_feature_levels", 4),
        # 方案B: FiLo到Decoder回灌
        "filo_to_decoder": getattr(args, "filo_to_decoder", False),
        "filo_decoder_mode": getattr(args, "filo_decoder_mode", "memory"),
        "filo_decoder_tokens": getattr(args, "filo_decoder_tokens", 64),
        # 方案C: 置信度融合头
        "enable_conf_fusion_head": getattr(args, "enable_conf_fusion_head", False),
        "conf_fusion_hidden_dim": getattr(args, "conf_fusion_hidden_dim", 64),
    })
    model = Constructor(**ctor_kwargs).to(device).eval()

    # ==================== TIE 模块集成（必须在加载 checkpoint 之前！）====================
    # 原因：checkpoint 中包含 TIE 权重（前缀为 tie_module.* 或 segmentation_head.tie_module.*）
    # 如果先加载 checkpoint 再构建 TIE，这些权重会被忽略（报告为 unexpected keys）
    tie_mode = getattr(args, "tie_mode", "none")
    if tie_mode != "none" and TIE_AVAILABLE:
        print(f"[INFO] Building TIE module for inference (mode={tie_mode})...")
        tie_module = _build_tie_module_for_inference(args, model, device)
        
        if tie_module is not None:
            if tie_mode == "in_forward":
                # 包装 segmentation_head 以在 forward 中应用 TIE
                try:
                    model.segmentation_head = SegmentationHeadWithTIE(
                        seg_head=model.segmentation_head,
                        tie_module=tie_module,
                        apply_to_features=bool(getattr(args, "tie_apply_to_features", False)),
                        infer_scale=float(getattr(args, "tie_infer_scale", 1.0)),
                    ).to(device)
                    print("[INFO] Segmentation head wrapped with TIE (in_forward mode)")
                except Exception as e:
                    print(f"[WARN] Failed to wrap segmentation_head with TIE: {e}")
                    model.tie_module = tie_module
            else:
                # loss_only 模式：仅附加模块
                model.tie_module = tie_module
                print("[INFO] TIE module attached (loss_only mode)")
    elif tie_mode != "none" and not TIE_AVAILABLE:
        print(f"[WARN] TIE mode={tie_mode} requested but TIE module not available")
    # ==================== End TIE 模块集成 ====================

    # 现在加载 checkpoint（TIE 结构已经就位，权重会正确匹配）
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[INFO] Loading fine-tuned checkpoint {args.ckpt} ...")
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            state = ckpt
        
        # ==================== 过滤形状不匹配的权重 ====================
        # 当跨数据集推理时（如 MVTec checkpoint 用于 VisA），prompt_learner 相关权重可能不兼容
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
                # 允许额外的键（如 TIE 模块）
                filtered_state[k] = v
        
        if skipped_keys:
            print(f"[WARN] Skipped {len(skipped_keys)} mismatched weights (cross-dataset inference):")
            for sk in skipped_keys[:5]:  # 只打印前5个
                print(f"       {sk}")
            if len(skipped_keys) > 5:
                print(f"       ... and {len(skipped_keys) - 5} more")
        # ==================== End 过滤 ====================
        
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"[INFO] Loaded fine-tuned weights. missing={len(missing)} unexpected={len(unexpected)}")
        
        # 打印 TIE 相关的加载情况
        if tie_mode != "none":
            tie_keys_loaded = [k for k in state.keys() if "tie_module" in k or "tie" in k.lower()]
            if tie_keys_loaded:
                print(f"[INFO] TIE weights found in checkpoint: {len(tie_keys_loaded)} keys")
            else:
                print("[WARN] No TIE weights found in checkpoint (tie_module.* keys)")
    else:
        print("[INFO] No fine-tuned checkpoint provided. Using base SAM3 weights.")

    return model


def _get_sample_id_from_meta(ds, global_idx: int, fallback_cls: str, fallback_specie: str):
    """
    Try to recover the meta.json entry for naming.
    
    从 img_path 提取序号，例如：
    - "zipper/test/fabric_border/006.png" -> stem="006"
    - 最终文件名: {specie_name}_{cls_name}_{stem}.png
    """
    meta_entry = None
    # 尝试从不同的数据集属性中获取 entry
    for attr in ("entries", "items", "meta_list", "data_list", "samples", "records", "metas", "data"):
        v = getattr(ds, attr, None)
        if isinstance(v, list) and global_idx < len(v):
            meta_entry = v[global_idx]
            break

    cls_name = fallback_cls
    specie_name = fallback_specie
    stem = f"{global_idx:03d}"  # 默认使用全局索引
    ext = ".png"

    if meta_entry is not None:
        # 处理 dict 类型的 entry (来自原始 meta.json)
        if isinstance(meta_entry, dict):
            cls_name = meta_entry.get("cls_name", cls_name)
            specie_name = meta_entry.get("specie_name", specie_name)
            img_path = meta_entry.get("img_path") or meta_entry.get("image_path") or meta_entry.get("img")
        # 处理 SampleEntry 或其他对象类型 (来自 MVTecMetaDataset.entries)
        else:
            cls_name = getattr(meta_entry, "cls_name", cls_name)
            specie_name = getattr(meta_entry, "specie_name", specie_name)
            img_path = getattr(meta_entry, "img_path", None)
        
        # 从 img_path 提取文件名序号
        # 例如: "zipper/test/fabric_border/006.png" -> stem="006", ext=".png"
        if isinstance(img_path, str) and img_path:
            base = os.path.basename(img_path)  # "006.png"
            stem2, ext2 = os.path.splitext(base)  # stem2="006", ext2=".png"
            if stem2:
                stem = stem2
            if ext2:
                ext = ext2

    if not specie_name:
        specie_name = "unknown"
    if not cls_name:
        cls_name = "unknown"
    return specie_name, cls_name, stem, ext


# ---------- main inference ----------
@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确定数据集类型和元数据路径
    dataset_type = getattr(args, "dataset", "mvtec").lower()
    if dataset_type == "visa":
        # VisA: 使用 CSV 文件
        meta_path = args.meta_path or os.path.join(args.data_root, "split_csv", "1cls.csv")
    else:
        # MVTec-AD: 使用 meta.json
        meta_path = args.meta_path or os.path.join(args.data_root, "meta.json")
    
    loader = build_loader(
        args.data_root,
        meta_path,
        args.mode,
        args.batch_size,
        include_test_defects=getattr(args, "include_test_defects", False),
        train_from_test=getattr(args, "train_from_test", False),
        specie_split_ratio=getattr(args, "specie_split_ratio", 0.8),
        specie_split_seed=getattr(args, "specie_split_seed", 42),
        save_dir=getattr(args, "save_dir", None),
        dataset_type=dataset_type,
        obj_name=getattr(args, "obj_name", None),
        prompt_mode=getattr(args, "prompt_mode", "simple"),
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

    # Initialize metrics accumulator
    metrics_acc = MetricsAccumulator()

    dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else 0
    pbar = tqdm(
        total=dataset_len,
        desc="Inference",
        unit="img",
        leave=False,          # ⭐ 关键：不保留历史行
        dynamic_ncols=True,   # ⭐ 自动适配终端宽度
        mininterval=0.2,      # ⭐ 最少 0.2s 才刷新一次
    )


    total_time = 0.0
    total_imgs = 0
    dataset_pos = 0

    for images, masks, prompt_lists, is_anomaly, class_names, specie_names in loader:
        images = images.to(device)

        # optional global prompt override
        custom_prompt: List[str] = []
        if getattr(args, "prompt", None):
            custom_prompt = [w.strip() for w in args.prompt.split(",") if w.strip()]
        if custom_prompt:
            # Override all samples with user-provided prompts (comma-separated)
            prompt_lists = [custom_prompt for _ in prompt_lists]
        # 注意：当 custom_prompt 为空时，prompt_lists 来自数据集但不会被使用
        # 因为下面会根据 prompt_mode 重新构建 anomaly_lists 和 normal_lists

        # measure GPU time
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # --- forward passes ---
        # PROMPT-driven inference: run two prompts per image and pick the prompt that yields
        # the highest anomaly score (max query score), similar to reverse-prompt evaluation.
        
        # 为 TIE in_forward 模式准备 is_anomaly tensor
        # 注意：在实际部署中可能没有 ground truth，此时传 None 让 TIE 均匀应用
        is_anomaly_tensor = torch.tensor(is_anomaly, dtype=torch.bool, device=device) if is_anomaly is not None else None
        
        # FiLo输出配置（拆分成mask和conf两个独立alpha）
        use_filo_output = getattr(args, "use_filo_output", False)
        filo_mask_alpha = getattr(args, "filo_mask_alpha", 0.0)
        filo_conf_alpha = getattr(args, "filo_conf_alpha", 0.0)
        
        if custom_prompt:
            pred_masks_1, query_scores_1, _ = _forward_once(
                model, images, prompt_lists, class_names, masks_size=masks.shape[-2:], device=device,
                is_anomaly=is_anomaly_tensor,
                use_filo_output=use_filo_output, 
                filo_mask_alpha=filo_mask_alpha, filo_conf_alpha=filo_conf_alpha
            )
            pred_masks_2, query_scores_2 = None, None
        else:
            # ===== 根据 prompt_mode 选择推理时的 prompt 格式 =====
            # 关键原则：测试时的 prompt 必须与训练时 prompt[0] 一致！
            prompt_mode = getattr(args, "prompt_mode", "simple").lower()
            dataset_type = getattr(args, "dataset", "mvtec").lower()
            
            if prompt_mode == "simple":
                # ===== simple 模式（推荐）=====
                # 训练时：anomaly 样本用 ["anomaly {cls}"], normal 样本用 ["normal {cls}"]
                # 测试时：用这两个 prompt 分别推理，选择得分更高的
                anomaly_lists = [[f"anomaly {class_names[i]}"] for i in range(len(class_names))]
                normal_lists = [[f"normal {class_names[i]}"] for i in range(len(class_names))]
            elif dataset_type == "visa":
                # ===== VisA full 模式 =====
                # 训练时：anomaly 样本用 ["damaged {cls}", ...], normal 样本用 ["normal {cls}", ...]
                anomaly_lists = [[f"damaged {class_names[i]}"] for i in range(len(class_names))]
                normal_lists = [[f"normal {class_names[i]}"] for i in range(len(class_names))]
            else:
                # ===== MVTec-AD full 模式 =====
                # 训练时：anomaly 样本用 ["anomaly {cls}", ...], normal 样本用 ["normal {cls}", ...]
                anomaly_lists = [[f"anomaly {class_names[i]}"] for i in range(len(class_names))]
                normal_lists = [[f"normal {class_names[i]}"] for i in range(len(class_names))]

            # 推理：分别用 anomaly prompt 和 normal prompt 跑一遍
            pred_masks_1, query_scores_1, _ = _forward_once(
                model, images, anomaly_lists, class_names, masks_size=masks.shape[-2:], device=device, upsample=False,
                is_anomaly=is_anomaly_tensor,
                use_filo_output=use_filo_output, 
                filo_mask_alpha=filo_mask_alpha, filo_conf_alpha=filo_conf_alpha
            )
            pred_masks_2, query_scores_2, _ = _forward_once(
                model, images, normal_lists, class_names, masks_size=masks.shape[-2:], device=device, upsample=False,
                is_anomaly=is_anomaly_tensor,
                use_filo_output=use_filo_output, 
                filo_mask_alpha=filo_mask_alpha, filo_conf_alpha=filo_conf_alpha
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_time = t1 - t0
        total_time += batch_time

        batch_imgs = images.size(0)
        total_imgs += batch_imgs

        gt = (masks > 0.5).float().to(device)

        for b in range(batch_imgs):
            cls_name = class_names[b]
            is_anom = is_anomaly[b]

            gm = gt[b].squeeze(0)

            # ---------------- PROMPT-driven candidates selection (UPSAMPLE ONLY WINNER) ----------------
            conf_thresh = float(getattr(args, "conf_thresh", 0.3))
            top_k = int(getattr(args, "top_k", 5))
            
            # 目标尺寸：用 GT mask 的空间尺寸最稳（gm 必须是 (H,W)，即 gm = gt[b].squeeze(0)）
            masks_size = tuple(gm.shape[-2:])  # (H, W)
            
            def _max_prompt_score(pm_low: torch.Tensor, scores_vec: Optional[torch.Tensor]) -> float:
                """
                pm_low: (Q, h, w) 低分辨率 masks
                scores_vec: (Q,) 或 None
                选择 winner prompt 时，只需要一个标量分数：max query score
                """
                if scores_vec is not None:
                    return float(scores_vec.view(-1).max().item())
                # fallback: 没有 query_scores 才用 mask 均值近似
                return float(pm_low.mean(dim=(1, 2)).max().item())
            
            if custom_prompt:
                chosen_prompt = ",".join(custom_prompt)
            
                pm_low = pred_masks_1[b]  # (Q, h, w)
                sv = query_scores_1[b].view(-1) if query_scores_1 is not None else None
            
                max_score = _max_prompt_score(pm_low, sv)
                alt_prompt, alt_score = "", 0.0
            
            else:
                # ===== 根据 prompt_mode 选择 prompt 标签（与推理时一致）=====
                prompt_mode = getattr(args, "prompt_mode", "simple").lower()
                dataset_type = getattr(args, "dataset", "mvtec").lower()
                
                if prompt_mode == "simple":
                    # simple 模式：使用 anomaly/normal
                    p1 = f"anomaly {cls_name}"
                    p2 = f"normal {cls_name}"
                elif dataset_type == "visa":
                    # VisA full 模式
                    p1 = f"damaged {cls_name}"
                    p2 = f"normal {cls_name}"
                else:
                    # MVTec full 模式
                    p1 = f"anomaly {cls_name}"
                    p2 = f"normal {cls_name}"
            
                pm1_low = pred_masks_1[b]  # (Q, h, w)
                pm2_low = pred_masks_2[b]  # (Q, h, w)
            
                sv1 = query_scores_1[b].view(-1) if query_scores_1 is not None else None
                sv2 = query_scores_2[b].view(-1) if query_scores_2 is not None else None
            
                max_score1 = _max_prompt_score(pm1_low, sv1)
                max_score2 = _max_prompt_score(pm2_low, sv2)
            
                # winner prompt：只看 max query score
                if max_score2 > max_score1:
                    chosen_prompt, alt_prompt = p2, p1
                    max_score, alt_score = max_score2, max_score1
                    pm_low, sv = pm2_low, sv2
                else:
                    chosen_prompt, alt_prompt = p1, p2
                    max_score, alt_score = max_score1, max_score2
                    pm_low, sv = pm1_low, sv1
            
            # 只对 winner 的这一张图做高分辨率插值： (Q,h,w) -> (Q,H,W)
            pm = pm_low
            if pm.shape[-2:] != masks_size:
                pm = F.interpolate(pm.unsqueeze(0), size=masks_size, mode="bilinear", align_corners=False).squeeze(0)
            
            # 用 winner 的高分辨率 masks 生成候选与 union mask
            cand, pm_comb, pred_prob_map, _ = _select_candidates(pm, sv, chosen_prompt, conf_thresh, top_k)
            # ------------------------------------------------------------------------------------------

            # Compute metrics
            metrics = safe_binary_metrics(pm_comb, gm)
            biou = compute_biou(pm_comb, gm)
            
            total_pixels = int(gm.numel())
            
            # ========== Key change: separate anomaly vs normal ==========
            if is_anom:
                # Anomaly sample: compute segmentation metrics
                metrics_acc.update_anomaly(metrics, biou, cls_name, anomaly_score=max_score)
            else:
                # Normal sample: compute FPR metrics only
                metrics_acc.update_normal(metrics, cls_name, total_pixels, anomaly_score=max_score)
            
            # Update pixel-level AUC data (for both anomaly and normal)
            gt_map = gm.cpu().numpy()
            metrics_acc.update_pixel_auc(pred_prob_map, gt_map)

            # ========== Visualization ==========
            img_pil = to_pil(images[b].cpu())
            frame = np.array(img_pil.convert("RGB"), dtype=np.uint8)

            # 统一处理 normal 和 anomaly 样本的可视化
            # 区别仅在于标签颜色和文字
            if cand:
                # Anomaly sample with predictions - show masks
                masks_stack = np.stack([(c[1] > 0.5) for c in cand]).astype(bool)
                prompts_for_color = [c[2] for c in cand]
                scores_for_prompt = [c[0] for c in cand]
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
                # 根据样本类型显示不同颜色和前缀
                sample_type = "ANOMALY" if is_anom else "NORMAL"
                label_color = (255, 255, 255) if is_anom else (255, 165, 0)  # 白色/橙色
                header_text = f"[{sample_type}] top: {chosen_prompt} ({max_score:.2f})" + (f" | alt: {alt_prompt} ({alt_score:.2f})" if alt_prompt else "")
                draw.text((5, 5), header_text, font=font, fill=label_color)
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
                x0, y0 = 5, 20
                draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h], fill=(0, 0, 0, 160))
                for i_row, txt in enumerate(legend_items):
                    y = y0 + pad + i_row * row_h
                    col = tuple(colors[i_row].tolist())
                    draw.rectangle([x0 + pad, y, x0 + pad + box_w, y + box_w], fill=col)
                    draw.text((x0 + pad + box_w + 4, y - 1), txt, font=font, fill=(255, 255, 255))
            else:
                # No predictions above threshold
                overlay_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(overlay_pil)
                sample_type = "ANOMALY" if is_anom else "NORMAL"
                label_color = (255, 0, 0) if is_anom else (0, 255, 0)  # 红色(漏检)/绿色(正确)
                no_mask_text = f"[{sample_type}] no mask >= {conf_thresh:.2f}, top={chosen_prompt}:{max_score:.2f}" + (f" alt={alt_prompt}:{alt_score:.2f}" if alt_prompt else "")
                draw.text((5, 5), no_mask_text, font=font, fill=label_color)

            # save
            sample_dir = os.path.join(args.output_dir, cls_name)
            os.makedirs(sample_dir, exist_ok=True)

            global_i = dataset_pos + b
            specie_n = specie_names[b] if specie_names else ""
            cls_n = cls_name
            specie_n, cls_n, stem, ext = _get_sample_id_from_meta(loader.dataset, global_i, cls_n, specie_n)

            filename = f"{specie_n}_{cls_n}_{stem}{ext}"
            overlay_path = os.path.join(sample_dir, filename)
            overlay_pil.save(overlay_path)

        dataset_pos += batch_imgs
        pbar.update(batch_imgs)
        fps = (total_imgs / total_time) if total_time > 0 else 0.0
        pbar.set_postfix(
            {"imgs": int(total_imgs), "fps": f"{fps:.2f}"},
            refresh=False   # ⭐ 不要每次都强制 redraw
        )


    pbar.close()

    # ========== Compute and print summary ==========
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
        # Convert numpy types to Python types for JSON serialization
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
    parser = argparse.ArgumentParser("MSAM test (confidence-head ranking)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None,
                        help="Path to meta.json (MVTec) or CSV file (VisA). If None, uses default for dataset type.")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    # ==================== 数据集选择 ====================
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"],
                        help="Dataset type: mvtec (MVTec-AD) or visa (VisA)")
    parser.add_argument("--obj_name", type=str, default=None,
                        help="Filter by class/object name (e.g., 'bottle' for MVTec, 'candle' for VisA)")
    # ==================== End 数据集选择 ====================

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

    # prompt learner config - CoOp/CoCoOp 提示学习参数
    parser.add_argument("--prompt_learner_type", type=str, default="perclass",
                        choices=["averaged", "static", "perclass", "coop", "cocoop"],
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
    parser.add_argument("--prompt_mode", type=str, default="simple",
                        choices=["simple", "full"],
                        help="数据集prompt模式")

    # dataset split from test defects
    parser.add_argument("--train_from_test", action="store_true")
    parser.add_argument("--specie_split_ratio", type=float, default=0.8)
    parser.add_argument("--specie_split_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)

    # parallel lora args
    parser.add_argument("--enable_parallel_lora", action="store_true")
    parser.add_argument("--parallel_lora_rank", type=int, default=16)
    parser.add_argument("--parallel_lora_alpha", type=float, default=None)

    # ranking thresholds
    parser.add_argument("--conf_thresh", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=5)

    # visualization
    parser.add_argument("--mask_alpha_center", type=float, default=0.1)
    parser.add_argument("--mask_alpha_edge", type=float, default=0.9)
    parser.add_argument("--mask_alpha_power", type=float, default=1.0)
    parser.add_argument("--mask_alpha_blur", type=float, default=0.8)

    # ==================== TIE 模块参数 ====================
    parser.add_argument("--tie_mode", type=str, default="none",
                        choices=["none", "loss_only", "in_forward"],
                        help="TIE inference mode: none(禁用), loss_only(仅用于诊断), in_forward(应用TIE平移)")
    parser.add_argument("--tie_infer_scale", type=float, default=1.0,
                        help="TIE inference translation scale: 0=disable, 1=full, (0,1)=conservative")
    parser.add_argument("--tie_num_vectors", type=int, default=4,
                        help="TIE 伪相关向量数量")
    parser.add_argument("--tie_source", type=str, default="learnable",
                        choices=["learnable", "text", "hybrid"],
                        help="TIE 伪相关向量来源")
    parser.add_argument("--tie_apply_to_queries", action="store_true", default=False,
                        help="TIE 应用于 decoder queries (默认启用)")
    parser.add_argument("--no_tie_apply_to_queries", action="store_true", default=False,
                        help="显式禁用 TIE 在 queries 上的应用")
    parser.add_argument("--tie_apply_to_features", action="store_true", default=False,
                        help="TIE 应用于视觉特征 (默认关闭)")
    parser.add_argument("--tie_adaptive_scale", action="store_true", default=False,
                        help="TIE 使用自适应缩放")
    parser.add_argument("--no_tie_adaptive_scale", action="store_true", default=False,
                        help="显式禁用 TIE 自适应缩放")
    # ==================== End TIE 参数 ====================
    
    # ==================== VV Attention & FiLo 参数 ====================
    parser.add_argument("--enable_vv_attention", action="store_true",
                        help="启用 VV Attention 模块")
    parser.add_argument("--vv_num_heads", type=int, default=8,
                        help="VV Attention 的头数")
    parser.add_argument("--vv_dropout", type=float, default=0.1,
                        help="VV Attention 的 dropout")
    
    parser.add_argument("--enable_filo", action="store_true",
                        help="启用 FiLo 模块（官方6路卷积实现）")
    parser.add_argument("--filo_dim_out", type=int, default=768,
                        help="FiLo 输出维度")
    parser.add_argument("--filo_k_linear", type=int, default=4,
                        help="FiLo LinearLayer 层数")
    parser.add_argument("--filo_k_cov", type=int, default=4,
                        help="FiLo CovLayer 层数")
    parser.add_argument("--filo_image_size", type=int, default=518,
                        help="FiLo 输出图像尺寸")
    parser.add_argument("--filo_use_alternating", action="store_true", default=True,
                        help="FiLo 使用交替分配（偶数层->QKV，奇数层->VV）")
    parser.add_argument("--num_feature_levels", type=int, default=4,
                        help="使用的 FPN 特征层数")
    
    # FiLo输出融合参数（独立控制mask和置信度）
    parser.add_argument("--use_filo_output", action="store_true",
                        help="使用FiLo的anomaly_map输出进行推理")
    parser.add_argument("--filo_mask_alpha", type=float, default=0.0,
                        help="FiLo mask与SAM3 mask的融合比例 (0=纯SAM3, 1=纯FiLo)。只对最可信query生效。")
    parser.add_argument("--filo_conf_alpha", type=float, default=0.0,
                        help="FiLo置信度与原始置信度的融合比例 (0=纯原始, 1=纯FiLo)。只对最可信query生效。")
    
    # 方案B: FiLo到Decoder回灌
    parser.add_argument("--filo_to_decoder", action="store_true",
                        help="启用FiLo特征到Decoder的回灌")
    parser.add_argument("--filo_decoder_mode", type=str, default="memory",
                        choices=["memory", "query_bias", "cross_attn"],
                        help="FiLo到Decoder的回灌模式")
    parser.add_argument("--filo_decoder_tokens", type=int, default=64,
                        help="FiLo压缩后的token数量")
    
    # 方案C: 置信度融合头
    parser.add_argument("--enable_conf_fusion_head", action="store_true",
                        help="启用置信度融合头")
    parser.add_argument("--conf_fusion_hidden_dim", type=int, default=64,
                        help="置信度融合头的隐藏维度")
    # ==================== End VV & FiLo 参数 ====================

    args = parser.parse_args()
    run_inference(args)