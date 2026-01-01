"""
FiLo-SAM3 修复版实现
====================

修复问题：
1. detach() 切断梯度 → 添加训练/推理模式开关
2. QKV取点位置错误 → 使用pre_hook获取正确的输入token
3. 明确区分训练模式和推理模式

核心架构（按FiLo论文）：
    ViT layers → Hook提取 → QKV分支 (LinearLayer) ──┬─> cosine_sim ─> 异常图
                          → VV分支 (CovLayer/MMCI) ─┘

关键理解：
1. QKV branch：取 attention 输入 token（norm1(x)），不是 block 输出
2. VV branch：取 attention 输出（softmax(QK^T)V 的近似）
3. 训练时不 detach，推理时可选 detach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math


# ==============================================================================
# FiLo LinearLayer (QKV分支)
# ==============================================================================

class FiLoLinearLayer(nn.Module):
    """
    QKV分支：k个独立线性层
    
    输入：k个ViT层的token特征 (B, N, C)
    输出：k个投影后特征 (B, N, D_text)
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.layers[i](x) for i, x in enumerate(x_list)]


# ==============================================================================
# FiLo CovLayer (VV分支/MMCI核心)
# ==============================================================================

class FiLoCovLayer(nn.Module):
    """
    VV分支：6路多形状卷积（MMCI核心）
    
    6种卷积核用于检测不同形状异常：
    - 1×1: 点状异常
    - 3×3, 5×5, 7×7: 块状异常（不同尺度）
    - 1×5: 水平划痕
    - 5×1: 垂直裂缝
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        
        self.conv_11 = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 1, bias=False) for _ in range(num_layers)])
        self.conv_33 = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False) for _ in range(num_layers)])
        self.conv_55 = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 5, padding=2, bias=False) for _ in range(num_layers)])
        self.conv_77 = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 7, padding=3, bias=False) for _ in range(num_layers)])
        self.conv_15 = nn.ModuleList([nn.Conv2d(in_dim, out_dim, (1,5), padding=(0,2), bias=False) for _ in range(num_layers)])
        self.conv_51 = nn.ModuleList([nn.Conv2d(in_dim, out_dim, (5,1), padding=(2,0), bias=False) for _ in range(num_layers)])
        
    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for i, x in enumerate(x_list):
            y = (self.conv_11[i](x) + self.conv_33[i](x) + self.conv_55[i](x) + 
                 self.conv_77[i](x) + self.conv_15[i](x) + self.conv_51[i](x))
            out.append(y)
        return out


# ==============================================================================
# SAM3 Feature Extractor - 修复版
# ==============================================================================

class SAM3FeatureExtractor:
    """
    从SAM3 ViT中提取特征（修复版）
    
    修复点：
    1. 添加 detach_features 开关：训练时不detach，推理时detach
    2. QKV分支：使用 pre_hook 获取 attention 输入（更接近 FiLo 定义）
    3. VV分支：获取 attention 输出（softmax(QK^T)V 的近似）
    
    使用方法：
        # 训练时
        extractor = SAM3FeatureExtractor(model, layers, detach_features=False)
        
        # 推理/可视化时  
        extractor = SAM3FeatureExtractor(model, layers, detach_features=True)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        extract_layers: List[int],
        detach_features: bool = False,  # 关键修复：默认不detach
    ):
        """
        Args:
            model: SAM3模型
            extract_layers: 要提取的ViT block索引，如[7, 15, 23, 31]
            detach_features: 是否detach特征（训练时False，推理时可True）
        """
        self.model = model
        self.extract_layers = extract_layers
        self.detach_features = detach_features
        
        self.qkv_features = {}   # QKV分支特征（attention输入）
        self.vv_features = {}    # VV分支特征（attention输出）
        self.hooks = []
        
    def _get_vit_blocks(self):
        """获取SAM3 ViT blocks"""
        try:
            return self.model.backbone.vision_backbone.trunk.blocks
        except AttributeError:
            try:
                return self.model.backbone.trunk.blocks
            except AttributeError:
                raise RuntimeError("Cannot locate ViT blocks in SAM3 model")
    
    def register_hooks(self):
        """注册hooks"""
        blocks = self._get_vit_blocks()
        
        for idx in self.extract_layers:
            if idx >= len(blocks):
                print(f"[Warning] Layer {idx} >= num_blocks {len(blocks)}, skipping")
                continue
            
            block = blocks[idx]
            
            # ===== QKV分支：使用 pre_hook 获取 attention 输入 =====
            # 这比取 block 输出更接近 FiLo 的 QKV token 定义
            def make_qkv_hook(layer_idx):
                def hook(module, args):
                    # args[0] 是 attention 模块的输入（已经过 norm1）
                    x = args[0]
                    if self.detach_features:
                        x = x.detach()
                    self.qkv_features[layer_idx] = x
                return hook
            
            # ===== VV分支：获取 attention 输出 =====
            # attention 输出是 softmax(QK^T)V @ out_proj 的结果
            # 这是 V·A 的近似（实际 FiLo 需要 pre-proj，但这需要侵入式修改）
            def make_vv_hook(layer_idx):
                def hook(module, input, output):
                    x = output
                    if self.detach_features:
                        x = x.detach()
                    self.vv_features[layer_idx] = x
                return hook
            
            # 注册 pre_hook 到 attention 模块（获取输入 token）
            self.hooks.append(
                block.attn.register_forward_pre_hook(make_qkv_hook(idx))
            )
            # 注册 forward_hook 到 attention 模块（获取输出）
            self.hooks.append(
                block.attn.register_forward_hook(make_vv_hook(idx))
            )
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def clear(self):
        self.qkv_features = {}
        self.vv_features = {}
    
    def set_detach_mode(self, detach: bool):
        """动态切换detach模式"""
        self.detach_features = detach
    
    def get_qkv_features(self) -> List[torch.Tensor]:
        """获取QKV分支特征，按层索引排序"""
        return [self.qkv_features[i] for i in sorted(self.qkv_features.keys())]
    
    def get_vv_features(self) -> List[torch.Tensor]:
        """获取VV分支特征，按层索引排序"""
        return [self.vv_features[i] for i in sorted(self.vv_features.keys())]
    
    def debug_print(self):
        """调试打印：验证特征是否正确提取"""
        print(f"[FiLo Extractor Debug]")
        print(f"  detach_features: {self.detach_features}")
        print(f"  extract_layers: {self.extract_layers}")
        print(f"  qkv_features count: {len(self.qkv_features)}")
        print(f"  vv_features count: {len(self.vv_features)}")
        
        for k, v in self.qkv_features.items():
            print(f"  qkv[{k}]: shape={tuple(v.shape)}, requires_grad={v.requires_grad}")
        for k, v in self.vv_features.items():
            print(f"  vv[{k}]: shape={tuple(v.shape)}, requires_grad={v.requires_grad}")


# ==============================================================================
# FiLo Anomaly Detection Head
# ==============================================================================

class FiLoHead(nn.Module):
    """
    FiLo异常检测头
    
    流程：
    1. QKV分支: LinearLayer处理 → cosine similarity with text
    2. VV分支:  CovLayer处理 → cosine similarity with text  
    3. 融合: (sim_abnormal + 1 - sim_normal) / 2 → 多层平均
    """
    
    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 768,
        num_layers: int = 4,
        output_size: int = 518,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.output_size = output_size
        
        # QKV分支: 所有层
        self.qkv_branch = FiLoLinearLayer(vision_dim, text_dim, num_layers)
        
        # VV分支: 跳过第一层 (FiLo official)
        self.vv_branch = FiLoCovLayer(vision_dim, text_dim, num_layers - 1)
        
    def forward(
        self,
        qkv_feats: List[torch.Tensor],
        vv_feats: List[torch.Tensor],
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算异常图
        
        Args:
            qkv_feats: k个 (B, H, W, C) 或 (B, N, C)
            vv_feats: k-1个 (同上)，跳过第一层
            text_normal: (D,) 或 (B, D)
            text_abnormal: (D,) 或 (B, D)
        """
        B = qkv_feats[0].shape[0]
        
        # 获取空间尺寸
        if qkv_feats[0].dim() == 4:
            H, W = qkv_feats[0].shape[1:3]
        else:
            N = qkv_feats[0].shape[1]
            H = W = int(math.sqrt(N))
        
        # 规范化文本嵌入
        if text_normal.dim() == 1:
            text_normal = text_normal.unsqueeze(0).expand(B, -1)
        if text_abnormal.dim() == 1:
            text_abnormal = text_abnormal.unsqueeze(0).expand(B, -1)
        
        text_n = F.normalize(text_normal, dim=-1)
        text_a = F.normalize(text_abnormal, dim=-1)
        
        # QKV分支
        qkv_input = self._to_bnc(qkv_feats)
        qkv_proj = self.qkv_branch(qkv_input)
        qkv_maps = [self._compute_anomaly_map(f, text_n, text_a, H, W) for f in qkv_proj]
        
        # VV分支
        vv_input = self._to_bchw(vv_feats, H, W)
        vv_proj = self.vv_branch(vv_input)
        vv_maps = [self._compute_anomaly_map(f.flatten(2).permute(0,2,1), text_n, text_a, H, W) 
                   for f in vv_proj]
        
        # 多层融合
        all_maps = qkv_maps + vv_maps
        all_maps_up = [
            F.interpolate(m.unsqueeze(1), size=(self.output_size, self.output_size), 
                         mode='bilinear', align_corners=False).squeeze(1)
            for m in all_maps
        ]
        anomaly_map = torch.stack(all_maps_up).mean(dim=0)
        
        return {
            'anomaly_map': anomaly_map,
            'qkv_maps': qkv_maps,
            'vv_maps': vv_maps,
        }
    
    def _to_bnc(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for f in feats:
            if f.dim() == 4:
                f = f.flatten(1, 2)
            out.append(f)
        return out
    
    def _to_bchw(self, feats: List[torch.Tensor], H: int, W: int) -> List[torch.Tensor]:
        out = []
        for f in feats:
            if f.dim() == 3:
                B, N, C = f.shape
                f = f.permute(0, 2, 1).view(B, C, H, W)
            elif f.dim() == 4 and f.shape[-1] != f.shape[1]:
                f = f.permute(0, 3, 1, 2)
            out.append(f)
        return out
    
    def _compute_anomaly_map(self, feat, text_n, text_a, H, W):
        B = feat.shape[0]
        feat = F.normalize(feat, dim=-1)
        
        sim_n = torch.bmm(feat, text_n.unsqueeze(-1)).squeeze(-1)
        sim_a = torch.bmm(feat, text_a.unsqueeze(-1)).squeeze(-1)
        
        anomaly = (sim_a + 1 - sim_n) / 2
        return anomaly.view(B, H, W)


# ==============================================================================
# FiLo-SAM3 完整模块
# ==============================================================================

class FiLoSAM3(nn.Module):
    """
    FiLo-SAM3 完整模块（修复版）
    
    关键修复：
    1. 训练时 detach_features=False，保留梯度
    2. 推理时 detach_features=True，节省内存
    3. QKV分支取点改为 attention 输入
    """
    
    def __init__(
        self,
        sam3_model: nn.Module,
        extract_layers: List[int] = [7, 15, 23, 31],
        vision_dim: int = 1024,
        text_dim: int = 768,
        output_size: int = 518,
        detach_features: bool = False,  # 默认不detach（训练模式）
    ):
        super().__init__()
        
        self.extractor = SAM3FeatureExtractor(
            sam3_model, 
            extract_layers,
            detach_features=detach_features,
        )
        self.head = FiLoHead(
            vision_dim=vision_dim,
            text_dim=text_dim,
            num_layers=len(extract_layers),
            output_size=output_size,
        )
        
        self._registered = False
        
    def register_hooks(self):
        if not self._registered:
            self.extractor.register_hooks()
            self._registered = True
        
    def remove_hooks(self):
        self.extractor.remove_hooks()
        self._registered = False
    
    def train(self, mode: bool = True):
        """重写train方法，自动切换detach模式"""
        super().train(mode)
        # 训练时不detach，推理时detach
        self.extractor.set_detach_mode(not mode)
        return self
    
    def eval(self):
        """重写eval方法"""
        return self.train(False)
        
    def forward(
        self,
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算异常图（需要先运行SAM3 forward以触发hooks）
        """
        qkv_feats = self.extractor.get_qkv_features()
        vv_feats = self.extractor.get_vv_features()[1:]  # 跳过第一层
        
        return self.head(qkv_feats, vv_feats, text_normal, text_abnormal)
    
    def debug_print(self):
        """调试打印"""
        self.extractor.debug_print()


# ==============================================================================
# 工厂函数
# ==============================================================================

def build_filo_for_sam3(
    sam3_model: nn.Module,
    extract_layers: List[int] = None,
    vision_dim: int = 1024,
    text_dim: int = 768,
    output_size: int = 518,
    detach_features: bool = False,  # 训练时False，推理时可True
) -> FiLoSAM3:
    """
    构建FiLo模块
    
    Args:
        sam3_model: SAM3模型实例
        extract_layers: 要提取的ViT层，默认[7, 15, 23, 31]
        vision_dim: SAM3 ViT隐藏维度
        text_dim: 文本嵌入维度
        output_size: 输出分辨率
        detach_features: 是否detach特征（训练False，推理True）
    """
    if extract_layers is None:
        extract_layers = [7, 15, 23, 31]
    
    filo = FiLoSAM3(
        sam3_model=sam3_model,
        extract_layers=extract_layers,
        vision_dim=vision_dim,
        text_dim=text_dim,
        output_size=output_size,
        detach_features=detach_features,
    )
    
    filo.register_hooks()
    return filo


# ==============================================================================
# 验证工具
# ==============================================================================

def verify_filo_gradients(filo_module: FiLoSAM3, verbose: bool = True):
    """
    验证FiLo模块的梯度流是否正常
    
    Returns:
        dict: 包含各部分的梯度状态
    """
    result = {
        'qkv_features_grad': [],
        'vv_features_grad': [],
        'head_params_grad': [],
    }
    
    # 检查提取的特征
    for k, v in filo_module.extractor.qkv_features.items():
        result['qkv_features_grad'].append((k, v.requires_grad))
    for k, v in filo_module.extractor.vv_features.items():
        result['vv_features_grad'].append((k, v.requires_grad))
    
    # 检查head参数
    for name, param in filo_module.head.named_parameters():
        result['head_params_grad'].append((name, param.requires_grad))
    
    if verbose:
        print("[FiLo Gradient Verification]")
        print(f"  detach_mode: {filo_module.extractor.detach_features}")
        print(f"  QKV features requires_grad: {result['qkv_features_grad']}")
        print(f"  VV features requires_grad: {result['vv_features_grad']}")
        print(f"  Head trainable params: {sum(1 for _, g in result['head_params_grad'] if g)}")
    
    return result