"""
MSAD: Multi-Shape Anomaly Detection Module for SAM3
====================================================

研究动机：
1. 工业异常形态多样性：点状(污点)、块状(缺损)、线状(划痕/裂缝)
2. 不同卷积核对不同形状异常的检测敏感度不同
3. 多尺度特征融合可以同时捕获局部细节和全局上下文

核心创新：
1. Multi-Shape Convolution Integration (MSCI): 6种形状卷积核的特征融合
2. Learnable Shape Attention (LSA): 自适应学习不同形状特征的重要性权重
3. Multi-Scale Hierarchical Aggregation (MSHA): 层次化多尺度特征聚合

与原FiLo的区别：
- FiLo: 256 → 768 维投影（为CLIP设计）
- MSAD: 保持256维（与SAM3原生对齐），无冗余投影
- 新增: 可学习的形状注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import math


class ConvAdapter2d(nn.Module):
    def __init__(self, dim: int, reduction: int = 2):
        super().__init__()
        red = max(1, int(dim // max(1, int(reduction))))
        self.fc = nn.Sequential(
            nn.Conv2d(dim, red, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, dim, kernel_size=1, bias=False),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + x


# ==============================================================================
# 核心模块1: Multi-Shape Convolution Integration (MSCI)
# ==============================================================================

class MultiShapeConvBlock(nn.Module):
    """
    多形状卷积块 - 单层实现
    
    6种卷积核设计理念：
    - 1×1: 点状异常（污点、小孔洞）
    - 3×3: 小块状异常（小缺损）
    - 5×5: 中块状异常（中等缺损、斑块）
    - 7×7: 大块状异常（大面积损伤）
    - 1×5: 水平线状异常（水平划痕、接缝）
    - 5×1: 垂直线状异常（垂直裂缝、划痕）
    """
    
    def __init__(
        self,
        dim: int = 256,
        use_bn: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_residual = use_residual
        
        # 6路多形状卷积
        self.conv_1x1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, bias=False)
        self.conv_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.conv_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), bias=False)
        
        # 归一化层
        if use_bn:
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.norm = nn.LayerNorm([dim])  # 需要在forward中处理
        self.use_bn = use_bn
        
        # 激活函数
        self.act = nn.GELU()
        
        # 残差缩放因子（可学习）
        if use_residual:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 6路卷积
        f_point = self.conv_1x1(x)    # 点状
        f_small = self.conv_3x3(x)    # 小块
        f_medium = self.conv_5x5(x)   # 中块
        f_large = self.conv_7x7(x)    # 大块
        f_horizontal = self.conv_1x5(x)  # 水平线
        f_vertical = self.conv_5x1(x)    # 垂直线
        
        # 特征融合（求和）
        out = f_point + f_small + f_medium + f_large + f_horizontal + f_vertical
        
        # 归一化 + 激活
        if self.use_bn:
            out = self.norm(out)
        out = self.act(out)
        
        # 残差连接
        if self.use_residual:
            out = x + self.residual_scale * out
        
        return out


class LearnableShapeAttention(nn.Module):
    """
    可学习的形状注意力机制 (Learnable Shape Attention, LSA)
    
    动态学习6种形状特征的重要性权重
    
    研究意义：
    - 不同类别的异常可能需要不同的形状特征组合
    - 自适应学习比固定权重（如等权求和）更灵活
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_shapes: int = 6,
        reduction: int = 16,
    ):
        super().__init__()
        self.num_shapes = num_shapes
        
        # 6路卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, bias=False),
            nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), bias=False),
        ])
        
        # 通道注意力生成权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, num_shapes),
            nn.Softmax(dim=-1),
        )
        
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) 融合后的特征
            weights: (B, 6) 形状注意力权重（用于可视化/分析）
        """
        B, C, H, W = x.shape
        
        # 计算6路特征
        shape_features = [conv(x) for conv in self.convs]  # List of (B, C, H, W)
        
        # 计算形状注意力权重
        weights = self.channel_attention(x)  # (B, 6)
        
        # 加权融合
        out = torch.zeros_like(x)
        for i, feat in enumerate(shape_features):
            # weights[:, i] 是 (B,)，需要扩展到 (B, 1, 1, 1)
            w = weights[:, i].view(B, 1, 1, 1)
            out = out + w * feat
        
        out = self.norm(out)
        out = self.act(out)
        
        return out, weights


# ==============================================================================
# 核心模块2: Multi-Scale Hierarchical Aggregation (MSHA)
# ==============================================================================

class MultiScaleAggregator(nn.Module):
    """
    多尺度层次聚合模块
    
    SAM3 FPN 提供4个尺度的特征：
    - Level 0: 4x上采样（最高分辨率，细节丰富）
    - Level 1: 2x上采样
    - Level 2: 原始分辨率
    - Level 3: 2x下采样（最低分辨率，上下文丰富）
    
    聚合策略：
    1. 每层独立生成异常图
    2. 可学习的层级权重融合
    """
    
    def __init__(
        self,
        num_levels: int = 4,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # 可学习的层级权重
            self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        else:
            # 固定等权重
            self.register_buffer('level_weights', torch.ones(num_levels) / num_levels)
    
    def forward(
        self,
        anomaly_maps: List[torch.Tensor],  # [(B, 2, H, W), ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            anomaly_maps: 每层的异常图，已上采样到相同尺寸
            
        Returns:
            aggregated: (B, 2, H, W) 聚合后的异常图
            weights: (num_levels,) 层级权重（用于分析）
        """
        # 归一化权重
        weights = F.softmax(self.level_weights, dim=0)
        
        # 加权聚合
        aggregated = torch.zeros_like(anomaly_maps[0])
        for i, amap in enumerate(anomaly_maps):
            aggregated = aggregated + weights[i] * amap
        
        return aggregated, weights


# ==============================================================================
# 核心模块3: Vision-Language Anomaly Scoring
# ==============================================================================

class VLAnomalyScorer(nn.Module):
    """
    视觉-语言异常评分模块
    
    计算视觉特征与文本特征（normal/abnormal）的相似度
    生成像素级异常分数图
    """
    
    def __init__(
        self,
        dim: int = 256,
        temperature: float = 100.0,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.dim = dim
        
        if learnable_temp:
            # 可学习的温度参数
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(
        self,
        patch_features: torch.Tensor,  # (B, N, C)
        text_features: torch.Tensor,   # (B, 2, C) [normal, abnormal]
        spatial_size: Tuple[int, int] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            patch_features: 视觉patch特征
            text_features: 文本特征 [normal, abnormal]
            spatial_size: 空间尺寸 (H, W)
            
        Returns:
            anomaly_map: (B, 2, H, W) 异常图 [normal_sim, abnormal_sim]
        """
        B, N, C = patch_features.shape
        
        # L2归一化
        patch_norm = F.normalize(patch_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        
        # 计算相似度: (B, N, C) @ (B, C, 2) -> (B, N, 2)
        similarity = self.temperature * torch.bmm(patch_norm, text_norm.transpose(-2, -1))
        
        # 推断空间尺寸
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
        
        # 重塑: (B, N, 2) -> (B, 2, H, W)
        logits_map = similarity.transpose(1, 2).view(B, 2, H, W)
        anomaly_map = F.softmax(logits_map, dim=1)
        if return_logits:
            return anomaly_map, logits_map
        return anomaly_map


# ==============================================================================
# 完整模块: MSAD (Multi-Shape Anomaly Detection)
# ==============================================================================

class MSAD(nn.Module):
    """
    MSAD: Multi-Shape Anomaly Detection Module
    
    完整的多形状多尺度异常检测模块
    
    架构：
    ```
    FPN Features (B, 256, H_i, W_i) × 4 levels
           │
           ↓
    ┌──────────────────────────────────────┐
    │  Multi-Shape Conv Integration (MSCI) │
    │  - 6种形状卷积: 1×1, 3×3, 5×5, 7×7,  │
    │                 1×5, 5×1             │
    │  - 可学习形状注意力 (LSA)             │
    └──────────────────────────────────────┘
           │
           ↓
    ┌──────────────────────────────────────┐
    │  Vision-Language Anomaly Scoring     │
    │  - L2 normalize                      │
    │  - τ × (patch @ text.T)              │
    │  - Softmax                           │
    └──────────────────────────────────────┘
           │
           ↓
    ┌──────────────────────────────────────┐
    │  Multi-Scale Hierarchical Aggregation│
    │  - 4层异常图上采样                    │
    │  - 可学习层级权重融合                 │
    └──────────────────────────────────────┘
           │
           ↓
    Anomaly Map (B, H, W)
    ```
    
    论文可写的创新点：
    1. 多形状卷积检测不同类型工业异常
    2. 可学习形状注意力自适应融合
    3. 与SAM3原生256维特征直接对齐，无冗余投影
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_levels: int = 4,
        output_size: int = 518,
        use_shape_attention: bool = True,
        learnable_level_weights: bool = True,
        learnable_temperature: bool = True,
        temperature: float = 100.0,
        use_vision_adapter: bool = False,
        vision_adapter_reduction: int = 2,
        vision_adapter_shared: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_levels = num_levels
        self.output_size = output_size
        self.use_shape_attention = use_shape_attention
        self.use_vision_adapter = bool(use_vision_adapter)

        self.vision_adapters: Optional[nn.Module] = None
        if self.use_vision_adapter:
            if bool(vision_adapter_shared):
                self.vision_adapters = ConvAdapter2d(dim=dim, reduction=int(vision_adapter_reduction))
            else:
                self.vision_adapters = nn.ModuleList(
                    [ConvAdapter2d(dim=dim, reduction=int(vision_adapter_reduction)) for _ in range(num_levels)]
                )
        
        # 每层的多形状卷积模块
        if use_shape_attention:
            self.shape_convs = nn.ModuleList([
                LearnableShapeAttention(dim=dim) for _ in range(num_levels)
            ])
        else:
            self.shape_convs = nn.ModuleList([
                MultiShapeConvBlock(dim=dim, use_residual=True) for _ in range(num_levels)
            ])
        
        # 异常评分模块
        self.scorer = VLAnomalyScorer(
            dim=dim,
            temperature=temperature,
            learnable_temp=learnable_temperature,
        )
        
        # 多尺度聚合
        self.aggregator = MultiScaleAggregator(
            num_levels=num_levels,
            learnable_weights=learnable_level_weights,
        )
    
    def forward(
        self,
        fpn_features: List[torch.Tensor],  # [(B, 256, H_i, W_i), ...]
        text_features: torch.Tensor,        # (B, 2, 256)
        return_intermediate: bool = False,
        return_similarity_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fpn_features: SAM3 FPN输出，4层多尺度特征
            text_features: 文本特征 (B, 2, C) [normal, abnormal]
            return_intermediate: 是否返回中间结果
            
        Returns:
            dict:
                - anomaly_score: (B, H, W) 最终异常分数
                - aggregated_map: (B, 2, H, W) 聚合后的异常图
                - anomaly_maps: List[(B, 2, H, W)] 每层异常图（可选）
                - shape_weights: List[(B, 6)] 形状注意力权重（可选）
                - level_weights: (num_levels,) 层级权重（可选）
        """
        B = fpn_features[0].shape[0]
        anomaly_maps = []
        similarity_logits_maps = []
        shape_weights_list = []
        
        for i, feat in enumerate(fpn_features):
            _, C, H, W = feat.shape

            if self.use_vision_adapter and self.vision_adapters is not None:
                if isinstance(self.vision_adapters, nn.ModuleList):
                    feat = self.vision_adapters[i](feat)
                else:
                    feat = self.vision_adapters(feat)
            
            # 1. 多形状卷积处理
            if self.use_shape_attention:
                enhanced, shape_weights = self.shape_convs[i](feat)
                shape_weights_list.append(shape_weights)
            else:
                enhanced = self.shape_convs[i](feat)
            
            # 2. 展平为patch tokens: (B, C, H, W) -> (B, H*W, C)
            patch_tokens = enhanced.flatten(2).transpose(1, 2)
            
            # 3. 计算异常图
            if return_intermediate and return_similarity_logits:
                anomaly_map, logits_map = self.scorer(
                    patch_tokens,
                    text_features,
                    spatial_size=(H, W),
                    return_logits=True,
                )
            else:
                anomaly_map = self.scorer(patch_tokens, text_features, spatial_size=(H, W))
                logits_map = None
            
            # 4. 上采样到目标尺寸
            anomaly_map = F.interpolate(
                anomaly_map,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
            )
            
            anomaly_maps.append(anomaly_map)
            if logits_map is not None:
                logits_map = F.interpolate(
                    logits_map,
                    size=(self.output_size, self.output_size),
                    mode='bilinear',
                    align_corners=False,
                )
                similarity_logits_maps.append(logits_map)
        
        # 5. 多尺度聚合
        aggregated_map, level_weights = self.aggregator(anomaly_maps)
        
        # 6. 提取异常分数（abnormal通道）
        anomaly_score = aggregated_map[:, 1, :, :]  # (B, H, W)
        
        # 构建输出
        result = {
            'anomaly_score': anomaly_score,
            'aggregated_map': aggregated_map,
        }
        
        if return_intermediate:
            result['anomaly_maps'] = anomaly_maps
            result['level_weights'] = level_weights
            if self.use_shape_attention:
                result['shape_weights'] = shape_weights_list
            if return_similarity_logits:
                result['similarity_logits_maps'] = similarity_logits_maps
                if len(similarity_logits_maps) > 0:
                    w = level_weights
                    if w.dim() == 1:
                        w = w.view(1, -1, 1, 1, 1)
                    logits_stack = torch.stack(similarity_logits_maps, dim=1)
                    result['aggregated_logits_map'] = (logits_stack * w).sum(dim=1)
        
        return result
    
    def get_interpretable_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取可解释的权重（用于论文分析）
        
        Returns:
            dict:
                - level_weights: 层级权重
                - temperature: 温度参数
        """
        return {
            'level_weights': F.softmax(self.aggregator.level_weights, dim=0),
            'temperature': self.scorer.temperature,
        }


# ==============================================================================
# 工厂函数
# ==============================================================================

def build_msad(
    dim: int = 256,
    num_levels: int = 4,
    output_size: int = 518,
    use_shape_attention: bool = True,
    config: str = 'default',
) -> MSAD:
    """
    构建MSAD模块
    
    Args:
        dim: 特征维度（与SAM3对齐=256）
        num_levels: FPN层数
        output_size: 输出尺寸
        use_shape_attention: 是否使用形状注意力
        config: 配置模式
            - 'default': 默认配置
            - 'lightweight': 轻量级（无形状注意力）
            - 'full': 完整配置（所有可学习参数）
    """
    configs = {
        'default': {
            'use_shape_attention': True,
            'learnable_level_weights': True,
            'learnable_temperature': True,
        },
        'lightweight': {
            'use_shape_attention': False,
            'learnable_level_weights': False,
            'learnable_temperature': False,
        },
        'full': {
            'use_shape_attention': True,
            'learnable_level_weights': True,
            'learnable_temperature': True,
        },
    }
    
    cfg = configs.get(config, configs['default'])
    
    return MSAD(
        dim=dim,
        num_levels=num_levels,
        output_size=output_size,
        **cfg,
    )


# ==============================================================================
# 参数量统计
# ==============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 分模块统计
    module_params = {}
    for name, module in model.named_children():
        module_params[name] = sum(p.numel() for p in module.parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'by_module': module_params,
    }


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == "__main__":
    # 模拟输入
    B, C = 2, 256
    fpn_features = [
        torch.randn(B, C, 256, 256),  # scale 4.0
        torch.randn(B, C, 128, 128),  # scale 2.0
        torch.randn(B, C, 64, 64),    # scale 1.0
        torch.randn(B, C, 32, 32),    # scale 0.5
    ]
    text_features = torch.randn(B, 2, 256)  # [normal, abnormal]
    
    # 创建模块
    msad = build_msad(dim=256, num_levels=4, output_size=518, config='default')
    
    # 前向传播
    out = msad(fpn_features, text_features, return_intermediate=True)
    
    print("MSAD Output:")
    print(f"  anomaly_score: {out['anomaly_score'].shape}")
    print(f"  aggregated_map: {out['aggregated_map'].shape}")
    print(f"  anomaly_maps: {[m.shape for m in out['anomaly_maps']]}")
    print(f"  level_weights: {out['level_weights']}")
    if 'shape_weights' in out:
        print(f"  shape_weights: {[w.shape for w in out['shape_weights']]}")
    
    # 参数统计
    params = count_parameters(msad)
    print(f"\nParameter Count:")
    print(f"  Total: {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  By module: {params['by_module']}")
    
    # 可解释权重
    weights = msad.get_interpretable_weights()
    print(f"\nInterpretable Weights:")
    print(f"  Level weights: {weights['level_weights']}")
    print(f"  Temperature: {weights['temperature'].item():.2f}")
