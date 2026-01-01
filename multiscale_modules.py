"""
multiscale_modules.py - FiLo官方实现移植版
==========================================

完全按照 FiLo/models/FiLo.py 官方代码移植

核心模块：
1. CovLayer (MMCI): 6路多形状卷积 - FiLo的核心创新
2. LinearLayer: QKV分支的线性投影
3. FiLoDecoder: 完整的双分支解码器

官方数据流：
    patch_tokens[::2]  → LinearLayer → patch_tokens_qkv → normalize → 100*(patch@text.T) → softmax
    patch_tokens[1::2] → CovLayer    → patch_tokens_vv  → normalize → 100*(patch@text.T) → softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


# ==============================================================================
# LinearLayer - 完全按照FiLo官方实现
# 来源: FiLo/models/FiLo.py - class LinearLayer
# ==============================================================================

class LinearLayer(nn.Module):
    """
    FiLo官方LinearLayer - 用于QKV分支
    
    原版代码:
    ```python
    class LinearLayer(nn.Module):
        def __init__(self, dim_in, dim_out, k):
            super(LinearLayer, self).__init__()
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

        def forward(self, tokens):
            for i in range(len(tokens)):
                if len(tokens[i].shape) == 3:
                    tokens[i] = self.fc[i](tokens[i][:, 1:, :])  # 去掉CLS token
                else:
                    assert 0 == 1
            return tokens
    ```
    """
    
    def __init__(self, dim_in: int, dim_out: int, k: int):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])
        self.k = k
        
    def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        处理多层patch tokens
        
        Args:
            tokens: List of (B, N, C) 或 (B, C, H, W)
            
        Returns:
            processed: List of (B, N-1, C_out) 或 (B, H*W, C_out)
        """
        result = []
        for i in range(len(tokens)):
            layer_idx = min(i, self.k - 1)
            
            if tokens[i].dim() == 3:
                # (B, N, C) - 原版会去掉第一个token (CLS)
                # SAM3没有CLS token，所以直接处理
                x = tokens[i]
                if x.shape[1] == int(np.sqrt(x.shape[1] - 1)) ** 2 + 1:
                    # 有CLS token，去掉
                    x = x[:, 1:, :]
                result.append(self.fc[layer_idx](x))
            elif tokens[i].dim() == 4:
                # (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = tokens[i].shape
                x = tokens[i].permute(0, 2, 3, 1).reshape(B, H * W, C)
                result.append(self.fc[layer_idx](x))
            else:
                raise ValueError(f"Unexpected token dim: {tokens[i].dim()}")
        
        return result


# ==============================================================================
# CovLayer (MMCI) - 完全按照FiLo官方实现
# 来源: FiLo/models/FiLo.py - class CovLayer
# ==============================================================================

class CovLayer(nn.Module):
    """
    FiLo官方CovLayer - 6路多形状卷积 (这才是真正的MMCI!)
    
    原版代码:
    ```python
    class CovLayer(nn.Module):
        def __init__(self, dim_in, dim_out, k):
            super(CovLayer, self).__init__()
            self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same") for _ in range(k)])
            self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same") for _ in range(k)])
            self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same") for _ in range(k)])
            self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same") for _ in range(k)])
            self.fc_51 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same") for _ in range(k)])
            self.fc_15 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same") for _ in range(k)])

        def forward(self, tokens):
            for i in range(len(tokens)):
                if len(tokens[i].shape) == 3:
                    x = tokens[i][:, 1:, :]
                    x = x.view(x.shape[0], int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])), x.shape[2])
                    x_temp = (self.fc_11[i](x.permute(0, 3, 1, 2))
                            + self.fc_33[i](x.permute(0, 3, 1, 2))
                            + self.fc_55[i](x.permute(0, 3, 1, 2))
                            + self.fc_77[i](x.permute(0, 3, 1, 2))
                            + self.fc_15[i](x.permute(0, 3, 1, 2))
                            + self.fc_51[i](x.permute(0, 3, 1, 2)))
                    tokens[i] = x_temp
                    tokens[i] = tokens[i].permute(0, 2, 3, 1).view(tokens[i].shape[0], -1, tokens[i].shape[1])
            return tokens
    ```
    
    6种卷积核的作用:
    - 1×1: 点特征（细节）
    - 3×3: 局部纹理
    - 5×5: 中等感受野
    - 7×7: 大感受野（上下文）
    - 5×1: 垂直方向特征（划痕、裂缝）
    - 1×5: 水平方向特征（划痕、裂缝）
    """
    
    def __init__(self, dim_in: int, dim_out: int, k: int):
        super().__init__()
        self.k = k
        
        # 完全按照官方实现
        self.fc_11 = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0)
            for _ in range(k)
        ])
        self.fc_33 = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
            for _ in range(k)
        ])
        self.fc_55 = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=5, padding=2)
            for _ in range(k)
        ])
        self.fc_77 = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=7, padding=3)
            for _ in range(k)
        ])
        self.fc_51 = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding=(2, 0))
            for _ in range(k)
        ])
        self.fc_15 = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding=(0, 2))
            for _ in range(k)
        ])
        
    def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        处理多层patch tokens
        
        Args:
            tokens: List of (B, N, C) 或 (B, C, H, W)
            
        Returns:
            processed: List of (B, N, C_out)
        """
        result = []
        
        for i in range(len(tokens)):
            layer_idx = min(i, self.k - 1)
            
            if tokens[i].dim() == 3:
                # (B, N, C) -> 需要reshape成2D
                x = tokens[i]
                B, N, C = x.shape
                
                # 检查是否有CLS token
                if N == int(np.sqrt(N - 1)) ** 2 + 1:
                    x = x[:, 1:, :]  # 去掉CLS token
                    N = N - 1
                
                H = W = int(np.sqrt(N))
                x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
                
            elif tokens[i].dim() == 4:
                # (B, C, H, W)
                x = tokens[i]
                B, C, H, W = x.shape
            else:
                raise ValueError(f"Unexpected token dim: {tokens[i].dim()}")
            
            # 6路卷积求和 - FiLo的核心操作
            x_out = (
                self.fc_11[layer_idx](x) +
                self.fc_33[layer_idx](x) +
                self.fc_55[layer_idx](x) +
                self.fc_77[layer_idx](x) +
                self.fc_51[layer_idx](x) +
                self.fc_15[layer_idx](x)
            )
            
            # 转回 (B, N, C) 格式
            B, C_out, H_out, W_out = x_out.shape
            x_out = x_out.permute(0, 2, 3, 1).reshape(B, H_out * W_out, C_out)
            
            result.append(x_out)
        
        return result


# ==============================================================================
# FiLoDecoder - 整合LinearLayer和CovLayer的解码器
# ==============================================================================

class FiLoDecoder(nn.Module):
    """
    FiLo解码器 - 整合QKV分支(LinearLayer)和VV分支(CovLayer)
    
    官方数据流:
        patch_tokens[::2]  → LinearLayer → patch_tokens_qkv
        patch_tokens[1::2] → CovLayer    → patch_tokens_vv
    """
    
    def __init__(
        self,
        dim_in: int = 256,      # SAM3 FPN特征维度
        dim_out: int = 768,     # 输出维度（对齐文本特征）
        k_linear: int = 4,      # LinearLayer层数
        k_cov: int = 4,         # CovLayer层数
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # QKV分支: LinearLayer
        self.decoder_linear = LinearLayer(dim_in, dim_out, k_linear)
        
        # VV分支: CovLayer (MMCI)
        self.decoder_cov = CovLayer(dim_in, dim_out, k_cov)
        
    def forward(
        self,
        fpn_features: List[torch.Tensor],
        use_alternating: bool = True,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        处理FPN特征
        
        Args:
            fpn_features: List of (B, C, H, W) FPN特征
            use_alternating: 是否交替分配（FiLo风格）
            
        Returns:
            dict containing:
                - 'patch_tokens_qkv': QKV分支输出 [(B, N, C), ...]
                - 'patch_tokens_vv': VV分支输出 [(B, N, C), ...]
                - 'level_sizes': 各层空间尺寸
        """
        level_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn_features]
        
        if use_alternating and len(fpn_features) >= 2:
            # FiLo风格：交替分配
            # 偶数索引 [0, 2, ...] → LinearLayer (QKV)
            # 奇数索引 [1, 3, ...] → CovLayer (VV)
            qkv_inputs = [fpn_features[i] for i in range(0, len(fpn_features), 2)]
            vv_inputs = [fpn_features[i] for i in range(1, len(fpn_features), 2)]
            qkv_sizes = [level_sizes[i] for i in range(0, len(level_sizes), 2)]
            vv_sizes = [level_sizes[i] for i in range(1, len(level_sizes), 2)]
        else:
            # 所有层都送入两个分支
            qkv_inputs = fpn_features
            vv_inputs = fpn_features
            qkv_sizes = level_sizes
            vv_sizes = level_sizes
        
        result = {
            'patch_tokens_qkv': [],
            'patch_tokens_vv': [],
            'level_sizes': level_sizes,
            'qkv_sizes': qkv_sizes,
            'vv_sizes': vv_sizes,
        }
        
        if len(qkv_inputs) > 0:
            result['patch_tokens_qkv'] = self.decoder_linear(qkv_inputs)
        
        if len(vv_inputs) > 0:
            result['patch_tokens_vv'] = self.decoder_cov(vv_inputs)
        
        return result


# ==============================================================================
# FiLoAnomalyHead - 异常图生成
# 完全按照FiLo官方forward中的计算
# ==============================================================================

class FiLoAnomalyHead(nn.Module):
    """
    FiLo异常图生成 - 按照官方实现
    
    官方代码:
    ```python
    patch_tokens_qkv[layer] = patch_tokens_qkv[layer] / patch_tokens_qkv[layer].norm(dim=-1, keepdim=True)
    anomaly_map = 100.0 * patch_tokens_qkv[layer] @ text_features.transpose(-2, -1)
    B, L, C = anomaly_map.shape
    H = int(np.sqrt(L))
    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H), size=image_size, mode="bilinear", align_corners=True)
    anomaly_map = torch.softmax(anomaly_map, dim=1)
    ```
    """
    
    def __init__(self, image_size: int = 518):
        super().__init__()
        self.image_size = image_size
        
    def forward(
        self,
        patch_tokens: List[torch.Tensor],  # [(B, N, C), ...]
        text_features: torch.Tensor,        # (B, 2, C) 或 (1, 2, C)
        level_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        """
        生成异常图
        
        Args:
            patch_tokens: 多层patch特征 [(B, N, C), ...]
            text_features: 文本特征 (B, 2, C) [normal, abnormal]
            level_sizes: 各层空间尺寸
            
        Returns:
            anomaly_maps: List of (B, 2, image_size, image_size)
        """
        anomaly_maps = []
        
        for layer_idx, patch_feat in enumerate(patch_tokens):
            B, L, C = patch_feat.shape
            
            # L2归一化 - 官方实现
            patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
            
            # 计算相似度: (B, N, C) @ (B, C, 2) -> (B, N, 2)
            # 官方使用温度100.0
            anomaly_map = 100.0 * patch_feat @ text_features.transpose(-2, -1)
            
            # 获取空间尺寸
            if level_sizes is not None and layer_idx < len(level_sizes):
                H, W = level_sizes[layer_idx]
            else:
                H = int(np.sqrt(L))
                W = H
            
            # 重塑并上采样 - 官方实现
            # anomaly_map: (B, L, 2) -> (B, 2, L) -> (B, 2, H, W)
            anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, W)
            
            anomaly_map = F.interpolate(
                anomaly_map,
                size=self.image_size,
                mode='bilinear',
                align_corners=True,
            )
            
            # Softmax归一化 - 官方实现
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            
            anomaly_maps.append(anomaly_map)
        
        return anomaly_maps


# ==============================================================================
# FiLoModule - 完整的FiLo模块
# ==============================================================================

class FiLoModule(nn.Module):
    """
    完整的FiLo模块 - 整合解码器和异常图生成
    
    用法:
        filo = FiLoModule(dim_in=256, dim_out=768)
        
        # 需要提供text_features (来自prompt_learner)
        outputs = filo(fpn_features, text_features)
        
        # 训练时
        anomaly_maps = outputs['anomaly_maps']  # List of (B, 2, H, W)
        
        # 推理时
        final_map = outputs['aggregated_map']   # (B, 2, H, W)
        anomaly_score = final_map[:, 1, :, :]   # 异常分数图
    """
    
    def __init__(
        self,
        dim_in: int = 256,
        dim_out: int = 768,
        text_dim: int = 256,     # SAM3的text维度
        k_linear: int = 4,
        k_cov: int = 4,
        image_size: int = 518,
        use_alternating: bool = True,
    ):
        super().__init__()
        
        self.use_alternating = use_alternating
        self.dim_out = dim_out
        
        self.decoder = FiLoDecoder(
            dim_in=dim_in,
            dim_out=dim_out,
            k_linear=k_linear,
            k_cov=k_cov,
        )
        
        self.anomaly_head = FiLoAnomalyHead(image_size=image_size)
        
        # 文本投影层：将SAM3的text_dim投影到dim_out
        # 这是必要的，因为SAM3的hidden_dim=256，而FiLo需要768
        if text_dim != dim_out:
            self.text_proj = nn.Linear(text_dim, dim_out)
            print(f"[FiLoModule] text_proj: {text_dim} -> {dim_out}")
        else:
            self.text_proj = nn.Identity()
        
    def forward(
        self,
        fpn_features: List[torch.Tensor],
        text_features: torch.Tensor,
        return_intermediate: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        完整的FiLo前向传播
        
        Args:
            fpn_features: List of (B, C, H, W) FPN特征
            text_features: (B, 2, C) 或 (1, 2, C) 文本特征 [normal, abnormal]
            return_intermediate: 是否返回中间结果
            
        Returns:
            dict containing:
                - 'patch_tokens_qkv': QKV分支特征
                - 'patch_tokens_vv': VV分支特征
                - 'anomaly_maps': 所有异常图
                - 'aggregated_map': 聚合后的最终异常图
        """
        # 1. 解码器处理
        decoder_out = self.decoder(fpn_features, self.use_alternating)
        
        patch_tokens_qkv = decoder_out['patch_tokens_qkv']
        patch_tokens_vv = decoder_out['patch_tokens_vv']
        qkv_sizes = decoder_out['qkv_sizes']
        vv_sizes = decoder_out['vv_sizes']
        
        # 2. 处理text_features维度
        if text_features.dim() == 2:
            # (2, C) -> (1, 2, C)
            text_features = text_features.unsqueeze(0)
        
        B = fpn_features[0].shape[0]
        if text_features.shape[0] == 1 and B > 1:
            text_features = text_features.expand(B, -1, -1)
        
        # 3. 投影text_features到dim_out维度
        # SAM3的hidden_dim=256，需要投影到768才能与patch_tokens匹配
        text_features = self.text_proj(text_features)  # (B, 2, 256) -> (B, 2, 768)
        
        # 4. 生成异常图 - 按照官方流程
        anomaly_maps = []
        
        # QKV分支的异常图
        if len(patch_tokens_qkv) > 0:
            qkv_maps = self.anomaly_head(patch_tokens_qkv, text_features, qkv_sizes)
            anomaly_maps.extend(qkv_maps)
        
        # VV分支的异常图
        if len(patch_tokens_vv) > 0:
            vv_maps = self.anomaly_head(patch_tokens_vv, text_features, vv_sizes)
            anomaly_maps.extend(vv_maps)
        
        # 4. 聚合所有异常图（等权重平均）
        if len(anomaly_maps) > 0:
            aggregated_map = sum(anomaly_maps) / len(anomaly_maps)
        else:
            aggregated_map = None
        
        result = {
            'anomaly_maps': anomaly_maps,
            'aggregated_map': aggregated_map,
        }
        
        if return_intermediate:
            result.update({
                'patch_tokens_qkv': patch_tokens_qkv,
                'patch_tokens_vv': patch_tokens_vv,
                'level_sizes': decoder_out['level_sizes'],
            })
        
        return result


# ==============================================================================
# 可选辅助模块：VVAttention（自注意力增强，与FiLo无关）
# ==============================================================================

class VVAttention(nn.Module):
    """
    Visual-Visual Self-Attention（可选增强模块）
    
    注意：这与FiLo的VV分支(CovLayer)完全不同！
    - FiLo的VV: CovLayer - 6路多形状卷积
    - 这里的VVAttention: 自注意力 - 可选的额外增强
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) 视觉特征
            
        Returns:
            enhanced: (B, C, H, W)
            attn_weights: 注意力权重（可选）
        """
        B, C, H, W = x.shape
        
        # Flatten: (B, C, H, W) -> (B, HW, C)
        x_flat = x.flatten(2).permute(0, 2, 1)
        
        if pos is not None:
            pos_flat = pos.flatten(2).permute(0, 2, 1)
            q = k = x_flat + pos_flat
        else:
            q = k = x_flat
        
        attn_out, attn_weights = self.self_attn(
            query=self.norm1(q),
            key=self.norm1(k),
            value=x_flat,
            need_weights=return_attention,
        )
        
        x_flat = x_flat + self.scale * attn_out
        x_flat = x_flat + self.scale * self.ffn(self.norm2(x_flat))
        
        enhanced = x_flat.permute(0, 2, 1).view(B, C, H, W)
        
        return enhanced, attn_weights


# ==============================================================================
# Stages消融实验配置
# ==============================================================================

class StagesAblationConfig:
    """Stages消融实验配置"""
    
    CONFIGS = {
        # 单层
        'single_0': [0],
        'single_1': [1],
        'single_2': [2],
        'single_3': [3],
        # FiLo风格
        'qkv_only': [0, 2],      # 偶数层 → QKV(LinearLayer)
        'vv_only': [1, 3],       # 奇数层 → VV(CovLayer)
        # 组合
        'low_res': [2, 3],
        'high_res': [0, 1],
        'all': [0, 1, 2, 3],
        'sowa_style': [1, 2, 3],
    }
    
    @classmethod
    def get_config(cls, name: str) -> List[int]:
        return cls.CONFIGS.get(name, [0, 1, 2, 3])


# ==============================================================================
# 工厂函数
# ==============================================================================

def build_filo_module(
    dim_in: int = 256,
    dim_out: int = 768,
    text_dim: int = 256,
    k_linear: int = 4,
    k_cov: int = 4,
    image_size: int = 518,
) -> FiLoModule:
    """构建完整FiLo模块"""
    return FiLoModule(
        dim_in=dim_in,
        dim_out=dim_out,
        text_dim=text_dim,
        k_linear=k_linear,
        k_cov=k_cov,
        image_size=image_size,
    )


# ==============================================================================
# 模块说明
# ==============================================================================

"""
FiLo模块说明（完全按照官方FiLo/models/FiLo.py移植）
====================================================

对应关系：
---------
| 本文件 | FiLo官方 | 作用 |
|--------|----------|------|
| LinearLayer | LinearLayer | QKV分支线性投影 |
| CovLayer | CovLayer | VV分支6路卷积(MMCI) |
| FiLoDecoder | decoder_linear + decoder_cov | 双分支解码 |
| FiLoAnomalyHead | forward中的异常图计算 | 生成异常图 |
| FiLoModule | FiLo类 | 完整流程 |

官方数据流：
-----------
# FiLo.forward中:
patch_tokens_qkv = self.decoder_linear(patch_tokens[::2])   # 偶数层
patch_tokens_vv = self.decoder_cov(patch_tokens[1::2])      # 奇数层

# 两个分支都:
patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
anomaly_map = 100.0 * patch_tokens @ text_features.transpose(-2, -1)
anomaly_map = F.interpolate(..., size=image_size, mode="bilinear")
anomaly_map = torch.softmax(anomaly_map, dim=1)

适配SAM3：
---------
FPN[0,2,...] → LinearLayer → QKV特征 → normalize → 100*(patch@text.T) → softmax
FPN[1,3,...] → CovLayer    → VV特征  → normalize → 100*(patch@text.T) → softmax

使用示例：
---------
# 在model_wrapper.py中
self.filo_module = FiLoModule(dim_in=256, dim_out=768)

# forward中
text_features = ...  # 从prompt_learner获取, shape (B, 2, C)
filo_out = self.filo_module(fpn_features, text_features)
anomaly_map = filo_out['aggregated_map']  # (B, 2, H, W)
anomaly_score = anomaly_map[:, 1, :, :]   # 取异常通道
"""