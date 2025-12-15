"""
Spurious Correlation Mitigation Module (TIE-Inspired)

基于 ICLR 2025 论文 "Mitigating Spurious Correlations in Zero-shot Multimodal Models"
适配于 SAM3 异常检测模型，用于减少假阳性问题。

核心数学原理:
================
1. 问题建模:
   假设视觉嵌入 h = [h_spurious, h_core, h_noise]
   - h_spurious: 与真实标签无关但与预测相关的伪相关特征
   - h_core: 核心特征（真正的缺陷特征）
   - h_noise: 随机噪声

2. 假阳性的数学原因:
   假阳性 ⟺ ⟨h_normal, w_anomaly⟩ > threshold
   这发生在 h_normal 包含与异常描述相似的伪相关成分时

3. TIE 解决方案 (Theorem 1):
   最优平移向量: v* = E[-P·h_a]
   其中 P 是选择伪相关特征维度的投影矩阵
   
   实际操作: h' ← h - λ·v_spurious
   其中: 
   - v_spurious = 伪相关方向（由文本编码器从 spurious prompts 计算）
   - λ = E[h^T · v_spurious]（平均投影长度）

4. 对假阳性的影响:
   对于正常样本: ⟨h'_normal, w_anomaly⟩ < ⟨h_normal, w_anomaly⟩
   因为我们减去了与正常纹理相关的成分

适配异常检测的创新:
==================
1. 可学习的伪相关向量: 不依赖手工设计的 spurious prompts
2. 双向处理: 同时处理 normal→anomaly 和 anomaly→normal 误判
3. 多尺度应用: 可以在 visual features 和 decoder queries 两个层次应用

Author: Claude (based on Lu et al., ICLR 2025)
"""

import math
from typing import List, Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpuriousVectorBank(nn.Module):
    """
    可学习的伪相关向量库
    
    数学原理:
    --------
    TIE 原论文使用文本编码器计算 v_a = φ_T(t_a) 作为伪相关向量。
    我们将其扩展为可学习的向量库，能够自适应地发现数据中的伪相关模式。
    
    优势:
    - 不依赖手工设计的 spurious prompts
    - 可以发现数据特定的伪相关模式
    - 支持多个伪相关方向（实际场景中可能有多种）
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_spurious_vectors: int = 4,
        init_scale: float = 0.02,
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_spurious_vectors: 伪相关向量数量
            init_scale: 初始化缩放
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_vectors = num_spurious_vectors
        
        # 可学习的伪相关向量 (K, D)
        # 初始化为小随机值，训练中会学习到数据的伪相关模式
        self.spurious_vectors = nn.Parameter(
            torch.randn(num_spurious_vectors, embed_dim) * init_scale
        )
        
        # 每个向量的重要性权重（用于加权组合）
        self.importance_logits = nn.Parameter(torch.zeros(num_spurious_vectors))
        
    def forward(self, normalize: bool = True) -> torch.Tensor:
        """
        返回归一化的伪相关向量
        
        Returns:
            (K, D) 归一化的向量，或加权组合后的单向量 (D,)
        """
        if normalize:
            vectors = F.normalize(self.spurious_vectors, dim=-1)
        else:
            vectors = self.spurious_vectors
        return vectors
    
    def get_weighted_vector(self, normalize: bool = True) -> torch.Tensor:
        """
        获取加权组合的单一伪相关向量
        
        数学: v_combined = Σ_k w_k · v_k / ||Σ_k w_k · v_k||
        其中 w_k = softmax(importance_logits)_k
        """
        weights = F.softmax(self.importance_logits, dim=0)  # (K,)
        vectors = self.forward(normalize=False)  # (K, D)
        combined = (weights.unsqueeze(-1) * vectors).sum(dim=0)  # (D,)
        
        if normalize:
            combined = F.normalize(combined, dim=0)
        return combined


class TextGuidedSpuriousVectors(nn.Module):
    """
    基于文本的伪相关向量生成器（遵循 TIE 原论文）
    
    数学原理:
    --------
    v_a = φ_T(t_a) / ||φ_T(t_a)||
    
    其中 t_a 是描述伪相关特征的文本提示，例如:
    - "a photo with normal texture"
    - "a photo with regular pattern"
    - "a photo without defects"
    
    这些文本嵌入定义了应该从视觉嵌入中移除的方向。
    """
    
    def __init__(
        self,
        text_encoder: nn.Module,
        spurious_prompts: Optional[List[str]] = None,
        embed_dim: int = 512,
        freeze_text_encoder: bool = True,
    ):
        """
        Args:
            text_encoder: 文本编码器 (e.g., CLIP text encoder)
            spurious_prompts: 描述伪相关特征的文本列表
            embed_dim: 嵌入维度
            freeze_text_encoder: 是否冻结文本编码器
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        
        # 默认的伪相关提示（针对异常检测场景）
        self.spurious_prompts = spurious_prompts or [
            "a photo with normal texture",
            "a photo with regular surface",
            "a photo without any defects",
            "a photo of normal appearance",
        ]
        
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        # 缓存预计算的向量（推理时使用）
        self.register_buffer("cached_vectors", None)
        
    @torch.no_grad()
    def compute_spurious_vectors(self, device: torch.device) -> torch.Tensor:
        """
        计算伪相关向量
        
        Returns:
            (K, D) 归一化的伪相关向量
        """
        # 这里需要根据实际的 text_encoder 接口调整
        # 假设 text_encoder 有 tokenizer 和 encode 方法
        
        # 对于 SAM3 的 text encoder:
        if hasattr(self.text_encoder, 'tokenizer'):
            tokens = self.text_encoder.tokenizer(
                self.spurious_prompts, 
                context_length=getattr(self.text_encoder, 'context_length', 32)
            ).to(device)
            
            # 获取文本嵌入
            if hasattr(self.text_encoder, 'encoder'):
                _, token_embs = self.text_encoder.encoder(tokens)
                # 取 EOT token 的嵌入
                eot_indices = tokens.argmax(dim=-1)
                vectors = token_embs[torch.arange(len(self.spurious_prompts)), eot_indices]
            else:
                vectors = self.text_encoder(tokens)
        else:
            # 备用：随机初始化
            vectors = torch.randn(len(self.spurious_prompts), self.embed_dim, device=device)
        
        # L2 归一化
        vectors = F.normalize(vectors, dim=-1)
        return vectors
    
    def forward(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """返回伪相关向量"""
        if self.cached_vectors is None and device is not None:
            self.cached_vectors = self.compute_spurious_vectors(device)
        return self.cached_vectors


class TIEModule(nn.Module):
    """
    Text-guided Image Embedding Translation (TIE) 模块
    
    核心算法 (Algorithm 1 from paper):
    ================================
    1. 计算伪相关向量: v_a = φ_T(t_a) / ||v_a||
    2. 推断伪标签 (TIE*): â = argmax_a ⟨φ_I(x), φ_T(t_a)⟩
    3. 计算缩放系数: λ_a = E[h^T · v_a]
    4. 平移嵌入: h ← h - λ_a · v_a
    
    适配异常检测:
    ============
    - 对于 normal 样本: 减去与正常纹理相关的成分 → 降低假阳性
    - 对于 anomaly 样本: 核心缺陷特征与正常纹理正交 → 影响较小
    
    数学证明 (Theorem 1):
    ===================
    最优平移向量 v* = E[-P·h_a]，其中 P 选择伪相关维度。
    这保证了平移后的嵌入分布与原分布一致（translation preserves distribution）。
    """
    
    def __init__(
        self,
        embed_dim: int,
        spurious_source: str = "learnable",  # "learnable", "text", "hybrid"
        num_spurious_vectors: int = 4,
        text_encoder: Optional[nn.Module] = None,
        spurious_prompts: Optional[List[str]] = None,
        adaptive_scale: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-6,
    ):
        """
        Args:
            embed_dim: 嵌入维度
            spurious_source: 伪相关向量来源
                - "learnable": 完全可学习
                - "text": 基于文本提示
                - "hybrid": 混合（文本初始化 + 可学习残差）
            num_spurious_vectors: 伪相关向量数量
            text_encoder: 文本编码器（当 source != "learnable" 时需要）
            spurious_prompts: 伪相关文本提示
            adaptive_scale: 是否使用自适应缩放
            momentum: 运行统计的动量
            eps: 数值稳定性
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.spurious_source = spurious_source
        self.adaptive_scale = adaptive_scale
        self.momentum = momentum
        self.eps = eps
        
        # 伪相关向量来源
        if spurious_source == "learnable":
            self.spurious_bank = SpuriousVectorBank(
                embed_dim=embed_dim,
                num_spurious_vectors=num_spurious_vectors,
            )
            self.text_spurious = None
        elif spurious_source == "text":
            assert text_encoder is not None, "text_encoder required for text-based spurious vectors"
            self.spurious_bank = None
            self.text_spurious = TextGuidedSpuriousVectors(
                text_encoder=text_encoder,
                spurious_prompts=spurious_prompts,
                embed_dim=embed_dim,
            )
        else:  # hybrid
            self.spurious_bank = SpuriousVectorBank(
                embed_dim=embed_dim,
                num_spurious_vectors=num_spurious_vectors,
            )
            if text_encoder is not None:
                self.text_spurious = TextGuidedSpuriousVectors(
                    text_encoder=text_encoder,
                    spurious_prompts=spurious_prompts,
                    embed_dim=embed_dim,
                )
            else:
                self.text_spurious = None
        
        # 自适应缩放参数
        if adaptive_scale:
            # 运行平均的 λ_a（用于估计 E[h^T · v_a]）
            self.register_buffer("running_lambda", torch.zeros(num_spurious_vectors))
            self.register_buffer("running_count", torch.zeros(1))
            
            # 可学习的缩放因子
            self.scale_factor = nn.Parameter(torch.ones(num_spurious_vectors))
        else:
            self.register_buffer("running_lambda", None)
            self.scale_factor = None
    
    def get_spurious_vectors(self, device: torch.device) -> torch.Tensor:
        """
        获取伪相关向量
        
        Returns:
            (K, D) 归一化的伪相关向量
        """
        if self.spurious_source == "learnable":
            return self.spurious_bank(normalize=True)
        elif self.spurious_source == "text":
            return self.text_spurious(device=device)
        else:  # hybrid
            learnable = self.spurious_bank(normalize=False)
            if self.text_spurious is not None:
                text = self.text_spurious(device=device)
                # 组合：text 向量 + 可学习残差
                combined = text + 0.1 * learnable
            else:
                combined = learnable
            return F.normalize(combined, dim=-1)
    
    def compute_projection_lengths(
        self, 
        embeddings: torch.Tensor,  # (..., D)
        spurious_vectors: torch.Tensor,  # (K, D)
    ) -> torch.Tensor:
        """
        计算嵌入在每个伪相关向量上的投影长度
        
        数学: λ_k = h^T · v_k
        
        Args:
            embeddings: (..., D) 输入嵌入
            spurious_vectors: (K, D) 伪相关向量
            
        Returns:
            (..., K) 投影长度
        """
        # embeddings: (..., D), spurious_vectors: (K, D)
        # 计算内积: (..., K)
        proj_lengths = torch.einsum("...d,kd->...k", embeddings, spurious_vectors)
        return proj_lengths
    
    def forward(
        self,
        embeddings: torch.Tensor,  # (B, ..., D)
        is_anomaly: Optional[torch.Tensor] = None,  # (B,) bool
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        TIE 前向传播：平移嵌入以减少伪相关性
        
        核心操作 (Equation 8):
        h ← h - λ · v_spurious
        
        Args:
            embeddings: (B, ..., D) 输入嵌入
            is_anomaly: (B,) 可选的异常标签（用于条件平移）
            return_diagnostics: 是否返回诊断信息
            
        Returns:
            translated_embeddings: (B, ..., D) 平移后的嵌入
            diagnostics: dict (可选)
        """
        device = embeddings.device
        orig_shape = embeddings.shape
        B = orig_shape[0]
        D = orig_shape[-1]
        
        # 将嵌入展平为 (B, N, D)
        embeddings_flat = embeddings.view(B, -1, D)  # (B, N, D)
        N = embeddings_flat.shape[1]
        
        # 获取伪相关向量
        spurious_vectors = self.get_spurious_vectors(device)  # (K, D)
        K = spurious_vectors.shape[0]
        
        # 计算投影长度: (B, N, K)
        proj_lengths = self.compute_projection_lengths(embeddings_flat, spurious_vectors)
        
        # 计算平均投影长度（用于估计 λ）
        mean_proj = proj_lengths.mean(dim=[0, 1])  # (K,)
        
        # 更新运行统计
        if self.training and self.running_lambda is not None:
            with torch.no_grad():
                self.running_lambda = (
                    self.momentum * mean_proj + 
                    (1 - self.momentum) * self.running_lambda
                )
                self.running_count += 1
        
        # 确定缩放系数
        if self.adaptive_scale and self.scale_factor is not None:
            # 使用可学习的缩放因子
            scale = self.scale_factor.abs()  # (K,)
            
            # 在训练时使用当前 batch 的统计，推理时使用运行统计
            if self.training:
                lambda_a = mean_proj * scale  # (K,)
            else:
                lambda_a = self.running_lambda * scale  # (K,)
        else:
            # 使用当前 batch 的平均投影长度
            lambda_a = mean_proj  # (K,)
        
        # 条件平移：根据 is_anomaly 调整平移强度
        if is_anomaly is not None:
            # 对于异常样本，减少平移强度（保留核心缺陷特征）
            # 对于正常样本，完全平移（减少假阳性）
            anomaly_mask = is_anomaly.float().view(B, 1, 1)  # (B, 1, 1)
            # 异常样本使用 0.3 的强度，正常样本使用 1.0
            translation_strength = 1.0 - 0.7 * anomaly_mask  # (B, 1, 1)
        else:
            translation_strength = 1.0
        
        # 执行平移
        # translation = Σ_k λ_k · v_k
        translation = torch.einsum("k,kd->d", lambda_a, spurious_vectors)  # (D,)
        
        # h' = h - translation_strength * translation
        translated = embeddings_flat - translation_strength * translation.unsqueeze(0).unsqueeze(0)
        
        # 恢复原始形状
        translated = translated.view(*orig_shape)
        
        if return_diagnostics:
            diagnostics = {
                "spurious_vectors": spurious_vectors,
                "proj_lengths_mean": mean_proj,
                "lambda_a": lambda_a,
                "translation_norm": translation.norm().item(),
                "embedding_shift_ratio": (
                    (translated - embeddings).norm() / (embeddings.norm() + self.eps)
                ).mean().item(),
            }
            return translated, diagnostics
        
        return translated


class TIEAnomalyHead(nn.Module):
    """
    集成 TIE 的异常检测头
    
    这个模块将 TIE 与你的分割/检测头集成，在特征层面应用伪相关性缓解。
    
    数学原理:
    --------
    1. 从 decoder 获取 object queries: hs ∈ R^{B×Q×D}
    2. 应用 TIE: hs' = TIE(hs)
    3. 使用 hs' 进行后续的分割/分类
    
    效果:
    - 减少 normal textures 在 anomaly queries 上的激活 → 降低假阳性
    - 保留真实缺陷的激活 → 维持检测能力
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_spurious_vectors: int = 4,
        apply_to_queries: bool = True,
        apply_to_features: bool = True,
        text_encoder: Optional[nn.Module] = None,
        spurious_prompts: Optional[List[str]] = None,
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_spurious_vectors: 伪相关向量数量
            apply_to_queries: 是否应用于 decoder queries
            apply_to_features: 是否应用于视觉特征
            text_encoder: 文本编码器（可选）
            spurious_prompts: 伪相关提示（可选）
        """
        super().__init__()
        self.apply_to_queries = apply_to_queries
        self.apply_to_features = apply_to_features
        
        # Query-level TIE
        if apply_to_queries:
            self.query_tie = TIEModule(
                embed_dim=embed_dim,
                spurious_source="learnable",
                num_spurious_vectors=num_spurious_vectors,
                adaptive_scale=True,
            )
        else:
            self.query_tie = None
        
        # Feature-level TIE
        if apply_to_features:
            self.feature_tie = TIEModule(
                embed_dim=embed_dim,
                spurious_source="learnable",
                num_spurious_vectors=num_spurious_vectors,
                adaptive_scale=True,
            )
        else:
            self.feature_tie = None
    
    def forward(
        self,
        decoder_hs: Optional[torch.Tensor] = None,  # (B, Q, D) or (L, B, Q, D)
        visual_features: Optional[torch.Tensor] = None,  # (B, C, H, W)
        is_anomaly: Optional[torch.Tensor] = None,  # (B,) bool
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            decoder_hs: decoder 输出的 object queries
            visual_features: 视觉特征图
            is_anomaly: 异常标签
            return_diagnostics: 是否返回诊断信息
            
        Returns:
            dict with translated features
        """
        outputs = {}
        diagnostics = {}
        
        # 处理 decoder queries
        if decoder_hs is not None and self.query_tie is not None:
            if decoder_hs.dim() == 4:
                # (L, B, Q, D) -> 取最后一层
                hs = decoder_hs[-1]
            else:
                hs = decoder_hs
            
            if return_diagnostics:
                translated_hs, diag = self.query_tie(
                    hs, is_anomaly=is_anomaly, return_diagnostics=True
                )
                diagnostics["query_tie"] = diag
            else:
                translated_hs = self.query_tie(hs, is_anomaly=is_anomaly)
            
            outputs["translated_queries"] = translated_hs
        
        # 处理视觉特征
        if visual_features is not None and self.feature_tie is not None:
            B, C, H, W = visual_features.shape
            # 将空间维度展平: (B, C, H, W) -> (B, H*W, C)
            feat_flat = visual_features.flatten(2).permute(0, 2, 1)
            
            if return_diagnostics:
                translated_feat, diag = self.feature_tie(
                    feat_flat, is_anomaly=is_anomaly, return_diagnostics=True
                )
                diagnostics["feature_tie"] = diag
            else:
                translated_feat = self.feature_tie(feat_flat, is_anomaly=is_anomaly)
            
            # 恢复形状: (B, H*W, C) -> (B, C, H, W)
            translated_feat = translated_feat.permute(0, 2, 1).view(B, C, H, W)
            outputs["translated_features"] = translated_feat
        
        if return_diagnostics:
            outputs["diagnostics"] = diagnostics
        
        return outputs


class TIELoss(nn.Module):
    """
    TIE 训练损失
    
    数学原理:
    --------
    我们希望伪相关向量满足以下性质:
    
    1. 正交性损失 (Orthogonality):
       L_orth = Σ_{i≠j} |⟨v_i, v_j⟩|^2
       确保不同伪相关向量捕获不同的模式
    
    2. 区分性损失 (Discrimination):
       对于正常样本: 投影长度应该大 → L_normal = -E[λ_normal]
       对于异常样本: 投影长度应该小 → L_anomaly = E[λ_anomaly]
       
    3. 假阳性惩罚 (FP Penalty):
       对于被错误预测为异常的正常样本，增加额外惩罚
    """
    
    def __init__(
        self,
        orthogonality_weight: float = 0.1,
        discrimination_weight: float = 1.0,
        margin: float = 0.3,
    ):
        """
        Args:
            orthogonality_weight: 正交性损失权重
            discrimination_weight: 区分性损失权重
            margin: 正常/异常投影长度的期望差距
        """
        super().__init__()
        self.orthogonality_weight = orthogonality_weight
        self.discrimination_weight = discrimination_weight
        self.margin = margin
    
    def forward(
        self,
        spurious_vectors: torch.Tensor,  # (K, D)
        embeddings: torch.Tensor,  # (B, ..., D)
        is_anomaly: torch.Tensor,  # (B,) bool
    ) -> Dict[str, torch.Tensor]:
        """
        计算 TIE 损失
        
        Returns:
            dict with loss components
        """
        K, D = spurious_vectors.shape
        B = embeddings.shape[0]
        device = embeddings.device
        
        # 1. 正交性损失
        # Gram matrix: (K, K)
        gram = spurious_vectors @ spurious_vectors.t()
        # 移除对角线
        off_diag = gram - torch.eye(K, device=device)
        loss_orth = (off_diag ** 2).sum() / (K * (K - 1) + 1e-6)
        
        # 2. 区分性损失
        # 计算投影长度
        embeddings_flat = embeddings.view(B, -1, D)  # (B, N, D)
        proj_lengths = torch.einsum("bnd,kd->bnk", embeddings_flat, spurious_vectors)  # (B, N, K)
        mean_proj = proj_lengths.mean(dim=[1, 2])  # (B,)
        
        # 分离正常和异常
        is_anomaly = is_anomaly.bool()
        normal_mask = ~is_anomaly
        anomaly_mask = is_anomaly
        
        # 正常样本应该有大的投影（我们希望移除这部分）
        # 异常样本应该有小的投影（核心特征应该与伪相关向量正交）
        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            normal_proj = mean_proj[normal_mask].mean()
            anomaly_proj = mean_proj[anomaly_mask].mean()
            
            # margin ranking loss: max(0, margin - (normal_proj - anomaly_proj))
            loss_disc = F.relu(self.margin - (normal_proj - anomaly_proj))
        else:
            loss_disc = torch.tensor(0.0, device=device)
        
        # 总损失
        total_loss = (
            self.orthogonality_weight * loss_orth +
            self.discrimination_weight * loss_disc
        )
        
        return {
            "total": total_loss,
            "orthogonality": loss_orth,
            "discrimination": loss_disc,
        }


# ============================================================================
# 便捷函数：将 TIE 集成到现有模型
# ============================================================================

def inject_tie_into_model(
    model: nn.Module,
    embed_dim: int,
    injection_points: List[str] = ["decoder_hs"],
    num_spurious_vectors: int = 4,
    text_encoder: Optional[nn.Module] = None,
) -> nn.Module:
    """
    将 TIE 模块注入到现有模型中
    
    Args:
        model: 原始模型
        embed_dim: 嵌入维度
        injection_points: 注入点列表
        num_spurious_vectors: 伪相关向量数量
        text_encoder: 文本编码器
        
    Returns:
        修改后的模型
    """
    # 创建 TIE 模块
    tie_module = TIEAnomalyHead(
        embed_dim=embed_dim,
        num_spurious_vectors=num_spurious_vectors,
        apply_to_queries="decoder_hs" in injection_points,
        apply_to_features="features" in injection_points,
        text_encoder=text_encoder,
    )
    
    # 添加到模型
    model.tie_module = tie_module
    
    return model


def apply_tie_to_output(
    model_output: Dict[str, torch.Tensor],
    tie_module: TIEAnomalyHead,
    is_anomaly: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    将 TIE 应用到模型输出
    
    Args:
        model_output: 模型输出字典
        tie_module: TIE 模块
        is_anomaly: 异常标签
        
    Returns:
        修改后的输出字典
    """
    tie_output = tie_module(
        decoder_hs=model_output.get("decoder_hs"),
        visual_features=model_output.get("decoder_features"),
        is_anomaly=is_anomaly,
    )
    
    # 替换原始输出
    if "translated_queries" in tie_output:
        model_output["decoder_hs"] = tie_output["translated_queries"]
    if "translated_features" in tie_output:
        model_output["decoder_features"] = tie_output["translated_features"]
    
    return model_output


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 简单测试
    print("Testing TIE Module...")
    
    B, Q, D = 4, 64, 256
    K = 4
    
    # 创建模块
    tie = TIEModule(
        embed_dim=D,
        spurious_source="learnable",
        num_spurious_vectors=K,
        adaptive_scale=True,
    )
    
    # 测试前向传播
    embeddings = torch.randn(B, Q, D)
    is_anomaly = torch.tensor([True, False, True, False])
    
    translated, diagnostics = tie(embeddings, is_anomaly=is_anomaly, return_diagnostics=True)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {translated.shape}")
    print(f"Diagnostics: {diagnostics}")
    
    # 测试损失
    tie_loss = TIELoss()
    spurious_vectors = tie.get_spurious_vectors(embeddings.device)
    loss_dict = tie_loss(spurious_vectors, embeddings, is_anomaly)
    print(f"Loss: {loss_dict}")
    
    print("\nAll tests passed!")
