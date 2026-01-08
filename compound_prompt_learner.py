"""
compound_prompt_learner.py (V3 - 按用户公式设计)
==========================================

按用户设计的公式实现：

P^n = [V_1(x)...V_i(x)][w_1(x)...w_i(x)][normal][object]
P^a = [V_1(x)...V_i(x)][W_1(x)...W_i(x)][anomaly][object]

其中：
- V: 共享的正常基础向量（从正常样本学习），normal/abnormal 都用
- w (小写): 疑似异常向量，用于 normal prompt（处理 hard negatives）
- W (大写): 异常向量，用于 abnormal prompt（K个独立的）

关键约束：
- w 与 V 正交（避免影响正常判断）
- W 与 V 正交（异常特征独立）
- 不同的 W_k 之间鼓励正交（覆盖不同异常类型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union
import math


class PatchMetaNet(nn.Module):
    """
    Patch-based Meta-Net (参考FAPrompt设计)
    
    输入: 拼接的top-k patch特征 (B, k * vis_dim)
    输出: 上下文bias (B, ctx_dim)
    """
    
    def __init__(
        self,
        input_dim: int,    # k * vis_dim
        ctx_dim: int,      # 输出维度
        reduction: int = 16,
    ):
        super().__init__()
        hidden_dim = max(input_dim // reduction, 64)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ctx_dim),
        )
        
        # 初始化为0，训练初期bias很小
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompoundPromptLearnerV3(nn.Module):
    """
    Compound Prompt Learner V3 - 按用户公式设计
    
    Prompt结构：
    - P^n = [V] + [w] + [normal][object]    → 正常 prompt
    - P^a = [V] + [W_k] + [anomaly][object] → K 个独立的异常 prompt
    
    其中：
    - V (n_V tokens): 共享的正常基础向量，两种 prompt 都用
    - w (n_w tokens): 疑似异常向量，只用于 normal prompt，处理 hard negatives
    - W (K × n_W tokens): 异常向量，只用于 abnormal prompt
    
    正交约束：
    - w ⊥ V: 疑似异常不影响正常判断
    - W_k ⊥ V: 异常特征独立于正常基础
    - W_i ⊥ W_j (i≠j): 不同异常类型相互独立
    """
    
    def __init__(
        self,
        text_encoder: nn.Module,
        n_V: int = 4,                       # V 向量数量（共享正常基础）
        n_w: int = 2,                       # w 向量数量（疑似异常）
        n_W: int = 2,                       # W 向量数量（异常偏移）
        num_abnormal_prompts: int = 10,     # K 个独立的 W
        mode: str = "cocoop",               # "coop" or "cocoop"
        vis_dim: int = 256,                 # 视觉特征维度
        top_k: int = 10,                    # 选择的 top-k patch 数量
        meta_net_reduction: int = 16,       # Meta-Net 缩减因子
        freeze_text_encoder: bool = True,
        output_dim: int = 256,              # 输出维度（SAM3 hidden_dim）
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.n_V = n_V
        self.n_w = n_w
        self.n_W = n_W
        self.num_abnormal_prompts = num_abnormal_prompts  # K
        self.mode = mode
        self.vis_dim = vis_dim
        self.top_k = top_k
        self.output_dim = output_dim
        
        # 获取文本编码器的维度
        self.ctx_dim = self._get_embedding_dim()
        
        # 冻结文本编码器
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # =====================================================================
        # 1. V 向量 [共享的正常基础] - 从正常样本学习，共享给两种 prompt
        # =====================================================================
        V_vectors = torch.empty(n_V, self.ctx_dim)
        nn.init.normal_(V_vectors, std=0.02)
        self.V = nn.Parameter(V_vectors)  # (n_V, ctx_dim)
        
        # =====================================================================
        # 2. w 向量 [疑似异常] - 只用于 normal prompt，处理 hard negatives
        # =====================================================================
        w_vectors = torch.empty(n_w, self.ctx_dim)
        nn.init.normal_(w_vectors, std=0.02)
        self.w = nn.Parameter(w_vectors)  # (n_w, ctx_dim)
        
        # =====================================================================
        # 3. W 向量 [异常偏移] - K 个独立的，只用于 abnormal prompt
        # =====================================================================
        W_vectors = torch.empty(num_abnormal_prompts, n_W, self.ctx_dim)
        nn.init.normal_(W_vectors, std=0.02)
        self.W = nn.Parameter(W_vectors)  # (K, n_W, ctx_dim)
        
        # =====================================================================
        # 4. Patch Meta-Net (CoCoOp 模式)
        # =====================================================================
        self.patch_meta_net = None
        if mode == "cocoop":
            meta_input_dim = vis_dim * top_k
            self.patch_meta_net = PatchMetaNet(
                input_dim=meta_input_dim,
                ctx_dim=self.ctx_dim,
                reduction=meta_net_reduction,
            )
            print(f"[CompoundPromptLearnerV3] CoCoOp mode: PatchMetaNet input_dim={meta_input_dim}")
        
        # =====================================================================
        # 5. 输出投影层 (ctx_dim -> output_dim)
        # =====================================================================
        self.output_proj = None
        if self.ctx_dim != self.output_dim:
            self.output_proj = nn.Linear(self.ctx_dim, self.output_dim)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            print(f"[CompoundPromptLearnerV3] Output projection: {self.ctx_dim} -> {self.output_dim}")
        
        # DAP 相关（可选）
        self.enable_dap = False
        
        print(f"[CompoundPromptLearnerV3] Initialized:")
        print(f"  V={n_V} tokens (shared normal basis)")
        print(f"  w={n_w} tokens (suspected anomaly for normal prompt)")
        print(f"  W={n_W} tokens × K={num_abnormal_prompts} (anomaly offsets)")
        print(f"  mode={mode}, ctx_dim={self.ctx_dim}")
    
    def _get_embedding_dim(self) -> int:
        """获取文本嵌入维度"""
        if hasattr(self.text_encoder, 'token_embedding'):
            return self.text_encoder.token_embedding.embedding_dim
        elif hasattr(self.text_encoder, 'model') and hasattr(self.text_encoder.model, 'token_embedding'):
            return self.text_encoder.model.token_embedding.embedding_dim
        else:
            for name, module in self.text_encoder.named_modules():
                if isinstance(module, nn.Embedding):
                    return module.embedding_dim
            return 512  # 默认值
    
    def select_top_k_patches(
        self,
        patch_features: torch.Tensor,
        vis_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        选择异常分数最高的 top-k 个 patch 特征
        
        Args:
            patch_features: (B, N, C) patch 特征
            vis_global: (B, C) 全局特征
        
        Returns:
            selected: (B, k, C) 选中的 patch 特征
        """
        B, N, C = patch_features.shape
        
        # 计算异常分数：与全局特征的差异
        patch_diff = patch_features - vis_global.unsqueeze(1)  # (B, N, C)
        anomaly_scores = patch_diff.norm(dim=-1)  # (B, N)
        
        # 选择 top-k
        k = min(self.top_k, N)
        _, top_indices = anomaly_scores.topk(k, dim=1)  # (B, k)
        
        # 获取对应的 patch 特征
        batch_indices = torch.arange(B, device=patch_features.device).unsqueeze(1).expand(-1, k)
        selected = patch_features[batch_indices, top_indices]  # (B, k, C)
        
        return selected
    
    def forward(
        self,
        prompt_lists,
        vis_feats=None,
        patch_features=None,
        device=None,
    ):
        """
        生成 compound prompts
        
        Returns:
            all_prefixes: (B, 1+K, prefix_len, output_dim)
                - [0]: normal prompt = [V] + [w]
                - [1:K+1]: K 个独立的 abnormal prompts = [V] + [W_k]
            prompt_mask: (B, 1+K, prefix_len)
            selected_patches: (B, k, C) 用于可视化（可选）
        """
        if device is None:
            device = self.V.device
        
        B = len(prompt_lists)
        K = self.num_abnormal_prompts
        
        # =====================================================================
        # 1. 计算 bias (CoCoOp 模式)
        # =====================================================================
        bias = None
        selected_patches = None
        
        if self.mode == "cocoop" and self.patch_meta_net is not None:
            if patch_features is not None and vis_feats is not None:
                # 获取全局特征
                if vis_feats.dim() == 4:
                    vis_global = vis_feats.mean(dim=[2, 3])  # (B, C)
                else:
                    vis_global = vis_feats.mean(dim=1)
                
                # 选择 top-k patch
                selected_patches = self.select_top_k_patches(patch_features, vis_global)  # (B, k, C)
                
                # 拼接并通过 Meta-Net
                selected_flat = selected_patches.view(B, -1)  # (B, k * C)
                bias = self.patch_meta_net(selected_flat)  # (B, ctx_dim)
                bias = bias.unsqueeze(1)  # (B, 1, ctx_dim)
        
        # =====================================================================
        # 2. 构建 Normal Prompt: P^n = [V] + [w]
        # =====================================================================
        V_batch = self.V.unsqueeze(0).expand(B, -1, -1)  # (B, n_V, ctx_dim)
        w_batch = self.w.unsqueeze(0).expand(B, -1, -1)  # (B, n_w, ctx_dim)
        
        normal_prefix = torch.cat([V_batch, w_batch], dim=1)  # (B, n_V + n_w, ctx_dim)
        
        # =====================================================================
        # 3. 构建 K 个独立的 Abnormal Prompts: P^a_k = [V] + [W_k]
        # =====================================================================
        all_abnormal_prefixes = []
        
        for k in range(K):
            W_k = self.W[k]  # (n_W, ctx_dim)
            W_k_batch = W_k.unsqueeze(0).expand(B, -1, -1)  # (B, n_W, ctx_dim)
            
            # 如果有 bias，加到 W_k 上（图像条件化）
            if bias is not None:
                W_k_batch = W_k_batch + bias
            
            abnormal_prefix_k = torch.cat([V_batch, W_k_batch], dim=1)  # (B, n_V + n_W, ctx_dim)
            all_abnormal_prefixes.append(abnormal_prefix_k)
        
        # Stack 所有 abnormal prefixes: (B, K, prefix_len, ctx_dim)
        abnormal_prefixes_stacked = torch.stack(all_abnormal_prefixes, dim=1)
        
        # =====================================================================
        # 4. 合并 normal + K 个 abnormal
        # =====================================================================
        # 注意：normal 和 abnormal 的 prefix_len 可能不同
        # normal: n_V + n_w
        # abnormal: n_V + n_W
        # 需要 padding 到相同长度
        
        normal_len = self.n_V + self.n_w
        abnormal_len = self.n_V + self.n_W
        max_len = max(normal_len, abnormal_len)
        
        if normal_len < max_len:
            pad_normal = torch.zeros(B, max_len - normal_len, self.ctx_dim, device=device)
            normal_prefix = torch.cat([normal_prefix, pad_normal], dim=1)
        
        if abnormal_len < max_len:
            pad_abnormal = torch.zeros(B, K, max_len - abnormal_len, self.ctx_dim, device=device)
            abnormal_prefixes_stacked = torch.cat([abnormal_prefixes_stacked, pad_abnormal], dim=2)
        
        # (B, 1+K, max_len, ctx_dim)
        all_prefixes = torch.cat([
            normal_prefix.unsqueeze(1),
            abnormal_prefixes_stacked
        ], dim=1)
        
        # 投影到输出维度
        if self.output_proj is not None:
            original_shape = all_prefixes.shape
            all_prefixes = all_prefixes.reshape(-1, original_shape[-1])
            all_prefixes = self.output_proj(all_prefixes)
            all_prefixes = all_prefixes.view(original_shape[0], original_shape[1], original_shape[2], -1)
        
        # =====================================================================
        # 5. 创建 mask
        # =====================================================================
        prompt_mask = torch.zeros(B, 1 + K, max_len, dtype=torch.bool, device=device)
        # 标记 padding 位置
        if normal_len < max_len:
            prompt_mask[:, 0, normal_len:] = True
        if abnormal_len < max_len:
            prompt_mask[:, 1:, abnormal_len:] = True
        
        return all_prefixes, prompt_mask, selected_patches
    
    def compute_orthogonal_loss_prototype_level(self) -> torch.Tensor:
        """
        【改进版】在 prototype 级别计算正交损失
        
        按分析文档的建议：
        - 对最终用于相似度打分的 prototype 做正交，而不是仅对参数做正交
        - t_n: normal prototype（由 V 产生）
        - t_a: abnormal prototype（由 W_k 聚合产生）
        - t_s: suspicious prototype（由 w 产生）
        
        正则：
        - cos 去相关: λ(|cos(t_s, t_n)| + |cos(t_s, t_a)|)
        - W_i ⊥ W_j: Gram 矩阵约束
        """
        loss = torch.tensor(0.0, device=self.V.device)
        
        # 计算 prototypes（token 均值）
        t_n = F.normalize(self.V.mean(dim=0), dim=-1)   # normal prototype
        t_s = F.normalize(self.w.mean(dim=0), dim=-1)   # suspicious prototype
        
        # W_k 的均值作为 abnormal prototype
        W_means = []
        K = self.W.shape[0]
        for k in range(K):
            W_k_mean = F.normalize(self.W[k].mean(dim=0), dim=-1)
            W_means.append(W_k_mean)
        W_means = torch.stack(W_means, dim=0)  # (K, D)
        t_a = F.normalize(W_means.mean(dim=0), dim=-1)  # aggregated abnormal prototype
        
        # =====================================================================
        # 1. t_s ⊥ t_n 和 t_s ⊥ t_a
        # =====================================================================
        loss_s_n = (t_s @ t_n).abs()
        loss_s_a = (t_s @ t_a).abs()
        loss = loss + loss_s_n + loss_s_a
        
        # =====================================================================
        # 2. W_i ⊥ W_j (Gram 矩阵约束)
        # =====================================================================
        if K > 1:
            W_means_norm = F.normalize(W_means, dim=-1)  # (K, D)
            gram = W_means_norm @ W_means_norm.t()  # (K, K)
            
            # 惩罚非对角项
            off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=gram.device)
            off_diag = gram[off_diag_mask]
            loss_W_W = off_diag.abs().mean()
            loss = loss + loss_W_W
        
        return loss

    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        计算正交损失，确保：
        1. w ⊥ V: 疑似异常不影响正常判断
        2. W_k ⊥ V: 异常特征独立于正常基础
        3. W_i ⊥ W_j (i≠j): 不同异常类型相互独立
        
        Returns:
            loss: 正交损失
        """
        loss = torch.tensor(0.0, device=self.V.device)
        
        # 归一化
        V_norm = F.normalize(self.V, dim=-1)  # (n_V, D)
        w_norm = F.normalize(self.w, dim=-1)  # (n_w, D)
        
        # 1. w ⊥ V
        # V 和 w 的平均表示
        V_mean = V_norm.mean(dim=0)  # (D,)
        w_mean = w_norm.mean(dim=0)  # (D,)
        loss_w_V = (V_mean @ w_mean).abs()
        loss = loss + loss_w_V
        
        # 2. W_k ⊥ V (对每个 W_k)
        K = self.W.shape[0]
        for k in range(K):
            W_k_norm = F.normalize(self.W[k], dim=-1)  # (n_W, D)
            W_k_mean = W_k_norm.mean(dim=0)  # (D,)
            loss_W_V = (V_mean @ W_k_mean).abs()
            loss = loss + loss_W_V / K
        
        # 3. W_i ⊥ W_j (i≠j)
        if K > 1:
            W_means = []
            for k in range(K):
                W_k_norm = F.normalize(self.W[k], dim=-1)
                W_means.append(W_k_norm.mean(dim=0))
            W_means = torch.stack(W_means, dim=0)  # (K, D)
            W_means_norm = F.normalize(W_means, dim=-1)
            
            # 计算两两相似度
            sim_matrix = W_means_norm @ W_means_norm.t()  # (K, K)
            # 排除对角线
            off_diag = sim_matrix - torch.eye(K, device=sim_matrix.device)
            loss_W_W = off_diag.abs().sum() / (K * (K - 1))
            loss = loss + loss_W_W
        
        return loss
    
    def compute_prior_loss(self) -> torch.Tensor:
        """
        计算先验正则化损失，防止向量过大
        """
        loss = torch.tensor(0.0, device=self.V.device)
        
        # L2 正则
        loss = loss + self.V.pow(2).mean()
        loss = loss + self.w.pow(2).mean()
        loss = loss + self.W.pow(2).mean()
        
        return loss * 0.1  # 小权重
    
    def compute_contrast_loss(
        self,
        is_anomaly: Optional[List[bool]] = None,
        decoder_hs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算对比损失，确保：
        - 正常样本的 query 更接近 V
        - 异常样本的 matched query 更接近对应的 W_k
        """
        # 简化版：仅基于参数的对比
        loss = torch.tensor(0.0, device=self.V.device)
        
        # V 和 W 应该分离
        V_mean = F.normalize(self.V.mean(dim=0), dim=-1)
        W_mean = F.normalize(self.W.mean(dim=(0, 1)), dim=-1)
        
        # 希望 V 和 W 的平均距离尽可能大
        sim = (V_mean @ W_mean).abs()
        loss = 1.0 - sim + 0.3  # margin = 0.3
        loss = F.relu(loss)
        
        return loss


# =====================================================================
# 兼容旧接口的别名
# =====================================================================
CompoundPromptLearnerV2 = CompoundPromptLearnerV3


def build_compound_prompt_learner(
    text_encoder: nn.Module,
    mode: str = "cocoop",
    n_ctx: int = 4,
    n_ctx_offset: int = 2,
    num_abnormal: int = 10,
    vis_dim: int = 256,
    output_dim: int = 256,
    **kwargs,
) -> CompoundPromptLearnerV3:
    """
    工厂函数，创建 Compound Prompt Learner
    
    映射关系：
    - n_ctx -> n_V (共享正常基础)
    - n_ctx_offset -> n_w 和 n_W (统一大小)
    - num_abnormal -> num_abnormal_prompts (K)
    """
    return CompoundPromptLearnerV3(
        text_encoder=text_encoder,
        n_V=n_ctx,
        n_w=n_ctx_offset,
        n_W=n_ctx_offset,
        num_abnormal_prompts=num_abnormal,
        mode=mode,
        vis_dim=vis_dim,
        output_dim=output_dim,
        **kwargs,
    )


# =====================================================================
# 兼容旧接口的别名
# =====================================================================
CompoundPromptLearner = CompoundPromptLearnerV3
CompoundPromptLearnerV2 = CompoundPromptLearnerV3