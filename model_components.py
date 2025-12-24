"""
model_components.py

包含：
- LoRA 相关组件
- CoOpPromptLearner: CoOp风格可学习提示（静态）
- CoCoOpPromptLearner: CoCoOp风格可学习提示（图像条件化）
- 原有的 AveragedPromptLearner, PerClassTemplatePromptLearner
"""

import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================
# LoRA 组件
# ================================================================================

class ParallelLoRA(nn.Module):
    """Parallel low-rank adapter (side-branch)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: Optional[float] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha or rank)
        self.scaling = self.alpha / float(rank)

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        lora_mid = x_flat @ self.lora_A.t()
        update = lora_mid @ self.lora_B.t()
        update = update.view(*orig_shape[:-1], -1)
        return update * self.scaling


class LoRALinear(nn.Module):
    """Lightweight LoRA adapter around a Linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: Optional[float] = None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha or float(rank)
        self.scaling = self.alpha / float(rank)

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        lora_update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return y + lora_update * self.scaling


def apply_lora_to_sam(
    module: nn.Module,
    target_substrings: Sequence[str] = ("qkv",),
    rank: int = 16,
    alpha: Optional[float] = None,
) -> List[str]:
    """Replace Linear layers containing target substrings with LoRA-wrapped versions."""
    wrapped: List[str] = []
    for name, child in list(module.named_children()):
        wrapped.extend(
            apply_lora_to_sam(child, target_substrings=target_substrings, rank=rank, alpha=alpha)
        )
        if isinstance(child, nn.Linear) and any(s in name for s in target_substrings):
            lora = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, lora)
            wrapped.append(name)
    return wrapped


# ================================================================================
# CoOp 提示学习器 (静态可学习向量)
# 论文: Learning to Prompt for Vision-Language Models (Zhou et al., 2022)
# ================================================================================

class CoOpPromptLearner(nn.Module):
    """
    CoOp-style Prompt Learner for Anomaly Detection.
    
    简化版提示格式（推荐）:
        [v1][v2]...[vM] + [anomaly/normal {cls_name}]
    
    完整版提示格式（可选，use_keywords=True）:
        [v1][v2]...[vM] + [anomaly/normal {cls_name}] + [kw_pooled]
    
    Args:
        text_encoder: SAM3的文本编码器
        n_ctx: 可学习上下文向量数量 (默认4)
        ctx_init: 用于初始化的文本 (如"a photo of a")
        freeze_text_encoder: 是否冻结文本编码器
        proj: 投影层
        class_token_position: 类别token位置 ("end"/"middle"/"front")
        use_keywords: 是否使用关键词聚合 (默认False，推荐关闭)
    """
    
    def __init__(
        self,
        text_encoder,
        n_ctx: int = 4,
        ctx_init: str = "",
        freeze_text_encoder: bool = True,
        proj: Optional[nn.Module] = None,
        class_token_position: str = "end",
        use_keywords: bool = False,  # 默认关闭，使用简化版
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.context_length = getattr(text_encoder, "context_length", 32)
        
        # 获取 text encoder 的 width
        self.width = getattr(text_encoder.encoder, "width", None)
        if self.width is None:
            self.width = getattr(text_encoder, "width", None)
        assert self.width is not None, "Cannot determine text encoder embedding width"
        
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.proj = proj if proj is not None else getattr(text_encoder, "resizer", None)
        self.use_keywords = use_keywords
        
        # ===== 核心: 可学习的上下文向量 =====
        if ctx_init and len(ctx_init) > 0:
            # 使用指定文本的word embedding初始化
            ctx_init_tokens = text_encoder.tokenizer([ctx_init], context_length=self.context_length)
            with torch.no_grad():
                _, ctx_init_embeds = text_encoder.encoder(ctx_init_tokens)
                n_available = min(n_ctx, ctx_init_embeds.shape[1] - 1)
                ctx_init_embeds = ctx_init_embeds[0, 1:n_available+1, :]
                if n_available < n_ctx:
                    pad = torch.randn(n_ctx - n_available, self.width) * 0.02
                    ctx_init_embeds = torch.cat([ctx_init_embeds, pad], dim=0)
            self.ctx = nn.Parameter(ctx_init_embeds.clone())
            print(f"[CoOpPromptLearner] 使用 '{ctx_init}' 初始化 {n_ctx} 个上下文向量")
        else:
            # 随机初始化 (std=0.02，与原始CoOp一致)
            self.ctx = nn.Parameter(torch.randn(n_ctx, self.width) * 0.02)
            print(f"[CoOpPromptLearner] 随机初始化 {n_ctx} 个上下文向量")
        
        # 关键词attention (仅当 use_keywords=True 时使用)
        if use_keywords:
            self.q_keyword = nn.Parameter(torch.randn(self.width) * 0.02)
            self.keyword_attn_scale = math.sqrt(self.width)
        
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        print(f"[CoOpPromptLearner] use_keywords={use_keywords}, position={class_token_position}")
    
    @torch.no_grad()
    def _encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """编码文本列表，返回EOT位置的embedding"""
        if len(texts) == 0:
            return torch.zeros((0, self.width), device=device)
        
        tokenized = self.text_encoder.tokenizer(texts, context_length=self.context_length).to(device)
        eot_indices = tokenized.argmax(dim=-1)
        if eot_indices.dim() > 1:
            eot_indices = eot_indices.argmax(dim=1)
        _, token_embeds = self.text_encoder.encoder(tokenized)
        
        batch_indices = torch.arange(len(texts), device=device)
        text_features = token_embeds[batch_indices, eot_indices]
        return text_features.to(device)
    
    def forward(
        self, 
        prompt_lists: Sequence[List[str]],
        image_features: Optional[torch.Tensor] = None,  # CoOp不使用
        class_ids: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prompt_lists: [B]个列表
                简化版: [["{anomaly/normal} {cls}"], ...]
                完整版: [["{anomaly/normal} {cls}", "kw1", "kw2", ...], ...]
            image_features: 未使用 (CoOp是静态的)
            device: 计算设备
        
        Returns:
            prompt_seq: (L, B, W) 提示序列
            prompt_mask: (B, L) 注意力mask
        """
        device = device or self.ctx.device
        B = len(prompt_lists)
        
        batch_features = []
        
        for prompts in prompt_lists:
            if len(prompts) == 0:
                prompts = ["normal"]
            
            # 1) 编码主描述 (第一个prompt: "anomaly bottle" 或 "normal bottle")
            main_desc = prompts[0]
            main_embed = self._encode_text([main_desc], device)  # (1, W)
            
            # 2) 可学习上下文向量
            ctx = self.ctx.to(device)  # (n_ctx, W)
            
            # 3) 关键词聚合 (仅当 use_keywords=True)
            if self.use_keywords and len(prompts) > 1:
                keywords = prompts[1:]
                kw_embeds = self._encode_text(keywords, device)  # (K, W)
                qk = self.q_keyword.view(self.width, 1).to(device)
                kw_scores = torch.matmul(kw_embeds, qk).squeeze(-1) / self.keyword_attn_scale
                kw_weights = torch.softmax(kw_scores, dim=0)
                kw_prototype = (kw_weights.unsqueeze(-1) * kw_embeds).sum(dim=0, keepdim=True)  # (1, W)
            else:
                kw_prototype = None
            
            # 4) 组合: [ctx] + [main_embed] (+ [kw_prototype])
            if self.class_token_position == "end":
                if kw_prototype is not None:
                    combined = torch.cat([ctx, main_embed, kw_prototype], dim=0)
                else:
                    combined = torch.cat([ctx, main_embed], dim=0)
            elif self.class_token_position == "front":
                if kw_prototype is not None:
                    combined = torch.cat([main_embed, ctx, kw_prototype], dim=0)
                else:
                    combined = torch.cat([main_embed, ctx], dim=0)
            else:  # middle
                mid = self.n_ctx // 2
                if kw_prototype is not None:
                    combined = torch.cat([ctx[:mid], main_embed, ctx[mid:], kw_prototype], dim=0)
                else:
                    combined = torch.cat([ctx[:mid], main_embed, ctx[mid:]], dim=0)
            
            batch_features.append(combined.unsqueeze(0))
        
        prompt_batch = torch.cat(batch_features, dim=0)  # (B, L, W)
        
        if self.proj is not None:
            prompt_batch = self.proj(prompt_batch)
        
        L = prompt_batch.shape[1]
        prompt_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        
        return prompt_batch.transpose(0, 1), prompt_mask


# ================================================================================
# CoCoOp 提示学习器 (图像条件化可学习向量)
# 论文: Conditional Prompt Learning for Vision-Language Models (Zhou et al., 2022)
# ================================================================================

class CoCoOpPromptLearner(nn.Module):
    """
    CoCoOp-style Conditional Prompt Learner for Anomaly Detection.
    
    简化版提示格式（推荐）:
        [v1+π][v2+π]...[vM+π] + [anomaly/normal {cls_name}]
    
    其中 π = Meta-Net(image_features) 是图像条件化的token
    
    关键优势 (论文 Table 1):
    - CoOp unseen classes: 63.22%
    - CoCoOp unseen classes: 71.69% (+8.47%)
    
    Meta-Net 架构 (与官方一致):
        Linear(vis_dim, vis_dim//16) → ReLU → Linear(vis_dim//16, text_width)
    
    Args:
        text_encoder: SAM3的文本编码器
        n_ctx: 可学习上下文向量数量 (默认4)
        ctx_init: 用于初始化的文本
        freeze_text_encoder: 是否冻结文本编码器
        proj: 投影层
        class_token_position: 类别token位置
        vis_dim: 视觉特征维度 (SAM3 backbone输出，默认256)
        reduction_factor: Meta-Net瓶颈缩减因子 (默认16，与官方一致)
        use_keywords: 是否使用关键词聚合 (默认False)
    """
    
    def __init__(
        self,
        text_encoder,
        n_ctx: int = 4,
        ctx_init: str = "",
        freeze_text_encoder: bool = True,
        proj: Optional[nn.Module] = None,
        class_token_position: str = "end",
        vis_dim: int = 256,
        reduction_factor: int = 16,
        use_keywords: bool = False,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.context_length = getattr(text_encoder, "context_length", 32)
        
        self.width = getattr(text_encoder.encoder, "width", None)
        if self.width is None:
            self.width = getattr(text_encoder, "width", None)
        assert self.width is not None, "Cannot determine text encoder embedding width"
        
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.proj = proj if proj is not None else getattr(text_encoder, "resizer", None)
        self.vis_dim = vis_dim
        self.use_keywords = use_keywords
        
        # ===== 可学习的上下文向量 =====
        if ctx_init and len(ctx_init) > 0:
            ctx_init_tokens = text_encoder.tokenizer([ctx_init], context_length=self.context_length)
            with torch.no_grad():
                _, ctx_init_embeds = text_encoder.encoder(ctx_init_tokens)
                n_available = min(n_ctx, ctx_init_embeds.shape[1] - 1)
                ctx_init_embeds = ctx_init_embeds[0, 1:n_available+1, :]
                if n_available < n_ctx:
                    pad = torch.randn(n_ctx - n_available, self.width) * 0.02
                    ctx_init_embeds = torch.cat([ctx_init_embeds, pad], dim=0)
            self.ctx = nn.Parameter(ctx_init_embeds.clone())
            print(f"[CoCoOpPromptLearner] 使用 '{ctx_init}' 初始化 {n_ctx} 个上下文向量")
        else:
            self.ctx = nn.Parameter(torch.randn(n_ctx, self.width) * 0.02)
            print(f"[CoCoOpPromptLearner] 随机初始化 {n_ctx} 个上下文向量")
        
        # ===== Meta-Net: 与官方实现完全一致 =====
        # 官方代码: nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        # ]))
        hidden_dim = vis_dim // reduction_factor
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, hidden_dim)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_dim, self.width))
        ]))
        print(f"[CoCoOpPromptLearner] Meta-Net: {vis_dim} → {hidden_dim} → {self.width}")
        
        # 关键词attention
        if use_keywords:
            self.q_keyword = nn.Parameter(torch.randn(self.width) * 0.02)
            self.keyword_attn_scale = math.sqrt(self.width)
        
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        print(f"[CoCoOpPromptLearner] use_keywords={use_keywords}, position={class_token_position}")
    
    @torch.no_grad()
    def _encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros((0, self.width), device=device)
        
        tokenized = self.text_encoder.tokenizer(texts, context_length=self.context_length).to(device)
        eot_indices = tokenized.argmax(dim=-1)
        if eot_indices.dim() > 1:
            eot_indices = eot_indices.argmax(dim=1)
        _, token_embeds = self.text_encoder.encoder(tokenized)
        
        batch_indices = torch.arange(len(texts), device=device)
        text_features = token_embeds[batch_indices, eot_indices]
        return text_features.to(device)
    
    def forward(
        self,
        prompt_lists: Sequence[List[str]],
        image_features: Optional[torch.Tensor] = None,
        class_ids: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prompt_lists: [B]个列表，每个包含 ["{anomaly/normal} {cls}", ...]
            image_features: (B, vis_dim) 或 (B, C, H, W) 图像特征
                          如果为None，退化为CoOp（π=0）
            device: 计算设备
        
        Returns:
            prompt_seq: (L, B, W) 条件化提示序列
            prompt_mask: (B, L) 注意力mask
        """
        device = device or self.ctx.device
        B = len(prompt_lists)
        
        # ===== 生成图像条件token π =====
        if image_features is not None:
            image_features = image_features.to(device)
            
            # 处理不同维度的输入
            if image_features.dim() == 4:
                # (B, C, H, W) → (B, C) via global average pooling
                image_features = image_features.mean(dim=[2, 3])
            
            # 确保维度匹配
            feat_dim = image_features.shape[-1]
            if feat_dim != self.vis_dim:
                if feat_dim > self.vis_dim:
                    image_features = image_features[..., :self.vis_dim]
                else:
                    # 填充到 vis_dim
                    pad = torch.zeros(B, self.vis_dim - feat_dim, device=device)
                    image_features = torch.cat([image_features, pad], dim=-1)
            
            # Meta-Net: 与官方一致的使用方式
            # bias = self.meta_net(im_features)  # (batch, ctx_dim)
            # bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
            # ctx_shifted = ctx + bias           # 广播加法
            pi = self.meta_net(image_features)  # (B, width)
        else:
            pi = torch.zeros((B, self.width), device=device)
        
        batch_features = []
        
        for i, prompts in enumerate(prompt_lists):
            if len(prompts) == 0:
                prompts = ["normal"]
            
            # 1) 编码主描述
            main_desc = prompts[0]
            main_embed = self._encode_text([main_desc], device)  # (1, W)
            
            # 2) 条件化上下文向量: ctx + π (与官方一致)
            # 官方: ctx_shifted = ctx + bias, 其中 bias 对所有 ctx token 相同
            ctx_conditioned = self.ctx.to(device) + pi[i].unsqueeze(0)  # (n_ctx, W)
            
            # 3) 关键词聚合 (可选)
            if self.use_keywords and len(prompts) > 1:
                keywords = prompts[1:]
                kw_embeds = self._encode_text(keywords, device)
                qk = self.q_keyword.view(self.width, 1).to(device)
                kw_scores = torch.matmul(kw_embeds, qk).squeeze(-1) / self.keyword_attn_scale
                kw_weights = torch.softmax(kw_scores, dim=0)
                kw_prototype = (kw_weights.unsqueeze(-1) * kw_embeds).sum(dim=0, keepdim=True)
            else:
                kw_prototype = None
            
            # 4) 组合
            if self.class_token_position == "end":
                if kw_prototype is not None:
                    combined = torch.cat([ctx_conditioned, main_embed, kw_prototype], dim=0)
                else:
                    combined = torch.cat([ctx_conditioned, main_embed], dim=0)
            elif self.class_token_position == "front":
                if kw_prototype is not None:
                    combined = torch.cat([main_embed, ctx_conditioned, kw_prototype], dim=0)
                else:
                    combined = torch.cat([main_embed, ctx_conditioned], dim=0)
            else:  # middle
                mid = self.n_ctx // 2
                if kw_prototype is not None:
                    combined = torch.cat([ctx_conditioned[:mid], main_embed, ctx_conditioned[mid:], kw_prototype], dim=0)
                else:
                    combined = torch.cat([ctx_conditioned[:mid], main_embed, ctx_conditioned[mid:]], dim=0)
            
            batch_features.append(combined.unsqueeze(0))
        
        prompt_batch = torch.cat(batch_features, dim=0)  # (B, L, W)
        
        if self.proj is not None:
            prompt_batch = self.proj(prompt_batch)
        
        L = prompt_batch.shape[1]
        prompt_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        
        return prompt_batch.transpose(0, 1), prompt_mask


# ================================================================================
# 原有的 PerClassTemplatePromptLearner
# ================================================================================

class PerClassTemplatePromptLearner(nn.Module):
    """Per-class template + SoWA-style prompt learner."""

    def __init__(
        self,
        text_encoder,
        class_names: Sequence[str],
        n_ctx: int = 4,
        num_templates: int = 4,
        freeze_text_encoder: bool = True,
        proj: Optional[nn.Module] = None,
        token_attn_scale: Optional[float] = None,
        keyword_attn_scale: Optional[float] = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.context_length = getattr(text_encoder, "context_length", 32)
        self.width = getattr(text_encoder.encoder, "width", None)
        assert self.width is not None, "Cannot find text encoder width"
        self.n_ctx = n_ctx
        self.num_templates = num_templates
        self.class_names = list(class_names)
        self.class_to_idx = {c.lower(): i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.ctx = nn.Parameter(torch.randn(n_ctx, self.width) * 0.02)
        self.class_templates = nn.Parameter(
            torch.randn(self.num_classes, num_templates, self.width) * 0.02
        )

        self.proj = proj if proj is not None else getattr(text_encoder, "resizer", None)
        self.q_token = nn.Parameter(torch.randn(self.width) * 0.02)
        self.q_keyword = nn.Parameter(torch.randn(self.width) * 0.02)
        self.token_attn_scale = token_attn_scale if token_attn_scale is not None else math.sqrt(self.width)
        self.keyword_attn_scale = keyword_attn_scale if keyword_attn_scale is not None else math.sqrt(self.width)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, prompt_lists: Sequence[List[str]], class_ids: Optional[Sequence[int]] = None,
                image_features: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        device = device or self.ctx.device
        B = len(prompt_lists)

        all_keywords = []
        counts = []
        for kws in prompt_lists:
            kws_clean = [w for w in kws if w]
            counts.append(len(kws_clean))
            all_keywords.extend(kws_clean)
        N = len(all_keywords)

        if N == 0:
            keyword_prototypes = torch.zeros((B, 1, self.width), device=device)
        else:
            tokenized = self.text_encoder.tokenizer(all_keywords, context_length=self.context_length).to(device)
            eot = tokenized.argmax(dim=-1)
            if eot.dim() > 1:
                eot = eot.argmax(dim=1)
            _, tokens = self.text_encoder.encoder(tokenized)
            tokens = tokens.to(device)
            seq_len = tokens.shape[1]
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            token_mask = pos <= eot.unsqueeze(1)

            q = self.q_token.view(self.width, 1).to(device)
            scores = torch.matmul(tokens, q).squeeze(-1) / self.token_attn_scale
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~token_mask, min_val)
            token_w = torch.softmax(scores, dim=1)
            keyword_embeds_all = (token_w.unsqueeze(-1) * tokens).sum(dim=1)

            max_k = max(counts) if max(counts) > 0 else 1
            kw_padded = torch.zeros((B, max_k, self.width), device=device)
            kw_mask = torch.ones((B, max_k), dtype=torch.bool, device=device)
            idx = 0
            for i, k in enumerate(counts):
                if k > 0:
                    kw_padded[i, :k, :] = keyword_embeds_all[idx: idx + k]
                    kw_mask[i, :k] = False
                    idx += k

            qk = self.q_keyword.view(self.width, 1).to(device)
            kw_scores = torch.matmul(kw_padded, qk).squeeze(-1) / self.keyword_attn_scale
            kw_scores = kw_scores.masked_fill(kw_mask, min_val)
            kw_w = torch.softmax(kw_scores, dim=1)
            keyword_prototypes = (kw_w.unsqueeze(-1) * kw_padded).sum(dim=1, keepdim=True)

        if class_ids is None:
            class_templates_batch = self.class_templates[0].unsqueeze(0).repeat(B, 1, 1)
        else:
            ids = torch.tensor(class_ids, dtype=torch.long, device=device)
            class_templates_batch = self.class_templates[ids]

        ctx_b = self.ctx.unsqueeze(0).to(device).repeat(B, 1, 1)
        stacked = torch.cat([ctx_b, class_templates_batch, keyword_prototypes], dim=1)
        projected = self.proj(stacked) if self.proj is not None else stacked
        prompt_mask = torch.zeros((projected.shape[0], projected.shape[1]), dtype=torch.bool, device=projected.device)

        return projected.transpose(0, 1), prompt_mask


# ================================================================================
# 原有的 AveragedPromptLearner (SoWA风格)
# ================================================================================

class AveragedPromptLearner(nn.Module):
    """SoWA-style prompt learner (原有静态提示)"""

    def __init__(
        self,
        text_encoder,
        n_ctx: int = 4,
        freeze_text_encoder: bool = True,
        proj: Optional[nn.Module] = None,
        token_attn_scale: Optional[float] = None,
        keyword_attn_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.context_length = getattr(text_encoder, "context_length", 32)
        self.width = getattr(text_encoder.encoder, "width", None)
        if self.width is None:
            self.width = getattr(text_encoder, "width", None)
        assert self.width is not None

        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(torch.randn(n_ctx, self.width) * 0.02)
        self.proj = proj if proj is not None else getattr(text_encoder, "resizer", None)

        self.q_token = nn.Parameter(torch.randn(self.width) * 0.02)
        self.q_keyword = nn.Parameter(torch.randn(self.width) * 0.02)
        self.token_attn_scale = token_attn_scale if token_attn_scale is not None else math.sqrt(self.width)
        self.keyword_attn_scale = keyword_attn_scale if keyword_attn_scale is not None else math.sqrt(self.width)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, prompt_lists: Sequence[List[str]], image_features: Optional[torch.Tensor] = None,
                class_ids: Optional[Sequence[int]] = None, device: Optional[torch.device] = None):
        device = device or self.ctx.device
        batch_features = []

        for words in prompt_lists:
            if len(words) == 0:
                words = ["normal"]
            tokenized = self.text_encoder.tokenizer(words, context_length=self.context_length).to(device)
            eot_indices = tokenized.argmax(dim=-1)
            if eot_indices.dim() > 1:
                eot_indices = eot_indices.argmax(dim=1)
            _, tokens = self.text_encoder.encoder(tokenized)
            tokens = tokens.to(device)
            n_words = tokens.shape[0]

            if n_words == 0:
                prototype = torch.zeros((1, self.width), device=device)
            else:
                seq_len = tokens.shape[1]
                pos = torch.arange(seq_len, device=device).unsqueeze(0)
                token_mask = pos <= eot_indices.unsqueeze(1)
                
                q = self.q_token.view(self.width, 1)
                scores = torch.matmul(tokens, q).squeeze(-1) / self.token_attn_scale
                min_val = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~token_mask, min_val)
                token_weights = torch.softmax(scores, dim=1)
                keyword_embeds = (token_weights.unsqueeze(-1) * tokens).sum(dim=1)

                qk = self.q_keyword.view(self.width, 1)
                kw_scores = torch.matmul(keyword_embeds, qk).squeeze(-1) / self.keyword_attn_scale
                kw_weights = torch.softmax(kw_scores, dim=0)
                prototype = (kw_weights.unsqueeze(-1) * keyword_embeds).sum(dim=0, keepdim=True)

            ctx = self.ctx.unsqueeze(0).to(device)
            prototype = prototype.unsqueeze(0)
            stacked = torch.cat([ctx, prototype], dim=1)
            batch_features.append(stacked)

        prompt_batch = torch.cat(batch_features, dim=0)
        projected = self.proj(prompt_batch) if self.proj is not None else prompt_batch
        prompt_mask = torch.zeros((projected.shape[0], projected.shape[1]), dtype=torch.bool, device=projected.device)
        return projected.transpose(0, 1), prompt_mask