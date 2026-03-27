"""
model_components.py

包含：
- LoRA 相关组件
- CoOpPromptLearner: CoOp风格可学习提示（静态）
- CoCoOpPromptLearner: CoCoOp风格可学习提示（图像条件化）
- 原有的 AveragedPromptLearner, PerClassTemplatePromptLearner
"""

import math
import re
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


class QKVLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: Optional[float] = None, target: str = "qv_only"):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha or rank)
        self.scaling = self.alpha / float(self.rank)
        self.target = str(target).lower()

        if self.base.out_features % 3 != 0:
            raise ValueError("QKVLoRALinear requires out_features divisible by 3")
        self.dim = int(self.base.out_features // 3)

        out_dim = int(self.base.out_features) if self.target in ("qkv_all", "all") else int(2 * self.dim)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        upd = (x @ self.lora_A.t()) @ self.lora_B.t()
        upd = upd * self.scaling
        if self.target in ("qkv_all", "all"):
            return y + upd
        if upd.shape[-1] != 2 * self.dim:
            return y
        z = torch.zeros_like(y)
        z[..., : self.dim] = upd[..., : self.dim]
        z[..., 2 * self.dim :] = upd[..., self.dim :]
        return y + z


def apply_lora_to_sam(
    module: nn.Module,
    target_substrings: Sequence[str] = ("qkv",),
    rank: int = 16,
    alpha: Optional[float] = None,
    layer_ids: Optional[Sequence[int]] = None,
    _prefix: str = "",
) -> List[str]:
    """Replace Linear layers containing target substrings with LoRA-wrapped versions.

    Notes:
    - By default, wraps *all* matching Linear layers (e.g. qkv).
    - If `layer_ids` is provided, only wraps layers whose module path contains `blocks.{id}.`
      (works for SAM3-style ViT/Transformer trunks that name blocks this way).
    """
    wrapped: List[str] = []
    layer_id_set = set(layer_ids) if layer_ids is not None else None

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name

        # recurse first
        wrapped.extend(
            apply_lora_to_sam(
                child,
                target_substrings=target_substrings,
                rank=rank,
                alpha=alpha,
                layer_ids=layer_ids,
                _prefix=full_name,
            )
        )

        if isinstance(child, nn.Linear) and any(s in name for s in target_substrings):
            if layer_id_set is not None:
                m = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", full_name)
                if m is None or int(m.group(1)) not in layer_id_set:
                    continue

            lora = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, lora)
            wrapped.append(full_name)

    return wrapped


def apply_qkv_lora_to_sam(
    module: nn.Module,
    rank: int = 16,
    alpha: Optional[float] = None,
    layer_ids: Optional[Sequence[int]] = None,
    target: str = "qv_only",
    _prefix: str = "",
) -> List[str]:
    wrapped: List[str] = []
    layer_id_set = set(layer_ids) if layer_ids is not None else None

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name

        wrapped.extend(
            apply_qkv_lora_to_sam(
                child,
                rank=rank,
                alpha=alpha,
                layer_ids=layer_ids,
                target=target,
                _prefix=full_name,
            )
        )

        if name == "qkv" and isinstance(child, nn.Linear):
            if layer_id_set is not None:
                m = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", full_name)
                if m is None or int(m.group(1)) not in layer_id_set:
                    continue
            setattr(module, name, QKVLoRALinear(child, rank=rank, alpha=alpha, target=target))
            wrapped.append(full_name)

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


# ================================================================================
# 方案C：置信度融合头 (Confidence Fusion Head)
# 融合 presence_logit, iou_pred, filo_conf -> final_conf
# ================================================================================

class ConfidenceFusionHead(nn.Module):
    """
    置信度融合头：学习如何融合SAM3的presence/iou和FiLo的置信度
    
    输入：
        - presence_logit: (B, Q) 或 (B, Q, 1)
        - iou_pred: (B, Q) 或 (B, Q, 1)
        - filo_conf: (B,) 或 (B, 1) - FiLo异常图的最大响应
        
    输出：
        - final_conf: (B, Q) 融合后的置信度
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        # 输入: presence(1) + iou(1) + filo(1) = 3
        self.input_dim = 3
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 输出单个置信度值
        )
        
        # 可学习的融合权重（作为backup/residual）
        self.alpha = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))  # presence, iou, filo
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        presence_logit: torch.Tensor,
        iou_pred: torch.Tensor,
        filo_conf: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            presence_logit: 各种可能形状 - (B, Q), (B, Q, 1), (B, 1), (B,) 等
            iou_pred: 各种可能形状 - (B, Q), (B, Q, 1), (B, 1), (B,) 等
            filo_conf: (B,) 或 (B, 1) - FiLo置信度
            
        Returns:
            final_conf: (B, Q) 融合后的置信度（logits）
        """
        # 获取batch size
        B = presence_logit.shape[0]
        
        # 处理 iou_pred 的形状，确定 Q
        if iou_pred.dim() == 3:
            iou_pred = iou_pred.squeeze(-1)  # (B, Q, 1) -> (B, Q)
        elif iou_pred.dim() == 1:
            iou_pred = iou_pred.unsqueeze(-1)  # (B,) -> (B, 1)
        
        Q = iou_pred.shape[1] if iou_pred.dim() == 2 else 1
        
        # 处理 presence_logit 的形状
        if presence_logit.dim() == 3:
            presence_logit = presence_logit.squeeze(-1)  # (B, Q, 1) -> (B, Q)
        elif presence_logit.dim() == 1:
            presence_logit = presence_logit.unsqueeze(-1)  # (B,) -> (B, 1)
        
        # 如果 presence_logit 是 (B, 1) 但 iou_pred 是 (B, Q)，需要扩展
        if presence_logit.shape[1] == 1 and Q > 1:
            presence_logit = presence_logit.expand(B, Q)  # (B, 1) -> (B, Q)
        
        # 如果 iou_pred 是 (B, 1) 但需要 Q 更大
        if iou_pred.shape[1] == 1 and presence_logit.shape[1] > 1:
            Q = presence_logit.shape[1]
            iou_pred = iou_pred.expand(B, Q)
        
        # 处理 filo_conf
        if filo_conf.dim() == 1:
            filo_conf = filo_conf.unsqueeze(-1)  # (B,) -> (B, 1)
        
        # 确保 Q 一致
        Q = max(presence_logit.shape[1], iou_pred.shape[1])
        
        # 将presence和iou转为概率
        presence_prob = torch.sigmoid(presence_logit)  # (B, Q) 或 (B, 1)
        iou_prob = torch.sigmoid(iou_pred) if iou_pred.max() > 1.0 else iou_pred  # (B, Q) 或 (B, 1)
        
        # 扩展到相同形状
        if presence_prob.shape[1] < Q:
            presence_prob = presence_prob.expand(B, Q)
        if iou_prob.shape[1] < Q:
            iou_prob = iou_prob.expand(B, Q)
        
        # 扩展filo_conf到所有query
        filo_expanded = filo_conf.expand(B, Q)  # (B, 1) -> (B, Q)
        
        # 拼接输入: (B, Q, 3)
        fusion_input = torch.stack([presence_prob, iou_prob, filo_expanded], dim=-1)
        
        # MLP融合
        mlp_out = self.mlp(fusion_input).squeeze(-1)  # (B, Q)
        
        # 加权平均作为残差
        alpha_normalized = F.softmax(self.alpha, dim=0)
        weighted_avg = (
            alpha_normalized[0] * presence_prob +
            alpha_normalized[1] * iou_prob +
            alpha_normalized[2] * filo_expanded
        )
        
        # 最终输出 = MLP输出 + 残差
        final_conf = mlp_out + 0.1 * weighted_avg
        
        if return_weights:
            return final_conf, alpha_normalized
        return final_conf


# ================================================================================
# 方案B：FiLo到Decoder的适配器 (FiLo-to-Decoder Adapter)
# 把FiLo特征编码成decoder可用的memory
# ================================================================================

class FiLoDecoderAdapter(nn.Module):
    """
    FiLo到Decoder的适配器：把FiLo的patch_tokens转换为decoder的额外memory
    
    方式1: 作为额外的memory tokens（扩展memory）
    方式2: 作为query的bias/conditioning
    方式3: 通过cross-attention融合到memory
    """
    
    def __init__(
        self,
        filo_dim: int = 768,           # FiLo输出维度
        decoder_dim: int = 256,        # Decoder hidden dim
        num_filo_tokens: int = 64,     # 压缩后的FiLo token数量
        mode: str = "memory",          # "memory", "query_bias", "cross_attn"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.num_filo_tokens = num_filo_tokens
        self.decoder_dim = decoder_dim
        
        # FiLo特征投影
        self.filo_proj = nn.Sequential(
            nn.Linear(filo_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        if mode == "memory":
            # 用于压缩FiLo tokens的可学习query
            self.compress_queries = nn.Parameter(
                torch.randn(num_filo_tokens, decoder_dim) * 0.02
            )
            self.compress_attn = nn.MultiheadAttention(
                embed_dim=decoder_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=False,
            )
            
        elif mode == "query_bias":
            # 把FiLo特征编码成query的bias
            self.bias_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.LayerNorm(decoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_dim, decoder_dim),
            )
            
        elif mode == "cross_attn":
            # 用于和memory做cross-attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=decoder_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=False,
            )
            self.cross_norm = nn.LayerNorm(decoder_dim)
    
    def forward(
        self,
        filo_tokens: torch.Tensor,      # (B, N_filo, D_filo)
        memory: Optional[torch.Tensor] = None,  # (N_mem, B, D_dec) for cross_attn mode
        query: Optional[torch.Tensor] = None,   # (Q, B, D_dec) for query_bias mode
    ) -> dict:
        """
        Returns:
            dict with keys depending on mode:
            - "memory": extra_memory (num_filo_tokens, B, D_dec)
            - "query_bias": query_bias (Q, B, D_dec)
            - "cross_attn": enhanced_memory (N_mem, B, D_dec)
        """
        B = filo_tokens.shape[0]
        
        # 投影FiLo特征
        filo_proj = self.filo_proj(filo_tokens)  # (B, N_filo, D_dec)
        filo_proj = filo_proj.permute(1, 0, 2)   # (N_filo, B, D_dec)
        
        result = {}
        
        if self.mode == "memory":
            # 用可学习query压缩FiLo tokens
            queries = self.compress_queries.unsqueeze(1).expand(-1, B, -1)  # (num_tokens, B, D)
            compressed, _ = self.compress_attn(
                query=queries,
                key=filo_proj,
                value=filo_proj,
            )  # (num_tokens, B, D)
            result["extra_memory"] = compressed
            
        elif self.mode == "query_bias":
            # 对FiLo特征做平均池化，然后生成bias
            filo_pooled = filo_proj.mean(dim=0)  # (B, D)
            bias = self.bias_mlp(filo_pooled)    # (B, D)
            # 扩展到所有query
            if query is not None:
                Q = query.shape[0]
                result["query_bias"] = bias.unsqueeze(0).expand(Q, -1, -1)  # (Q, B, D)
            else:
                result["query_bias"] = bias
                
        elif self.mode == "cross_attn":
            # 和memory做cross-attention
            if memory is not None:
                enhanced, _ = self.cross_attn(
                    query=memory,
                    key=filo_proj,
                    value=filo_proj,
                )
                result["enhanced_memory"] = self.cross_norm(memory + enhanced)
            else:
                result["enhanced_memory"] = None
        
        # 同时返回投影后的FiLo特征
        result["filo_proj"] = filo_proj
        
        return result


# ================================================================================
# FiLo异常图编码器（用于方案B的变体）
# 把2D anomaly_map编码成memory tokens
# ================================================================================

class FiLoMapEncoder(nn.Module):
    """
    把FiLo的anomaly_map (B, 2, H, W) 编码成memory tokens
    """
    
    def __init__(
        self,
        in_channels: int = 2,          # normal + abnormal
        hidden_dim: int = 256,
        num_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 轻量级CNN编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 可学习query用于token化
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_tokens = num_tokens
    
    def forward(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anomaly_map: (B, 2, H, W)
            
        Returns:
            tokens: (num_tokens, B, hidden_dim)
        """
        B = anomaly_map.shape[0]
        
        # CNN编码
        features = self.encoder(anomaly_map)  # (B, hidden_dim, H', W')
        H, W = features.shape[-2:]
        
        # Flatten并转置
        features = features.flatten(2).permute(2, 0, 1)  # (H'*W', B, hidden_dim)
        
        # 用可学习query做cross-attention
        queries = self.queries.unsqueeze(1).expand(-1, B, -1)  # (num_tokens, B, hidden_dim)
        tokens, _ = self.cross_attn(
            query=queries,
            key=features,
            value=features,
        )
        tokens = self.norm(tokens)
        
        return tokens
