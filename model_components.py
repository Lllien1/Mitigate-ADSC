import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.text_encoder_ve import VETextEncoder

class ParallelLoRA(nn.Module):
    """
    Parallel low-rank adapter (side-branch).
    Computes update = scaling * ((x @ A.T) @ B.T)
    and returns update (to be added to main output).
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: Optional[float] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha or rank)
        self.scaling = self.alpha / float(rank)

        # LoRA params: shapes chosen to match (x @ A^T) -> (N, rank), then @ B^T -> (N, out_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        returns: (..., out_features)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (N, in_features)
        lora_mid = x_flat @ self.lora_A.t()  # (N, rank)
        update = lora_mid @ self.lora_B.t()  # (N, out_features)
        update = update.view(*orig_shape[:-1], -1)  # (..., out_features)
        return update * self.scaling

    def fold_into_linear(self, linear: nn.Linear):
        """
        Fold the adapter into a given Linear layer in-place:
          linear.weight.data += scaling * (B @ A)
        Preconditions:
          - linear.weight.shape == (out_features, in_features)
        """
        assert linear.weight.shape[0] == self.out_features and linear.weight.shape[1] == self.in_features
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
            linear.weight.data += delta


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

        # Freeze the base weights to train only LoRA (unless caller overrides).
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten last dim for matmul; supports extra leading dims
        y = self.base(x)
        lora_update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return y + lora_update * self.scaling


def apply_lora_to_sam(
    module: nn.Module,
    target_substrings: Sequence[str] = ("qkv",),
    rank: int = 16,
    alpha: Optional[float] = None,
) -> List[str]:
    """Replace Linear layers containing target substrings with LoRA-wrapped versions.

    Designed for SAM3 ViT-Det attention blocks (`sam3/model/vitdet.py:Attention.qkv`).
    Returns list of module names that were wrapped.
    """
    wrapped: List[str] = []
    for name, child in list(module.named_children()):
        # Recurse first
        wrapped.extend(
            apply_lora_to_sam(
                child, target_substrings=target_substrings, rank=rank, alpha=alpha
            )
        )
        if isinstance(child, nn.Linear) and any(s in name for s in target_substrings):
            lora = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, lora)
            wrapped.append(name)
    return wrapped

class PerClassTemplatePromptLearner(nn.Module):
    """
    Per-class template + SoWA-style prompt learner.

    Usage:
      pl = PerClassTemplatePromptLearner(text_encoder, class_names=class_list, n_ctx=4, num_templates=4)
      prompt_seq, prompt_mask = pl(prompt_lists, class_ids=[cls_idx_0, cls_idx_1, ...], device=device)
    """

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
        # class templates
        self.class_templates = nn.Parameter(
            torch.randn(self.num_classes, num_templates, self.width) * 0.02
        )  # (C, T, W)

        self.proj = proj if proj is not None else getattr(text_encoder, "resizer", None)

        # SoWA-like: token-level and keyword-level queries
        self.q_token = nn.Parameter(torch.randn(self.width) * 0.02)
        self.q_keyword = nn.Parameter(torch.randn(self.width) * 0.02)
        self.token_attn_scale = token_attn_scale if token_attn_scale is not None else math.sqrt(self.width)
        self.keyword_attn_scale = keyword_attn_scale if keyword_attn_scale is not None else math.sqrt(self.width)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _tokenize_and_encode(self, keywords: List[str], device: torch.device):
        """Tokenize and run text encoder return tokens and token_mask (True for valid tokens)."""
        if len(keywords) == 0:
            keywords = ["normal"]
        tokenized = self.text_encoder.tokenizer(keywords, context_length=self.context_length).to(device)
        eot_indices = tokenized.argmax(dim=-1)
        if eot_indices.dim() > 1:
            eot_indices = eot_indices.argmax(dim=1)
        _, tokens = self.text_encoder.encoder(tokenized)  # (N, seq_len, W)
        tokens = tokens.to(device)
        seq_len = tokens.shape[1]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        token_mask = pos <= eot_indices.unsqueeze(1)  # (N, seq_len)
        return tokens, token_mask

    def forward(self, prompt_lists: Sequence[List[str]], class_ids: Optional[Sequence[int]] = None, device: Optional[torch.device] = None):
        """
        prompt_lists: Sequence[B] of list[str]
        class_ids: optional Sequence[B] of ints (0..num_classes-1)
        returns prompt_seq (L,B,W), prompt_mask (B,L)
        """
        device = device or self.ctx.device
        B = len(prompt_lists)

        # flatten keywords
        all_keywords = []
        counts = []
        for kws in prompt_lists:
            kws_clean = [w for w in kws if w]
            counts.append(len(kws_clean))
            all_keywords.extend(kws_clean)
        N = len(all_keywords)

        if N == 0:
            # fallback: zero prototype
            keyword_prototypes = torch.zeros((B, 1, self.width), device=device)
        else:
            tokenized = self.text_encoder.tokenizer(all_keywords, context_length=self.context_length).to(device)
            eot = tokenized.argmax(dim=-1)
            if eot.dim() > 1:
                eot = eot.argmax(dim=1)
            _, tokens = self.text_encoder.encoder(tokenized)  # (N, seq_len, W)
            tokens = tokens.to(device)
            seq_len = tokens.shape[1]
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            token_mask = pos <= eot.unsqueeze(1)

            # token-level attention:
            q = self.q_token.view(self.width, 1).to(device)
            scores = torch.matmul(tokens, q).squeeze(-1) / (self.token_attn_scale or math.sqrt(self.width))
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~token_mask, min_val)
            token_w = torch.softmax(scores, dim=1)  # (N, seq_len)
            keyword_embeds_all = (token_w.unsqueeze(-1) * tokens).sum(dim=1)  # (N, W)

            # regroup per sample
            max_k = max(counts) if max(counts) > 0 else 1
            kw_padded = torch.zeros((B, max_k, self.width), device=device)
            kw_mask = torch.ones((B, max_k), dtype=torch.bool, device=device)
            idx = 0
            for i, k in enumerate(counts):
                if k > 0:
                    kw_padded[i, :k, :] = keyword_embeds_all[idx: idx + k]
                    kw_mask[i, :k] = False
                    idx += k

            # keyword-level attention
            qk = self.q_keyword.view(self.width, 1).to(device)
            kw_scores = torch.matmul(kw_padded, qk).squeeze(-1) / (self.keyword_attn_scale or math.sqrt(self.width))  # (B, max_k)
            kw_scores = kw_scores.masked_fill(kw_mask, min_val)
            kw_w = torch.softmax(kw_scores, dim=1)  # (B,max_k)
            keyword_prototypes = (kw_w.unsqueeze(-1) * kw_padded).sum(dim=1, keepdim=True)  # (B,1,W)

        # class templates batch
        if class_ids is None:
            class_templates_batch = self.class_templates[0].unsqueeze(0).repeat(B, 1, 1)
        else:
            ids = torch.tensor(class_ids, dtype=torch.long, device=device)
            class_templates_batch = self.class_templates[ids]  # (B, T, W)

        ctx_b = self.ctx.unsqueeze(0).to(device).repeat(B, 1, 1)  # (B, n_ctx, W)
        stacked = torch.cat([ctx_b, class_templates_batch, keyword_prototypes], dim=1)  # (B, n_ctx+T+1, W)
        projected = self.proj(stacked) if self.proj is not None else stacked
        prompt_mask = torch.zeros((projected.shape[0], projected.shape[1]), dtype=torch.bool, device=projected.device)

        return projected.transpose(0, 1), prompt_mask

class AveragedPromptLearner(nn.Module):
    """
    SoWA-style prompt learner:
      - token-level attention: for each keyword phrase, produce a keyword embedding
        by attending over its token embeddings (learned query).
      - keyword-level attention: attend over keyword embeddings to produce a single
        prototype for the sample (learned query).
      - prepend learnable ctx tokens to the prototype as before.

    Returns:
      prompt_seq: (n_ctx + 1, B, width)
      prompt_mask: (B, n_ctx + 1)  -- False = visible (same convention as original code)
    """

    def __init__(
        self,
        text_encoder: VETextEncoder,
        n_ctx: int = 4,
        freeze_text_encoder: bool = True,
        proj: Optional[nn.Module] = None,
        token_attn_scale: Optional[float] = None,
        keyword_attn_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.context_length = getattr(text_encoder, "context_length", 32)
        # width: embedding dim of text encoder tokens
        self.width = getattr(text_encoder.encoder, "width", None)
        if self.width is None:
            # fallback to known attribute if layout differs
            self.width = getattr(text_encoder, "width", None)
        assert self.width is not None, "Cannot determine text encoder embedding width"

        self.n_ctx = n_ctx
        # learnable context tokens
        self.ctx = nn.Parameter(torch.randn(n_ctx, self.width) * 0.02)
        self.proj = proj if proj is not None else getattr(text_encoder, "resizer", None)

        # token-level learned query vector (used to score tokens within each keyword)
        # we store as a parameter vector of size (width,)
        self.q_token = nn.Parameter(torch.randn(self.width) * 0.02)
        # keyword-level learned query vector (used to score keywords)
        self.q_keyword = nn.Parameter(torch.randn(self.width) * 0.02)

        # optional temperature / scaling factors (learnable hyperparams are optional)
        self.token_attn_scale = (
            token_attn_scale if token_attn_scale is not None else math.sqrt(self.width)
        )
        self.keyword_attn_scale = (
            keyword_attn_scale if keyword_attn_scale is not None else math.sqrt(self.width)
        )

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _tokenize_and_encode(self, keywords: List[str], device: torch.device):
        """
        Tokenize a list of keywords and return:
           token_ids: tensor (n_words, seq_len) -- token id per position (uses .argmax as before)
           eot_indices: tensor (n_words,) -- position index of EOT / last meaningful token (same as before)
           token_embs: tensor (n_words, seq_len, width) -- token embeddings from text_encoder.encoder
        This mirrors the original pipeline but returns token-level embeddings (not just EOT).
        """
        if len(keywords) == 0:
            keywords = ["normal"]
        # tokenizer existing interface in repo:
        tokenized = self.text_encoder.tokenizer(keywords, context_length=self.context_length).to(device)
        # tokenized is used previously with argmax to get eot indices
        eot_indices = tokenized.argmax(dim=-1)  # shape: (n_words, ) or (n_words, seq_len)? repo's original code worked
        # In original code they did tokens[range, eot_indices] so eot_indices must be (n_words,) giving the index.
        # To be robust: if eot_indices is 2D, take argmax across seq to get position index per word:
        if eot_indices.dim() > 1:
            # eot_indices maybe shape (n_words, seq_len) if tokenizer returns weird shape
            # take the position of the highest value across seq dimension
            eot_indices = eot_indices.argmax(dim=1)

        # run text encoder to get token embeddings: tokens: (n_words, seq_len, width)
        _, tokens = self.text_encoder.encoder(tokenized)
        # Ensure tokens is float tensor on device
        tokens = tokens.to(device)

        # Build mask for valid token positions for each keyword using eot_indices:
        seq_len = tokens.shape[1]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        # token_mask True for positions <= eot_index (i.e., valid tokens), False otherwise
        token_mask = pos <= eot_indices.unsqueeze(1)  # shape (n_words, seq_len), dtype=bool

        return eot_indices, tokens, token_mask

    def forward(self, prompt_lists: Sequence[List[str]], device: Optional[torch.device] = None):
        """
        Input:
          prompt_lists: Sequence of lists of keyword strings, length B
        Output:
          prompt_seq: (n_ctx+1, B, width)
          prompt_mask: (B, n_ctx+1)  (False = visible)
        """
        device = device or self.ctx.device
        batch_features = []

        for words in prompt_lists:
            # encode tokens and compute token-level keyword embeddings
            eot_indices, tokens, token_mask = self._tokenize_and_encode(words, device=device)
            # tokens: (n_words, seq_len, width)
            n_words = tokens.shape[0]

            if n_words == 0:
                # fallback to zero prototype
                prototype = torch.zeros((1, self.width), device=device)
            else:
                # token-level attention:
                # scores = (tokens @ q_token) / scale  -> (n_words, seq_len)
                # mask out positions after EOT with -1e9
                q = self.q_token.view(self.width, 1)  # (width,1)
                # compute dot product along width: (n_words, seq_len, width) @ (width,1) -> (n_words, seq_len, 1)
                # faster as torch.einsum or matmul
                scores = torch.matmul(tokens, q).squeeze(-1)  # (n_words, seq_len)
                scores = scores / (self.token_attn_scale if self.token_attn_scale is not None else math.sqrt(self.width))

                # mask positions beyond EOT
                min_val = torch.tensor(torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype)
                scores = scores.masked_fill(~token_mask, min_val)
                # softmax over seq_len
                token_weights = torch.softmax(scores, dim=1)  # (n_words, seq_len)
                # compute keyword embedding as weighted sum of token embeddings
                keyword_embeds = (token_weights.unsqueeze(-1) * tokens).sum(dim=1)  # (n_words, width)

                # keyword-level attention: compute scores over the n_words
                qk = self.q_keyword.view(self.width, 1)  # (width,1)
                kw_scores = torch.matmul(keyword_embeds, qk).squeeze(-1)  # (n_words,)
                kw_scores = kw_scores / (self.keyword_attn_scale if self.keyword_attn_scale is not None else math.sqrt(self.width))
                kw_weights = torch.softmax(kw_scores, dim=0)  # (n_words,)

                # produce prototype as weighted sum of keyword embeddings
                prototype = (kw_weights.unsqueeze(-1) * keyword_embeds).sum(dim=0, keepdim=True)  # (1, width)

            # prepend ctx tokens as before
            ctx = self.ctx.unsqueeze(0).to(device)  # (1, n_ctx, width)
            prototype = prototype.unsqueeze(0)  # (1,1,width)
            stacked = torch.cat([ctx, prototype], dim=1)  # (1, n_ctx+1, width)
            batch_features.append(stacked)

        prompt_batch = torch.cat(batch_features, dim=0)  # (B, n_ctx+1, width)
        projected = self.proj(prompt_batch) if self.proj is not None else prompt_batch
        # prompt_mask: False = visible (same convention); all ctx + prototype are visible (False)
        prompt_mask = torch.zeros((projected.shape[0], projected.shape[1]), dtype=torch.bool, device=projected.device)
        # return seq-first for transformer encoder (seq,len first)
        return projected.transpose(0, 1), prompt_mask
# -------------------------------------------------------------------------------
