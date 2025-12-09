import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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


class AveragedPromptLearner(nn.Module):
    """Encode short-word lists with SAM3's text encoder, mean-pool, and prepend learnable ctx."""

    def __init__(
        self,
        text_encoder: VETextEncoder,
        n_ctx: int = 4,
        freeze_text_encoder: bool = True,
        proj: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.context_length = getattr(text_encoder, "context_length", 32)
        self.width = text_encoder.encoder.width
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(torch.randn(n_ctx, self.width) * 0.02)
        self.proj = proj if proj is not None else text_encoder.resizer

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _encode_keywords(self, keywords: List[str], device: torch.device) -> torch.Tensor:
        if len(keywords) == 0:
            keywords = ["normal"]
        tokenized = self.text_encoder.tokenizer(
            keywords, context_length=self.context_length
        ).to(device)
        _, tokens = self.text_encoder.encoder(tokenized)  # [b, seq, width]
        # take EOT (argmax token id) for each word
        eot_indices = tokenized.argmax(dim=-1)
        word_embeds = tokens[torch.arange(len(keywords), device=device), eot_indices]
        return word_embeds  # [n_words, width]

    def forward(
        self, prompt_lists: Sequence[List[str]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.ctx.device
        batch_features: List[torch.Tensor] = []
        for words in prompt_lists:
            with torch.no_grad():
                word_embeds = self._encode_keywords(words, device)  # (n_words, width)
                prototype = word_embeds.mean(dim=0, keepdim=True)  # (1, width)
            ctx = self.ctx.unsqueeze(0).to(device)  # (1, n_ctx, width)
            prototype = prototype.unsqueeze(0)  # (1, 1, width)
            stacked = torch.cat([ctx, prototype], dim=1)  # (1, n_ctx+1, width)
            batch_features.append(stacked)

        prompt_batch = torch.cat(batch_features, dim=0)  # (B, seq, width)
        projected = self.proj(prompt_batch) if self.proj is not None else prompt_batch
        prompt_mask = torch.zeros(
            (projected.shape[0], projected.shape[1]),
            dtype=torch.bool,
            device=projected.device,
        )  # False = visible
        # return seq-first for transformer encoder
        return projected.transpose(0, 1), prompt_mask
