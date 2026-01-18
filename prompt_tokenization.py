from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class TokenizedText:
    tokenized: torch.Tensor
    attention_mask: torch.Tensor
    base_embeds: torch.Tensor


def build_compound_template_texts(
    class_names: List[str],
    num_abnormal: int,
    ctx_len: int,
    abnormal_word: str,
) -> List[str]:
    placeholder = " ".join(["X"] * int(ctx_len))
    texts: List[str] = []
    for cls in class_names:
        cls_s = str(cls)
        texts.append(f"{placeholder} normal {cls_s}.")
        for _ in range(int(num_abnormal)):
            texts.append(f"{placeholder} {str(abnormal_word)} {cls_s}.")
    return texts


def tokenize_ve(
    text_encoder,
    texts: List[str],
    device: torch.device,
) -> TokenizedText:
    if not hasattr(text_encoder, "tokenizer") or not hasattr(text_encoder, "context_length"):
        raise RuntimeError("tokenize_ve expects VETextEncoder-like object with tokenizer/context_length")
    if not hasattr(text_encoder, "encoder") or not hasattr(text_encoder.encoder, "token_embedding"):
        raise RuntimeError("tokenize_ve expects text_encoder.encoder.token_embedding")

    tokenized = text_encoder.tokenizer(texts, context_length=int(text_encoder.context_length)).to(device)
    base_embeds = text_encoder.encoder.token_embedding(tokenized).to(device)
    attention_mask = (tokenized != 0).bool()
    attention_mask = attention_mask.ne(1)
    return TokenizedText(tokenized=tokenized, attention_mask=attention_mask, base_embeds=base_embeds)


def encode_ve_with_optional_inputs_embeds(
    text_encoder,
    tokenized: torch.Tensor,
    base_embeds: torch.Tensor,
    device: torch.device,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(text_encoder, "encode_with_inputs_embeds"):
        raise RuntimeError("encode_ve_with_optional_inputs_embeds expects encode_with_inputs_embeds")
    ie = base_embeds if inputs_embeds is None else inputs_embeds
    text_attention_mask, text_memory_resized, _ = text_encoder.encode_with_inputs_embeds(
        tokenized=tokenized,
        inputs_embeds=ie,
        device=device,
    )
    return text_attention_mask, text_memory_resized

