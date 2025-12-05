from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from model_components import AveragedPromptLearner, apply_lora_to_sam
from sam3.model_builder import (
    _create_segmentation_head,
    _create_sam3_transformer,
    _create_text_encoder,
    _create_vision_backbone,
    build_sam3_image_model,
)


class FineTuneSAM3(nn.Module):
    """Simplified build (custom) with LoRA + prompt learner."""

    def __init__(
        self,
        bpe_path: Optional[str] = None,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vision_backbone = _create_vision_backbone()
        self.text_encoder = _create_text_encoder(
            bpe_path or "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )
        self.transformer = _create_sam3_transformer()
        self.segmentation_head = _create_segmentation_head()
        self.hidden_dim = self.transformer.d_model
        self.num_feature_levels = 1

        if enable_lora:
            apply_lora_to_sam(
                self.vision_backbone.trunk,
                target_substrings=("qkv",),
                rank=lora_rank,
                alpha=lora_alpha,
            )

        if freeze_vision:
            for n, p in self.vision_backbone.trunk.named_parameters():
                if "lora" in n:
                    continue
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.prompt_learner = AveragedPromptLearner(
            text_encoder=self.text_encoder,
            n_ctx=4,
            freeze_text_encoder=freeze_text,
            proj=self.text_encoder.resizer,
        )
        self.to(self.device)

    def forward(self, images: torch.Tensor, prompt_lists: Sequence[List[str]]) -> dict:
        images = images.to(self.device)
        sam3_features, sam3_pos, _, _ = self.vision_backbone(images)
        vis_feats = sam3_features[-self.num_feature_levels :]
        vis_pos = sam3_pos[-self.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos]

        prompt_seq, prompt_mask = self.prompt_learner(prompt_lists, device=self.device)
        prompt_pos = torch.zeros_like(prompt_seq)

        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos = [x.flatten(2).permute(2, 0, 1) for x in vis_pos]

        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos.copy(),
            prompt=prompt_seq,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None,
        )

        bs = images.shape[0]
        tgt = self.transformer.decoder.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(
            tgt=tgt,
            memory=memory["memory"],
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt_seq,
            text_attention_mask=prompt_mask,
        )

        seg_out = self.segmentation_head(
            backbone_feats=vis_feats,
            obj_queries=hs,
            image_ids=torch.arange(bs, device=self.device),
            encoder_hidden_states=memory["memory"],
            prompt=prompt_seq,
            prompt_mask=prompt_mask,
        )

        return {
            "pred_masks": seg_out.get("pred_masks"),
            "semantic_seg": seg_out.get("semantic_seg"),
            "presence_logit": seg_out.get("presence_logit"),
            "iou_predictions": seg_out.get("iou_predictions"),
            "decoder_hs": hs,
            "reference_boxes": reference_boxes,
        }


class FineTuneSAM3Official(nn.Module):
    """Use official build_sam3_image_model then add LoRA + prompt learner."""

    def __init__(
        self,
        bpe_path: Optional[str] = None,
        sam3_ckpt: Optional[str] = None,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=False,
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        if sam3_ckpt:
            state = self._load_ckpt_state(sam3_ckpt)
            missing, unexpected = full_model.load_state_dict(state, strict=False)
            print(
                f"[INFO] Loaded SAM3 ckpt {sam3_ckpt}, mapped={len(state)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self.backbone = full_model.backbone
        self.transformer = full_model.transformer
        self.segmentation_head = full_model.segmentation_head
        self.hidden_dim = self.transformer.d_model
        self.num_feature_levels = full_model.num_feature_levels

        if enable_lora:
            apply_lora_to_sam(
                self.backbone.vision_backbone.trunk,
                target_substrings=("qkv",),
                rank=lora_rank,
                alpha=lora_alpha,
            )

        if freeze_vision:
            for n, p in self.backbone.vision_backbone.trunk.named_parameters():
                if "lora" in n:
                    continue
                p.requires_grad = False
        if freeze_text:
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        text_encoder = self.backbone.language_backbone
        self.prompt_learner = AveragedPromptLearner(
            text_encoder=text_encoder,
            n_ctx=4,
            freeze_text_encoder=freeze_text,
            proj=text_encoder.resizer if hasattr(text_encoder, "resizer") else None,
        )

        self.to(self.device)

    @staticmethod
    def _load_ckpt_state(ckpt_path: str):
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError("Please install safetensors to load .safetensors weights: pip install safetensors") from e
            raw_state = load_file(ckpt_path, device="cpu")
        else:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw_state = raw.get("model", raw) if isinstance(raw, dict) else raw

        mapped = {}
        for k, v in raw_state.items():
            if k.startswith("detector."):
                nk = k[len("detector.") :]
                mapped[nk] = v
            elif k.startswith("backbone."):
                nk = k[len("backbone.") :]
                mapped[nk] = v
            else:
                mapped[k] = v
        return mapped

    def forward(self, images: torch.Tensor, prompt_lists: Sequence[List[str]]) -> dict:
        images = images.to(self.device)
        backbone_out = self.backbone.forward_image(images)
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vis_pos = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos]

        prompt_seq, prompt_mask = self.prompt_learner(prompt_lists, device=self.device)
        prompt_pos = torch.zeros_like(prompt_seq)

        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos = [x.flatten(2).permute(2, 0, 1) for x in vis_pos]

        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos.copy(),
            prompt=prompt_seq,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None,
        )

        bs = images.shape[0]
        tgt = self.transformer.decoder.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(
            tgt=tgt,
            memory=memory["memory"],
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt_seq,
            text_attention_mask=prompt_mask,
        )
        hs = hs.permute(0, 2, 1, 3).contiguous()

        seg_out = self.segmentation_head(
            backbone_feats=vis_feats,
            obj_queries=hs,
            image_ids=torch.arange(bs, device=self.device),
            encoder_hidden_states=memory["memory"],
            prompt=prompt_seq,
            prompt_mask=prompt_mask,
        )

        return {
            "pred_masks": seg_out.get("pred_masks"),
            "semantic_seg": seg_out.get("semantic_seg"),
            "presence_logit": seg_out.get("presence_logit"),
            "iou_predictions": seg_out.get("iou_predictions"),
            "decoder_hs": hs,
            "reference_boxes": reference_boxes,
        }
