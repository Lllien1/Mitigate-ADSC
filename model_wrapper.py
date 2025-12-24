"""
model_wrapper.py

SAM3 微调模型封装，支持多种提示学习器：
- averaged/static: 原有SoWA风格
- perclass: Per-class template
- coop: CoOp可学习静态向量
- cocoop: CoCoOp图像条件化向量
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from model_components import (
    AveragedPromptLearner, 
    apply_lora_to_sam, 
    PerClassTemplatePromptLearner,
    CoOpPromptLearner,
    CoCoOpPromptLearner,
    ParallelLoRA,
)
import json
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
    """
    Use official build_sam3_image_model then add LoRA + prompt learner.
    
    支持的 prompt_learner_type:
    - "averaged" / "static": 原有的SoWA风格静态提示
    - "perclass": Per-class template提示
    - "coop": CoOp风格可学习提示向量
    - "cocoop": CoCoOp风格图像条件化提示
    """

    def __init__(
        self,
        bpe_path: Optional[str] = None,
        sam3_ckpt: Optional[str] = None,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        enable_parallel_lora: bool = False,
        parallel_lora_rank: int = 16,
        parallel_lora_alpha: Optional[float] = None,
        device: Optional[torch.device] = None,
        # Per-class 相关参数
        class_list: Optional[Sequence[str]] = None,
        num_templates: int = 4,
        # ========== 提示学习器配置 ==========
        prompt_learner_type: str = "averaged",
        n_ctx: int = 4,
        ctx_init: str = "",
        class_token_position: str = "end",
        use_keywords: bool = False,  # 是否使用关键词聚合
        # CoCoOp特有参数
        cocoop_vis_dim: int = 256,
        cocoop_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_learner_type = prompt_learner_type.lower()

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
            wrapped = apply_lora_to_sam(
                self.backbone.vision_backbone.trunk,
                target_substrings=("qkv",),
                rank=lora_rank,
                alpha=lora_alpha,
            )
            print(f"[INFO] Applied qkv-LoRA to {len(wrapped)} linear layers in vision trunk")        

        # parallel LoRA injection
        if enable_parallel_lora:
            from sam3.model.vitdet import Attention
            for module in self.backbone.vision_backbone.trunk.modules():
                if isinstance(module, Attention):
                    in_dim = module.proj.in_features
                    out_dim = module.proj.out_features
                    module.out_adapter = ParallelLoRA(in_features=in_dim, out_features=out_dim,
                                                      rank=parallel_lora_rank, alpha=parallel_lora_alpha)
                    module.enable_parallel_lora = True

        # freeze vision except LoRA weights
        if freeze_vision:
            for n, p in self.backbone.vision_backbone.trunk.named_parameters():
                if "lora" in n:
                    continue
                p.requires_grad = False
        if freeze_text:
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        self.text_encoder = self.backbone.language_backbone

        # ========== 构建 prompt_learner ==========
        if self.prompt_learner_type == "perclass":
            if class_list is None:
                raise ValueError("Per-class prompt learner requested but class_list is None")
            self.prompt_learner = PerClassTemplatePromptLearner(
                text_encoder=self.text_encoder,
                class_names=class_list,
                n_ctx=n_ctx,
                num_templates=num_templates,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_encoder, "resizer", None),
            )
            print(f"[INFO] Using PerClassTemplatePromptLearner with {len(class_list)} classes")
            
        elif self.prompt_learner_type == "coop":
            self.prompt_learner = CoOpPromptLearner(
                text_encoder=self.text_encoder,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_encoder, "resizer", None),
                class_token_position=class_token_position,
                use_keywords=use_keywords,
            )
            print(f"[INFO] Using CoOpPromptLearner: n_ctx={n_ctx}, ctx_init='{ctx_init}', use_keywords={use_keywords}")
            
        elif self.prompt_learner_type == "cocoop":
            self.prompt_learner = CoCoOpPromptLearner(
                text_encoder=self.text_encoder,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_encoder, "resizer", None),
                class_token_position=class_token_position,
                vis_dim=cocoop_vis_dim,
                reduction_factor=cocoop_reduction,
                use_keywords=use_keywords,
            )
            print(f"[INFO] Using CoCoOpPromptLearner: n_ctx={n_ctx}, vis_dim={cocoop_vis_dim}, use_keywords={use_keywords}")
            
        else:
            # 默认: averaged / static
            self.prompt_learner = AveragedPromptLearner(
                text_encoder=self.text_encoder,
                n_ctx=n_ctx,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_encoder, "resizer", None),
            )
            print(f"[INFO] Using AveragedPromptLearner (static/SoWA style)")

        self.to(self.device)

    @staticmethod
    def _load_ckpt_state(ckpt_path: str):
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError("Please install safetensors: pip install safetensors") from e
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

    def forward(
        self, 
        images: torch.Tensor, 
        prompt_lists: Sequence[List[str]], 
        class_names: Optional[Sequence[str]] = None
    ) -> dict:
        """
        Forward pass with support for different prompt learner types.
        
        Args:
            images: (B, 3, H, W) 输入图像
            prompt_lists: [B]个列表，每个包含提示词
                CoOp/CoCoOp简化版: [["{anomaly/normal} {cls_name}"], ...]
                CoOp/CoCoOp完整版: [["{anomaly/normal} {cls_name}", "kw1", "kw2"], ...]
            class_names: 可选，用于per-class prompt learner
        
        Returns:
            dict包含分割输出和中间特征
        """
        images = images.to(self.device)
        backbone_out = self.backbone.forward_image(images)
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vis_pos = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos]

        B = images.shape[0]
        
        # 提取图像特征 (用于CoCoOp)
        image_features_for_cocoop = None
        if self.prompt_learner_type == "cocoop":
            # vis_feats[0]: (B, C, H, W) → (B, C)
            image_features_for_cocoop = vis_feats[0].mean(dim=[2, 3])
        
        # 调用 prompt_learner
        if hasattr(self.prompt_learner, "class_to_idx") and class_names is not None:
            # PerClassTemplatePromptLearner
            cls_ids = [
                self.prompt_learner.class_to_idx.get(c.lower(), 0) if c is not None else 0 
                for c in class_names
            ]
            prompt_seq, prompt_mask = self.prompt_learner(
                prompt_lists, 
                class_ids=cls_ids, 
                image_features=image_features_for_cocoop,
                device=self.device
            )
        else:
            # CoOp / CoCoOp / Averaged
            prompt_seq, prompt_mask = self.prompt_learner(
                prompt_lists, 
                image_features=image_features_for_cocoop,
                device=self.device
            )
        
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

        out = {
            "pred_masks": seg_out.get("pred_masks"),
            "semantic_seg": seg_out.get("semantic_seg"),
            "presence_logit": seg_out.get("presence_logit"),
            "iou_predictions": seg_out.get("iou_predictions"),
            "decoder_hs": hs,
            "reference_boxes": reference_boxes,
            "prompt_seq": prompt_seq,
        }

        # decoder features
        decoder_feat = None
        for key in ("mask_features", "mask_feat", "mask_pred_feat", "decoder_features"):
            v = seg_out.get(key, None)
            if isinstance(v, torch.Tensor) and v is not None and v.dim() == 4:
                decoder_feat = v
                break

        if decoder_feat is None:
            try:
                decoder_feat = vis_feats[0]
            except Exception:
                decoder_feat = None

        out["decoder_features"] = decoder_feat

        return out