"""
model_wrapper.py - SAM3 + MSAD + Spurious-Gating + CompoundPromptLearnerV3
=====================================================

This wrapper intentionally does NOT restore FiLo.
It supports:
  - Official SAM3 image model wrapper (FineTuneSAM3Official)
  - Minimal custom wrapper (FineTuneSAM3) to satisfy train.py import
  - CompoundPromptLearnerV3 (V + w + W_k), where eta only modulates w
  - Fixed Spurious Prompt Set -> spurious map/presence -> eta feedback
  - MSAD module (Multi-Shape Anomaly Detection) as a replacement of FiLo

Key conventions:
  - Output keys for MSAD:
      out["msad_anomaly_score"] : (B, H, W) in [0,1]
      out["msad_aggregated_map"]: (B, 2, H, W) softmax over {normal, abnormal}
      out["msad_anomaly_maps"]  : list of per-level maps (optional)
  - Spurious gating:
      out["spurious_map"]       : (B, H, W) in [0,1]
      out["spurious_presence"]  : (B,)
      out["eta_spurious"]       : (B,)
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model

# Reuse your existing helpers
from model_components import (
    AveragedPromptLearner,
    apply_lora_to_sam,
    PerClassTemplatePromptLearner,
    CoOpPromptLearner,
    CoCoOpPromptLearner,
    ParallelLoRA,
    apply_qkv_lora_to_sam,
)

from compound_prompt_learner import CompoundPromptLearnerV3

# MSAD (your migrated module)
from msad_module import MSAD


# -------------------------
# Utilities
# -------------------------

def _load_ckpt_state(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    for k in ("model", "state_dict"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            return state[k]
    if isinstance(state, dict):
        return state
    raise RuntimeError(f"Unsupported checkpoint format at: {path}")


def _filter_state_dict_for_model(
    model: nn.Module,
    raw_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int, int, int]:
    """
    Keep only keys that exist in model.state_dict() and shape matches.
    This avoids huge 'unexpected' due to format mismatch.
    """
    model_state = model.state_dict()
    new_state: Dict[str, torch.Tensor] = {}

    # strip common wrappers
    def _strip_prefix(k: str) -> str:
        if k.startswith("module."):
            return k[len("module.") :]
        return k

    kept = 0
    for k, v in raw_state.items():
        kk = _strip_prefix(k)
        if kk in model_state and hasattr(v, "shape") and model_state[kk].shape == v.shape:
            new_state[kk] = v
            kept += 1

    missing = len([k for k in model_state.keys() if k not in new_state])
    unexpected = len([k for k in raw_state.keys() if _strip_prefix(k) not in model_state])
    return new_state, kept, missing, unexpected


def _install_textenc_compat(text_enc: nn.Module) -> nn.Module:
    """
    Your VETextEncoder has `.tokenizer` but not `.tokenize`.
    Some prompt learners / old code paths expect `.tokenize`.
    We add a lightweight `tokenize()` wrapper if missing.
    """
    if not hasattr(text_enc, "tokenize"):

        def _tokenize(texts, context_length: Optional[int] = None, device: Optional[torch.device] = None):
            # VETextEncoder.tokenizer(texts, context_length=...) -> LongTensor[B, L]
            tok = text_enc.tokenizer(
                texts,
                context_length=context_length if context_length is not None else getattr(text_enc, "context_length", 32),
            )
            if device is not None:
                tok = tok.to(device)
            return tok

        setattr(text_enc, "tokenize", _tokenize)

    return text_enc


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# -------------------------
# Minimal wrapper (keep for train.py import safety)
# -------------------------

class FineTuneSAM3(nn.Module):
    """
    Minimal wrapper to satisfy:
      from model_wrapper import FineTuneSAM3, FineTuneSAM3Official

    Not used in your current --use_official pipeline, but must exist.
    """

    def __init__(
        self,
        bpe_path: Optional[str] = None,
        sam3_ckpt: Optional[str] = None,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        lora_layer_ids: Optional[Sequence[int]] = None,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        device: Optional[torch.device] = None,
        n_ctx: int = 4,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=False,
            checkpoint_path=sam3_ckpt,   # ✅ 让官方加载
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )


        self.backbone = full_model.backbone
        self.transformer = full_model.transformer
        self.segmentation_head = full_model.segmentation_head
        self.hidden_dim = self.transformer.d_model
        self.num_feature_levels = 1

        if enable_lora:
            wrapped_layers = apply_lora_to_sam(
                self.backbone.vision_backbone.trunk,
                target_substrings=("qkv",),
                rank=lora_rank,
                alpha=lora_alpha,
                layer_ids=lora_layer_ids,
            )
            print(f"[INFO] Applied qkv-LoRA to {len(wrapped_layers)} linear layers")

        if freeze_vision:
            for n, p in self.backbone.vision_backbone.trunk.named_parameters():
                if "lora" in n:
                    continue
                p.requires_grad = False

        if freeze_text:
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        self.text_enc = _install_textenc_compat(self.backbone.language_backbone)

        self.prompt_learner = AveragedPromptLearner(
            text_encoder=self.text_enc,
            n_ctx=n_ctx,
            freeze_text_encoder=freeze_text,
            proj=getattr(self.text_enc, "resizer", None),
        )

    def forward(self, images: torch.Tensor, prompt_lists: List[List[str]], class_names: Optional[List[str]] = None) -> dict:
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
        hs, reference_boxes, _, _ = self.transformer.decoder(
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
            backbone_feats=all_fpn_feats,
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
            "prompt_seq": prompt_seq,
        }


# -------------------------
# Official wrapper with MSAD + Spurious gating + CompoundPromptLearnerV3
# -------------------------

class FineTuneSAM3Official(nn.Module):
    def __init__(
        self,
        bpe_path: Optional[str] = None,
        sam3_ckpt: Optional[str] = None,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        lora_layer_ids: Optional[Sequence[int]] = None,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        device: Optional[torch.device] = None,
        num_feature_levels: int = 1,
        selected_levels: Optional[List[int]] = None,

        # prompt learner
        prompt_learner_type: str = "compound",
        n_ctx: int = 4,
        ctx_init: str = "",
        num_templates: int = 3,
        class_list: Optional[List[str]] = None,
        class_token_position: str = "end",
        use_keywords: bool = True,
        cocoop_vis_dim: int = 256,
        cocoop_reduction: int = 16,

        # compound
        compound_mode: str = "cocoop",
        compound_n_ctx: int = 4,
        compound_n_ctx_offset: int = 4,
        compound_num_abnormal: int = 10,
        compound_enable_dap: bool = False,
        compound_dap_top_k: int = 10,
        compound_meta_reduction: int = 16,

        # MSAD
        enable_msad: bool = False,
        msad_use_shape_attention: bool = True,
        msad_learnable_level_weights: bool = True,
        msad_learnable_temperature: bool = True,
        msad_temperature: float = 100.0,
        msad_output_size: int = 518,
        msad_num_levels: Optional[int] = None,

        # spurious gating
        enable_spurious_gating: bool = False,
        spurious_score_threshold: float = 0.20,
        spurious_topk_ratio: float = 0.02,
        spurious_prompt_set: Optional[List[str]] = None,

        # misc
        enable_multiscale_output: bool = False,

        # accept extra args safely
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_learner_type = str(prompt_learner_type).lower()
        self.enable_multiscale_output = bool(enable_multiscale_output)
        self.selected_levels = selected_levels
        self.compound_dap_use_multilevel = bool(kwargs.get("compound_dap_use_multilevel", False))
        self.compound_dap_num_levels = int(kwargs.get("compound_dap_num_levels", 0) or 0)
        self.compound_use_text_encoder = bool(kwargs.get("compound_use_text_encoder", False))
        self.compound_abnormal_word = str(kwargs.get("compound_abnormal_word", "anomaly"))
        self.compound_pooling = str(kwargs.get("compound_pooling", "ctx_only")).lower()
        self.compound_abnormal_order = str(kwargs.get("compound_abnormal_order", "v_then_wk")).lower()
        self.compound_dap_spurious_filter = bool(kwargs.get("compound_dap_spurious_filter", False))
        self.compound_dap_spurious_alpha = float(kwargs.get("compound_dap_spurious_alpha", 1.0))
        self.use_sam3_text_prompt = bool(kwargs.get("use_sam3_text_prompt", False))
        self.trace_dump = bool(kwargs.get("trace_dump", False))
        self.compound_disable_w = bool(kwargs.get("compound_disable_w", False))
        self.msad_return_similarity_logits = bool(kwargs.get("msad_return_similarity_logits", False))

        # build base model
        full_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=False,
            checkpoint_path=sam3_ckpt,   # ✅ 让官方加载
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )


        self.backbone = full_model.backbone
        self.transformer = full_model.transformer
        self.segmentation_head = full_model.segmentation_head
        self.hidden_dim = self.transformer.d_model
        self.max_feature_levels = getattr(full_model, "num_feature_levels", 1)

        self.num_feature_levels = min(int(num_feature_levels), int(self.max_feature_levels))
        print(f"[INFO] Using {self.num_feature_levels} feature levels")
        print(f"[DEBUG] full_model.num_feature_levels = {full_model.num_feature_levels}")
        # full_model build 完之后
        sd = full_model.state_dict()
        print("[DEBUG] example key:", next(iter(sd.keys())))
        print("[DEBUG] num keys:", len(sd))


        # LoRA (optional; make sure NOT assigning returned list to trunk)
        if enable_lora:
            wrapped_layers = apply_lora_to_sam(
                self.backbone.vision_backbone.trunk,
                target_substrings=("qkv",),
                rank=lora_rank,
                alpha=lora_alpha,
                layer_ids=lora_layer_ids,
            )
            print(f"[INFO] Applied qkv-LoRA to {len(wrapped_layers)} linear layers")

        if bool(kwargs.get("enable_out_adapter_lora", False)):
            from sam3.model.vitdet import Attention
            out_adapter_rank = int(kwargs.get("parallel_lora_rank", 16))
            out_adapter_alpha = kwargs.get("parallel_lora_alpha", None)
            for module in self.backbone.vision_backbone.trunk.modules():
                if isinstance(module, Attention) and hasattr(module, "proj"):
                    in_dim = int(module.proj.in_features)
                    out_dim = int(module.proj.out_features)
                    module.out_adapter = ParallelLoRA(
                        in_features=in_dim,
                        out_features=out_dim,
                        rank=out_adapter_rank,
                        alpha=out_adapter_alpha,
                    )
                    module.enable_parallel_lora = True

        if bool(kwargs.get("enable_parallel_lora", False)):
            qkv_rank = int(kwargs.get("parallel_lora_rank", 16))
            qkv_alpha = kwargs.get("parallel_lora_alpha", None)
            qkv_target = str(kwargs.get("parallel_lora_target", "qv_only"))
            wrapped_layers = apply_qkv_lora_to_sam(
                self.backbone.vision_backbone.trunk,
                rank=qkv_rank,
                alpha=qkv_alpha,
                layer_ids=kwargs.get("parallel_lora_layer_ids", None),
                target=qkv_target,
            )
            print(f"[INFO] Applied qkv-parallel LoRA to {len(wrapped_layers)} layers (target={qkv_target})")

        # Freeze
        if freeze_vision:
            for n, p in self.backbone.vision_backbone.trunk.named_parameters():
                if "lora" in n:
                    continue
                p.requires_grad = False
        if freeze_text:
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        # text encoder compat
        self.text_enc = _install_textenc_compat(self.backbone.language_backbone)

        # -------------------------
        # MSAD init (前移到 __init__，避免 lazy init 影响优化器)
        # -------------------------
        self.enable_msad = bool(enable_msad)
        self.msad: Optional[MSAD] = None
        if self.enable_msad:
            default_lv = 4
            use_lv = default_lv if msad_num_levels is None else int(msad_num_levels)
            use_lv = max(1, int(use_lv))
            try:
                self.msad = MSAD(
                    dim=int(self.hidden_dim),
                    num_levels=int(use_lv),
                    output_size=int(msad_output_size),
                    use_shape_attention=bool(msad_use_shape_attention),
                    learnable_level_weights=bool(msad_learnable_level_weights),
                    learnable_temperature=bool(msad_learnable_temperature),
                    temperature=float(msad_temperature),
                    use_vision_adapter=bool(kwargs.get("msad_use_vision_adapter", False)),
                    vision_adapter_reduction=int(kwargs.get("msad_vision_adapter_reduction", 2)),
                    vision_adapter_shared=bool(kwargs.get("msad_vision_adapter_shared", True)),
                ).to(self.device)
                print(f"[INFO] MSAD enabled: dim={self.hidden_dim}, num_levels={use_lv}, "
                      f"shape_attn={msad_use_shape_attention}")
            except Exception as e:
                print(f"[WARN] MSAD init failed: {e}")
                self.enable_msad = False
                self.msad = None

        # -------------------------
        # Spurious gating config
        # -------------------------
        self.enable_spurious_gating = bool(enable_spurious_gating)
        self.spurious_score_threshold = float(spurious_score_threshold)
        self.spurious_topk_ratio = float(spurious_topk_ratio)

        if spurious_prompt_set is None:
            # Fixed universal spurious prompt set (generic; you can edit safely)
            spurious_prompt_set = [
                "background clutter",
                "illumination change",
                "shadow",
                "specular highlight",
                "reflection",
                "texture change",
                "material difference",
                "color cast",
                "sensor noise",
                "blur",
                "compression artifact",
                "manufacturing trace",
                "camera viewpoint change",
            ]
        self.spurious_prompt_set = list(spurious_prompt_set)

        if self.enable_spurious_gating:
            print(f"[INFO] Spurious gating enabled: threshold={self.spurious_score_threshold}, topk_ratio={self.spurious_topk_ratio}")

        # -------------------------
        # Prompt learner init
        # -------------------------
        if self.prompt_learner_type == "perclass":
            if class_list is None:
                raise ValueError("Per-class prompt learner requested but class_list is None")
            self.prompt_learner = PerClassTemplatePromptLearner(
                text_encoder=self.text_enc,
                class_names=class_list,
                n_ctx=n_ctx,
                num_templates=num_templates,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_enc, "resizer", None),
            )
            print(f"[INFO] Using PerClassTemplatePromptLearner with {len(class_list)} classes")

        elif self.prompt_learner_type == "coop":
            self.prompt_learner = CoOpPromptLearner(
                text_encoder=self.text_enc,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_enc, "resizer", None),
                class_token_position=class_token_position,
                use_keywords=use_keywords,
            )
            print(f"[INFO] Using CoOpPromptLearner: n_ctx={n_ctx}")

        elif self.prompt_learner_type == "cocoop":
            vis_dim = cocoop_vis_dim if cocoop_vis_dim > 0 else self.hidden_dim
            self.prompt_learner = CoCoOpPromptLearner(
                text_encoder=self.text_enc,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_enc, "resizer", None),
                class_token_position=class_token_position,
                use_keywords=use_keywords,
                vis_dim=vis_dim,
                reduction_factor=cocoop_reduction,
            )
            print(f"[INFO] Using CoCoOpPromptLearner: n_ctx={n_ctx}")

        elif self.prompt_learner_type == "compound":
            # CompoundPromptLearnerV3
            vis_dim = cocoop_vis_dim if cocoop_vis_dim > 0 else self.hidden_dim
            self.prompt_learner = CompoundPromptLearnerV3(
                text_encoder=self.text_enc,
                n_V=int(compound_n_ctx),
                n_w=int(compound_n_ctx_offset),
                n_W=int(compound_n_ctx_offset),
                num_abnormal_prompts=int(compound_num_abnormal),
                mode=str(compound_mode),
                vis_dim=int(vis_dim),
                top_k=int(compound_dap_top_k),
                enable_dap=bool(compound_enable_dap),
                dap_top_k=int(compound_dap_top_k),
                meta_net_reduction=int(compound_meta_reduction),
                freeze_text_encoder=freeze_text,
                output_dim=int(self.hidden_dim),
                disable_w=bool(self.compound_disable_w),
            )
            print(
                f"[INFO] Using CompoundPromptLearnerV3: mode={compound_mode}, n_V={compound_n_ctx}, "
                f"n_w={compound_n_ctx_offset}, num_abnormal={compound_num_abnormal}, enable_dap={compound_enable_dap}"
            )
        else:
            self.prompt_learner = AveragedPromptLearner(
                text_encoder=self.text_enc,
                n_ctx=n_ctx,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_enc, "resizer", None),
            )
            print(f"[INFO] Using AveragedPromptLearner: n_ctx={n_ctx}")

    # -------------------------
    # Spurious gating core
    # -------------------------
    @torch.no_grad()
    def _encode_text_global(self, texts: List[str]) -> torch.Tensor:
        """
        Use SAM3 backbone's forward_text to obtain a global embedding per prompt.
        Returns: (T, D)
        """
        out = self.backbone.forward_text(texts, device=self.device)
        feats = out.get("language_features", out.get("text_features", None))
        if feats is None:
            # fallback: VETextEncoder forward returns (attn_mask, memory, inputs_embeds)
            attn_mask, text_mem, _ = self.text_enc(texts, device=self.device)
            # text_mem: (L, B, D); take CLS-like average over tokens
            feats = text_mem.mean(dim=0)  # (B, D)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return feats  # (T, D)

    def _compute_spurious_eta(self, vis_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        vis_feat: (B, C, H, W), C == hidden_dim
        Returns:
          spurious_map: (B, H, W) in [0,1]
          spurious_presence: (B,)
          eta: (B,)
        """
        B, C, H, W = vis_feat.shape

        # text: (T, D)
        sp_txt = self._encode_text_global(self.spurious_prompt_set)  # (T, D)
        sp_txt = _l2_normalize(sp_txt, dim=-1).to(vis_feat.device)

        # vision patches: (B, HW, C)
        patches = vis_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        patches = _l2_normalize(patches, dim=-1)

        # sim: (B, HW, T)
        sim = torch.matmul(patches, sp_txt.t())
        # max over spurious prompts: (B, HW)
        sim_max = sim.max(dim=-1).values
        sp_map = sim_max.view(B, H, W)

        # normalize to [0,1] (stable across batches)
        sp_map01 = torch.sigmoid(sp_map)

        # presence: mean of top-k pixels
        HW = H * W
        k = max(1, int(self.spurious_topk_ratio * HW))
        topv = sp_map01.view(B, -1).topk(k, dim=1).values
        presence = topv.mean(dim=1)  # (B,)

        # eta feedback: only modulates w
        thr = self.spurious_score_threshold
        eta = (presence - thr) / max(1e-6, (1.0 - thr))
        eta = eta.clamp(0.0, 1.0)

        return sp_map01, presence, eta

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        images: torch.Tensor,
        prompt_lists: List[List[str]],
        class_names: Optional[List[str]] = None,
    ) -> dict:
        images = images.to(self.device)
        backbone_out = self.backbone.forward_image(images)

        all_fpn_feats = backbone_out["backbone_fpn"]

        # -------------------------
        # MSAD forward (Route B: uses multi-level FPN features)
        # -------------------------
        msad_out = None


        all_pos_enc = backbone_out["vision_pos_enc"]

        if not hasattr(self, "_debug_once"):
            self._debug_once = True
            print("[DEBUG] len(backbone_fpn) =", len(all_fpn_feats))
            print("[DEBUG] len(vision_pos_enc) =", len(all_pos_enc))
            for i, f in enumerate(all_fpn_feats):
                print(f"[DEBUG] fpn[{i}] shape =", tuple(f.shape))

        # encoder uses last N feature levels (memory-friendly)
        vis_feats = all_fpn_feats[-self.num_feature_levels :]
        vis_pos = all_pos_enc[-self.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos]
        B = images.shape[0]

        # spurious gating -> eta (B,)
        sp_map = None
        sp_presence = None
        eta = None
        if self.enable_spurious_gating:
            sp_map, sp_presence, eta = self._compute_spurious_eta(vis_feats[0])

        # compound prompt (eta only modulates w) or official text prompt path
        compound_dap_weights = None
        text_features_structured = None
        trace_info = None

        if self.use_sam3_text_prompt:
            captions = []
            for pl in prompt_lists:
                if isinstance(pl, list) and len(pl) > 0:
                    captions.append(str(pl[0]))
                else:
                    captions.append("object")
            text_out = self.backbone.forward_text(captions, device=self.device)
            prompt_seq = text_out["language_features"]
            prompt_mask = text_out["language_mask"]
        if (not self.use_sam3_text_prompt) and self.prompt_learner_type == "compound":
            # vis_global + patch_features for DAP
            vis_global = vis_feats[0].mean(dim=[2, 3])  # (B, C)

            patch_features = None
            dap_spurious_scores = None
            if getattr(self.prompt_learner, "enable_dap", False):
                if self.compound_dap_use_multilevel and len(vis_feats) > 0:
                    lv = vis_feats
                    if self.compound_dap_num_levels > 0:
                        lv = vis_feats[: self.compound_dap_num_levels]
                    patch_features = torch.cat([f.flatten(2).transpose(1, 2) for f in lv], dim=1)
                    if self.compound_dap_spurious_filter and isinstance(sp_map, torch.Tensor):
                        sp_levels = []
                        for f in lv:
                            sp_lv = F.interpolate(
                                sp_map.unsqueeze(1),
                                size=f.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(1)
                            sp_levels.append(sp_lv.flatten(1))
                        dap_spurious_scores = torch.cat(sp_levels, dim=1)
                else:
                    feat = vis_feats[0]
                    patch_features = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
                    if self.compound_dap_spurious_filter and isinstance(sp_map, torch.Tensor):
                        dap_spurious_scores = sp_map.flatten(1)

            prompt_result = self.prompt_learner(
                prompt_lists,
                vis_feats=vis_feats[0],
                patch_features=patch_features,
                device=self.device,
                eta=eta,  # eta only modulates w inside CompoundPromptLearnerV3
                class_names=class_names,
                abnormal_word=self.compound_abnormal_word,
                use_text_encoder=self.compound_use_text_encoder,
                pooling=self.compound_pooling,
                dap_spurious_scores=dap_spurious_scores,
                dap_spurious_alpha=self.compound_dap_spurious_alpha,
                abnormal_order=self.compound_abnormal_order,
            )

            # (B, 1+K, max_len, D), (B, 1+K, max_len), optional selected_patches
            prompt_prefixes = prompt_result[0]
            _prompt_mask3d = prompt_result[1]
            if len(prompt_result) >= 3:
                compound_dap_weights = prompt_result[2]
            proto_suspicious_pl = prompt_result[3] if len(prompt_result) >= 4 else None

            # prototype per prompt: mean over prefix_len -> (B, 1+K, D)
            proto = prompt_prefixes.mean(dim=2)  # (B, 1+K, D)
            # transformer expects (P, B, D)
            prompt_seq = proto.permute(1, 0, 2).contiguous()
            prompt_mask = torch.zeros(
                prompt_seq.shape[1], prompt_seq.shape[0],
                dtype=torch.bool, device=prompt_seq.device
            )

            P = prompt_seq.shape[0]
            text_features_structured = {
                "normal": prompt_seq[0],                          # (B, D)
                "abnormal_all": prompt_seq[1:].permute(1, 0, 2),  # (B, K, D)
                "abnormal_mean": prompt_seq[1:].mean(dim=0),      # (B, D)
                "num_abnormal": P - 1,
            }

            # proto_suspicious: mean of w tokens in normal prefix
            if isinstance(proto_suspicious_pl, torch.Tensor):
                text_features_structured["proto_suspicious"] = proto_suspicious_pl
            else:
                n_V = getattr(self.prompt_learner, "n_V", 0)
                n_w = getattr(self.prompt_learner, "n_w", 0)
                if (not bool(getattr(self.prompt_learner, "disable_w", False))) and n_V > 0 and n_w > 0:
                    normal_prefix_tokens = prompt_prefixes[:, 0, :, :]  # (B, max_len, D)
                    w_tokens = normal_prefix_tokens[:, n_V:n_V + n_w, :]
                    text_features_structured["proto_suspicious"] = w_tokens.mean(dim=1)  # (B, D)

        else:
            # other prompt learners
            image_features_for_cocoop = None
            if self.prompt_learner_type == "cocoop":
                image_features_for_cocoop = vis_feats[0].mean(dim=[2, 3])

            prompt_seq, prompt_mask = self.prompt_learner(
                prompt_lists,
                image_features=image_features_for_cocoop,
                device=self.device,
            )

        if self.trace_dump:
            with torch.no_grad():
                p = prompt_seq
                p_stats = {
                    "shape": tuple(p.shape) if isinstance(p, torch.Tensor) else None,
                    "mean": float(p.detach().float().mean().item()) if isinstance(p, torch.Tensor) else None,
                    "std": float(p.detach().float().std().item()) if isinstance(p, torch.Tensor) else None,
                }
                trace_info = {
                    "fpn_levels": int(len(all_fpn_feats)) if isinstance(all_fpn_feats, list) else None,
                    "pos_levels": int(len(all_pos_enc)) if isinstance(all_pos_enc, list) else None,
                    "encoder_levels_used": int(len(vis_feats)) if isinstance(vis_feats, list) else None,
                    "seg_head_levels": int(len(all_fpn_feats)) if isinstance(all_fpn_feats, list) else None,
                    "use_sam3_text_prompt": bool(self.use_sam3_text_prompt),
                    "prompt_stats": p_stats,
                }

        if self.enable_msad and self.msad is not None:
            feats = list(all_fpn_feats)
            avail_lv = len(feats)
            need_lv = int(self.msad.num_levels)

            if avail_lv >= need_lv:
                msad_feats = feats[:need_lv]
            else:
                msad_feats = feats + [feats[-1]] * (need_lv - avail_lv)

            if isinstance(text_features_structured, dict) and "normal" in text_features_structured:
                t_normal = text_features_structured["normal"]
                t_abn = text_features_structured["abnormal_mean"]
                text_features_for_msad = torch.stack([t_normal, t_abn], dim=1)
            else:
                if prompt_seq.dim() == 3 and prompt_seq.shape[0] >= 2:
                    text_features_for_msad = prompt_seq[:2].permute(1, 0, 2).contiguous()
                else:
                    t = prompt_seq.mean(dim=0, keepdim=True)
                    if t.dim() == 3:
                        t = t.squeeze(0)
                    text_features_for_msad = torch.stack([t, t], dim=1)

            msad_out = self.msad(
                fpn_features=msad_feats,
                text_features=text_features_for_msad,
                return_intermediate=True,
                return_similarity_logits=bool(getattr(self, "msad_return_similarity_logits", False)),
            )
                
        if msad_out is not None and not hasattr(self, "_msad_debug_once"):
            self._msad_debug_once = True
            print("[DEBUG] MSAD got feats:", [tuple(f.shape) for f in msad_feats])

        # -------------------------
        # SAM3 encoder/decoder
        # -------------------------
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
            "compound_dap_weights": compound_dap_weights,
        }
        if trace_info is not None:
            out["trace_info"] = trace_info

        # decoder features (for your losses)
        decoder_feat = None
        for key in ("mask_features", "mask_feat", "mask_pred_feat", "decoder_features"):
            v = seg_out.get(key, None)
            if isinstance(v, torch.Tensor) and v is not None and v.dim() == 4:
                decoder_feat = v
                break
        if decoder_feat is None:
            decoder_feat = vis_feats[0] if vis_feats else None
        out["decoder_features"] = decoder_feat

        # compound structured outputs
        if isinstance(text_features_structured, dict):
            out["text_features_structured"] = text_features_structured
            if "proto_suspicious" in text_features_structured:
                out["proto_suspicious"] = text_features_structured["proto_suspicious"]

        # spurious gating outputs
        if self.enable_spurious_gating:
            out["spurious_map"] = sp_map
            out["spurious_presence"] = sp_presence
            out["eta_spurious"] = eta

        # MSAD outputs
        if msad_out is not None:
            out["msad_anomaly_score"] = msad_out.get("anomaly_score", None)
            out["msad_aggregated_map"] = msad_out.get("aggregated_map", None)
            out["msad_anomaly_maps"] = msad_out.get("anomaly_maps", None)
            out["msad_similarity_logits_maps"] = msad_out.get("similarity_logits_maps", None)
            out["msad_aggregated_logits_map"] = msad_out.get("aggregated_logits_map", None)
            out["msad_debug"] = msad_out.get("debug", None)

        # optional multiscale debug
        if self.enable_multiscale_output:
            out["multiscale_features"] = {
                "used_features": vis_feats,
                "all_fpn_features": all_fpn_feats,
                "feature_sizes": vis_feat_sizes,
                "num_levels": len(vis_feats),
                "selected_levels": self.selected_levels,
                "pos_encodings": vis_pos,
            }

        return out


# Alias for older code paths
MultiscaleSAM3 = FineTuneSAM3Official
