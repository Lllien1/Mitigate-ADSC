"""
model_wrapper.py - FiLo集成版
==============================

SAM3 微调模型封装，支持：
1. 多种提示学习器（CoOp/CoCoOp/PerClass等）
2. VVAttention（视觉自注意力增强）
3. FiLoModule（FiLo官方的6路卷积MMCI模块）
   - LinearLayer: QKV分支（处理偶数层FPN特征）
   - CovLayer: VV分支（处理奇数层FPN特征，6种卷积核）
   - 输出: anomaly_maps用于训练/推理

FiLo数据流（来自官方实现）：
    FPN[0,2,...] → LinearLayer → patch_tokens_qkv → 100*(patch@text.T) → softmax → anomaly_map
    FPN[1,3,...] → CovLayer    → patch_tokens_vv  → 100*(patch@text.T) → softmax → anomaly_map
"""

from typing import List, Optional, Sequence, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, images, prompt_lists):
        backbone_out = self.vision_backbone.forward_image(images)
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
            "prompt_seq": prompt_seq,
        }


# FiLoModule现在在multiscale_modules.py中定义
# 不再需要单独的filo_sam3_minimal


class FineTuneSAM3Official(nn.Module):
    """
    完整SAM3微调模型
    
    支持：
    - VVAttention: 视觉自注意力增强（可选）
    - FiLoModule: FiLo风格的6路卷积MMCI（用于异常检测）
      - LinearLayer: QKV分支（偶数层）
      - CovLayer: VV分支（奇数层，6路多形状卷积）
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
        device: Optional[torch.device] = None,
        num_feature_levels: int = 4,
        selected_levels: Optional[List[int]] = None,
        prompt_learner_type: str = "averaged",
        n_ctx: int = 4,
        ctx_init: str = "",
        num_templates: int = 3,
        class_list: Optional[List[str]] = None,
        class_token_position: str = "end",
        use_keywords: bool = True,
        cocoop_vis_dim: int = 256,
        cocoop_reduction: int = 16,
        # ===== VVAttention 参数 =====
        enable_vv_attention: bool = False,
        vv_num_heads: int = 8,
        vv_dropout: float = 0.1,
        # ===== FiLo模块参数 (6路卷积MMCI) =====
        enable_filo: bool = False,
        filo_dim_out: int = 768,         # 输出维度（对齐文本特征）
        filo_k_linear: int = 4,          # LinearLayer层数
        filo_k_cov: int = 4,             # CovLayer层数
        filo_image_size: int = 518,      # 异常图输出尺寸
        filo_use_alternating: bool = True,  # 是否交替分配（FiLo风格）
        # ===== 方案B: FiLo到Decoder的回灌 =====
        filo_to_decoder: bool = False,       # 是否把FiLo特征喂给decoder
        filo_decoder_mode: str = "memory",   # "memory", "query_bias", "cross_attn"
        filo_decoder_tokens: int = 64,       # 压缩后的FiLo token数量
        # ===== 方案C: 置信度融合头 =====
        enable_conf_fusion_head: bool = False,  # 是否启用置信度融合头
        conf_fusion_hidden_dim: int = 64,       # 融合头隐藏维度
        # ===== Parallel LoRA =====
        enable_parallel_lora: bool = False,
        parallel_lora_rank: int = 16,
        parallel_lora_alpha: float = 1.0,
        # ===== 其他 =====
        enable_multiscale_output: bool = False,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_learner_type = prompt_learner_type.lower()
        self.enable_multiscale_output = enable_multiscale_output
        self.selected_levels = selected_levels
        
        # ===== 保存启用标志 =====
        self.enable_vv_attention = enable_vv_attention
        self.enable_filo = enable_filo
        self.filo_to_decoder = filo_to_decoder and enable_filo  # 需要FiLo启用
        self.filo_decoder_mode = filo_decoder_mode
        self.enable_conf_fusion_head = enable_conf_fusion_head
        
        # 保存FiLo参数供后续使用
        self.filo_dim_out = filo_dim_out
        self.filo_k_linear = filo_k_linear
        self.filo_k_cov = filo_k_cov
        self.filo_image_size = filo_image_size
        self.filo_use_alternating = filo_use_alternating

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
        self.max_feature_levels = full_model.num_feature_levels
        
        self.num_feature_levels = min(num_feature_levels, full_model.num_feature_levels)
        print(f"[INFO] Using {self.num_feature_levels} feature levels")
        
        if selected_levels is not None:
            print(f"[INFO] Selected specific levels for ablation: {selected_levels}")
        
        # ===== VVAttention 模块 =====
        self.vv_attention = None
        if self.enable_vv_attention:
            try:
                from multiscale_modules import VVAttention
                self.vv_attention = VVAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=vv_num_heads,
                    dropout=vv_dropout,
                )
                print(f"[INFO] VVAttention enabled: embed_dim={self.hidden_dim}, num_heads={vv_num_heads}")
            except ImportError as e:
                print(f"[WARN] VVAttention import failed: {e}")
                self.enable_vv_attention = False
        
        # ===== FiLo模块 (6路卷积MMCI) =====
        self.filo_module = None
        if self.enable_filo:
            try:
                from multiscale_modules import FiLoModule
                self.filo_module = FiLoModule(
                    dim_in=self.hidden_dim,          # SAM3 FPN维度 (256)
                    dim_out=self.filo_dim_out,       # 输出维度 (768)
                    text_dim=self.hidden_dim,        # SAM3 text维度 (256)
                    k_linear=self.filo_k_linear,     # LinearLayer层数
                    k_cov=self.filo_k_cov,           # CovLayer层数
                    image_size=self.filo_image_size, # 异常图尺寸
                    use_alternating=self.filo_use_alternating,
                )
                print(f"[INFO] FiLoModule enabled: dim_in={self.hidden_dim}, dim_out={self.filo_dim_out}, text_dim={self.hidden_dim}")
            except ImportError as e:
                print(f"[WARN] FiLoModule import failed: {e}")
                self.enable_filo = False

        if enable_lora:
            wrapped = apply_lora_to_sam(
                self.backbone.vision_backbone.trunk,
                target_substrings=("qkv",),
                rank=lora_rank,
                alpha=lora_alpha,
            )
            print(f"[INFO] Applied qkv-LoRA to {len(wrapped)} linear layers")        

        if enable_parallel_lora:
            from sam3.model.vitdet import Attention
            for module in self.backbone.vision_backbone.trunk.modules():
                if isinstance(module, Attention):
                    in_dim = module.proj.in_features
                    out_dim = module.proj.out_features
                    module.out_adapter = ParallelLoRA(
                        in_features=in_dim, out_features=out_dim,
                        rank=parallel_lora_rank, alpha=parallel_lora_alpha
                    )
                    module.enable_parallel_lora = True

        if freeze_vision:
            for n, p in self.backbone.vision_backbone.trunk.named_parameters():
                if "lora" in n:
                    continue
                p.requires_grad = False
        if freeze_text:
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        self.text_encoder = self.backbone.language_backbone

        # ===== Prompt Learner =====
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
            print(f"[INFO] Using CoOpPromptLearner: n_ctx={n_ctx}")
            
        elif self.prompt_learner_type == "cocoop":
            vis_dim = cocoop_vis_dim if cocoop_vis_dim > 0 else self.hidden_dim
            self.prompt_learner = CoCoOpPromptLearner(
                text_encoder=self.text_encoder,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_encoder, "resizer", None),
                class_token_position=class_token_position,
                use_keywords=use_keywords,
                vis_dim=vis_dim,
                reduction_factor=cocoop_reduction,  # 修正：参数名是reduction_factor
            )
            print(f"[INFO] Using CoCoOpPromptLearner: n_ctx={n_ctx}")
            
        else:
            self.prompt_learner = AveragedPromptLearner(
                text_encoder=self.text_encoder,
                n_ctx=n_ctx,
                freeze_text_encoder=freeze_text,
                proj=getattr(self.text_encoder, "resizer", None),
            )
            print(f"[INFO] Using AveragedPromptLearner: n_ctx={n_ctx}")
        
        # ===== 方案B: FiLo到Decoder的适配器 =====
        self.filo_decoder_adapter = None
        if self.filo_to_decoder:
            try:
                from model_components import FiLoDecoderAdapter
                self.filo_decoder_adapter = FiLoDecoderAdapter(
                    filo_dim=self.filo_dim_out,      # FiLo输出768
                    decoder_dim=self.hidden_dim,     # Decoder 256
                    num_filo_tokens=filo_decoder_tokens,
                    mode=filo_decoder_mode,
                    dropout=0.1,
                )
                print(f"[INFO] FiLoDecoderAdapter enabled: mode={filo_decoder_mode}, tokens={filo_decoder_tokens}")
            except Exception as e:
                print(f"[WARN] FiLoDecoderAdapter init failed: {e}")
                self.filo_to_decoder = False
        
        # ===== 方案C: 置信度融合头 =====
        self.conf_fusion_head = None
        if self.enable_conf_fusion_head:
            try:
                from model_components import ConfidenceFusionHead
                self.conf_fusion_head = ConfidenceFusionHead(
                    hidden_dim=conf_fusion_hidden_dim,
                    dropout=0.1,
                    use_layer_norm=True,
                )
                print(f"[INFO] ConfidenceFusionHead enabled: hidden_dim={conf_fusion_hidden_dim}")
            except Exception as e:
                print(f"[WARN] ConfidenceFusionHead init failed: {e}")
                self.enable_conf_fusion_head = False
        
        # 打印最终状态
        print(f"[INFO] Final module status: VVAttention={self.enable_vv_attention}, FiLo={self.enable_filo}, "
              f"FiLoToDecoder={self.filo_to_decoder}, ConfFusionHead={self.enable_conf_fusion_head}")

    def _load_ckpt_state(self, path):
        state = torch.load(path, map_location="cpu")
        for k in ("model", "state_dict"):
            if k in state:
                return state[k]
        return state

    def forward(
        self,
        images: torch.Tensor,
        prompt_lists: List[List[str]],
        class_names: Optional[List[str]] = None,
    ) -> dict:
        images = images.to(self.device)
        
        backbone_out = self.backbone.forward_image(images)
        
        all_fpn_feats = backbone_out["backbone_fpn"]
        all_pos_enc = backbone_out["vision_pos_enc"]
        
        # ===== 特征层选择策略 =====
        # 1. FiLo可以使用多层（selected_levels指定）
        # 2. SAM3 encoder/decoder只用最后num_feature_levels层（显存友好）
        
        # 为FiLo准备多尺度特征（如果有selected_levels）
        if self.selected_levels is not None:
            filo_selected_idx = [i for i in self.selected_levels if i < len(all_fpn_feats)]
            filo_feats = [all_fpn_feats[i] for i in filo_selected_idx]
        else:
            # 默认使用所有层给FiLo
            filo_feats = all_fpn_feats
        
        # 为SAM3 encoder准备特征（只用最后num_feature_levels层）
        vis_feats = all_fpn_feats[-self.num_feature_levels:]
        vis_pos = all_pos_enc[-self.num_feature_levels:]
        
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos]
        B = images.shape[0]
        
        # ===== VVAttention =====
        vv_attn_weights = None
        if self.enable_vv_attention and self.vv_attention is not None:
            enhanced_feats = []
            vv_attn_weights = []
            for i, (feat, pos) in enumerate(zip(vis_feats, vis_pos)):
                enhanced, attn = self.vv_attention(feat, pos, return_attention=True)
                enhanced_feats.append(enhanced)
                if attn is not None:
                    vv_attn_weights.append(attn)
            vis_feats = enhanced_feats
        
        image_features_for_cocoop = None
        if self.prompt_learner_type == "cocoop":
            image_features_for_cocoop = vis_feats[0].mean(dim=[2, 3])
        
        if hasattr(self.prompt_learner, "class_to_idx") and class_names is not None:
            cls_ids = [
                self.prompt_learner.class_to_idx.get(c.lower(), 0) if c is not None else 0 
                for c in class_names
            ]
            prompt_seq, prompt_mask = self.prompt_learner(
                prompt_lists, class_ids=cls_ids, 
                image_features=image_features_for_cocoop, device=self.device
            )
        else:
            prompt_seq, prompt_mask = self.prompt_learner(
                prompt_lists, image_features=image_features_for_cocoop, device=self.device
            )
        
        # ===== FiLo模块 (6路卷积MMCI) =====
        # FiLo使用filo_feats（可以是多层），encoder使用vis_feats（只有最后一层）
        filo_outputs = None
        if self.enable_filo and self.filo_module is not None:
            # FiLo需要text_features: (B, 2, C) 格式 [normal, abnormal]
            # 从prompt_seq中提取（假设prompt_seq包含normal和abnormal的表示）
            # 或者使用外部提供的text_features
            
            # 方式1: 如果prompt_seq是 (L, B, D) 格式，提取前两个作为text_features
            if prompt_seq.dim() == 3:
                if prompt_seq.shape[0] >= 2 and prompt_seq.shape[1] == B:
                    # (L, B, D) -> 取前2个作为normal/abnormal
                    text_features_for_filo = prompt_seq[:2].permute(1, 0, 2)  # (B, 2, D)
                elif prompt_seq.shape[0] == B:
                    # (B, L, D) -> 取前2个token
                    text_features_for_filo = prompt_seq[:, :2, :]  # (B, 2, D)
                else:
                    # 使用全局平均作为fallback
                    text_features_for_filo = prompt_seq.mean(dim=0, keepdim=True)  # (1, L, D)
                    text_features_for_filo = text_features_for_filo[:, :2, :]
                    text_features_for_filo = text_features_for_filo.expand(B, -1, -1)
            else:
                # 2D情况: (L, D) -> 扩展
                text_features_for_filo = prompt_seq[:2].unsqueeze(0).expand(B, -1, -1)
            
            # 调用FiLo模块 - 使用filo_feats（多尺度特征）
            filo_outputs = self.filo_module(
                fpn_features=filo_feats,  # 使用filo_feats而不是vis_feats
                text_features=text_features_for_filo,
                return_intermediate=True,
            )
        
        prompt_pos = torch.zeros_like(prompt_seq)
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos = [x.flatten(2).permute(2, 0, 1) for x in vis_pos]

        memory = self.transformer.encoder(
            src=img_feats.copy(), src_key_padding_mask=None, src_pos=img_pos.copy(),
            prompt=prompt_seq, prompt_pos=prompt_pos, prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes, encoder_extra_kwargs=None,
        )

        bs = images.shape[0]
        tgt = self.transformer.decoder.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        
        # ===== 方案B: FiLo特征喂给Decoder =====
        decoder_memory = memory["memory"]
        filo_adapter_out = None
        
        if self.filo_to_decoder and self.filo_decoder_adapter is not None and filo_outputs is not None:
            # 获取FiLo的patch_tokens
            filo_qkv = filo_outputs.get('patch_tokens_qkv', [])
            filo_vv = filo_outputs.get('patch_tokens_vv', [])
            
            if len(filo_qkv) > 0 or len(filo_vv) > 0:
                # 合并所有FiLo tokens (取第一层或平均)
                all_tokens = filo_qkv + filo_vv
                filo_tokens = all_tokens[0]  # (B, N, 768)
                
                # 通过适配器处理
                filo_adapter_out = self.filo_decoder_adapter(
                    filo_tokens=filo_tokens,
                    memory=decoder_memory,
                    query=tgt,
                )
                
                if self.filo_decoder_mode == "memory" and "extra_memory" in filo_adapter_out:
                    # memory模式：使用cross-attention增强原始memory（不改变长度）
                    # 这避免了SAM3 decoder内部mask维度不匹配的问题
                    extra_mem = filo_adapter_out["extra_memory"]  # (num_tokens, B, D)
                    
                    # 用原始memory attend to FiLo tokens
                    # decoder_memory: (N, B, D), extra_mem: (num_tokens, B, D)
                    # 简单残差融合：让memory增强
                    filo_mean = extra_mem.mean(dim=0, keepdim=True)  # (1, B, D)
                    # 广播加到所有memory位置（轻量级融合）
                    decoder_memory = decoder_memory + 0.1 * filo_mean.expand_as(decoder_memory)
                    
                elif self.filo_decoder_mode == "query_bias" and "query_bias" in filo_adapter_out:
                    # 添加query bias
                    query_bias = filo_adapter_out["query_bias"]  # (Q, B, D)
                    tgt = tgt + query_bias
                    
                elif self.filo_decoder_mode == "cross_attn" and "enhanced_memory" in filo_adapter_out:
                    # 使用增强的memory（长度不变）
                    if filo_adapter_out["enhanced_memory"] is not None:
                        decoder_memory = filo_adapter_out["enhanced_memory"]
        
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(
            tgt=tgt, memory=decoder_memory, memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"], reference_boxes=None,
            level_start_index=memory["level_start_index"], spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"], tgt_mask=None,
            memory_text=prompt_seq, text_attention_mask=prompt_mask,
        )
        hs = hs.permute(0, 2, 1, 3).contiguous()

        seg_out = self.segmentation_head(
            backbone_feats=vis_feats, obj_queries=hs,
            image_ids=torch.arange(bs, device=self.device),
            encoder_hidden_states=memory["memory"],
            prompt=prompt_seq, prompt_mask=prompt_mask,
        )

        # ===== 方案C: 置信度融合头 =====
        presence_logit = seg_out.get("presence_logit")
        iou_predictions = seg_out.get("iou_predictions")
        fused_conf = None
        
        if self.enable_conf_fusion_head and self.conf_fusion_head is not None:
            # 计算FiLo置信度
            filo_conf = None
            if filo_outputs is not None:
                filo_agg = filo_outputs.get('aggregated_map', None)
                filo_maps = filo_outputs.get('anomaly_maps', [])
                
                if filo_agg is not None:
                    # (B, 2, H, W) -> 取abnormal通道的最大值
                    filo_conf = filo_agg[:, 1].view(bs, -1).max(dim=-1)[0]  # (B,)
                elif len(filo_maps) > 0:
                    filo_conf = filo_maps[-1][:, 1].view(bs, -1).max(dim=-1)[0]  # (B,)
            
            if filo_conf is not None and presence_logit is not None and iou_predictions is not None:
                # 确保filo_conf在0-1范围
                if filo_conf.min() < 0 or filo_conf.max() > 1:
                    filo_conf = torch.sigmoid(filo_conf)
                
                fused_conf = self.conf_fusion_head(
                    presence_logit=presence_logit,
                    iou_pred=iou_predictions,
                    filo_conf=filo_conf,
                )

        out = {
            "pred_masks": seg_out.get("pred_masks"),
            "semantic_seg": seg_out.get("semantic_seg"),
            "presence_logit": presence_logit,
            "iou_predictions": iou_predictions,
            "fused_conf": fused_conf,  # 方案C: 融合后的置信度
            "decoder_hs": hs,
            "reference_boxes": reference_boxes,
            "prompt_seq": prompt_seq,
        }

        decoder_feat = None
        for key in ("mask_features", "mask_feat", "mask_pred_feat", "decoder_features"):
            v = seg_out.get(key, None)
            if isinstance(v, torch.Tensor) and v is not None and v.dim() == 4:
                decoder_feat = v
                break
        if decoder_feat is None:
            decoder_feat = vis_feats[0] if vis_feats else None
        out["decoder_features"] = decoder_feat
        
        if self.enable_multiscale_output:
            out["multiscale_features"] = {
                "used_features": vis_feats, "all_fpn_features": all_fpn_feats,
                "feature_sizes": vis_feat_sizes, "num_levels": len(vis_feats),
                "selected_levels": self.selected_levels, "pos_encodings": vis_pos,
            }
        
        
        if vv_attn_weights is not None:
            out["vv_attention_weights"] = vv_attn_weights
        
        # FiLo输出
        if filo_outputs is not None:
            out["filo_anomaly_maps"] = filo_outputs.get('anomaly_maps', [])
            out["filo_aggregated_map"] = filo_outputs.get('aggregated_map', None)
            out["filo_patch_tokens_qkv"] = filo_outputs.get('patch_tokens_qkv', [])
            out["filo_patch_tokens_vv"] = filo_outputs.get('patch_tokens_vv', [])

        return out

    def forward_anomaly(self, images, text_features):
        """
        使用FiLo模块生成异常图
        
        Args:
            images: (B, C, H, W) 输入图像
            text_features: (B, 2, D) 或 (2, D) 文本特征 [normal, abnormal]
            
        Returns:
            dict with 'anomaly_maps', 'aggregated_map' etc.
        """
        if not self.enable_filo or self.filo_module is None:
            raise RuntimeError("FiLo not enabled")
        
        images = images.to(self.device)
        
        # 提取FPN特征
        backbone_out = self.backbone.forward_image(images)
        fpn_feats = backbone_out["backbone_fpn"]
        
        # 选择使用的层
        if self.selected_levels is not None:
            vis_feats = [fpn_feats[i] for i in self.selected_levels if i < len(fpn_feats)]
        else:
            vis_feats = fpn_feats[-self.num_feature_levels:]
        
        # VVAttention增强（如果启用）
        if self.enable_vv_attention and self.vv_attention is not None:
            enhanced_feats = []
            for i, feat in enumerate(vis_feats):
                enhanced, _ = self.vv_attention(feat, None, return_attention=False)
                enhanced_feats.append(enhanced)
            vis_feats = enhanced_feats
        
        # 调用FiLo模块
        return self.filo_module(
            fpn_features=vis_feats,
            text_features=text_features,
            return_intermediate=True,
        )

    def get_text_embeddings(self, prompts, device=None):
        device = device or self.device
        text_out = self.backbone.forward_text(prompts, device=device)
        text_features = text_out.get('language_features', text_out.get('text_features'))
        if text_features.dim() == 3:
            text_features = text_features[:, 0]
        return text_features
    
    def debug_print_status(self):
        print(f"[DEBUG] VVAttention={self.enable_vv_attention}(module={self.vv_attention is not None})")
        print(f"[DEBUG] FiLo={self.enable_filo}(module={self.filo_module is not None})")


MultiscaleSAM3 = FineTuneSAM3Official