"""
TIE Module Integration Guide for SAM3 Anomaly Detection
=======================================================

本文档详细说明如何将 TIE (Text-guided Image Embedding Translation) 模块
集成到你的 SAM3 异常检测训练流程中，以减少假阳性问题。

数学原理总结:
=============

1. 问题: 假阳性的数学本质
   -------------------------
   假阳性发生时: ⟨h_normal, w_anomaly⟩ > threshold
   
   原因: 正常样本的视觉嵌入 h_normal 包含与异常描述相似的伪相关成分
   例如: 正常纹理、光照变化、边缘模式等

2. TIE 解决方案
   -------------------------
   操作: h' = h - λ · v_spurious
   
   其中:
   - v_spurious: 伪相关方向（正常纹理的文本嵌入）
   - λ = E[h^T · v_spurious]: 平均投影长度
   
   效果:
   - 正常样本: 减少与异常描述的相似度 → 降低假阳性
   - 异常样本: 核心缺陷特征与 v_spurious 正交 → 影响较小

3. 最优性证明 (Theorem 1)
   -------------------------
   最优平移向量: v* = E[-P·h_a]
   保证: 平移后的嵌入分布与原分布一致
   
   Lemma 1 推导: 
   group-wise accuracy A(h, w; y) = (1/2)erfc(-w^T μ / √(2w^T Σ w))
   
   通过调整 μ（平移）而非 Σ（投影），可以更稳定地提升 worst-group accuracy


集成步骤:
=========
"""

# ============================================================================
# Step 1: 修改 model_wrapper.py - 添加 TIE 模块
# ============================================================================

STEP1_CODE = '''
# 在 model_wrapper.py 中添加以下导入和修改

# --- 在文件顶部添加 ---
from spurious_mitigation import TIEAnomalyHead, TIELoss

# --- 在 FineTuneSAM3Official.__init__ 中添加（在 prompt_learner 之后）---

class FineTuneSAM3Official(nn.Module):
    def __init__(self, ...):
        # ... 原有代码 ...
        
        # === 新增: TIE 模块 ===
        # 获取嵌入维度（从 transformer 或 decoder 获取）
        self.tie_embed_dim = getattr(self.transformer, 'd_model', 256)
        
        # 创建 TIE 模块
        self.tie_module = TIEAnomalyHead(
            embed_dim=self.tie_embed_dim,
            num_spurious_vectors=4,  # 可通过参数调整
            apply_to_queries=True,   # 应用于 decoder queries
            apply_to_features=False, # 可选：也应用于视觉特征
            text_encoder=self.text_encoder,
            spurious_prompts=[
                "a photo with normal texture",
                "a photo of regular surface",
                "a photo without defects",
                "an image of undamaged material",
            ],
        )
        
        # TIE 损失
        self.tie_loss = TIELoss(
            orthogonality_weight=0.1,
            discrimination_weight=1.0,
            margin=0.3,
        )
        
        self.to(self.device)
'''

# ============================================================================
# Step 2: 修改 forward 方法 - 应用 TIE
# ============================================================================

STEP2_CODE = '''
# 在 FineTuneSAM3Official.forward 中修改

def forward(self, images, prompt_lists, class_names=None, is_anomaly=None):
    """
    Modified forward with TIE integration.
    
    新增参数:
        is_anomaly: (B,) 可选的异常标签，用于条件 TIE
    """
    # ... 原有代码直到 decoder 输出 ...
    
    hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(...)
    hs = hs.permute(0, 2, 1, 3).contiguous()  # 原有代码
    
    # === 新增: 应用 TIE ===
    if hasattr(self, 'tie_module') and self.tie_module is not None:
        # 获取最后一层 decoder hidden states
        if hs.dim() == 4:
            hs_last = hs[-1]  # (B, Q, D)
        else:
            hs_last = hs
        
        # 应用 TIE 平移
        tie_output = self.tie_module(
            decoder_hs=hs_last,
            visual_features=None,  # 可选：vis_feats[0] 如果要处理特征
            is_anomaly=is_anomaly,
            return_diagnostics=self.training,  # 训练时返回诊断信息
        )
        
        # 替换 hidden states
        if "translated_queries" in tie_output:
            hs_last = tie_output["translated_queries"]
            if hs.dim() == 4:
                hs = torch.cat([hs[:-1], hs_last.unsqueeze(0)], dim=0)
            else:
                hs = hs_last
    
    # ... 继续原有的 segmentation_head 调用 ...
    
    # === 新增: 返回 TIE 诊断信息 ===
    out = {...}  # 原有输出
    
    if hasattr(self, 'tie_module') and self.training:
        out["tie_diagnostics"] = tie_output.get("diagnostics", {})
        out["tie_spurious_vectors"] = self.tie_module.query_tie.get_spurious_vectors(self.device)
    
    return out
'''

# ============================================================================
# Step 3: 修改 train.py - 添加 TIE 损失
# ============================================================================

STEP3_CODE = '''
# 在 train.py 中添加以下修改

# --- 在 main() 中的模型创建后添加 ---

# 添加 TIE 相关参数
parser.add_argument("--enable_tie", action="store_true", 
                    help="Enable TIE module for spurious correlation mitigation")
parser.add_argument("--tie_num_vectors", type=int, default=4,
                    help="Number of spurious vectors in TIE")
parser.add_argument("--lambda_tie", type=float, default=0.05,
                    help="Weight for TIE loss")
parser.add_argument("--tie_orth_weight", type=float, default=0.1,
                    help="Orthogonality loss weight in TIE")
parser.add_argument("--tie_disc_weight", type=float, default=1.0,
                    help="Discrimination loss weight in TIE")

# --- 在训练循环中添加 TIE 损失计算 ---

def train_one_epoch(...):
    # ... 原有代码 ...
    
    for step, (images, masks, prompts, anomalies, classes) in enumerate(pbar):
        # ... 原有前向传播 ...
        
        # 将 anomalies 转换为 tensor
        is_anomaly = torch.tensor(anomalies, dtype=torch.bool, device=device)
        
        # 修改 forward 调用，传入 is_anomaly
        out = model(images, prompts, class_names=classes, is_anomaly=is_anomaly)
        
        # ... 原有损失计算 ...
        
        # === 新增: TIE 损失 ===
        loss_tie = torch.tensor(0.0, device=device)
        if args.enable_tie and hasattr(model, 'tie_module'):
            # 获取必要的数据
            decoder_hs = out.get("decoder_hs")
            tie_spurious_vectors = out.get("tie_spurious_vectors")
            
            if decoder_hs is not None and tie_spurious_vectors is not None:
                # 获取最后一层
                if decoder_hs.dim() == 4:
                    hs_last = decoder_hs[-1]
                else:
                    hs_last = decoder_hs
                
                # 计算 TIE 损失
                tie_loss_dict = model.tie_loss(
                    spurious_vectors=tie_spurious_vectors,
                    embeddings=hs_last,
                    is_anomaly=is_anomaly,
                )
                loss_tie = tie_loss_dict["total"]
        
        # 将 TIE 损失加入总损失
        loss = (
            args.loss_alpha * loss_focal +
            args.loss_beta * loss_dice +
            args.loss_gamma * loss_iou +
            args.presence_weight * loss_presence +
            args.lambda_align * align_loss +
            args.lambda_query_align * query_align_loss +
            args.lambda_tie * loss_tie  # 新增
        )
        
        # ... 后续代码 ...
        
        # === 新增: 记录 TIE 损失 ===
        if is_main_process and writer is not None:
            writer.add_scalar("loss/tie", loss_tie.item(), global_step)
            
            # 可选：记录 TIE 诊断信息
            if "tie_diagnostics" in out:
                for key, val in out["tie_diagnostics"].items():
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if isinstance(v2, (int, float)):
                                writer.add_scalar(f"tie/{key}_{k2}", v2, global_step)
'''

# ============================================================================
# Step 4: 推荐的训练参数配置
# ============================================================================

TRAINING_CONFIG = '''
# 推荐的训练命令（基于你之前的配置）

python train.py \\
    --use_official \\
    --sam3_ckpt /path/to/sam3.pt \\
    --data_root /path/to/mvtec \\
    --meta_path /path/to/meta.json \\
    --mode train --train_from_test \\
    --batch_size 6 --epochs 20 \\
    --mask_downsample 256 \\
    --unfreeze_decoder last_layer --lr_decoder 5e-6 \\
    \\
    # Align loss (基于之前的调整)
    --lambda_align 0.05 \\
    --align_temp 0.15 \\
    --align_margin 0.2 \\
    --use_anomaly_grouping \\
    \\
    # Query align (基于之前的调整)
    --lambda_query_align 0.02 \\
    --query_align_top_k 16 \\
    --query_align_temp 0.5 \\
    \\
    # === 新增: TIE 参数 ===
    --enable_tie \\
    --tie_num_vectors 4 \\
    --lambda_tie 0.05 \\
    --tie_orth_weight 0.1 \\
    --tie_disc_weight 1.0 \\
    \\
    # 其他参数
    --loss_alpha 0.5 --loss_beta 1.0 --loss_gamma 0.5 \\
    --lr_prompt 3e-5 --lr_main 1e-5 \\
    --presence_weight 0.3 \\
    --grad_clip_norm 1.0
'''

# ============================================================================
# Step 5: 验证 TIE 效果的诊断方法
# ============================================================================

DIAGNOSTIC_CODE = '''
# 在训练过程中添加诊断代码

def diagnose_tie_effect(model, dataloader, device, num_samples=100):
    """
    诊断 TIE 模块的效果
    
    这个函数计算以下指标:
    1. 投影长度分布: 正常 vs 异常样本
    2. 假阳性降低幅度估计
    3. 伪相关向量的正交性
    """
    model.eval()
    
    normal_proj_lengths = []
    anomaly_proj_lengths = []
    
    with torch.no_grad():
        for i, (images, masks, prompts, anomalies, classes) in enumerate(dataloader):
            if i * len(images) >= num_samples:
                break
            
            images = images.to(device)
            is_anomaly = torch.tensor(anomalies, dtype=torch.bool, device=device)
            
            # 前向传播
            out = model(images, prompts, is_anomaly=is_anomaly)
            
            # 获取 TIE 诊断信息
            if "tie_diagnostics" in out:
                diag = out["tie_diagnostics"]
                if "query_tie" in diag:
                    proj_mean = diag["query_tie"]["proj_lengths_mean"]
                    
                    for j, is_anom in enumerate(anomalies):
                        if is_anom:
                            anomaly_proj_lengths.append(proj_mean[j].item())
                        else:
                            normal_proj_lengths.append(proj_mean[j].item())
    
    # 统计分析
    import numpy as np
    normal_proj = np.array(normal_proj_lengths)
    anomaly_proj = np.array(anomaly_proj_lengths)
    
    print("=" * 50)
    print("TIE Effect Diagnostics")
    print("=" * 50)
    print(f"Normal samples projection: mean={normal_proj.mean():.4f}, std={normal_proj.std():.4f}")
    print(f"Anomaly samples projection: mean={anomaly_proj.mean():.4f}, std={anomaly_proj.std():.4f}")
    print(f"Separation: {normal_proj.mean() - anomaly_proj.mean():.4f}")
    
    # 理想情况: 正常样本有更大的投影长度（会被移除更多）
    if normal_proj.mean() > anomaly_proj.mean():
        print("✓ Good: Normal samples have higher projection (will be translated more)")
    else:
        print("⚠ Warning: Anomaly samples have higher projection (may affect detection)")
    
    return {
        "normal_proj_mean": normal_proj.mean(),
        "normal_proj_std": normal_proj.std(),
        "anomaly_proj_mean": anomaly_proj.mean(),
        "anomaly_proj_std": anomaly_proj.std(),
        "separation": normal_proj.mean() - anomaly_proj.mean(),
    }
'''

# ============================================================================
# 完整的修改后的 model_wrapper.py 片段
# ============================================================================

COMPLETE_MODEL_WRAPPER_SNIPPET = '''
# === 完整的 FineTuneSAM3Official 类修改 ===

class FineTuneSAM3Official(nn.Module):
    """Use official build_sam3_image_model then add LoRA + prompt learner + TIE."""

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
        class_list: Optional[Sequence[str]] = None,
        prompt_learner_type: str = "averaged",
        num_templates: int = 4,
        n_ctx: int = 4,
        # === 新增: TIE 参数 ===
        enable_tie: bool = False,
        tie_num_vectors: int = 4,
        tie_spurious_prompts: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        
        # ... 原有初始化代码 ...
        
        # === 新增: TIE 模块 ===
        self.enable_tie = enable_tie
        if enable_tie:
            from spurious_mitigation import TIEAnomalyHead, TIELoss
            
            # 获取嵌入维度
            tie_embed_dim = self.hidden_dim
            
            # 默认的伪相关提示
            default_spurious_prompts = [
                "a photo with normal texture",
                "a photo of regular surface", 
                "a photo without defects",
                "an image of undamaged material",
            ]
            spurious_prompts = tie_spurious_prompts or default_spurious_prompts
            
            self.tie_module = TIEAnomalyHead(
                embed_dim=tie_embed_dim,
                num_spurious_vectors=tie_num_vectors,
                apply_to_queries=True,
                apply_to_features=False,
                text_encoder=self.text_encoder,
                spurious_prompts=spurious_prompts,
            )
            
            self.tie_loss_fn = TIELoss(
                orthogonality_weight=0.1,
                discrimination_weight=1.0,
                margin=0.3,
            )
        else:
            self.tie_module = None
            self.tie_loss_fn = None
        
        self.to(self.device)

    def forward(
        self, 
        images: torch.Tensor, 
        prompt_lists: Sequence[List[str]], 
        class_names: Optional[Sequence[str]] = None,
        is_anomaly: Optional[torch.Tensor] = None,  # 新增
    ) -> dict:
        """Modified forward with TIE integration."""
        
        # ... 原有代码直到 decoder 输出 ...
        
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(...)
        hs = hs.permute(0, 2, 1, 3).contiguous()
        
        # === 新增: 应用 TIE ===
        tie_output = {}
        if self.enable_tie and self.tie_module is not None:
            if hs.dim() == 4:
                hs_for_tie = hs[-1]
            else:
                hs_for_tie = hs
            
            tie_output = self.tie_module(
                decoder_hs=hs_for_tie,
                is_anomaly=is_anomaly,
                return_diagnostics=self.training,
            )
            
            if "translated_queries" in tie_output:
                hs_translated = tie_output["translated_queries"]
                if hs.dim() == 4:
                    hs = torch.cat([hs[:-1], hs_translated.unsqueeze(0)], dim=0)
                else:
                    hs = hs_translated
        
        # ... segmentation_head 调用 ...
        
        out = {
            # ... 原有输出 ...
        }
        
        # === 新增: TIE 相关输出 ===
        if self.enable_tie and self.tie_module is not None:
            out["tie_diagnostics"] = tie_output.get("diagnostics", {})
            if hasattr(self.tie_module, 'query_tie') and self.tie_module.query_tie is not None:
                out["tie_spurious_vectors"] = self.tie_module.query_tie.get_spurious_vectors(self.device)
        
        return out
    
    def compute_tie_loss(
        self, 
        decoder_hs: torch.Tensor, 
        is_anomaly: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算 TIE 损失"""
        if not self.enable_tie or self.tie_loss_fn is None:
            return {"total": torch.tensor(0.0, device=decoder_hs.device)}
        
        # 获取伪相关向量
        spurious_vectors = self.tie_module.query_tie.get_spurious_vectors(decoder_hs.device)
        
        # 获取最后一层
        if decoder_hs.dim() == 4:
            hs = decoder_hs[-1]
        else:
            hs = decoder_hs
        
        return self.tie_loss_fn(spurious_vectors, hs, is_anomaly)
'''

# ============================================================================
# 打印指南
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TIE Module Integration Guide for SAM3 Anomaly Detection")
    print("=" * 80)
    
    print("\n" + "=" * 40)
    print("Step 1: Modify model_wrapper.py")
    print("=" * 40)
    print(STEP1_CODE)
    
    print("\n" + "=" * 40)
    print("Step 2: Modify forward method")
    print("=" * 40)
    print(STEP2_CODE)
    
    print("\n" + "=" * 40)
    print("Step 3: Modify train.py")
    print("=" * 40)
    print(STEP3_CODE)
    
    print("\n" + "=" * 40)
    print("Step 4: Recommended Training Config")
    print("=" * 40)
    print(TRAINING_CONFIG)
    
    print("\n" + "=" * 40)
    print("Step 5: Diagnostic Code")
    print("=" * 40)
    print(DIAGNOSTIC_CODE)
