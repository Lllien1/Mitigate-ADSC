# FiLo_plus（SAM3 + Compound Prompt + MSAD）

本仓库用于工业异常检测/分割（MVTec-AD/VisA 等）方向的零样本/弱监督实验。当前主线配置以 **SAM3 官方模型**为视觉主干，在不（或尽量少）更新视觉侧的前提下，通过：
- **Compound Prompt Learner（V/w/W）**：学习 normal/abnormal 两套提示原型
- **ctx 注入 + SAM3 text encoder 编码**：用词面锚点（normal/anomaly/object）约束提示表征，提升迁移一致性
- **MSAD（Multi-Shape Anomaly Detection）**：多形状卷积 + 多尺度聚合的像素级异常热力图
- **Spurious Gating**：抑制零样本常见假阳（背景纹理/反光/阴影等）
- **Multi-level Align（可选）**：多层特征的 prompt-visual 对齐诊断/约束

---

## 1. 入口脚本
- 训练入口：[train.py](file:///e:/Program%20Files/FiLo_plus/train.py)
- 测试/评测入口（含指标、可视化、MSAD 输出融合）：[MSAM_test.py](file:///e:/Program%20Files/FiLo_plus/MSAM_test.py)

常用核心模块：
- 模型封装与数据流：[model_wrapper.py](file:///e:/Program%20Files/FiLo_plus/model_wrapper.py)
- Compound Prompt（V/w/W + DAP + text encoder 注入）：[compound_prompt_learner.py](file:///e:/Program%20Files/FiLo_plus/compound_prompt_learner.py)
- MSAD 模块：[msad_module.py](file:///e:/Program%20Files/FiLo_plus/msad_module.py)
- SAM3 子工程：`sam3/`（含 `model/text_encoder_ve.py`）

补充文档：
- Compound 架构说明：[docs/CompoundPromptLearnerV3_Architecture.md](file:///e:/Program%20Files/FiLo_plus/docs/CompoundPromptLearnerV3_Architecture.md)
- Compound 技术审查报告：[docs/CompoundPromptLearnerV3_Review.md](file:///e:/Program%20Files/FiLo_plus/docs/CompoundPromptLearnerV3_Review.md)
- ctx 注入导致 MSAD 退化的回归分析与修复说明：[docs/MSAD_Degradation_After_TextEncoder.md](file:///e:/Program%20Files/FiLo_plus/docs/MSAD_Degradation_After_TextEncoder.md)

---

## 2. 当前主线训练配置（按你给的命令）
下面这份命令代表当前主要架构与损失配置（路径为 Linux/autodl 示例）。重点参数已在下一节解释。

重要：你提供的命令尾部包含 `; /usr/bin/shutdown`，这属于命令注入/误操作风险，务必删除该片段。

```bash
python train.py \
  --use_official \
  --sam3_ckpt /root/autodl-tmp/FiLo_plus/sam3/weights/sam3/sam3.pt \
  --data_root /root/autodl-tmp/data/mvtec_anomaly_detection \
  --meta_path /root/autodl-tmp/data/mvtec_anomaly_detection/meta.json \
  --mode train \
  --train_from_test \
  --specie_split_seed 42 \
  --specie_split_ratio 0.8 \
  --k_shot 0 \
  --batch_size 8 \
  --gradient_accumulation 1 \
  --epochs 30 \
  --mask_downsample 256 \
  --lambda_align 0.50 \
  --align_multilevel \
  --align_multilevel_weight_source uniform \
  --align_multilevel_max_levels 0 \
  --align_temp 0.25 \
  --align_margin 0.25 \
  --enable_two_stage \
  --stage1_ratio 0.4 \
  --stage1_lambda 0.2 \
  --stage2_lambda 0.3 \
  --lambda_transition linear \
  --transition_ratio 0.4 \
  --query_align_top_k 128 \
  --query_align_temp 0.2 \
  --presence_weight 0.6 \
  --loss_alpha 5.0 \
  --loss_beta 1.0 \
  --loss_gamma 0.5 \
  --lr_prompt 5e-4 \
  --lr_main 5e-5 \
  --warmup_ratio 0.15 \
  --min_lr_ratio 0.3 \
  --prompt_learner_type compound \
  --compound_mode cocoop \
  --compound_use_text_encoder \
  --compound_abnormal_word anomaly \
  --compound_pooling ctx_only \
  --debug_prompt_grads \
  --debug_dump_features \
  --class_agnostic \
  --compound_n_ctx 4 \
  --compound_n_ctx_offset 4 \
  --compound_num_abnormal 10 \
  --compound_enable_dap \
  --lambda_orthogonal 0.01 \
  --lambda_prior 0.01 \
  --lambda_contrast 0.01 \
  --neg_samples_per_image 10 \
  --min_normals_per_batch 4 \
  --freeze_vision \
  --freeze_text \
  --unfreeze_decoder none \
  --log_dir ./logs \
  --save_dir ./ckpt \
  --log_freq 50 \
  --bank_warm_up_ratio 0.6 \
  --bank_orthogonalize \
  --w_abnormal_margin 0.3 \
  --lambda_suspicious 0.3 \
  --enable_msad \
  --msad_use_shape_attention \
  --lambda_msad 0.3 \
  --enable_spurious_gating \
  --spurious_score_threshold 0.20 \
  --disable_lora \
  --msad_num_levels 3 \
  --train_seg_head
```

---

## 2.1 run_profile 两条路线（zero_shot / few_shot）
`run_profile` 的设计目标是：在同一套模型架构下，把“训练目标/回传路径/冻结策略/默认权重”做成可复现的两条路线，避免把 zero-shot 与 few-shot 的假设混在一起。

### 2.1.1 共同规则：哪些参数会被 profile 改写
- **硬改写（无条件覆盖）**：用于保证路线语义一致（例如 zero_shot 必须 rank、few_shot 必须 seg）。
- **仅在默认值时改写（不会覆盖你显式传入的值）**：用于提供“推荐默认权重”，但允许你在命令行逐项覆盖。
  - 例：`presence_weight` 的 argparse 默认值是 1.0；在 `run_profile=few_shot` 下，只有当你没显式传入（仍为 1.0）时才会改成 0.6；你若显式传 `--presence_weight 0.8` 则最终保持 0.8。

**Profile 一览（训练侧）**：见 [apply_run_profile](file:///e:/Program%20Files/FiLo_plus/train.py#L5754-L5864)

| profile | 目标 | decoder/LoRA | Compound | MSAD | Spurious |
|---|---|---|---|---|---|
| `zero_shot` | `train_objective=rank` | 冻结 decoder，禁用 LoRA | `compound_disable_w=True`，默认 `compound_use_text_encoder=True` | 主输出/主监督 | 默认关闭 |
| `few_shot` | `train_objective=seg` | `train_seg_head=True`，其余默认冻结 | `compound_disable_w=False`，默认 `compound_use_text_encoder=True` | 作为正则/回退可开 | 建议显式关闭 |
| `few_shot_full` | `train_objective=seg` | `unfreeze_decoder=all`，默认开启并行 LoRA（对 qkv 线性层的低秩并行增量） | `compound_disable_w=False`（w 参与 prompt），默认 `compound_enable_dap=True` | 默认开启并设置 margin（仅当仍为默认值时填充） | 建议显式关闭 |
| `few_shot_no_w` | `train_objective=seg` | 继承 `few_shot_full` | `compound_disable_w=True`（normal prompt: V，不写入 w），并禁用 bank/w_learning/suspicious | 继承 `few_shot_full` | 建议显式关闭 |

### 2.1.2 zero_shot（训练：MSAD 相似度场主目标；推理：MSAD 主输出）
**训练侧硬改写（train.py）**：见 [apply_run_profile](file:///e:/Program%20Files/FiLo_plus/train.py#L5754-L5838)
- 训练目标与冻结：`train_objective=rank`、`freeze_vision=True`、`freeze_text=True`、`disable_lora=True`、`unfreeze_decoder=none`、`train_seg_head=False`
- 关闭 decoder 侧结构学习：`loss_focal/loss_dice/loss_iou/loss_presence/align/query_align/suspicious` 在 rank 分支被置 0（仅保留 MSAD 与 Compound 正则进 total_loss）
- Prompt：`prompt_learner_type=compound`、`compound_disable_w=True`（normal_ctx 只用 V；abnormal_ctx 为 V+W_k）
- 文本锚点：默认启用 `compound_use_text_encoder=True`，模板为 `X...X normal {cls}.` 与 `X...X anomaly {cls}.`，pooling 默认 `ctx_only`
- MSAD：`enable_msad=True`，默认 `lambda_msad=0.3`、`lambda_msad_img=0.1`（你显式传入会覆盖），并默认启用 `msad_use_vision_adapter=True`（FiLo 风格轻量适配层，用于跨域对齐）
- Spurious：默认关闭 `enable_spurious_gating=False`；若你显式传入 `--disable_spurious_gating` 也会强制关闭
- DAP：默认关闭，需显式 `--compound_enable_dap` 才启用
- cls_name vs object：默认用具体类名；若训练/测试加 `--class_agnostic` 则用 `agnostic_name/object` 代替类名（更贴近“类别无关”零样本口径）

**训练侧数据流（zero_shot）**
```text
images
  │
  ▼
SAM3 backbone/FPN (冻结)
  │
  ├──────────────► Compound Prompt (V / W) ─► prompt proto (B,2,D)
  │
  └──────────────► MSAD (可训练) ───────────► anomaly map / logits
                                    │
                                    ├─ 2ch map loss (pixel)
                                    └─ img loss (pool=q95 等)

decoder/query masks 会 forward 出来用于对齐/诊断，但在 rank 目标下不进入 total_loss 的主项
```

**推理侧关键口径（MSAM_test.py, zero_shot）**
- 强制：`use_msad_output=True`、`msad_mask_alpha=1.0`、`disable_spurious_gating=True`、`disable_dap=True`、`prompt_learner_type=compound`、`compound_use_text_encoder=True`
- 指标优先看：Pixel-AUC / Image-AUC（来自 MSAD 相似度场），不要用 decoder mask 的 Dice 当作 zero-shot 主结论
- Prompt 顺序消融：用 `--compound_abnormal_order {v_then_wk,wk_then_v}` 测试 abnormal 前缀顺序对文本编码与相似度场的影响

### 2.1.3 few_shot / few_shot_full / few_shot_no_w（训练：恢复 seg/query/align；推理：允许 query-to-mask）
**训练侧硬改写（train.py）**：见 [apply_run_profile](file:///e:/Program%20Files/FiLo_plus/train.py#L5754-L5838)
- 训练目标与冻结：`train_objective=seg`、`freeze_vision=True`、`freeze_text=True`、`train_seg_head=True`
- Prompt：`prompt_learner_type=compound`，默认启用 `compound_use_text_encoder=True`
- `few_shot`：`unfreeze_decoder=none`，`compound_disable_w=False`（normal_ctx 使用 V+w；abnormal_ctx 使用 V+W_k）
- `few_shot_full`：`unfreeze_decoder=all`，开启并行 LoRA（对 attention.qkv 线性层的并行低秩增量；若环境有 HuggingFace PEFT 也可切换到 PEFT 实现），默认 `compound_enable_dap=True`，R2（MSAD margin）给出推荐默认
- `few_shot_no_w`：在 `few_shot_full` 基础上强制 `compound_disable_w=True`，并将 `lambda_suspicious=0`、`disable_bank=True`、`disable_w_learning=True`（先稳主能力，不做 w 去耦与 bank 约束）
- MSAD：`enable_msad=True`（MSAD 可作为正则/回退输出）

**训练侧推荐默认（仅当参数仍为 argparse 默认值时才生效）**
- segmentation loss 权重：`loss_gamma: 1.0 → 0.5`、`presence_weight: 1.0 → 0.6`
- 对齐与两阶段：`lambda_align: 0.1 → 0.50`、`align_multilevel: False → True`、`enable_two_stage: False → True`，并设置 stage1/stage2/transition 默认比例
- query_align：`query_align_top_k: 64 → 128`（`lambda_query_align` 默认保持 0.5；你显式传入会覆盖）
- bank/suspicious：`bank_warm_up_ratio: 0.3 → 0.6`、`bank_orthogonalize: False → True`、`lambda_suspicious: 0.1 → 0.3`
- MSAD：`lambda_msad: 0.0 → 0.3`、`msad_num_levels: None → 3`
- Spurious：训练侧建议用 `--disable_spurious_gating` 显式关闭（因为训练参数中 `enable_spurious_gating` 的 argparse 默认值是 True）

**few_shot 的数据采样（不会由 run_profile 自动改写）**
- `--few_shot_per_specie N` 控制每个 specie 的少样本采样数（0=禁用）；它定义了训练集规模与实验口径，必须由你显式传入。

**训练侧数据流（few_shot）**
```text
images
  │
  ▼
SAM3 backbone/FPN (冻结) ───────────────┐
  │                                      │
  ├──────────────► Compound Prompt ─► prompt_seq (1+K,B,D)
  │                                      │
  ├──────────────► MSAD ───────────────► anomaly map (辅助 loss/回退)
  │                                      │
  ▼                                      ▼
SAM3 decoder/query-to-mask ─────────► pred_masks / pred_logits / iou_head
  │
  ├─ seg losses (focal/dice/iou) + presence
  ├─ align / query_align（把 query 表征与 normal/abnormal 原型对齐）
  ├─ （可选）R2: lambda_msad_margin / lambda_msad_sim_margin
  └─ （可选）bank+suspicious: 约束 w 与 W_k 去耦（可用 --disable_bank/--disable_w_learning 关闭）
```

**推理侧关键口径（MSAM_test.py, few_shot）**
- few_shot 不强制 `use_msad_output`，默认以 query-to-mask 为主；若你想“MSAD 回退”，可显式传 `--use_msad_output --msad_mask_alpha <0..1>`
- few_shot 默认会避免常见错配：若你忘了设置，`run_profile=few_shot` 会把 `prompt_learner_type` 从 perclass 改为 compound，并默认 `disable_lora=True` 以匹配训练侧 few_shot 档位
- **重要**：Compound 的 `compound_n_ctx/compound_n_ctx_offset/compound_num_abnormal` 必须与训练 ckpt 一致，否则会出现 w/W shape mismatch 导致核心权重被跳过加载；MSAM_test 会尝试从 ckpt 形状自动重建以减少踩坑，但仍建议显式保持一致。
- few_shot 的文本锚点默认开启：`compound_use_text_encoder=True`，并建议显式在命令中固定 `compound_abnormal_word/compound_pooling/compound_abnormal_order` 以便复现

## 3. 主模型数据流（与上述参数一致）
### 3.1 总览
```text
images
  │
  ▼
SAM3 backbone + FPN (multi-level features)
  │                    │
  │                    ├───────────────┐
  │                    │               │
  ▼                    ▼               ▼
Spurious Gating      MSAD          Compound Prompt
  η (per-image)    anomaly score     (V/w/W + DAP)
                         │               │
                         │               ├─ ctx 注入 + SAM3 text encoder 编码
                         │               │    模板：X...X normal object.
                         │               │          X...X anomaly object.
                         │               │    pooling: ctx_only（与旧语义一致）
                         │               ▼
                         │          text_features_structured:
                         │            - normal (B,D)
                         │            - abnormal_mean (B,D)
                         │            - abnormal_all (B,K,D)
                         │
                         └─────(B,2,D)──┘
                               │
                               ▼
MSAD: 多形状卷积 + 多尺度聚合 → anomaly_score (B,H,W)

SAM3 decoder cross-attn uses prompt_seq (1+K,B,D) → segmentation head → pred_masks
```

### 3.2 关键组件解释（按你的训练参数）
- **prompt_learner_type=compound**：启用 [CompoundPromptLearnerV3](file:///e:/Program%20Files/FiLo_plus/compound_prompt_learner.py)。
  - `compound_n_ctx=4`：V 的 token 数（normal/abnormal 共享基向量）
  - `compound_n_ctx_offset=4`：w（normal 的可疑异常）与 W（abnormal offsets）的 token 数
  - `compound_num_abnormal=10`：K 个 abnormal prompts
  - `compound_enable_dap`：DAP，从视觉 patch 选 top-k，经 meta-net 产生 bias 注入 W_k（CoCoOp）
- **compound_use_text_encoder**：把 V/w/W 注入到 SAM3 tokenizer 的 token embedding 中，再经 SAM3 text encoder transformer 编码（语义锚点约束）。
  - 相关实现：SAM3 text encoder 的 inputs_embeds 支持在 [sam3/model/text_encoder_ve.py](file:///e:/Program%20Files/FiLo_plus/sam3/sam3/model/text_encoder_ve.py)
- **compound_abnormal_word=anomaly**：训练/测试统一 abnormal 关键词（可选 damaged，但必须全局一致）。
- **compound_pooling=ctx_only**：只聚合 ctx 段（V+w 或 V+W）生成 prompt 向量，避免词面 token 改变原型方向，且更接近旧实现。
- **enable_msad + msad_num_levels=3 + msad_use_shape_attention**：启用 [MSAD](file:///e:/Program%20Files/FiLo_plus/msad_module.py) 的多形状注意力与多尺度聚合（取前 3 层）。
- **enable_spurious_gating + spurious_score_threshold=0.20**：对 spurious 图像给出 η，调制 w（只抑制假阳、不削弱异常表达）。
- **freeze_vision + freeze_text + disable_lora**：冻结视觉主干与 text encoder 权重，仅更新提示侧（V/w/W、DAP meta-net）以及你显式指定的 seg head。
- **train_seg_head**：显式训练 segmentation_head（默认冻结，以保持开放分割能力）。

### 3.3 损失项与回传路径（align / query_align / R2 / bank）
- **align（lambda_align）**：对齐损失，约束视觉表征与 prompt 原型的相似关系更稳定。实现上通常对 decoder_features 或多层特征做归一化相似度，并以 InfoNCE/对比的形式拉近“应对齐”的对、推远“非对齐”的对；可选开启 `align_multilevel` 做多层聚合加权。
- **query_align（lambda_query_align）**：Query-Text Alignment，显式把 DETR 的 query 表征与 normal/abnormal 原型对齐。当前实现包含 top-k 竞争（降低负样本规模）与软标签/normal 图参与等改进；若启用 `enable_two_stage`，会先低权重稳定 segmentation，再提高对齐强度。
- **R2（lambda_msad_margin / lambda_msad_sim_margin）**：基于 MSAD 的 margin 约束。  
  - `lambda_msad_margin` 对像素级 abnormal/normal 分数做边界约束（抑制背景假阳）。  
  - `lambda_msad_sim_margin` 在返回相似度 logits 时，对“spurious/高风险区域”的相似度分布做额外 margin 约束（需配合 `msad_return_similarity_logits` 才能生效）。
- **bank + suspicious（lambda_suspicious）**：异常特征记忆库与可疑方向学习。  
  - bank 收集异常图中 matched query 的异常特征，形成更稳定的异常锚点。  
  - suspicious loss 用这些锚点约束 `w`（可疑子空间）与 `W_k`（异常原型）之间的关系，避免把伪相关直接写入异常原型；可用 `bank_warm_up_ratio` 控制 warm-up，`bank_orthogonalize` 强化去相关。  
  - 若启用 `compound_disable_w`（w 不写入 prompt），`proto_suspicious` 不存在，则该项对主链路可视为冗余，建议在 `few_shot_no_w` 中默认关闭。

---

## 3.4 DAP（数据依赖异常先验）与 Bank 的作用说明
- DAP（Data-dependent Abnormal Prior，CoCoOp 条件化）  
  - 选取每张图中“最异常”的 top-k patch：以 patch 特征与图像全局特征的差异范数为 anomaly_scores：[compound_prompt_learner.py:select_top_k_patches](file:///e:/Program%20Files/FiLo_plus/compound_prompt_learner.py#L179-L213)。  
  - 拼接 top-k patch 过 PatchMetaNet 得到 bias（B,ctx_dim），加到每个 W_k（只影响 abnormal prompts），使异常原型图像条件化：[compound_prompt_learner.py:forward](file:///e:/Program%20Files/FiLo_plus/compound_prompt_learner.py#L252-L338)。  
  - 可配合 `compound_dap_spurious_filter`/`compound_dap_spurious_alpha` 用 spurious 分数衰减 anomaly_scores，降低高光/阴影等伪异常被选入 DAP 的概率。
- Bank + suspicious  
  - bank（AnomalyFeatureBankV2）收集来自异常图的 matched queries 特征，并在入库前进行 warm-up 延迟与与 normal 原型的正交化去冗余：[train.py:AnomalyFeatureBankV2](file:///e:/Program%20Files/FiLo_plus/train.py#L88-L217)。  
  - suspicious loss 利用这些视觉侧 anchors 让 w 学习正常图里的 hard negatives，同时以 w-abnormal margin 做去耦，避免 w 演化成“异常子类”。  
  - 当 `compound_disable_w=True`（w 不参与 prompt）时，这条链路对主任务基本冗余，建议在 `few_shot_no_w` 中默认关闭。

---

## 4. 数据划分与复现（train_from_test）
你启用了：
- `--train_from_test --specie_split_seed 42 --specie_split_ratio 0.8`

划分文件会写入：`./ckpt/<run_name>/specie_splits_<cls>.json`（用于后续严格复现相同划分）。

---

## 5. 测试/评测建议（保持与训练一致）
测试时重点是“推理相关参数一致”，训练专用的 lr/lambda 调度无需带入。

建议最小一致性测试（示例）：
```bash
python MSAM_test.py \
  --use_official \
  --dataset mvtec \
  --sam3_ckpt /root/autodl-tmp/FiLo_plus/sam3/weights/sam3/sam3.pt \
  --data_root /root/autodl-tmp/data/mvtec_anomaly_detection \
  --meta_path /root/autodl-tmp/data/mvtec_anomaly_detection/meta.json \
  --ckpt ./ckpt/<run_name>/sam3_peft_best.pth \
  --output_dir ./outputs/<run_name> \
  --train_from_test \
  --splits_dir ./ckpt/<run_name>/ \
  --class_agnostic --agnostic_name object \
  --prompt_learner_type compound \
  --compound_mode cocoop \
  --compound_n_ctx 4 \
  --compound_n_ctx_offset 4 \
  --compound_num_abnormal 10 \
  --compound_enable_dap \
  --compound_use_text_encoder \
  --compound_abnormal_word anomaly \
  --compound_pooling ctx_only \
  --enable_msad --msad_use_shape_attention --msad_num_levels 3 --save_msad_vis \
  --enable_spurious_gating --spurious_score_threshold 0.20
```

如果你要让 MSAD 轻微参与最终 mask（便于看“MSAD 是否还在工作”）：
```bash
--use_msad_output --msad_mask_alpha 0.1
```

---

## 6. 调试与对比（你当前启用的两个开关）
### 6.1 梯度范数诊断
训练加：
- `--debug_prompt_grads`

会每 `log_freq` 步打印 `prompt_learner.*` 与 `patch_meta_net` 的 grad norm，用于确认：
- V/w/W 是否真的在学习
- DAP meta-net 是否有梯度

### 6.2 特征 dump（用于新旧版本对比）
训练加：
- `--debug_dump_features`

会在 `./ckpt/<run_name>/debug_features_step0.npz` 写入少量特征（tfs、msad score、eta）。

测试加：
- `--debug_dump_features`

会在 `--output_dir` 下写入 `debug_features_test.npz`。

对比脚本：
- [tools/compare_feature_dumps.py](file:///e:/Program%20Files/FiLo_plus/tools/compare_feature_dumps.py)

```bash
python tools/compare_feature_dumps.py --old <old.npz> --new <new.npz>
```

---

## 7. 旧版备份
用于精确对比的历史备份位于：
- `./备份/20260112-232051/`（含旧版 compound、train、msad、wrapper 等）

---

## 8. 依赖与环境提示
本仓库包含子工程：
- `sam3/`：SAM3 官方实现（见 `sam3/pyproject.toml`）
- `FiLo/`：FiLo 原仓库及其依赖（见 `FiLo/requirements.txt`）

建议在 autodl/conda 环境中确保：
- Python 版本与 torch/torchvision 版本匹配 CUDA
- 运行 `sam3` 与本仓库脚本时使用同一个环境，避免 tokenization/text encoder 行为差异

---

## 9. 更新记录
- 2026-01-15：新增登刊版“2 方案概述”正文稿与 MathType 可粘贴公式稿（docs/方案概述_登刊版.md、docs/方案概述_公式_MathType可粘贴版.txt）。
- 2026-01-15：新增 WPS 旧版公式编辑器操作手册（含快捷键、验证清单与截图占位；定版环境：WPS 2025 冬季更新 12.1.0.24657 32位）（docs/WPS旧版公式编辑器操作手册.md）。
- 2026-01-18：新增相关工作PDF抽取、文献卡片、章节标题映射与“摘要+绪论+第二章”整合稿（docs/相关工作_抽取/、docs/相关工作_文献卡片与写作风格摘要.md、docs/梁恩源_章节标题提取与映射.md、docs/论文_摘要_绪论_相关技术与方法概述_整合稿.md、docs/一致性检查_术语与引用清单.md），并添加抽取脚本（tools/extract_relatedwork_sections.py）。
- 2026-01-18：新增相关工作PDF的摘要/第一章/第二章字数统计脚本与报告（tools/wordcount_relatedwork_pdfs.py、tools/generate_wordcount_report.py、docs/相关工作_字数统计/wordcount_report.md 及其原始JSON/CSV数据）。
