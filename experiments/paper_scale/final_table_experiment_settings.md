# Final Table 实验设置与参数说明

## 实验目标

在统一代码版本、统一训练配置、完全隔离的 cache 环境下，公平对比 16 个数据选择方法在 3 个下游 benchmark 上的表现。所有方法共享相同的：模型、LoRA 配置、训练数据、budget、训练步数、评估协议。

---

## 基础设置

### 模型

| 项目 | 配置 |
|------|------|
| 基座模型 | Llama-3.1-8B (`Meta-Llama-3___1-8B`) |
| 微调方式 | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 8 |
| LoRA target | all (所有 attention + MLP 层) |
| 精度 | bfloat16 |
| 框架 | DataFlex (基于 LLaMA-Factory) |

### 训练数据

| 项目 | 配置 |
|------|------|
| 数据集 | Open-Hermes-2.5 (openhermes_10w, 100k 样本) |
| 模板 | llama3 |
| 最大长度 | 4096 tokens |
| 选择 budget | 5000 样本 |

### 训练超参数

| 项目 | 配置 |
|------|------|
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 1 |
| 全局 batch size | 8 (8 GPU × 1) |
| 学习率 | 1e-4 |
| LR scheduler | cosine |
| warmup ratio | 0.1 |
| 训练 epochs | 1.0 |
| 总训练步数 | 1260 (自动计算: warmup_step + update_step × update_times) |
| GPU | 8 × NVIDIA H20 (98 GB) |
| 分布式 | torchrun, nproc=8 |
| ddp_timeout | 180000000 |

### DataFlex 动态选择参数

| 项目 | 配置 | 说明 |
|------|------|------|
| train_type | dynamic_select | 动态数据选择模式 |
| warmup_step | 10 | 前 10 步使用全量数据随机训练 |
| update_step | 625 | 每次选择后训练 625 步 (= budget / nproc = 5000/8) |
| update_times | 2 | 执行 2 轮动态选择 |
| 评估间隔 | 420 步 (= 1260/3) | 训练中评估 3 次 eval_loss |
| eval_dataset | mmlu_valid_cot | 训练时的验证集（仅用于监控，不用于选择） |

### 训练流程

```
Step 0-10:    warmup（随机数据）
Step 10:      第 1 轮 selector 被调用 → 选择 5000 样本
Step 10-635:  用选出的 5000 样本训练
Step 635:     第 2 轮 selector 被调用 → 重新选择 5000 样本
Step 635-1260: 用新选出的 5000 样本训练
```

---

## 方法列表（16 个实验）

### Baselines（4 个）

| 方法名 | 组件名 | 说明 |
|--------|--------|------|
| `random_s42` | random | 随机选择 5000 样本 |
| `loss_s42` | loss | 按模型 loss 中等范围选择 |
| `less_s42` | less | LESS (需要 eval_dataset, target-aware) |
| `fisher_sft_s42` | fisher_sft | FisherSFT (last-layer embedding + logdet) |

### Negative Controls（4 个）

| 方法名 | 组件名 | 说明 |
|--------|--------|------|
| `grad_norm_topk_s42` | grad_norm_topk | 梯度范数 top-k（自己算梯度） |
| `rsub_own_seed1` | random_subspace_logdet_seed1 | 随机子空间 + logdet（自己算梯度, subspace seed=1） |
| `rsub_own_seed2` | random_subspace_logdet_seed2 | 随机子空间 + logdet（subspace seed=2） |
| `rsub_own_seed3` | random_subspace_logdet_seed3 | 随机子空间 + logdet（subspace seed=3） |

### Our Methods（4 个主方法）

| 方法名 | 组件名 | 说明 |
|--------|--------|------|
| `hybrid_add_l025_s42` | opt_gcs_hybrid_add_lambda0.25 | Hybrid Additive (λ=0.25) |
| `hybrid_mul_g025_s42` | opt_gcs_hybrid_mul_gamma0.25 | Hybrid Multiplicative (γ=0.25) |
| `hybrid_mul_g05_s42` | opt_gcs_hybrid_mul_gamma0.5 | Hybrid Multiplicative (γ=0.5) |
| `logdet_nopref_s42` | opt_gcs_logdet_no_prefilter | LogDet 无 prefilter |

### Multi-seed（4 个，用于方差估计）

| 方法名 | 组件名 | training seed | 说明 |
|--------|--------|--------------|------|
| `hybrid_add_l025_s1` | opt_gcs_hybrid_add_lambda0.25 | 1 | |
| `hybrid_add_l025_s2` | opt_gcs_hybrid_add_lambda0.25 | 2 | |
| `hybrid_mul_g025_s1` | opt_gcs_hybrid_mul_gamma0.25 | 1 | |
| `hybrid_mul_g025_s2` | opt_gcs_hybrid_mul_gamma0.25 | 2 | |

> **注**：multi-seed 改变的是训练随机性（dataloader shuffle, LoRA init），selector 的 projector_seed 固定为 42。

---

## Selector 核心参数

### 共享参数（所有 OptGCS 变体）

| 参数 | 值 | 说明 |
|------|-----|------|
| gradient_type | adam_diag | D_t · g_i (对角 AdamW 预处理) |
| proj_dim | 4096 | TRAK 随机投影维度 |
| projector_seed | 42 | TRAK Rademacher 投影器 seed |
| save_interval | 16 | 梯度分块保存间隔 |
| rank_method | effective | 自动 rank 确定 (r_eff = trace/λ_max) |
| whitening_beta | 0.5 | 谱白化强度 (0=不白化, 1=完全白化) |
| length_norm_alpha | 0.5 | token 长度归一化指数 |
| clipping_method | adaptive | 自适应裁剪 (95th percentile) |
| logdet_eps | 0.001 | LogDet 正则化参数 |
| whitening_eigen_floor | 1e-6 | 白化特征值下限 |
| whitening_max_weight | 100.0 | 白化权重上限 |

### Hybrid Additive (λ=0.25)

```
gain_i = z_normalize(log(1 + x_i^T A^{-1} x_i)) + λ · z_normalize(log(s_i))
```

| 参数 | 值 |
|------|-----|
| selection_method | hybrid_add |
| hybrid_lambda | 0.25 |
| prefilter_ratio | 5.0 |

### Hybrid Multiplicative (γ=0.25 / γ=0.5)

```
gain_i = log(1 + x_i^T A^{-1} x_i) × (s_i / mean(s))^γ
```

| 参数 | 值 |
|------|-----|
| selection_method | hybrid_mul |
| hybrid_gamma | 0.25 或 0.5 |
| prefilter_ratio | 5.0 |

### LogDet NoPrefilter

| 参数 | 值 |
|------|-----|
| selection_method | logdet |
| prefilter_ratio | -1 (无 prefilter, 全量候选) |

### Random Subspace LogDet

| 参数 | 值 | 说明 |
|------|-----|------|
| subspace_dim | 50 | 随机子空间维度 |
| seed | 1/2/3 | 控制随机子空间 Q 矩阵 |
| projector_seed | 42 | 固定 TRAK 投影（跨 seed 一致） |
| clipping_method | adaptive | 对齐 OptGCS preprocessing |
| compute_own_grads | true | 自己计算梯度，不复用其他方法 |

### Grad Norm Top-K

| 参数 | 值 | 说明 |
|------|-----|------|
| compute_own_grads | true | 自己计算梯度 |
| gradient_type | adam_diag | 和 OptGCS 相同的梯度表示 |
| length_norm_alpha | 0.5 | 长度归一化 |

---

## Cache 隔离机制

每个实验的 selector cache 完全隔离：

```
/jizhicfs/karonhe/dataflex_saves/final_table/
├── configs/
│   └── components_final.yaml   # 自动生成，所有 cache_dir 被覆盖
├── cache/
│   ├── random/                  # random selector 的 cache
│   ├── loss/
│   ├── less/
│   ├── opt_gcs_hybrid_add_lambda0.25/
│   │   └── gradients/step_10_xxx/   # 梯度 cache
│   ├── random_subspace_logdet_seed1/
│   │   └── gradients/step_10_xxx/   # 自己计算的梯度
│   └── ...
├── random_s42/                  # 训练输出（adapter）
├── loss_s42/
├── hybrid_add_l025_s42/
└── ...
```

**关键隔离点**：
- `components_final.yaml` 在运行时自动生成，将所有 selector 的 `cache_dir` 重写到 `$SAVE_DIR/cache/<selector_name>`
- Negative controls (`random_subspace`, `grad_norm_topk`) 设置 `compute_own_grads=true`，自己计算梯度
- 不存在跨方法的 gradient cache 复用
- Random subspace 的 `projector_seed=42` 固定（跨 subspace seed 一致），只有 subspace 本身随 `seed` 变化

---

## 评估协议

### Benchmarks

| Benchmark | 框架 | 任务类型 | Few-shot | 指标 |
|-----------|------|---------|----------|------|
| MMLU | lm_eval v0.4.12 | loglikelihood (选择) | 5-shot | accuracy |
| GSM8K | lm_eval v0.4.12 | generate_until (CoT) | 8-shot | exact_match (strict) |
| IFEval | lm_eval v0.4.12 | generate_until | 0-shot | prompt_level_strict_acc |

### 评估设置

| 项目 | 配置 |
|------|------|
| 模型加载 | HuggingFace + PEFT (peft=adapter_path) |
| 精度 | bfloat16 |
| batch_size | auto |
| 并行 | 8 GPU 并行评测不同模型 |
| log_samples | true |

### 评估命令示例

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
  --model hf \
  --model_args pretrained=/path/to/Llama-3.1-8B,peft=/path/to/adapter,dtype=bfloat16,trust_remote_code=True \
  --tasks gsm8k_cot \
  --num_fewshot 8 \
  --batch_size auto \
  --output_path /path/to/output \
  --log_samples
```

---

## 预处理流程对齐

所有使用梯度特征的方法（OptGCS variants, random_subspace, grad_norm_topk）共享完全相同的预处理流程：

```
1. 前向 + 反向 → 得到 per-sample gradient
2. Adam-diag 预处理: g_precond = g / sqrt(v + eps)  [v clamped >= 1e-16]
3. TRAK 随机投影: g_proj = Rademacher_matrix @ g_precond  [4096 维]
4. NaN/Inf → 0
5. Length normalization: h = g_proj / token_length^0.5
6. Adaptive clipping: τ = 95th percentile of norms, clip to τ
7. L2 normalization: h = h / ||h||
```

之后各方法分别进行：
- **OptGCS**: 谱分析 (SVD) → 白化投影 → hybrid/logdet 选择
- **Random Subspace**: 随机正交投影 → logdet 选择
- **Grad Norm Top-K**: 直接按 step 5 后的 norm 排序（不做 L2 norm）

---

## 硬件与环境

| 项目 | 配置 |
|------|------|
| GPU | 8 × NVIDIA H20 (98 GB HBM) |
| CPU | 64 核 |
| 内存 | 512 GB |
| OS | Linux (TENCENT64) |
| Python | 3.11 |
| PyTorch | 2.x |
| Conda 环境 | spec_gcs |
| 数据持久化 | /jizhicfs/karonhe/ |

---

## 文件结构

```
DataFlex/
├── src/dataflex/train/selector/
│   ├── spec_gcs_selector.py          # OptGCS 核心实现 (hybrid_add, hybrid_mul, logdet, score)
│   ├── negative_control_selectors.py  # random_subspace, grad_norm_topk
│   ├── fisher_sft_selector.py         # FisherSFT baseline
│   └── __init__.py
├── src/dataflex/configs/
│   └── components.yaml                # 所有方法配置
├── experiments/paper_scale/
│   ├── run_final_table.sh             # Final table 训练脚本
│   ├── run_lm_eval_final.sh           # Final table 评测脚本
│   ├── debug_set_results_round1.md    # Round 1 结果
│   └── debug_set_results_round2.md    # Round 2 结果 + 分析
└── data/
    ├── dataset_info.json              # 数据集注册
    ├── Openhermes_train.json          # 训练数据 (100k)
    └── MMLU_valid_cot.json            # 验证集
```

---

## 可复现性

```bash
# 1. 训练
bash experiments/paper_scale/run_final_table.sh

# 2. 评测
bash experiments/paper_scale/run_lm_eval_final.sh

# 3. 单个方法（filter 支持）
bash experiments/paper_scale/run_final_table.sh hybrid_add_l025_s42
```

所有随机性来源：
- `training seed`：控制 dataloader shuffle, LoRA 初始化
- `projector_seed=42`：控制 TRAK 投影矩阵（固定）
- `selector seed`：控制 random_subspace 的 Q 矩阵（对 OptGCS methods 无效）
