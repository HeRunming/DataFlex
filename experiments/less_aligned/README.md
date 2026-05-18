# LESS-Aligned Experiment Setting

This directory contains experiment configurations aligned with the LESS paper
(Xia et al., 2024) for fair comparison of data selection methods.

## Setting

| Component | Configuration |
|-----------|--------------|
| **Base model** | Llama-2-7B / Llama-3.1-8B |
| **Candidate pool** | Flan v2 (sampled 100k-500k) |
| **Target sets** | GSM8K train (64), MMLU dev (per-subject), BBH dev, TyDiQA dev |
| **Selection ratios** | 1%, 5%, 10% |
| **Training** | LoRA SFT, static selected subset |
| **Evaluation** | GSM8K test, MMLU test, BBH test, TyDiQA test |

## How to Run

### Step 1: Prepare data
Download Flan v2 and target datasets. Place them in `data/` directory.

### Step 2: Run selection
```bash
# Random baseline
python scripts/static_select_and_train.py select \
    --method random \
    --candidate_data data/flan_v2_100k.json \
    --target_data data/gsm8k_train_64.json \
    --selection_ratio 0.05 \
    --output_dir experiments/less_aligned/results/random_gsm8k_5pct

# MMD Embedding RBF
python scripts/static_select_and_train.py select \
    --method mmd_emb_rbf \
    --candidate_data data/flan_v2_100k.json \
    --target_data data/gsm8k_train_64.json \
    --selection_ratio 0.05 \
    --output_dir experiments/less_aligned/results/mmd_emb_rbf_gsm8k_5pct

# Embedding Nearest Neighbor (no redundancy penalty)
python scripts/static_select_and_train.py select \
    --method embedding_nn \
    --candidate_data data/flan_v2_100k.json \
    --target_data data/gsm8k_train_64.json \
    --selection_ratio 0.05 \
    --output_dir experiments/less_aligned/results/emb_nn_gsm8k_5pct
```

### Step 3: Train on selected subset
```bash
dataflex-cli train experiments/less_aligned/configs/train_selected_llama7b.yaml \
    dataset_dir=experiments/less_aligned/results/mmd_emb_rbf_gsm8k_5pct/selected_subset.json \
    output_dir=experiments/less_aligned/results/mmd_emb_rbf_gsm8k_5pct/model
```

### Step 4: Evaluate
Use lm-evaluation-harness or direct inference on held-out benchmarks.

## Methods Compared

| Method | `--method` | Description |
|--------|-----------|-------------|
| Random | `random` | Uniform random sampling (lower bound) |
| Full data | `full` | Train on full candidate pool |
| Embedding NN | `embedding_nn` | Nearest neighbor to target (no redundancy) |
| MMD-Emb-RBF | `mmd_emb_rbf` | RBF kernel on embeddings with greedy MMD |
| MMD-Grad-RBF | `mmd_grad_rbf` | RBF kernel on gradient features |
| MMD-GradCov | `mmd_grad_cov` | Degree-2 polynomial on gradients |
| LESS | (external) | Run via original LESS codebase or DataFlex LESS selector |
| TSDS | (external) | Run via DataFlex TSDS selector |

## Key Differences from Smoke Tests

1. **Static selection**: Selection done ONCE offline, not dynamically during training
2. **Held-out evaluation**: Final eval on test splits, NOT on target/selection set
3. **Controlled ratio**: Fixed % of candidates selected, token-budget matched
4. **Real scale**: 100k+ candidates, 7B+ model, real benchmarks
