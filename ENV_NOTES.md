# DataFlex 实验环境 (HeRunming)

新建的独立 conda 环境，不依赖任何他人环境。

## 激活
```bash
conda activate /jizhicfs/karonhe/envs/dataflex
# 或直接用绝对路径 bin:
PY=/jizhicfs/karonhe/envs/dataflex/bin/python
BIN=/jizhicfs/karonhe/envs/dataflex/bin   # dataflex-cli, lm_eval 在这里
```

## 已装组件
- python 3.11, torch 2.6.0+cu124 (8×H20 可见, CUDA available)
- dataflex 1.0.0 (editable, -e .), llamafactory 0.9.4
- transformers 4.54.1, numpy 1.26.4 (<2.0 ok)
- trak 0.3.2 + fast_jl (CudaProjector 可用; 注意必须先 import torch 再 import fast_jl)
- lm-eval 0.4.5 (BBH/MMLU 评测)

## 代理 (下载时)
```bash
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com"
```

## 缺失资产 (实验前需准备)
- data/less_train_all.jsonl  (270k 候选池) — 未下载
- /jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf — 未下载 (现有 Qwen3-8B/32B)
- /jizhicfs/karonhe/dataflex_saves/ — 全部运行产物缺失 (warmup adapter+optimizer state, 梯度缓存, SFT adapter, eval 输出)

## Git
- repo: github.com/HeRunming/DataFlex.git, 分支 fa
