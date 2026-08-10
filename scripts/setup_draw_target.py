#!/usr/bin/env python3
"""
Set up per-draw target-gradient extraction config + dataset registration for the target-draw pilot.
Emits, for one draw (e.g. stem80_draw0):
  - registers the draw jsonl as a llamafactory dataset key `<draw>_target`
  - writes a select YAML  experiments/less_aligned/configs/draws/select_<draw>.yaml
  - appends a components entry `<draw>` to src/dataflex/configs/components_draws.yaml
Target grads then come from:  dataflex-cli train <select yaml>  ->  <cache>/target/1/all_projected_grads.pt
Candidate grads are REUSED from the shared less_output cache (Adam), target uses SGD (LESS-aligned),
matching the frozen protocol / hum80_mirror_manifest provenance (warmup_seed42/checkpoint-1692).

This only writes config/registration files — it runs no gradients. Idempotent.
"""
import argparse, json, os

WARMUP = "/jizhicfs/karonhe/dataflex_saves/sft_results/warmup_seed42/checkpoint-1692"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"

SELECT_YAML = """model_name_or_path: {base}
adapter_name_or_path: {warmup}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 512
lora_target: q_proj,k_proj,v_proj,o_proj
lora_dropout: 0.1
dataset: less_train_all
template: llama2
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16
output_dir: {saves}/less_aligned/draw_{draw}
logging_steps: 10
overwrite_output_dir: true
save_steps: 99999
report_to: none
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 2.0e-05
num_train_epochs: 1.0
bf16: true
ddp_timeout: 180000000
train_step: 2
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components_draws.yaml
component_name: {draw}
warmup_step: 1
update_step: 1
update_times: 1
selection_ratio: 0.05
optimizer_state_path: {warmup}
target_dataset: {draw}_target
eval_dataset: {draw}_target
eval_strategy: 'no'
"""

COMPONENT = """  {draw}:
    name: mmd
    params:
      cache_dir: {saves}/draw_{draw}_output
      proj_dim: 8192
      save_interval: 16
      seed: 123
      candidate_subsample: -1
      greedy_device: auto
      gradient_type: adam
      sigma: null
      kernel_type: grad_cov
      target_gradient_type: sgd
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True, help="e.g. stem80_draw0")
    ap.add_argument("--cutoff_len", type=int, default=2048,
                    help="cutoff for TARGET-GRADIENT extraction. MMLU draws used 2048. BBH draws MUST "
                         "use 3072: at 2048, 7/192 BBH 3-shot CoT query prompts are tail-truncated so "
                         "badly the query itself is deleted (see bbh_token_truncation_audit.json and "
                         "code_review_0810_2). This does NOT change the downstream SFT cutoff.")
    args = ap.parse_args()
    draw = args.draw
    jsonl = f"{ROOT}/data/target_draws/{draw}.jsonl"
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)

    # 1. register dataset key
    info = json.load(open(f"{ROOT}/data/dataset_info.json"))
    key = f"{draw}_target"
    info[key] = {"file_name": jsonl, "formatting": "sharegpt",
                 "columns": {"messages": "messages"},
                 "tags": {"role_tag": "role", "content_tag": "content",
                          "user_tag": "user", "assistant_tag": "assistant"}}
    json.dump(info, open(f"{ROOT}/data/dataset_info.json", "w"), indent=2, ensure_ascii=False)

    # 2. select yaml
    os.makedirs(f"{ROOT}/experiments/less_aligned/configs/draws", exist_ok=True)
    open(f"{ROOT}/experiments/less_aligned/configs/draws/select_{draw}.yaml", "w").write(
        SELECT_YAML.format(base=BASE, warmup=WARMUP, saves=SAVES, draw=draw,
                           cutoff_len=args.cutoff_len))

    # 3. components file (create header once, append component if missing)
    comp_path = f"{ROOT}/src/dataflex/configs/components_draws.yaml"
    if not os.path.exists(comp_path):
        open(comp_path, "w").write("selectors:\n")
    body = open(comp_path).read()
    if f"\n  {draw}:" not in body and not body.rstrip().endswith(f"{draw}:"):
        with open(comp_path, "a") as f:
            f.write(COMPONENT.format(draw=draw, saves=SAVES))
    print(f"[setup] {draw}: registered dataset {key}, wrote select_{draw}.yaml, "
          f"component -> components_draws.yaml, target cache -> {SAVES}/draw_{draw}_output")


if __name__ == "__main__":
    main()
