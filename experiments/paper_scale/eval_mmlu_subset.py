#!/usr/bin/env python
"""
Evaluate LoRA models on MMLUSubset-test (10261 questions, 7 categories).
Uses vLLM for fast generation, extracts A/B/C/D from response.

Usage:
    python experiments/paper_scale/eval_mmlu_subset.py [--methods all] [--gpus 0,1,2,3,4,5,6,7]
"""

import os
import sys
import json
import re
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
BASE_MODEL = "/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
ADAPTER_DIR = "/jizhicfs/karonhe/dataflex_saves/debug_set"
DATA_PATH = "/jizhicfs/karonhe/DataFlex/data/MMLUSubset_test.json"
RESULT_DIR = "/jizhicfs/karonhe/dataflex_saves/eval_results"

ALL_METHODS = [
    "base", "random", "loss", "less", "fisher_sft",
    "opt_gcs_logdet", "opt_gcs_score", "opt_gcs_unwhitened",
    "opt_gcs_rank50", "random_subspace_logdet", "grad_norm_topk",
]

EVAL_LOSSES = {
    "random": 0.874, "loss": 0.881, "less": 0.822,
    "fisher_sft": 0.859, "opt_gcs_logdet": 0.861, "opt_gcs_score": 0.869,
    "opt_gcs_unwhitened": 0.880, "opt_gcs_rank50": 0.873,
    "random_subspace_logdet": 0.936, "grad_norm_topk": 0.813,
}


def extract_answer(text: str) -> str:
    """Extract A/B/C/D answer from model generation."""
    text = text.strip()
    if not text:
        return ""
    # Direct match: starts with A/B/C/D
    m = re.match(r'^([A-D])', text)
    if m:
        return m.group(1)
    # Match "The answer is X" pattern
    m = re.search(r'(?:answer|choice)\s+(?:is\s+)?([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Last resort: find first A-D in text
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return ""


def evaluate_single_model(method: str, gpu: int):
    """Evaluate one model on MMLUSubset-test using HF generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda:0")

    output_path = os.path.join(RESULT_DIR, method, "mmlu_subset_test")
    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, "results.json")

    if os.path.exists(result_file):
        print(f"[{method}] Results exist, skipping.", flush=True)
        with open(result_file) as f:
            return json.load(f)

    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Load model
    print(f"[{method}] Loading model on GPU {gpu}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if method != "base":
        adapter_path = os.path.join(ADAPTER_DIR, method)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    print(f"[{method}] Model loaded. Evaluating {len(data)} questions...", flush=True)

    # Evaluate
    correct = 0
    total = 0
    category_stats = {}
    errors = []

    for i, item in enumerate(data):
        instruction = item["instruction"]
        gold_answer = item["output"][0]  # First char: A/B/C/D

        # Build prompt - simple instruction format
        prompt = f"{instruction}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)

        is_correct = (pred == gold_answer)
        correct += is_correct
        total += 1

        cat = item.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        category_stats[cat]["correct"] += int(is_correct)

        if not is_correct and len(errors) < 10:
            errors.append({"idx": i, "gold": gold_answer, "pred": pred, "response": response[:100]})

        if (i + 1) % 500 == 0:
            acc = correct / total
            print(f"[{method}] {i+1}/{len(data)} acc={acc:.4f}", flush=True)

    overall_acc = correct / total if total > 0 else 0.0
    cat_accs = {cat: stats["correct"] / stats["total"] for cat, stats in sorted(category_stats.items())}

    result = {
        "method": method,
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "category_accuracy": cat_accs,
        "category_stats": category_stats,
        "sample_errors": errors,
    }

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{method}] Done. Accuracy: {overall_acc:.4f} ({correct}/{total})", flush=True)
    return result


def run_method_subprocess(method: str, gpu: int):
    """Run evaluation in a subprocess to isolate GPU memory."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["http_proxy"] = "http://hy-proxy.woa.com:3128"
    env["https_proxy"] = "http://hy-proxy.woa.com:3128"

    python = "/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs/bin/python"
    cmd = [python, __file__, "--single", method, "--gpus", str(gpu)]

    log_path = os.path.join(RESULT_DIR, "logs", f"{method}_mmlu_subset.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"[{method}] Launching on GPU {gpu}...", flush=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, timeout=7200)
    return method, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all", help="Comma-separated methods or 'all'")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--single", default=None, help="Run single method (internal use)")
    args = parser.parse_args()

    if args.single:
        gpu = int(args.gpus.split(",")[0])
        evaluate_single_model(args.single, gpu)
        return

    # Parse methods
    if args.methods == "all":
        methods = ALL_METHODS
    else:
        methods = args.methods.split(",")

    gpus = [int(g) for g in args.gpus.split(",")]
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"Evaluating {len(methods)} methods on MMLUSubset-test ({10261} questions)")
    print(f"GPUs: {gpus}")
    print()

    # Launch in parallel, 1 per GPU
    results = {}
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, method in enumerate(methods):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(run_method_subprocess, method, gpu)] = method

        for future in as_completed(futures):
            method = futures[future]
            try:
                name, retcode = future.result()
                if retcode == 0:
                    rpath = os.path.join(RESULT_DIR, method, "mmlu_subset_test", "results.json")
                    if os.path.exists(rpath):
                        with open(rpath) as f:
                            results[method] = json.load(f)
                        print(f"[{method}] OK: {results[method]['overall_accuracy']:.4f}")
                    else:
                        print(f"[{method}] Completed but no results file")
                else:
                    print(f"[{method}] FAILED (exit code {retcode})")
            except Exception as e:
                print(f"[{method}] ERROR: {e}")

    # Print summary
    print()
    print("=" * 70)
    print("MMLUSubset-test Results (10261 questions, 7 categories)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Method':<28} {'Accuracy':>10} {'eval_loss':>11}")
    print("-" * 56)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["overall_accuracy"])
    for rank, (method, r) in enumerate(sorted_results, 1):
        el = EVAL_LOSSES.get(method)
        el_s = f"{el:.3f}" if el else "N/A"
        print(f"{rank:<5} {method:<28} {r['overall_accuracy']:>10.4f} {el_s:>11}")

    # Category breakdown
    if sorted_results:
        print()
        print("Category breakdown:")
        cats = sorted(sorted_results[0][1].get("category_accuracy", {}).keys())
        header = f"{'Method':<25}" + "".join(f"{c:>10}" for c in cats)
        print(header)
        print("-" * len(header))
        for method, r in sorted_results:
            ca = r.get("category_accuracy", {})
            row = f"{method:<25}" + "".join(f"{ca.get(c, 0):>10.3f}" for c in cats)
            print(row)


if __name__ == "__main__":
    main()
