#!/usr/bin/env python3
"""TydiQA-GoldP 1-shot F1/EM evaluation (LESS protocol).

Not available in standard lm-eval, so implemented here. For each of the 9
languages we build a 1-shot prompt (one in-language demonstration drawn from
the target/dev set), generate an answer for every validation question, and
compute SQuAD-style F1 / Exact-Match. We report per-language and macro-average
F1 (matching LESS, which reports TydiQA-GoldP 1-shot F1).

Uses HF transformers generation directly (greedy) on the base model + LoRA
adapter. Run one model per GPU.

Usage:
  python scripts/eval_tydiqa.py --adapter <peft_dir> --output <json> [--limit N]
"""
import argparse
import json
import os
import re
import string
import collections
from pathlib import Path

import torch

LANGS = ["arabic", "russian", "bengali", "telugu", "finnish",
         "swahili", "korean", "indonesian", "english"]

# 1-shot demonstrations: one (context, question, answer) per language taken from
# data/tydiqa_target.jsonl (the same 1-shot examples LESS uses as the target set).
DEMO_PATH = "/jizhicfs/karonhe/DataFlex_fa/data/tydiqa_target.jsonl"


# ---------- SQuAD F1 / EM (official-style) ----------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def em_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def metric_max_over_gold(fn, pred, golds):
    return max(fn(pred, g) for g in golds) if golds else 0.0


def load_demos():
    """Return {lang: (context, question, answer)} from the LESS target jsonl."""
    demos = {}
    with open(DEMO_PATH) as f:
        for line in f:
            row = json.loads(line)
            lang = row["id"].split("-")[0]
            user = row["messages"][0]["content"]
            ans = row["messages"][1]["content"]
            demos[lang] = (user, ans)
    return demos


def build_prompt(demo_user, demo_ans, ctx, q):
    """1-shot prompt: demonstration + the actual question."""
    qa = (
        "Answer the following question based on the context.\n\n"
        f"Context: {ctx}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )
    # Use the same template surface as the target examples.
    # The explicit "Answer:" trigger is required: without it, transformers>=4.5x
    # emits EOS as the first token on non-Latin-script languages (bengali/telugu/
    # arabic), collapsing their F1 to ~0. The trigger restores normal answering
    # and is applied identically to every method (fair comparison).
    return f"{demo_user}\n{demo_ans}\n\n{qa}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (None = base model)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1, help="max examples per language (-1 = all)")
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[tydiqa] loading model {args.base_model} + adapter={args.adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    demos = load_demos()
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task",
                      split="validation", trust_remote_code=True)

    # group by language
    by_lang = collections.defaultdict(list)
    for ex in ds:
        lang = ex["id"].split("-")[0]
        by_lang[lang].append(ex)

    results = {}
    for lang in LANGS:
        examples = by_lang.get(lang, [])
        if args.limit > 0:
            examples = examples[: args.limit]
        if not examples or lang not in demos:
            continue
        demo_user, demo_ans = demos[lang]

        f1s, ems = [], []
        # batched generation
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i: i + args.batch_size]
            prompts = [build_prompt(demo_user, demo_ans, ex["context"][:3000], ex["question"]) for ex in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3500).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for ex, txt in zip(batch, texts):
                # answer = first line of generation
                pred = txt.strip().split("\n")[0].strip()
                golds = ex["answers"]["text"]
                f1s.append(metric_max_over_gold(f1_score, pred, golds))
                ems.append(metric_max_over_gold(em_score, pred, golds))
        results[lang] = {
            "f1": sum(f1s) / len(f1s),
            "em": sum(ems) / len(ems),
            "n": len(f1s),
        }
        print(f"[tydiqa] {lang}: F1={results[lang]['f1']:.4f} EM={results[lang]['em']:.4f} (n={results[lang]['n']})", flush=True)

    macro_f1 = sum(r["f1"] for r in results.values()) / len(results)
    macro_em = sum(r["em"] for r in results.values()) / len(results)
    summary = {"per_language": results, "macro_f1": macro_f1, "macro_em": macro_em,
               "adapter": args.adapter}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[tydiqa] MACRO F1={macro_f1:.4f}  EM={macro_em:.4f}")
    print(f"[tydiqa] saved to {args.output}")


if __name__ == "__main__":
    main()
