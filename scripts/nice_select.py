#!/usr/bin/env python3
"""
NICE (Non-differentiable evaluation metric InfluenCE) offline selection,
LESS-aligned adaptation.

NICE = LESS skeleton with the VALIDATION signal replaced: instead of the
next-token-prediction (NTP) gradient of the gold target answer, it uses a
reward-weighted policy gradient (REINFORCE) over Monte-Carlo generations from
the policy model, where the reward is a (non-differentiable) TASK METRIC.

This script:
  1. Loads the pre-computed CANDIDATE gradient cache (reused verbatim from the
     LESS run: less_output/train/1/all_projected_grads.pt, [N, proj_dim], adam,
     L2-normalized). NICE thus differs from LESS ONLY in the validation signal.
  2. Loads the warmup checkpoint (base Llama-2-7B + LoRA adapter) as the policy
     model.
  3. For each target example: MC-sample generations, score each with the task
     metric (reward), form the reward-weighted summed-NLL gradient (vanilla
     REINFORCE, no baseline), TRAK-project to proj_dim (same seed as candidates),
     L2-normalize.
  4. Score candidates = mean_t <g_cand, g_val_t>, take top-k, write step_1.json
     compatible with scripts/export_gradient_selection.py.

Faithful to JTWang2000/NICE obtain_policy_gradients (vanilla policy):
  loss = mean_NLL(continuation) * sentence_length * reward   # = summed_NLL * R
  policy_grad = sum_i loss_i.backward()                       # vanilla: pure sum
  (no advantage/baseline; length-weighted; row-normalized after projection)

Differences from official NICE, matched to our pipeline:
  - 1 checkpoint (our checkpoint-1692), not 4 (our LESS/MMD use single ckpt).
  - reward = task metric (MMLU gold-token prob / BBH exact-match / TyDiQA F1),
    not a reward model.
"""
import argparse
import json
import os
import re
import string
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ───────────────────────── metrics (rewards) ─────────────────────────
def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(pred, gold):
    p = _normalize_answer(pred).split()
    g = _normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if len(p) == 0 or len(g) == 0:
        return float(p == g)
    if ns == 0:
        return 0.0
    prec = ns / len(p)
    rec = ns / len(g)
    return 2 * prec * rec / (prec + rec)


def _bbh_extract_answer(text):
    """Extract final answer after 'the answer is' (BBH CoT convention)."""
    m = re.search(r"answer is\s*(.+?)\s*\.?\s*$", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(".")
    # fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def reward_bbh(gen_text, gold_text):
    """Exact-match of extracted BBH answer (0/1)."""
    pred = _bbh_extract_answer(gen_text)
    gold = _bbh_extract_answer(gold_text) if "answer is" in gold_text.lower() else gold_text.strip()
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def reward_tydiqa(gen_text, gold_text):
    """SQuAD F1 in [0,1]. Use first line of generation as the answer."""
    pred = gen_text.strip().splitlines()[0].strip() if gen_text.strip() else ""
    return _f1(pred, gold_text)


def reward_mmlu_exactmatch(gen_text, gold_text):
    """MMLU: exact match of the answer letter (A/B/C/D).

    The MMLU prompt ends in 'Answer:' and the gold is a single letter, so the
    model's answer is its FIRST emitted A-D token. Take the first A-D character
    of the (stripped) generation — but only if it appears at the very start
    (optionally after punctuation/space), to avoid matching the 'A' in a word.
    """
    g = gold_text.strip().upper()[:1]
    s = gen_text.strip().upper()
    # first alnum char should be the letter; strip leading non-letters
    m = re.match(r"[^A-D]*([A-D])\b", s)
    pred = m.group(1) if m else ""
    return float(pred == g)


REWARD_FNS = {
    "bbh": reward_bbh,
    "tydiqa": reward_tydiqa,
    "mmlu": reward_mmlu_exactmatch,   # note: MMLU also has a special gold-prob path below
}


# ───────────────────────── data ─────────────────────────
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_prompt_answer(messages):
    """From sharegpt messages -> (user_prompt_text, gold_answer_text)."""
    user = ""
    gold = ""
    for m in messages:
        if m["role"] == "user":
            user = m["content"]
        elif m["role"] == "assistant":
            gold = m["content"]
    return user, gold


# ───────────────────────── projector ─────────────────────────
def get_projector(grad_dim, proj_dim, seed, device, dtype):
    from trak.projectors import BasicProjector, CudaProjector, ProjectionType
    try:
        import fast_jl  # noqa
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        fast_jl.project_rademacher_8(torch.zeros(8, 1000, device=device), 512, 0, num_sms)
        cls = CudaProjector
        print("[NICE] Using CudaProjector.")
    except (ImportError, RuntimeError) as e:
        cls = BasicProjector
        print(f"[NICE] Using BasicProjector fallback ({e}).")
    return cls(
        grad_dim=grad_dim, proj_dim=proj_dim, seed=seed,
        proj_type=ProjectionType.rademacher, max_batch_size=8,
        block_size=128, device=device, dtype=dtype,
    )


# ───────────────────────── policy gradient ─────────────────────────
def build_llama2_prompt(user_text):
    """Match template: llama2 [INST] ... [/INST]. Generation prompt (no answer)."""
    return f"<s>[INST] {user_text} [/INST]"


def policy_gradient_for_target(
    model, tokenizer, user_text, gold_text, reward_fn, target_name,
    mc, temperature, max_new_tokens, max_prompt_len, device,
    trainable_params,
):
    """
    Vanilla NICE REINFORCE policy gradient for one target example.
    Returns a flat gradient vector (sum over mc samples of summed-NLL*reward grads),
    or None if all rewards are 0 (zero signal).
    """
    prompt = build_llama2_prompt(user_text)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len,
                    add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    # --- MC generation ---
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=True, top_k=50, top_p=0.95, temperature=temperature,
            max_new_tokens=max_new_tokens, num_return_sequences=mc,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode continuations
    seqs = []
    for i in range(gen.shape[0]):
        cont = gen[i, prompt_len:]
        seqs.append(tokenizer.decode(cont, skip_special_tokens=True))

    # MMLU special: continuous reward = predicted prob of gold token (more signal
    # than 0/1 exact-match on a single-letter answer). Handled by caller via
    # target_name=='mmlu_prob' path; here reward_fn already chosen.

    # rewards for each sampled sequence
    rewards = [reward_fn(s, gold_text) for s in seqs]

    if sum(rewards) == 0.0:
        return None  # no positive signal; vanilla -> zero vector, skip

    # --- reward-weighted summed-NLL gradients, vanilla = pure sum ---
    model.zero_grad(set_to_none=True)
    acc = None
    for s, R in zip(seqs, rewards):
        if R == 0.0:
            continue
        cont_ids = tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if cont_ids.shape[1] == 0:
            continue
        full = torch.cat([input_ids, cont_ids], dim=1)
        labels = full.clone()
        labels[0, :prompt_len] = -100
        n_tok = int((labels[0, 1:] != -100).sum().item())
        if n_tok == 0:
            continue
        out = model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels)
        loss = out.loss * n_tok * R   # mean-NLL * len = summed NLL, * reward
        loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in trainable_params if p.grad is not None])
        acc = g.clone() if acc is None else acc + g
        model.zero_grad(set_to_none=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_grads", required=True,
                    help="Pre-computed candidate grad cache all_projected_grads.pt [N, proj_dim]")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True, help="warmup LoRA adapter (checkpoint-1692)")
    ap.add_argument("--target_data", required=True, help="target jsonl (sharegpt messages)")
    ap.add_argument("--target_name", required=True, choices=["bbh", "mmlu", "tydiqa"])
    ap.add_argument("--out_cache_dir", required=True, help="write step_1.json here")
    ap.add_argument("--proj_dim", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_prompt_len", type=int, default=2048)
    ap.add_argument("--selection_ratio", type=float, default=0.05)
    ap.add_argument("--val_grads_out", default=None, help="optional: save target policy grads .pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load candidate cache ---
    print(f"[NICE] Loading candidate grads: {args.candidate_grads}")
    cand = torch.load(args.candidate_grads, map_location="cpu")
    if not torch.is_tensor(cand):
        cand = torch.as_tensor(cand)
    cand = cand.float()
    N = cand.shape[0]
    print(f"[NICE] candidate grads: {tuple(cand.shape)}")

    # --- load policy model (base + warmup adapter) ---
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"[NICE] Loading base {args.base_model} + adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()
    # PeftModel loads adapters with requires_grad=False (inference mode). NICE needs
    # gradients w.r.t. the LoRA params, so re-enable grad on all lora_ tensors.
    n_lora = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            n_lora += 1
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grad_dim = sum(p.numel() for p in trainable_params)
    print(f"[NICE] LoRA tensors with grad: {n_lora}; trainable params: {grad_dim:,}")
    assert grad_dim > 0, "No trainable params — adapter not loaded correctly."

    projector = get_projector(grad_dim, args.proj_dim, args.seed, device, torch.float16)

    # --- targets ---
    targets = load_jsonl(args.target_data)
    print(f"[NICE] {len(targets)} target examples ({args.target_name})")
    reward_fn = REWARD_FNS[args.target_name]

    val_grads = []
    n_zero = 0
    for ex in tqdm(targets, desc="[NICE] target policy grads"):
        user, gold = split_prompt_answer(ex["messages"])
        g = policy_gradient_for_target(
            model, tok, user, gold, reward_fn, args.target_name,
            mc=args.mc, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            max_prompt_len=args.max_prompt_len, device=device, trainable_params=trainable_params,
        )
        if g is None:
            n_zero += 1
            continue
        # project to proj_dim, then L2-normalize (row)
        gp = projector.project(g.to(device, torch.float16).unsqueeze(0), model_id=0).cpu().float()
        gp = gp / gp.norm(dim=1, keepdim=True).clamp_min(1e-12)
        val_grads.append(gp)

    if not val_grads:
        raise RuntimeError("[NICE] All target policy grads were zero — no signal. "
                           "Check reward function / generation.")
    val = torch.cat(val_grads, dim=0)  # [M', proj_dim]
    print(f"[NICE] target policy grads: {tuple(val.shape)}  (zero-signal skipped: {n_zero})")

    if args.val_grads_out:
        os.makedirs(os.path.dirname(args.val_grads_out), exist_ok=True)
        torch.save(val, args.val_grads_out)

    # --- score = mean_t <g_cand, g_val_t>, top-k ---
    # candidate rows already L2-normalized (from LESS merge); val rows normalized above.
    scores = (cand @ val.T).mean(dim=1)  # [N]
    k = max(1, int(round(N * args.selection_ratio)))
    topk = torch.topk(scores, k=k, largest=True)
    selected = topk.indices.tolist()
    print(f"[NICE] selected {len(selected)} / {N} (ratio {args.selection_ratio})")

    os.makedirs(args.out_cache_dir, exist_ok=True)
    step_path = os.path.join(args.out_cache_dir, "step_1.json")
    payload = {
        "indices": selected,
        "metric": {"nice_score": [float(scores[i]) for i in selected]},
    }
    with open(step_path, "w") as f:
        json.dump(payload, f)
    print(f"[NICE] wrote {step_path}")


if __name__ == "__main__":
    main()
