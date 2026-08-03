#!/usr/bin/env python3
"""Assemble the 4-draw master manifest for the target-draw pilot (advice_0731). Records, per draw:
target JSONL + target-gradient + 8 selection + 8 subset hashes, NICE val-grad hash / zero-signal IDs
/ reward diagnostics, Random-K seed, RR permutation seed, LengthMatched bucket counts, and the shared
candidate-cache / checkpoint / projection / environment hashes. No training. Read-only over artifacts."""
import json, glob, os, hashlib
import torch, transformers

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
DRAWS = ["stem80_draw0", "stem80_draw1", "stem80_draw2", "stem80_draw3", "stem80_draw4",
         "hum80_draw0", "hum80_draw1", "hum80_draw2", "hum80_draw3", "hum80_draw4"]
METHODS = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice", "randk", "randk_lenmatch"]
WARMUP = f"{SAVES}/sft_results/warmup_seed42/checkpoint-1692"


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tsha(p):
    v = torch.load(p, map_location="cpu")
    return hashlib.sha256(v.numpy().tobytes()).hexdigest(), tuple(v.shape)


def sidx(p):
    return hashlib.sha256(json.dumps(sorted(json.load(open(p))["indices"])).encode()).hexdigest()


cand = f"{SAVES}/less_output/train/1/all_projected_grads.pt"
cand_sha, cand_shape = tsha(os.path.realpath(cand))
man = {"env": {"torch": torch.__version__, "transformers": transformers.__version__,
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
       "candidate_cache": {"path": cand, "sha256": cand_sha, "shape": list(cand_shape)},
       "warmup_ckpt": {"adapter_sha256": fsha(f"{WARMUP}/adapter_model.safetensors"),
                       "optimizer_sha256": fsha(f"{WARMUP}/optimizer.pt")},
       "proj_dim": 8192, "proj_seed": 123,
       "gradient_type_candidate": "adam", "gradient_type_target": "sgd",
       "nice_mode": "strict_deterministic",
       "draws": {}}

for d in DRAWS:
    meta = json.load(open(f"{ROOT}/data/target_draws/{d}.meta.json"))
    tgt_grad = f"{SAVES}/draw_{d}_output/target/1/all_projected_grads.pt"
    tg_sha, tg_shape = tsha(tgt_grad)
    entry = {"train_seed": meta["train_seed"], "rr_perm_seed": meta["rr_perm_seed"],
             "target_jsonl_sha256": fsha(f"{ROOT}/data/target_draws/{d}.jsonl"),
             "target_jsonl_sha256_frozenmeta": meta["target_file_sha256"],
             "target_grad_sha256": tg_sha, "target_grad_shape": list(tg_shape),
             "randk_seed": 2000 + int(d.split("draw")[-1]),
             "selections": {}, "subsets": {}}
    assert entry["target_jsonl_sha256"] == entry["target_jsonl_sha256_frozenmeta"], f"{d} target hash drift"
    for m in METHODS:
        entry["selections"][m] = sidx(f"{SAVES}/sel_{d}_{m}/step_1.json")
        entry["subsets"][m] = fsha(f"{SAVES}/sft_subsets/{d}_{m}_sel.jsonl")
    # NICE specifics
    nm = json.load(open(f"{SAVES}/sel_{d}_nice/step_1.json"))["metric"]
    vg = f"{SAVES}/sel_{d}_nice/val_grads.pt"
    entry["nice"] = {"val_grad_sha256": tsha(vg)[0] if os.path.exists(vg) else None,
                     "n_zero_signal": nm["n_zero_signal"], "zero_signal_target_ids": nm["zero_signal_target_ids"],
                     "reward_mean": nm["reward_mean_overall"], "reward_hist": nm["reward_hist_counts"],
                     "mc": nm["mc"], "gen_seed": nm["gen_seed"]}
    # LengthMatched bucket counts
    lm = json.load(open(f"{SAVES}/sel_{d}_randk_lenmatch/step_1.json"))["metric"]
    entry["randk_lenmatch"] = {"per_bucket": lm["per_bucket"], "token_diff": lm["token_diff"]}
    man["draws"][d] = entry

out = f"{ROOT}/experiments/less_aligned/targetdraw_10draw_master_manifest.json"
json.dump(man, open(out, "w"), indent=2)
print(f"wrote {out}")
print(f"draws={len(man['draws'])}  env torch={man['env']['torch']} gpu={man['env']['gpu']}")
for d in DRAWS:
    e = man["draws"][d]
    print(f"  {d}: tgt_grad {e['target_grad_shape']} nice_zero={e['nice']['n_zero_signal']} "
          f"reward_mean={e['nice']['reward_mean']:.3f} lenmatch_tokdiff={e['randk_lenmatch']['token_diff']}")
