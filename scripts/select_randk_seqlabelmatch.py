#!/usr/bin/env python3
"""`Random-K-SeqLabelMatched`: a Random subset that jointly matches DSMC's (sequence-length,
loss-bearing-label-length) 2D histogram at fixed K (code_review_0811). Selection only — no SFT.

Why this REPLACES the sequence-only control rather than adding to it
--------------------------------------------------------------------
The sequence-only `Random-K-LengthMatched` matched DSMC's sequence tokens to 0.984x, but still carried
**7.14x** DSMC's loss-bearing label positions. So it could only rule out coarse *input* length; it could
not rule out the far more plausible explanation:

    DSMC's difference from Random is just that it selects a different instruction FORMAT --
    long context + very short response (classification / multiple-choice) rather than free-form
    generation.

Response length and supervision density are known to matter for SFT, so leaving that axis uncontrolled
would be the obvious reviewer objection. Matching BOTH axes answers the sharper question with the same
6-adapter budget, so the total stays at 36 adapters rather than growing to 42.

Bins are FIXED IN ADVANCE and must not be retuned after seeing anything:
    sequence: [0,256) [256,512) [512,1024) [1024,1536) [1536,inf)
    label:    [0,4)   [4,16)    [16,64)    [64,256)    [256,inf)
Lengths come only from the AUTHORITATIVE cache (built via the real LlamaFactory
`SupervisedDatasetProcessor`), never from a hand-reconstructed template.

If any DSMC-occupied cell has too few pool candidates, this FAILS LOUDLY instead of quietly widening a
bin. What is matched is length only; source composition is reported as a diagnostic and is NOT matched,
and no causal claim about provenance is made.
"""
import argparse, hashlib, json, math, os

import numpy as np

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
CACHE = f"{ROOT}/data/candidate_length_cache_authoritative.npz"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
SEQ_BINS = [(0, 256), (256, 512), (512, 1024), (1024, 1536), (1536, 10 ** 9)]
LAB_BINS = [(0, 4), (4, 16), (16, 64), (64, 256), (256, 10 ** 9)]


def bin_of(n, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= n < hi:
            return i
    return len(bins) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsmc_step1", required=True)
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="frozen: 7000 + draw_id")
    ap.add_argument("--length_npz", default=CACHE)
    ap.add_argument("--feasibility_only", action="store_true")
    args = ap.parse_args()

    z = np.load(args.length_npz)
    seq = z["sequence_tokens_after_cutoff"]
    lab = z["loss_bearing_label_positions"]
    N = len(seq)
    cell = np.array([bin_of(int(s), SEQ_BINS) * len(LAB_BINS) + bin_of(int(l), LAB_BINS)
                     for s, l in zip(seq, lab)])

    dsmc = np.array(json.load(open(args.dsmc_step1))["indices"])
    K = args.num_select
    if len(dsmc) != K:
        raise SystemExit(f"DSMC subset has {len(dsmc)} indices, expected K={K}")
    n_cells = len(SEQ_BINS) * len(LAB_BINS)
    need = np.bincount(cell[dsmc], minlength=n_cells)
    avail = np.bincount(cell, minlength=n_cells)

    # ---- feasibility gate: fail loudly rather than silently widening a bin ----
    short = [{"cell": int(c), "seq_bin": int(c // len(LAB_BINS)), "label_bin": int(c % len(LAB_BINS)),
              "need": int(need[c]), "available": int(avail[c])}
             for c in range(n_cells) if need[c] > avail[c]]
    occupied = int((need > 0).sum())
    print(f"[seqlab] DSMC occupies {occupied}/{n_cells} cells; K={K}")
    if short:
        print("[seqlab] INFEASIBLE cells:", json.dumps(short, indent=1))
        raise SystemExit("feasibility gate FAILED: some DSMC cells cannot be matched from the pool. "
                         "STOP and report -- do NOT widen the bins.")
    print("[seqlab] feasibility: OK (every DSMC-occupied cell has enough candidates)")
    if args.feasibility_only:
        return 0

    # ---- sample per cell, seeded, excluding nothing but the cell constraint ----
    rng = np.random.RandomState(args.seed)
    selected, per_cell = [], {}
    for c in range(n_cells):
        if need[c] == 0:
            continue
        pool = np.flatnonzero(cell == c)
        pick = rng.choice(pool, size=int(need[c]), replace=False)
        selected.extend(pick.tolist())
        per_cell[str(c)] = {"seq_bin": int(c // len(LAB_BINS)), "label_bin": int(c % len(LAB_BINS)),
                            "n": int(need[c]), "pool": int(len(pool))}
    selected = sorted(int(x) for x in selected)
    if len(selected) != K or len(set(selected)) != K:
        raise SystemExit(f"produced {len(selected)} ({len(set(selected))} unique), expected {K}")

    sel = np.array(selected)
    # verify the 2D histogram matches exactly, not approximately
    got = np.bincount(cell[sel], minlength=n_cells)
    if not np.array_equal(got, need):
        raise SystemExit("2D histogram mismatch after sampling")

    # source composition (diagnostic only, NOT matched)
    want = set(selected)
    comp = {}
    for i, line in enumerate(open(CAND_JSONL)):
        if i in want:
            s = json.loads(line).get("dataset", "unknown")
            comp[s] = comp.get(s, 0) + 1
    tot = sum(comp.values())
    H = -sum((n / tot) * math.log(n / tot) for n in comp.values() if n)

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {
        "kernel": "random_k_seqlabelmatched", "seed": args.seed, "num_select": K,
        "role": ("PRE-REGISTERED SECONDARY SENSITIVITY CONTROL. Replaces the sequence-only "
                 "Random-K-LengthMatched; not a seventh arm. Primary comparisons untouched."),
        "length_source": f"AUTHORITATIVE {os.path.relpath(args.length_npz, ROOT)} "
                         f"(real LlamaFactory SupervisedDatasetProcessor)",
        "seq_bins": SEQ_BINS, "label_bins": [(a, b if b < 10 ** 9 else None) for a, b in LAB_BINS],
        "bins_fixed_in_advance": True,
        "dsmc_step1": os.path.abspath(args.dsmc_step1),
        "n_cells_occupied": occupied, "per_cell": per_cell,
        "histogram_matches_exactly": True,
        "sequence_tokens_selected": int(seq[sel].sum()),
        "sequence_tokens_dsmc": int(seq[dsmc].sum()),
        "sequence_ratio_vs_dsmc": round(float(seq[sel].sum() / seq[dsmc].sum()), 4),
        "label_positions_selected": int(lab[sel].sum()),
        "label_positions_dsmc": int(lab[dsmc].sum()),
        "label_ratio_vs_dsmc": round(float(lab[sel].sum() / lab[dsmc].sum()), 4),
        "source_composition": dict(sorted(comp.items(), key=lambda x: -x[1])),
        "source_entropy_nats": round(H, 3),
        "matched_axes": "BOTH post-cutoff sequence length AND loss-bearing label positions (coarse, 2D)",
        "not_matched": ("source composition -- reported as a diagnostic only. Length and provenance are "
                        "correlated, so this control isolates the length/format axis and makes NO causal "
                        "claim about source."),
        "terminology": ("sequence tokens = post-cutoff sequence-token EXPOSURE (not GPU cost: batches pad "
                        "to the longest member). label positions = LOSS-BEARING LABEL POSITIONS (not "
                        "'amount of supervised signal': the loss ignores -100 then takes a "
                        "token-normalized mean CE)."),
    }
    json.dump({"indices": selected, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[seqlab] K={K} cells={occupied} seq_ratio={meta['sequence_ratio_vs_dsmc']} "
          f"label_ratio={meta['label_ratio_vs_dsmc']} H={H:.3f}")
    print(f"[seqlab] subset sha256 = "
          f"{hashlib.sha256(json.dumps(selected).encode()).hexdigest()[:16]}")
    print(f"[seqlab] wrote {args.out_cache_dir}/step_1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
