#!/usr/bin/env python3
"""
Joint integer subject-allocation for the target-draw protocol (code_review_0730_2).

Replaces the broken per-subject `round(w_s * 320)` (which summed to 318, not 320) with a
deterministic JOINT integer allocation over the ten blocks of a domain:
  domain STEM: 5 majority blocks of 51 (stem80 draws) + 5 minority blocks of 13 (hum80 draws)
  domain HUM : 5 majority blocks of 51 (hum80 draws) + 5 minority blocks of 13 (stem80 draws)
Constraints (per domain):
  * each block b sums EXACTLY to its size B_b (51 or 13)  -> domain total = 5*51+5*13 = 320 exactly
  * per-subject column sum <= validation reservoir count cap_s
  * minimize deviation from the ideal B_b * w_s (w_s = lm-eval micro weight = test-doc proportion)
Algorithm: per-block largest-remainder rounding of B_b*w_s (exact row sums), then a deterministic
repair pass that moves units, WITHIN a block, from over-cap subjects to the most under-ideal
subject with remaining cap+room. Guaranteed to satisfy caps when sum_s cap_s >= 320 (STEM 335,
HUM 518). Deterministic tie-breaking by subject index.

--plan_only prints/saves the allocation matrix + deviation stats (no example sampling, no draws).
Full draw generation (example IDs, meta, overlap) is a later step, gated on protocol freeze.
"""
import argparse, glob, json, os
from datasets import Dataset

STEM = ["abstract_algebra","anatomy","astronomy","college_biology","college_chemistry",
        "college_computer_science","college_mathematics","college_physics","computer_security",
        "conceptual_physics","electrical_engineering","elementary_mathematics","high_school_biology",
        "high_school_chemistry","high_school_computer_science","high_school_mathematics",
        "high_school_physics","high_school_statistics","machine_learning"]
HUM = ["formal_logic","high_school_european_history","high_school_us_history",
       "high_school_world_history","international_law","jurisprudence","logical_fallacies",
       "moral_disputes","moral_scenarios","philosophy","prehistory","professional_law","world_religions"]
CACHE = os.path.expanduser("~/.cache/huggingface/datasets/hails___mmlu_no_train")


def count(subj, split):
    fs = glob.glob(f"{CACHE}/{subj}/*/*/mmlu_no_train-{split}.arrow")
    if not fs:
        raise FileNotFoundError(f"no {split} arrow for {subj}")
    return len(Dataset.from_file(fs[0]))


def largest_remainder(weights, total):
    """Integer vector of length len(weights) summing to `total`, closest to weights*total
    (largest-remainder / Hamilton). Deterministic tie-break: lower index first."""
    ideal = [w * total for w in weights]
    floor = [int(x) for x in ideal]
    rem = total - sum(floor)
    order = sorted(range(len(weights)), key=lambda i: (-(ideal[i] - floor[i]), i))
    for k in range(rem):
        floor[order[k]] += 1
    return floor


def allocate_domain(subjects, caps, weights, block_sizes):
    """Return a[b][s] integer matrix: rows=blocks (sizes given), cols=subjects. Exact row sums,
    column sums <= caps, close to B_b*w_s, minimal cap-repair."""
    nb, ns = len(block_sizes), len(subjects)
    a = [largest_remainder(weights, B) for B in block_sizes]   # exact row sums
    col = [sum(a[b][s] for b in range(nb)) for s in range(ns)]
    # repair: reduce any subject over its cap by moving units within blocks to under-ideal subjects
    ideal_col = [sum(B * weights[s] for B in block_sizes) for s in range(ns)]
    guard = 0
    while any(col[s] > caps[s] for s in range(ns)):
        guard += 1
        if guard > 100000:
            raise RuntimeError("cap repair did not converge")
        # pick the most-over-cap subject (largest overflow, then lowest index)
        s_over = max(range(ns), key=lambda s: (col[s] - caps[s], -s))
        # find a block where s_over has a unit to give
        moved = False
        for b in range(nb):
            if a[b][s_over] <= 0:
                continue
            # recipient in same block: has cap room (col<cap), room in block (a<B_b), most under ideal
            cands = [s2 for s2 in range(ns)
                     if s2 != s_over and col[s2] < caps[s2] and a[b][s2] < block_sizes[b]]
            if not cands:
                continue
            s_to = min(cands, key=lambda s2: ((col[s2] - ideal_col[s2]), s2))
            a[b][s_over] -= 1; a[b][s_to] += 1
            col[s_over] -= 1; col[s_to] += 1
            moved = True
            break
        if not moved:
            raise RuntimeError(f"cannot repair cap for subject {subjects[s_over]} "
                               f"(col={col[s_over]} cap={caps[s_over]}); reservoir too small")
    return a, col, ideal_col


def build_domain(subjects, maj_blocks=5, maj_size=51, min_blocks=5, min_size=13):
    caps = [count(s, "validation") for s in subjects]
    test = [count(s, "test") for s in subjects]
    T = sum(test)
    weights = [t / T for t in test]
    block_sizes = [maj_size] * maj_blocks + [min_size] * min_blocks
    a, col, ideal_col = allocate_domain(subjects, caps, weights, block_sizes)
    # stats
    total = sum(sum(row) for row in a)
    l1_dev = sum(abs(a[b][s] - block_sizes[b] * weights[s])
                 for b in range(len(block_sizes)) for s in range(len(subjects)))
    tvd = 0.5 * sum(abs(col[s] / total - ideal_col[s] / total) for s in range(len(subjects)))
    return {"subjects": subjects, "caps": caps, "test": test, "weights": weights,
            "block_sizes": block_sizes, "alloc": a, "col_sum": col, "ideal_col": ideal_col,
            "domain_total": total, "l1_dev_vs_ideal": l1_dev, "tvd_col_vs_ideal": tvd,
            "cap_ok": all(col[s] <= caps[s] for s in range(len(subjects))),
            "row_sums_ok": all(sum(a[b]) == block_sizes[b] for b in range(len(block_sizes)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/target_draws/subject_allocation_plan.json")
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument("--maj_size", type=int, default=51)
    ap.add_argument("--min_size", type=int, default=13)
    args = ap.parse_args()
    plan = {}
    for name, subs in [("STEM", STEM), ("HUM", HUM)]:
        d = build_domain(subs, maj_size=args.maj_size, min_size=args.min_size)
        plan[name] = d
        print(f"=== {name}: total={d['domain_total']} (want 320)  cap_ok={d['cap_ok']}  "
              f"row_sums_ok={d['row_sums_ok']}  L1dev={d['l1_dev_vs_ideal']:.2f}  "
              f"col-TVD={d['tvd_col_vs_ideal']:.4f}")
        for s, sub in enumerate(subs):
            flag = "" if d["col_sum"][s] <= d["caps"][s] else "  !!CAP"
            print(f"  {sub:32s} w={d['weights'][s]*100:4.1f}%  col={d['col_sum'][s]:3d}/{d['caps'][s]:3d}{flag}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(plan, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    assert all(plan[g]["domain_total"] == 320 and plan[g]["cap_ok"] and plan[g]["row_sums_ok"]
               for g in plan), "allocation invariants violated"
    print("INVARIANTS OK: both domains total 320, exact block sums, caps satisfied")


if __name__ == "__main__":
    main()
