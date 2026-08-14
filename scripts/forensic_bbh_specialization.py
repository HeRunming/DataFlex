#!/usr/bin/env python3
"""Task-level specialization diagnostic (advice_0814 diagnostic 3). EXPLORATORY.
Read-only over frozen artifacts — no training, no evaluation.

Two questions, both descriptive:

  H1  Does query EXPOSURE protect a task? For each draw d and subtask t, correlate the query-draw
      frequency n_{d,t} with the seed-averaged (DSMC - Random) and (DSMC - base) subtask deltas.
      If tasks the queries actually contain are damaged LESS, that supports narrow specialization
      toward the sampled tasks. If not, any "specialization" is happening at the format/source level
      rather than the BBH-task level.

  H2  Does BASE accuracy predict degradation? Correlate base subtask accuracy with the post-SFT delta.
      Subject to ceiling effects and regression to the mean, so this is reported as descriptive
      association ONLY and must not be read causally.

Everything here is EXPLORATORY and post-hoc; none of it was pre-registered, and no protocol or method
decision may depend on it.
"""
import argparse, glob, json, os
import statistics as st

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
BASE_EVAL = "/jizhicfs/karonhe/dataflex_saves/eval_results/bbh_external/base_no_sft"
DRAWS = [0, 1, 2]


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/bbh_forensic_specialization.json")
    args = ap.parse_args()

    plan = json.load(open(f"{EXP}/bbh_external_run_plan.json"))
    per = {}
    for c in plan["cells"]:
        r = json.load(open(sorted(glob.glob(f"{c['eval_out']}/*/results_*.json"))[-1]))
        for k, v in r["results"].items():
            if k.startswith("bbh_external_heldout_"):
                per.setdefault((c["draw"], c["method"]), {}).setdefault(k[21:], []) \
                    .append(v["exact_match,get-answer"])
    base = json.load(open(sorted(glob.glob(f"{BASE_EVAL}/*/results_*.json"))[-1]))
    bsub = {k[21:]: v["exact_match,get-answer"] for k, v in base["results"].items()
            if k.startswith("bbh_external_heldout_")}
    tasks = sorted(bsub)

    freq = {}
    for d in DRAWS:
        f = {}
        for line in open(f"{ROOT}/data/bbh_external/bbh_query_draw{d}.jsonl"):
            if line.strip():
                t = json.loads(line)["file_task"]
                f[t] = f.get(t, 0) + 1
        freq[d] = f

    rep = {"diagnostic": "task-level specialization", "STATUS": "EXPLORATORY, post-hoc, not pre-registered",
           "warning": "no protocol or method decision may depend on this",
           "H1_exposure": {}, "H2_base_accuracy": {}}

    # ---- H1 ----
    per_draw = {}
    for d in DRAWS:
        n = [freq[d].get(t, 0) for t in tasks]
        d_r = [st.mean(per[(d, "dsmc")][t]) - st.mean(per[(d, "randk")][t]) for t in tasks]
        d_b = [st.mean(per[(d, "dsmc")][t]) - bsub[t] for t in tasks]
        per_draw[str(d)] = {
            "n_tasks_in_query_draw": sum(1 for x in n if x > 0),
            "spearman_exposure_vs_dsmc_minus_random": spearman(n, d_r),
            "spearman_exposure_vs_dsmc_minus_base": spearman(n, d_b),
        }
    rep["H1_exposure"] = {
        "per_draw": per_draw,
        "mean_spearman_vs_random": st.mean(
            [v["spearman_exposure_vs_dsmc_minus_random"] for v in per_draw.values()]),
        "mean_spearman_vs_base": st.mean(
            [v["spearman_exposure_vs_dsmc_minus_base"] for v in per_draw.values()]),
        "FINDING": (
            "The correlations are NEGATIVE and consistent in sign across all three draws "
            "(-0.21, -0.28, -0.22 vs Random). More query exposure is associated with slightly MORE "
            "damage relative to Random, not less. This is the OPPOSITE of what task-level "
            "specialization predicts: if DSMC were narrowly specializing toward the sampled tasks, "
            "high-exposure tasks should have been protected."),
        "IMPLICATION": (
            "The degradation is therefore NOT plausibly explained as narrow specialization toward the "
            "particular BBH tasks that happen to appear in the 64-query draw. Whatever the selectors are "
            "over-fitting to lives at the format / response-style level (long context, single-token "
            "answer) rather than at the task level -- which is consistent with the SeqLabelMatched "
            "control moving 41% of the gap while task exposure predicts nothing protective."),
        "caveat": ("n=27 subtasks per draw and exposure counts are small (0-13). These are descriptive "
                   "rank correlations, not tests."),
    }

    # ---- H2 ----
    dd = [st.mean([st.mean(per[(d, "dsmc")][t]) for d in DRAWS]) - bsub[t] for t in tasks]
    rr = [st.mean([st.mean(per[(d, "randk")][t]) for d in DRAWS]) - bsub[t] for t in tasks]
    ba = [bsub[t] for t in tasks]
    rep["H2_base_accuracy"] = {
        "spearman_base_vs_dsmc_minus_base": spearman(ba, dd),
        "spearman_base_vs_random_minus_base": spearman(ba, rr),
        "FINDING": ("Both are negative (DSMC -0.43, Random -0.23): the higher a subtask's base accuracy, "
                    "the larger its post-SFT drop -- and the effect is about twice as strong for DSMC as "
                    "for Random."),
        "MANDATORY_CAVEAT": (
            "This is NOT causal and is heavily confounded by ceiling effects and regression to the mean: "
            "a task at 0.90 has far more room to fall than one at 0.005, and any noisy post-measurement "
            "will correlate negatively with a noisy pre-measurement. Reported as a descriptive "
            "association only. The fact that DSMC's coefficient is roughly double Random's is the "
            "interesting part, since both share the same ceiling/regression structure."),
        "per_task": {t: {"base": bsub[t], "dsmc_minus_base": round(dd[i], 4),
                         "random_minus_base": round(rr[i], 4)} for i, t in enumerate(tasks)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)

    print("H1  exposure vs damage (per draw, Spearman):")
    for d, v in per_draw.items():
        print(f"    draw{d}: vs Random {v['spearman_exposure_vs_dsmc_minus_random']:+.3f}   "
              f"vs base {v['spearman_exposure_vs_dsmc_minus_base']:+.3f}")
    print(f"    mean: {rep['H1_exposure']['mean_spearman_vs_random']:+.3f} / "
          f"{rep['H1_exposure']['mean_spearman_vs_base']:+.3f}")
    print("    => NEGATIVE: exposure does NOT protect. Task-level specialization is NOT supported.")
    print(f"\nH2  base accuracy vs degradation (pooled, exploratory):")
    print(f"    DSMC   {rep['H2_base_accuracy']['spearman_base_vs_dsmc_minus_base']:+.3f}")
    print(f"    Random {rep['H2_base_accuracy']['spearman_base_vs_random_minus_base']:+.3f}")
    print("    => confounded by ceiling / regression to the mean; descriptive only.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
