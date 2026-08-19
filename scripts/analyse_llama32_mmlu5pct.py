#!/usr/bin/env python3
"""Pre-registered analysis of the Llama-3.2-3B x MMLU 5% arm (stop-rule amendment #1).

Runs ONLY after 35/35 adapter evals + the shared no-SFT reference. Everything below was fixed in
`prereg_llama32_mmlu5pct.md` before any computation.

Statistical unit, per the prereg
--------------------------------
The primary descriptive unit is the FIVE draw-index blocks. For each index the stem-majority and
hum-majority directions are averaged FIRST, then the five paired block differences are reported.
n=10 must NOT be claimed: the two directions of an index share a training seed, and Random-K even
shares one adapter between them.

Primary metric: balanced MMLU = (STEM + HUM) / 2.
Three pre-registered comparisons: DSMC - First-RR, DSMC - Second-RR, DSMC - Random-K.
Descriptive only -- no p-values, no significance claims.
"""
import argparse, json
import statistics as st

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
METHODS = ["dsmc", "first_rr", "second_rr", "randk"]
TARGETED = ["dsmc", "first_rr", "second_rr"]
IDX = [0, 1, 2, 3, 4]
SEED_OF = {0: 42, 1: 1, 2: 2, 3: 3, 4: 4}
DIRECTIONS = ["stem80", "hum80"]


def aid_of(direction, i, m):
    s = SEED_OF[i]
    if m == "randk":
        return f"l32_randk_drawidx{i}_seed{s}"          # shared by both directions
    return f"l32_{direction}_draw{i}_{m}_seed{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/llama32_mmlu5pct_results.json")
    a = ap.parse_args()

    stt = json.load(open(f"{EXP}/llama32_mmlu5pct_run_state.json"))
    ad = stt["adapters"]
    ne = sum(1 for x in ad.values() if x.get("evaluated"))
    if ne != 35 or not stt.get("base_eval"):
        raise SystemExit(f"refusing to unseal: {ne}/35 adapter evals, base_eval={stt.get('base_eval')}")
    if not all(x["n_subtasks"] == 57 for x in ad.values() if x.get("evaluated")):
        raise SystemExit("refusing to unseal: a cell is not 57 MMLU subtasks")

    base = stt["base_cell"]
    rep = {"arm": "Llama-3.2-3B x MMLU 5% (stop-rule amendment #1)",
           "prereg": "experiments/less_aligned/prereg_llama32_mmlu5pct.md",
           "primary_metric": "balanced MMLU = (mmlu_stem + mmlu_humanities) / 2",
           "statistical_unit": ("the FIVE draw-index blocks; the stem-majority and hum-majority "
                                "directions are averaged WITHIN an index first. n=10 is NOT claimed "
                                "-- the two directions share a training seed, and Random-K shares "
                                "one adapter between them."),
           "base_noSFT": {"mmlu": base["_mmlu_sealed"], "stem": base["_stem_sealed"],
                          "hum": base["_hum_sealed"],
                          "balanced": (base["_stem_sealed"] + base["_hum_sealed"]) / 2},
           "per_direction": {}, "per_block": {}, "methods": {}}

    bal_base = rep["base_noSFT"]["balanced"]

    # direction-level values (what each adapter actually scored)
    for direction in DIRECTIONS:
        for i in IDX:
            for m in METHODS:
                x = ad[aid_of(direction, i, m)]
                bal = (x["_stem_sealed"] + x["_hum_sealed"]) / 2
                rep["per_direction"][f"{direction}_draw{i}_{m}"] = {
                    "adapter_id": x["adapter_id"], "mmlu": x["_mmlu_sealed"],
                    "stem": x["_stem_sealed"], "hum": x["_hum_sealed"], "balanced": bal}

    # the five draw-index blocks: average the two directions first
    blocks = {m: [] for m in METHODS}
    for i in IDX:
        row = {}
        for m in METHODS:
            vals = [rep["per_direction"][f"{d}_draw{i}_{m}"]["balanced"] for d in DIRECTIONS]
            mu = st.mean(vals)
            blocks[m].append(mu)
            row[m] = {"direction_values": vals, "block_mean": mu, "delta_vs_base": mu - bal_base}
        row["best_method_this_block"] = max(METHODS, key=lambda m: row[m]["block_mean"])
        rep["per_block"][str(i)] = row

    for m in METHODS:
        b = blocks[m]
        rep["methods"][m] = {"per_block": b, "mean_over_blocks": st.mean(b),
                             "delta_vs_base": st.mean(b) - bal_base,
                             "sd_over_blocks": st.stdev(b), "n_blocks": len(b)}

    # the three pre-registered comparisons, as PAIRED block differences
    rep["comparisons"] = {}
    for name, other in (("delta_rep_dsmc_minus_first_rr", "first_rr"),
                        ("delta_mmd_dsmc_minus_second_rr", "second_rr"),
                        ("delta_rand_dsmc_minus_randk", "randk")):
        diffs = [blocks["dsmc"][k] - blocks[other][k] for k in range(len(IDX))]
        rep["comparisons"][name] = {
            "per_block_diff": diffs, "mean_diff": st.mean(diffs),
            "mean_diff_pp": round(st.mean(diffs) * 100, 3),
            "blocks_favouring_dsmc": sum(1 for x in diffs if x > 0), "n_blocks": len(diffs)}

    order = sorted(METHODS, key=lambda m: -rep["methods"][m]["mean_over_blocks"])
    rep["ranking_best_to_worst"] = order
    rep["all_below_base"] = all(rep["methods"][m]["delta_vs_base"] < 0 for m in METHODS)

    # which pre-registered outcome fired
    d_first = rep["comparisons"]["delta_rep_dsmc_minus_first_rr"]["mean_diff"]
    d_second = rep["comparisons"]["delta_mmd_dsmc_minus_second_rr"]["mean_diff"]
    d_rand = rep["comparisons"]["delta_rand_dsmc_minus_randk"]["mean_diff"]
    TIE = 0.002                                          # 0.2 pp, declared here explicitly
    if d_first > 0 and d_second > TIE:
        fired = ("1: DSMC > Second-RR and > First-RR -- the positive MMLU method result transfers "
                 "across two model stacks")
    elif abs(d_second) <= TIE and d_first > 0:
        fired = ("2: DSMC ~ Second-RR, both > First-RR -- what replicates is the SECOND-ORDER "
                 "REPRESENTATION, not the extra MMD-coreset gain; scope the DSMC claim accordingly")
    else:
        fired = ("3: First-RR or Second-RR >= DSMC -- DSMC's MMLU method advantage is itself "
                 "model-stack dependent; lower the method claim further")
    rep["outcome"] = {
        "fired": fired,
        "tie_threshold_pp": TIE * 100,
        "tie_threshold_note": ("the prereg wrote outcome 2 as 'DSMC ~ Second-RR' without a numeric "
                               "band; 0.2 pp is declared here as the reading rule and applies "
                               "symmetrically, so it cannot be tuned to favour an interpretation"),
        "outcome_4_random_ge_targeted": all(
            rep["methods"]["randk"]["mean_over_blocks"] >= rep["methods"][m]["mean_over_blocks"]
            for m in TARGETED),
        "binding": ("no outcome may trigger tuning, a 1% follow-up, a new method, or any change to "
                    "the paper's central framing")}
    json.dump(rep, open(a.out, "w"), indent=2)

    b = rep["base_noSFT"]
    print(f"no-SFT base: balanced {b['balanced']:.4f}  (stem {b['stem']:.4f}  hum {b['hum']:.4f}  "
          f"mmlu {b['mmlu']:.4f})\n")
    print(f"{'method':12s} " + " ".join(f"{'blk'+str(i):>8s}" for i in IDX) +
          f" {'mean':>8s} {'d base':>9s}")
    for m in order:
        v = rep["methods"][m]
        print(f"{m:12s} " + " ".join(f"{x:8.4f}" for x in v["per_block"]) +
              f" {v['mean_over_blocks']:8.4f} {v['delta_vs_base']:+9.4f}")
    print()
    for name, c in rep["comparisons"].items():
        print(f"{name:34s} {c['mean_diff_pp']:+7.3f} pp   "
              f"{c['blocks_favouring_dsmc']}/{c['n_blocks']} blocks favour DSMC   "
              f"{[round(x*100,2) for x in c['per_block_diff']]}")
    print(f"\nall methods below base: {rep['all_below_base']}")
    print(f"OUTCOME {rep['outcome']['fired']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
