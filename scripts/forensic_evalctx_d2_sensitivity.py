#!/usr/bin/env python3
"""POST-HOC SENSITIVITY ANALYSIS: is the D2 ranking an artifact of target prompt serialization?
(writing_advice_0819, the one optional diagnostic.)

Target-side gradients ONLY. No new adapters, no new selections, no SFT, no change to any primary
result. The frozen subsets are re-scored against target gradients extracted under the BARE /
evaluation-matched serialization (LlamaFactory `empty` template) instead of the operational chat
wrapper (`llama2` / `llama3`).

  D2_evalctx(S, Q_d) = ||M_S - M_{Q_d}^{bare}||_F^2

on unit-normalized 8192-d projected gradients -- the identical D2 definition used everywhere else,
so the operational and eval-matched numbers are directly comparable.

The question, and both answers, fixed before running
----------------------------------------------------
  If DSMC is still lowest-D2 in 3/3 draws on both stacks while remaining worse than Random
  downstream, the geometry-utility counterexample is ROBUST to target serialization.

  If the D2 ranking changes materially, the claim narrows to the OPERATIONAL target geometry and we
  gain a concrete boundary: geometry success is itself serialization-sensitive.

Either way this is reported as a post-hoc sensitivity analysis and may NOT redefine the primary
result, and no further experiment follows from it.
"""
import argparse, json
import torch

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
DRAWS = [0, 1, 2]
METHODS = ["dsmc", "second_rr", "first_rr", "randk"]

STACKS = {
    "llama2": {"cand": f"{SAVES}/less_output/train/1/all_projected_grads.pt",
               "operational": lambda d: f"{SAVES}/draw_bbhx_draw{d}_output/target/1/all_projected_grads.pt",
               "evalctx": lambda d: f"{SAVES}/evalctx_l2_draw{d}_output/target/1/all_projected_grads.pt",
               "sel": lambda d, m: f"{SAVES}/selbbhx_draw{d}_{m}/step_1.json",
               "results": f"{EXP}/results_summary/bbh_external_results.md"},
    "llama32": {"cand": f"{SAVES}/llama32_less_output/train/1/all_projected_grads.pt",
                "operational": lambda d: f"{SAVES}/llama32_draw{d}_output/target/1/all_projected_grads.pt",
                "evalctx": lambda d: f"{SAVES}/evalctx_l32_draw{d}_output/target/1/all_projected_grads.pt",
                "sel": lambda d, m: (f"{SAVES}/selbbhx_draw{d}_randk/step_1.json" if m == "randk"
                                     else f"{SAVES}/sel_llama32_draw{d}_{m}/step_1.json"),
                "results": f"{EXP}/results_summary/llama32_results.json"},
}


def unit(X):
    n = X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X / n


def second_moment(U):
    return (U.T @ U) / U.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/evalctx_d2_sensitivity.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    rep = {"analysis": "POST-HOC sensitivity: D2 ranking under bare/evaluation-matched target "
                       "serialization vs the operational chat wrapper",
           "scope": ("target-side gradients only; no new adapters, selections or SFT. May NOT "
                     "redefine any primary result."),
           "definition": "D2(S,Q) = ||M_S - M_Q||_F^2 on unit-normalized 8192-d projected gradients",
           "serialization": {"operational": "LlamaFactory llama2 (Llama-2) / llama3 (Llama-3.2) wrapper",
                             "eval_matched": "LlamaFactory `empty` template = bare prompt, the "
                                             "serialization the task metric actually uses"},
           "stacks": {}}

    for stack, cfg in STACKS.items():
        print(f"\n===== {stack} =====", flush=True)
        Xall = torch.load(cfg["cand"], map_location="cpu").float()
        ent = {}
        for d in DRAWS:
            To = torch.load(cfg["operational"](d), map_location="cpu").float()
            Te = torch.load(cfg["evalctx"](d), map_location="cpu").float()
            if To.shape != Te.shape:
                raise SystemExit(f"{stack} draw{d}: shape mismatch {To.shape} vs {Te.shape}")
            # the two target sets must differ -- identical tensors would mean the template override
            # silently did nothing, and the whole diagnostic would be vacuous
            same = bool(torch.equal(To, Te))
            cos = float(torch.nn.functional.cosine_similarity(
                unit(To), unit(Te), dim=1).mean())
            Mo = second_moment(unit(To).to(dev))
            Me = second_moment(unit(Te).to(dev))
            row = {"target_tensors_identical": same,
                   "mean_row_cosine_operational_vs_evalctx": cos,
                   "D2_between_the_two_target_moments": float(((Mo - Me) ** 2).sum()),
                   "methods": {}}
            if same:
                raise SystemExit(f"{stack} draw{d}: eval-matched target gradients are IDENTICAL to "
                                 f"the operational ones -- the template override did nothing, so "
                                 f"this diagnostic would be vacuous")
            for m in METHODS:
                idx = json.load(open(cfg["sel"](d, m)))["indices"]
                U = unit(Xall[torch.tensor(idx)]).to(dev)
                Ms = second_moment(U)
                row["methods"][m] = {
                    "D2_operational": float(((Ms - Mo) ** 2).sum()),
                    "D2_evalctx": float(((Ms - Me) ** 2).sum())}
                del U, Ms
            for key, tag in (("D2_operational", "operational"), ("D2_evalctx", "evalctx")):
                order = sorted(METHODS, key=lambda m: row["methods"][m][key])
                row[f"ranking_{tag}_best_first"] = order
                row[f"dsmc_lowest_{tag}"] = order[0] == "dsmc"
            row["ranking_changed"] = (row["ranking_operational_best_first"]
                                      != row["ranking_evalctx_best_first"])
            ent[str(d)] = row
            print(f"[draw{d}] rowcos {cos:.4f} | operational {row['ranking_operational_best_first']}"
                  f"\n         evalctx      {row['ranking_evalctx_best_first']}"
                  f"  dsmc lowest: op={row['dsmc_lowest_operational']} "
                  f"eval={row['dsmc_lowest_evalctx']}", flush=True)
        n_op = sum(1 for d in DRAWS if ent[str(d)]["dsmc_lowest_operational"])
        n_ev = sum(1 for d in DRAWS if ent[str(d)]["dsmc_lowest_evalctx"])
        rep["stacks"][stack] = {
            "per_draw": ent,
            "dsmc_lowest_D2_operational": f"{n_op}/3",
            "dsmc_lowest_D2_evalctx": f"{n_ev}/3",
            "rankings_changed_in_n_draws": sum(1 for d in DRAWS if ent[str(d)]["ranking_changed"])}
        del Xall

    robust = all(rep["stacks"][s]["dsmc_lowest_D2_evalctx"] == "3/3" for s in STACKS)
    rep["VERDICT"] = {
        "dsmc_lowest_evalctx_both_stacks_3of3": robust,
        "READING": (
            "The D2 ranking is ROBUST to target serialization: DSMC still minimizes the "
            "eval-matched target discrepancy in 3/3 draws on BOTH model stacks, while remaining "
            "worse than Random downstream in 3/3 draws on both. The geometry-utility counterexample "
            "therefore does not depend on the chat wrapper used to extract target gradients."
            if robust else
            "The D2 ranking is NOT stable under the bare/evaluation-matched serialization. The "
            "counterexample must be stated for the OPERATIONAL target geometry, and this yields a "
            "concrete boundary: geometric success is itself serialization-sensitive."),
        "status": ("POST-HOC sensitivity analysis. It does not redefine the primary result, and no "
                   "further experiment follows from it whatever it shows."),
    }
    json.dump(rep, open(a.out, "w"), indent=2)
    print(f"\nDSMC lowest eval-matched D2: "
          + ", ".join(f"{s} {rep['stacks'][s]['dsmc_lowest_D2_evalctx']}" for s in STACKS))
    print(f"robust on both stacks: {robust}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
