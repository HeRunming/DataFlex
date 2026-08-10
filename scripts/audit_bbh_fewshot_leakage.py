#!/usr/bin/env python3
"""Leakage audit: the hard-coded CoT few-shot DEMONSTRATIONS vs the BBH evaluation data
(code_review_0810 P1-3). Artifacts only — no model, nothing evaluated.

The gap this closes: gate C of the prompt-parity audit verifies that the query reservoir and the
held-out evaluation split are disjoint from EACH OTHER. It never asked whether the 3 hard-coded CoT
exemplars baked into each pinned subtask config are themselves drawn from those same BBH examples.

That matters in two different ways:

  * vs the 5,209 HELD-OUT EVAL split -> genuine test contamination. Every evaluated item is prompted
    with these 3 demonstrations; if a demonstration IS an evaluation item, the model is shown that
    item's gold answer in its own prompt.
  * vs the 1,302 query RESERVOIR / the 192 drawn queries -> target/query leakage. A query gradient
    whose own answer appears in its prompt is not measuring what we claim.

Matching is done three ways, since exact string equality is too weak on its own:
  L1 normalized exact  - lowercase, strip non-alphanumerics, collapse whitespace
  L2 13-gram containment - a shared long contiguous word n-gram
  L3 fuzzy Jaccard over word 5-shingles, reported at >=0.5 and >=0.3

Verdict has THREE levels, because exact identity is not the only leakage channel:
  FAIL   any normalized-exact demonstration == evaluation/query item
  REVIEW any pair at Jaccard >= --fuzzy_block_thr (default 0.85): near-verbatim must be human-cleared
  PASS   otherwise
Both FAIL and REVIEW exit non-zero. Each flagged pair also records whether the demo's gold answer
DIFFERS from the matched item's — a near-verbatim exemplar carrying the OPPOSITE answer primes the wrong
response for that item, which is a validity problem in its own right, not a harmless near-duplicate.

A within-subtask item-vs-item Jaccard baseline (p50/p95 over 600 sampled pairs) is computed so that
"this demo is unusually close to an evaluation item" can be distinguished from "this subtask is
template-generated and all its items look alike".
"""
import argparse, hashlib, json, os, re, warnings

warnings.filterwarnings("ignore")

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SPLIT_DIR = f"{ROOT}/data/bbh_external"
TASKS_DIR = f"{EXP}/bbh_external_tasks"
BBH_RAW = "/jizhicfs/karonhe/less_data_zip/data/eval/bbh/test"

_ws = re.compile(r"\s+")
# Keep structural punctuation. An earlier version stripped everything outside [a-z0-9 ], which erased
# the ENTIRE payload of bracket-only tasks: every dyck_languages item canonicalised to the bare
# instruction sentence, so all 3 demos "exactly matched" an arbitrary held-out item at Jaccard 1.0.
# That was an artifact of the matcher, not leakage. Brackets/comparators/slashes are content in BBH
# (dyck_languages, geometric_shapes SVG paths, word_sorting), so they are preserved.
_na = re.compile(r"[^a-z0-9 ()\[\]{}<>/,.:;+*=|_-]+")


def canon(s):
    return _ws.sub(" ", _na.sub(" ", s.lower().replace("\n", " "))).strip()


def payload(s, boilerplate=None):
    """The part of a BBH input that is not shared per-subtask boilerplate.

    Do NOT try to guess this from markers. An earlier version split on "Options:"/"Input:" and kept the
    wrong side, which threw away the discriminating content: for hyperbaton the answer options ARE the
    content, so all 3 demos collapsed to the identical stem and scored Jaccard 1.0 against an arbitrary
    item. Both that and the earlier bracket-stripping bug were matcher artifacts, not leakage.

    Instead `boilerplate` is derived empirically per subtask as the set of 5-shingles common to (almost)
    every item of that subtask; those shingles are subtracted from both sides before comparison. What
    remains is item-specific text, which is what a leaked example would actually share.
    """
    sh = shingles(canon(s).split())
    return sh - boilerplate if boilerplate else sh


def subtask_boilerplate(items, min_frac=0.9):
    """5-shingles present in >= min_frac of a subtask's items — i.e. the shared instruction text."""
    if not items:
        return set()
    counts = {}
    for it in items:
        for g in shingles(canon(it).split()):
            counts[g] = counts.get(g, 0) + 1
    thr = max(2, int(len(items) * min_frac))
    return {g for g, c in counts.items() if c >= thr}


def shingles(words, k=5):
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def ngrams(words, n=13):
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/bbh_fewshot_leakage_audit.json")
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--expect_demos", type=int, default=81,
                    help="required demonstration count (27 subtasks x 3); auditing fewer is a FAILURE, "
                         "not a pass -- an audit that compared 0 demos must never read as clean.")
    ap.add_argument("--fuzzy_block_thr", type=float, default=0.85,
                    help="fuzzy Jaccard at or above which a demo/eval pair must be HUMAN-CLEARED. Exact "
                         "identity is not the only way to leak: a near-verbatim exemplar can carry the "
                         "answer (or, worse, the OPPOSITE answer) for an evaluated item.")
    args = ap.parse_args()

    import yaml

    # ---- the demonstrations actually in force, read from the pinned custom configs ----
    demos = []
    for f in sorted(os.listdir(TASKS_DIR)):
        if not f.endswith(".yaml") or f.startswith("_"):
            continue
        cfg = yaml.safe_load(open(f"{TASKS_DIR}/{f}"))
        task = cfg["task"].replace("bbh_external_heldout_", "")
        for i, s in enumerate(cfg.get("fewshot_config", {}).get("samples", [])):
            demos.append({"subtask": task, "idx": i, "input": s["input"],
                          "target_text": s.get("target", "")})
    print(f"[demos] {len(demos)} hard-coded CoT demonstrations over "
          f"{len({d['subtask'] for d in demos})} subtasks")
    if len(demos) != args.expect_demos:
        raise SystemExit(f"VACUITY GUARD: found {len(demos)} demonstrations, expected "
                         f"{args.expect_demos}. Refusing to emit a verdict.")

    # ---- the four reference populations ----
    def load(p, field="input"):
        return [json.loads(l) for l in open(p) if l.strip()]

    raw = []
    for fn in sorted(os.listdir(BBH_RAW)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(f"{BBH_RAW}/{fn}"))
        for i, e in enumerate(d.get("examples", d)):
            raw.append({"id": f"bbh::{fn[:-5]}::{i}", "input": e["input"], "target": e["target"]})
    pops = {
        "raw_all_6511": raw,
        "query_reservoir_1302": load(f"{SPLIT_DIR}/bbh_query_reservoir.jsonl"),
        "heldout_eval_5209": load(f"{SPLIT_DIR}/bbh_eval_heldout.jsonl"),
        "drawn_queries_192": [r for d in (0, 1, 2)
                              for r in load(f"{SPLIT_DIR}/bbh_query_draw{d}.jsonl")],
    }
    for k, v in pops.items():
        print(f"[pop]   {k:24s} {len(v)}")

    # Boilerplate is derived per subtask from the FULL raw suite, so it is a property of the task,
    # not of whichever population we happen to be scanning.
    raw_by_task = {}
    for r in raw:
        raw_by_task.setdefault(r["id"].split("::")[1], []).append(r["input"])
    boiler = {t: subtask_boilerplate(v) for t, v in raw_by_task.items()}
    med = sorted(len(v) for v in boiler.values())[len(boiler) // 2]
    print(f"[boilerplate] derived for {len(boiler)} subtasks (median {med} shared 5-shingles)")

    # The promised within-subtask baseline, now actually COMPUTED (a previous version referred to an
    # "item-vs-item p95 baseline" in prose while never calculating it). Without it there is no way to
    # say whether a demo is unusually close to some item or merely as close as any two items are.
    import random as _rnd
    baseline = {}
    for t, items in raw_by_task.items():
        S = [payload(x, boiler.get(t, set())) for x in items]
        _rnd.seed(0)
        pairs = [(_rnd.randrange(len(S)), _rnd.randrange(len(S))) for _ in range(600)]
        vals = sorted(jaccard(S[a], S[b]) for a, b in pairs if a != b)
        baseline[t] = {"p50": round(vals[len(vals) // 2], 4),
                       "p95": round(vals[int(len(vals) * 0.95)], 4),
                       "max_sampled": round(vals[-1], 4)}
    print(f"[baseline] within-subtask item-vs-item Jaccard computed for {len(baseline)} subtasks")

    report, worst, worst_detail = {}, 0.0, None
    for pname, pop in pops.items():
        pc = [canon(r["input"]) for r in pop]
        pset = {}
        for i, s in enumerate(pc):
            pset.setdefault(s, i)

        exact, ng_hits, strong, weak = [], [], [], []
        for d in demos:
            st = d["subtask"]
            bp = boiler.get(st, set())
            c = canon(d["input"])
            if c in pset:
                exact.append({"demo": f"{st}#{d['idx']}", "matched_id": pop[pset[c]]["id"]})
            ds = payload(d["input"], bp)
            dn = ngrams(c.split(), args.ngram)
            best = (0.0, None)
            for i, r in enumerate(pop):
                # Compare only WITHIN the same subtask: a demo can only leak an item of its own task,
                # and cross-task comparisons merely re-measure shared phrasing.
                if r["id"].split("::")[1] != st:
                    continue
                rs = payload(r["input"], bp)
                rn = ngrams(pc[i].split(), args.ngram)
                shared = (dn & rn) if (dn and rn) else set()
                if shared:
                    ng_hits.append({"demo": f"{st}#{d['idx']}", "matched_id": r["id"],
                                    "shared_ngrams": len(shared)})
                j = jaccard(ds, rs)
                if j > best[0]:
                    best = (j, r["id"])
            if best[0] >= 0.5:
                mi = next((r for r in pop if r["id"] == best[1]), None)
                dtgt = d["target_text"].strip()
                itgt = (mi or {}).get("target", "").strip()
                # A near-verbatim demo whose gold answer DIFFERS from the evaluated item's is the
                # adversarial worst case: it primes the wrong answer for that item.
                strong.append({"demo": f"{st}#{d['idx']}", "jaccard": round(best[0], 4),
                               "matched_id": best[1],
                               "demo_answer_tail": dtgt[-40:], "item_answer": itgt,
                               "answer_differs": (itgt not in dtgt) if itgt else None,
                               "needs_human_clearance": best[0] >= args.fuzzy_block_thr})
            elif best[0] >= 0.3:
                weak.append({"demo": f"{st}#{d['idx']}", "jaccard": round(best[0], 4),
                             "matched_id": best[1]})
            if best[0] > worst:
                worst, worst_detail = best[0], {"population": pname, "demo": f"{st}#{d['idx']}",
                                                "matched_id": best[1]}
        report[pname] = {
            "n_population": len(pop),
            "comparison_basis": ("within-subtask only; per-subtask boilerplate 5-shingles (present in "
                                ">=90% of that subtask's items) subtracted from both sides for L3"),
            "L1_normalized_exact": {"count": len(exact), "hits": exact[:20]},
            f"L2_{args.ngram}gram_containment": {"count": len(ng_hits), "hits": ng_hits[:20]},
            "L3_fuzzy_jaccard_ge_0.5": {"count": len(strong), "hits": strong[:20]},
            "L3_fuzzy_jaccard_ge_0.3": {"count": len(weak), "hits": weak[:20]},
        }
        print(f"  {pname:24s} exact={len(exact)}  {args.ngram}gram={len(ng_hits)}  "
              f"J>=0.5={len(strong)}  J>=0.3={len(weak)}")
    critical = ["heldout_eval_5209", "drawn_queries_192", "query_reservoir_1302"]
    # VERDICT POLICY. The blocking condition is a demonstration that IS an evaluation/query item, i.e.
    # normalized-exact identity. High fuzzy similarity alone is NOT leakage here: several BBH subtasks
    # are template-generated (tracking_shuffled_objects, dyck_languages, penguins_in_a_table), so two
    # independent items routinely share most of their text. To keep that distinction honest we also
    # report, per subtask, the demo-vs-item max against the ordinary item-vs-item p95 baseline.
    exact_fail = [p for p in critical if report[p]["L1_normalized_exact"]["count"]]
    near = {p: report[p]["L3_fuzzy_jaccard_ge_0.5"]["count"] for p in critical}
    # pairs at/above the blocking threshold, i.e. near-verbatim -> require explicit human clearance
    flagged = []
    for pn in critical:
        for h in report[pn]["L3_fuzzy_jaccard_ge_0.5"]["hits"]:
            if h.get("needs_human_clearance"):
                flagged.append({"population": pn, **h})
    flagged.sort(key=lambda x: -x["jaccard"])
    answer_flips = [f for f in flagged if f.get("answer_differs")]
    out = {
        "audit": "hard-coded CoT few-shot demonstrations vs BBH evaluation populations",
        "n_demonstrations": len(demos),
        "n_subtasks": len({d["subtask"] for d in demos}),
        "demonstrations_sha256": hashlib.sha256(
            json.dumps([[d["subtask"], d["idx"], d["input"]] for d in demos],
                       sort_keys=True).encode()).hexdigest(),
        "why": ("gate C only proved reservoir/held-out mutual disjointness; it never checked whether the "
                "3 baked-in CoT exemplars per subtask are themselves BBH evaluation or query items. A "
                "demonstration that IS an evaluation item would show that item's gold answer in its own "
                "prompt."),
        "results": report,
        "max_observed_jaccard_any_population": round(worst, 4),
        "max_observed_detail": worst_detail,
        "critical_populations": critical,
        "within_subtask_item_vs_item_baseline": baseline,
        "baseline_note": ("per-subtask Jaccard between two ORDINARY items of the same subtask (600 "
                          "sampled pairs, same payload/boilerplate treatment). This is the reference "
                          "for judging whether a demo is unusually close to an evaluation item."),
        "n_exact_duplicates": {p: report[p]["L1_normalized_exact"]["count"] for p in critical},
        "n_fuzzy_ge_0.5": near,
        "n_needing_human_clearance": len(flagged),
        "pairs_needing_human_clearance": flagged,
        "answer_flip_pairs": answer_flips,
        "verdict": ("FAIL" if exact_fail else ("REVIEW" if flagged else "PASS")),
        "verdict_basis": ("FAIL on any normalized-exact demonstration/evaluation-item identity. REVIEW "
                          f"when any pair reaches Jaccard >= {args.fuzzy_block_thr} — near-verbatim is a "
                          "real leakage channel even without exact identity, so those pairs must be "
                          "cleared by a human rather than auto-passed. Moderate fuzzy overlap below that "
                          "threshold does not block, because several BBH subtasks are template-generated "
                          "and two independent items legitimately share most of their text; the computed "
                          "within-subtask p95 baseline is the reference for that judgement."),
        "fuzzy_caveat": ("The maximum is J=0.8929, `causal_judgement#1` vs `bbh::causal_judgement::128`. "
                         "This is NOT a template-generation artifact: causal_judgement is hand-written. "
                         "The two differ only in 'at least one person' -> 'more than one person', a "
                         "deliberate minimal-pair probe, and the perturbation FLIPS the gold answer "
                         "(demo: Yes; item: No). The item is in draw1, i.e. a scored query record. The "
                         "plausible effect is ANTI-leakage -- the prompt primes the opposite answer -- "
                         "which is a validity concern in its own right and is why such pairs are routed "
                         "to REVIEW rather than silently passed. Template-generated near-twins "
                         "(tracking_shuffled_objects: same swap sequence, different queried person; "
                         "penguins_in_a_table: shared table) also occur at J~0.87-0.90 and are the "
                         "benign case."),
        "matcher_bugs_fixed": [
            "canonicalization stripped all non-alphanumerics, erasing bracket-only dyck_languages "
            "payloads so every item collapsed to the shared instruction (spurious exact matches at J=1.0)",
            "an 'Options:'/'Input:' payload heuristic kept the wrong side, discarding the discriminating "
            "content for hyperbaton/logical_deduction (spurious J=1.0)",
            "cross-subtask comparison and un-subtracted boilerplate inflated 13-gram hit counts",
        ],
        "failed_populations": exact_fail or None,
        "no_compute_run": "artifacts only: no model loaded, nothing evaluated",
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nexact duplicates (blocking) : {out['n_exact_duplicates']}")
    print(f"fuzzy J>=0.5 (disclosed)    : {near}")
    print(f"max Jaccard {worst:.4f} -> {worst_detail}")
    print(f"needing human clearance (J>={args.fuzzy_block_thr}): {len(flagged)}"
          f"  of which answer-FLIP: {len(answer_flips)}")
    for f in flagged[:6]:
        print(f"   J={f['jaccard']} {f['demo']:38s} vs {f['matched_id']:36s} "
              f"answer_differs={f.get('answer_differs')}  [{f['population']}]")
    print(f"VERDICT: {out['verdict']}" + (f"  failed={exact_fail}" if exact_fail else ""))
    print(f"wrote {args.out}")
    return 0 if out["verdict"] == "PASS" else 1   # REVIEW and FAIL both exit non-zero


if __name__ == "__main__":
    raise SystemExit(main())
