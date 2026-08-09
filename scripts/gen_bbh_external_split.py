#!/usr/bin/env python3
"""BBH external-validation split + query draws (artifacts ONLY — no gradients, no selection, no SFT).

Implements the frozen design in `experiments/less_aligned/prereg_bbh_external.md` (code_review_0809):

  * official 23-task BBH suite. The local layout ships 27 files because `logical_deduction` and
    `tracking_shuffled_objects` each appear as three size-variants; we keep all 27 files but record the
    23-task grouping so no ad-hoc benchmark is invented.
  * ONE deterministic, per-task-stratified split: 20% -> query reservoir, 80% -> held-out evaluation.
    Query and evaluation are therefore exactly disjoint.
  * THREE query draws of M=64 sampled INDEPENDENTLY from the fixed reservoir (without replacement
    within a draw; overlap ACROSS draws is allowed and reported — we do not force global disjointness,
    which would induce negative correlation as it did in the MMLU design).
  * every draw is later trained under BOTH SFT seeds {42, 1} (fully crossed), so query-realization
    variance and seed variance are no longer confounded. Seeds are not attached to draws here.

Writes: bbh_split_manifest.json, bbh_eval_heldout.jsonl, bbh_query_reservoir.jsonl,
        bbh_query_draw{0,1,2}.jsonl (+ .meta.json), bbh_draw_overlap.csv
"""
import argparse, json, os, glob, hashlib, random

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
BBH_DIR = "/jizhicfs/karonhe/less_data_zip/data/eval/bbh/test"
OUT = f"{ROOT}/data/bbh_external"

# official 23-task suite -> the local file variants that compose each task
TASK_GROUPS = {
    "logical_deduction": ["logical_deduction_three_objects", "logical_deduction_five_objects",
                          "logical_deduction_seven_objects"],
    "tracking_shuffled_objects": ["tracking_shuffled_objects_three_objects",
                                  "tracking_shuffled_objects_five_objects",
                                  "tracking_shuffled_objects_seven_objects"],
}


def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def official_task(fname):
    for task, variants in TASK_GROUPS.items():
        if fname in variants:
            return task
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoir_frac", type=float, default=0.20)
    ap.add_argument("--n_draws", type=int, default=3)
    ap.add_argument("--draw_size", type=int, default=64)
    ap.add_argument("--master_seed", type=int, default=20260809)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    files = sorted(glob.glob(f"{BBH_DIR}/*.json"))
    assert len(files) == 27, f"expected 27 local BBH files, found {len(files)}"

    # ---- load, assign stable ids ----
    per_file = {}
    for f in files:
        name = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        ex = d.get("examples", d)
        rows = []
        for i, e in enumerate(ex):
            rows.append({"id": f"bbh::{name}::{i}", "file_task": name,
                         "official_task": official_task(name),
                         "input": e["input"], "target": e["target"]})
        per_file[name] = rows
    total = sum(len(v) for v in per_file.values())
    official = sorted({official_task(n) for n in per_file})
    assert len(official) == 23, f"expected 23 official tasks, got {len(official)}"
    print(f"[bbh] {len(files)} files -> {len(official)} official tasks, {total} examples")

    # ---- ONE deterministic per-task-stratified split (per FILE task, so size-variants stay balanced) ----
    rng = random.Random(args.master_seed)
    reservoir, heldout = [], []
    split_table = {}
    for name in sorted(per_file):
        rows = per_file[name][:]
        rng.shuffle(rows)
        n_res = max(1, round(len(rows) * args.reservoir_frac))
        reservoir += rows[:n_res]
        heldout += rows[n_res:]
        split_table[name] = {"official_task": official_task(name), "n_total": len(rows),
                             "n_reservoir": n_res, "n_heldout": len(rows) - n_res}
    print(f"[bbh] query reservoir {len(reservoir)}  held-out eval {len(heldout)}  "
          f"(disjoint: {len(set(r['id'] for r in reservoir) & set(h['id'] for h in heldout)) == 0})")

    # ---- three INDEPENDENT draws from the fixed reservoir ----
    draws = {}
    for d in range(args.n_draws):
        r = random.Random(args.master_seed * 1000 + d)
        draws[d] = r.sample(reservoir, args.draw_size)     # independent, w/o replacement within draw

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        return sha_file(path)

    h_eval = write_jsonl(f"{OUT}/bbh_eval_heldout.jsonl", heldout)
    h_res = write_jsonl(f"{OUT}/bbh_query_reservoir.jsonl", reservoir)

    draw_meta = {}
    for d, rows in draws.items():
        p = f"{OUT}/bbh_query_draw{d}.jsonl"
        hh = write_jsonl(p, rows)
        comp = {}
        for x in rows:
            comp[x["official_task"]] = comp.get(x["official_task"], 0) + 1
        meta = {"draw": d, "n": len(rows), "file_sha256": hh,
                "ordered_ids_sha256": sha_str(json.dumps([x["id"] for x in rows])),
                "unordered_ids_sha256": sha_str(json.dumps(sorted(x["id"] for x in rows))),
                "official_task_composition": comp,
                "sampling": "independent sample without replacement from the fixed reservoir",
                "sft_seeds_to_run": [42, 1],
                "note": "seeds are CROSSED with draws (each draw trained under both seeds)"}
        json.dump(meta, open(f"{OUT}/bbh_query_draw{d}.meta.json", "w"), indent=2)
        draw_meta[d] = meta
        print(f"[bbh] draw{d}: {len(rows)} queries over {len(comp)} official tasks  sha={hh[:10]}")

    # ---- pairwise overlap (expected small, NOT forced to zero) ----
    ids = {d: set(x["id"] for x in rows) for d, rows in draws.items()}
    with open(f"{OUT}/bbh_draw_overlap.csv", "w") as f:
        f.write("," + ",".join(f"draw{d}" for d in draws) + "\n")
        for a in draws:
            f.write(f"draw{a}," + ",".join(str(len(ids[a] & ids[b])) for b in draws) + "\n")
    off = [(a, b, len(ids[a] & ids[b])) for a in draws for b in draws if a < b]
    exp = args.draw_size ** 2 / len(reservoir)
    print(f"[bbh] pairwise draw overlap (examples): {[(f'{a}-{b}', n) for a, b, n in off]}  "
          f"(expected ≈ {exp:.1f})")

    man = {"family": "BBH", "source_dir": BBH_DIR,
           "n_local_files": len(files), "n_official_tasks": len(official),
           "official_tasks": official, "task_groups_note": TASK_GROUPS,
           "n_examples_total": total,
           "master_seed": args.master_seed, "reservoir_frac": args.reservoir_frac,
           "n_reservoir": len(reservoir), "n_heldout": len(heldout),
           "reservoir_sha256": h_res, "heldout_eval_sha256": h_eval,
           "per_task_split": split_table,
           "draws": {str(d): draw_meta[d] for d in draw_meta},
           "pairwise_overlap": {f"draw{a}-draw{b}": n for a, b, n in off},
           "expected_pairwise_overlap": exp,
           "query_eval_disjoint": True,
           "no_compute_run": "artifacts only: no gradients, no selection, no SFT"}
    json.dump(man, open(f"{OUT}/bbh_split_manifest.json", "w"), indent=2)
    print(f"\nwrote {OUT}/bbh_split_manifest.json")


if __name__ == "__main__":
    main()
