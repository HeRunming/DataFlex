#!/usr/bin/env python3
"""
Generate the 10 globally-disjoint skewed target draws (FROZEN protocol, review_0729→choice_0730_3).
Artifact generation ONLY: writes draw JSONL + per-draw meta + allocation matrix + overlap matrix.
No gradients / no selection / no SFT.

Design (see experiments/less_aligned/target_draw_protocol.md):
  * n_T=64, rho=0.8 -> 51 majority + 13 minority per draw. 5 stem80 + 5 hum80 draws.
  * subject composition per block from scripts/allocate_target_subjects.py (exact 320/domain,
    micro-weighted, caps respected). All 5 majority blocks of a domain share the same per-subject
    counts, likewise the 5 minority blocks (deterministic largest-remainder) -> draws differ in
    WHICH validation examples fill each subject quota, held at P*.
  * GLOBAL disjointness: for each subject, shuffle its validation pool once (master seed) and hand
    out consecutive, non-overlapping slices to every block that needs that subject (across BOTH
    domains and both maj/min roles). Feasible because total per-subject demand <= reservoir cap.
  * training seed per draw index: 42,1,2,3,4 (shared by all methods in a draw).
  * RR permutation seed per draw: 3000+draw_id.
Prompt/answer format identical to data/mmlu_target_stem80.jsonl.
"""
import argparse, glob, json, os, hashlib
from datasets import Dataset

CACHE = os.path.expanduser("~/.cache/huggingface/datasets/hails___mmlu_no_train")
LETTERS = ["A", "B", "C", "D"]
TRAIN_SEEDS = [42, 1, 2, 3, 4]        # per draw index
DIRECTIONS = ["stem80", "hum80"]      # dir_id 0,1


def subj_pretty(s):
    return s.replace("_", " ")


def load_val(subj):
    fs = glob.glob(f"{CACHE}/{subj}/*/*/mmlu_no_train-validation.arrow")
    if not fs:
        raise FileNotFoundError(subj)
    d = Dataset.from_file(fs[0])
    return [{"question": d[i]["question"], "choices": list(d[i]["choices"]),
             "answer": int(d[i]["answer"]), "subject": subj, "val_row": i} for i in range(len(d))]


def make_example(rec, idx):
    o = rec["choices"]
    prompt = (f"The following are multiple choice questions (with answers) about {subj_pretty(rec['subject'])}.\n\n"
              f"{rec['question']}\nA. {o[0]}\nB. {o[1]}\nC. {o[2]}\nD. {o[3]}\nAnswer:")
    return {"dataset": "mmlu_target", "id": f"mmlu_{rec['subject']}_val{rec['val_row']}",
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": f" {LETTERS[rec['answer']]}"}]}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/target_draws/subject_allocation_plan.json")
    ap.add_argument("--out_dir", default="data/target_draws")
    ap.add_argument("--master_seed", type=int, default=20260730)
    args = ap.parse_args()
    import random
    plan = json.load(open(args.plan))
    os.makedirs(args.out_dir, exist_ok=True)

    # per-domain: alloc[b][s], b in 0..9 (0-4 majority size51, 5-9 minority size13)
    # Global slice bookkeeping: for each subject, a cursor into its shuffled val pool.
    # Build total demand per subject across BOTH domains to verify feasibility, then slice.
    pools = {}            # subject -> shuffled list of records
    cursor = {}           # subject -> next unused index
    rng = random.Random(args.master_seed)

    # gather all subjects referenced
    demand = {}           # subject -> total examples needed across all 10+10 blocks
    for dom in ["STEM", "HUM"]:
        subs = plan[dom]["subjects"]; alloc = plan[dom]["alloc"]
        for b in range(len(alloc)):
            for s, subj in enumerate(subs):
                demand[subj] = demand.get(subj, 0) + alloc[b][s]
    for subj, need in demand.items():
        recs = load_val(subj)
        if need > len(recs):
            raise RuntimeError(f"{subj}: demand {need} > val pool {len(recs)}")
        rng.shuffle(recs)
        pools[subj] = recs; cursor[subj] = 0

    # Draw assembly. Draw d in a direction = majority block d (size51) of that direction's majority
    # domain + minority block d (size13) of the other domain.
    #   stem80 draw d: STEM majority block d (rows 0-4) + HUM minority block d (rows 5-9)
    #   hum80  draw d: HUM  majority block d (rows 0-4) + STEM minority block d (rows 5-9)
    def take(subj, n):
        c = cursor[subj]; out = pools[subj][c:c+n]; cursor[subj] = c + n
        if len(out) != n:
            raise RuntimeError(f"{subj}: ran out (need {n})")
        return out

    all_ids = {}          # (dir,draw) -> set of ids, for overlap matrix
    draws_meta = []
    for dir_id, direction in enumerate(DIRECTIONS):
        maj_dom = "STEM" if direction == "stem80" else "HUM"
        min_dom = "HUM" if direction == "stem80" else "STEM"
        for d in range(5):
            recs = []
            comp = {}
            # majority block d of maj_dom (block index d, size 51)
            for dom, blk in [(maj_dom, d), (min_dom, 5 + d)]:
                subs = plan[dom]["subjects"]; alloc = plan[dom]["alloc"]
                for s, subj in enumerate(subs):
                    n = alloc[blk][s]
                    if n:
                        recs += take(subj, n); comp[subj] = comp.get(subj, 0) + n
            assert len(recs) == 64, f"{direction} draw{d}: {len(recs)} != 64"
            maj_ct = sum(comp[s] for s in plan[maj_dom]["subjects"] if s in comp)
            min_ct = sum(comp[s] for s in plan[min_dom]["subjects"] if s in comp)
            assert maj_ct == 51 and min_ct == 13, f"{direction} draw{d}: {maj_ct}/{min_ct}"
            exs = [make_example(r, i) for i, r in enumerate(recs)]
            ids = [e["id"] for e in exs]
            assert len(set(ids)) == 64
            all_ids[(direction, d)] = set(ids)
            name = f"{direction}_draw{d}"
            with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in exs) + "\n")
            meta = {"name": name, "direction": direction, "draw_id": d,
                    "n_total": 64, "n_majority": 51, "n_minority": 13,
                    "majority_domain": maj_dom, "minority_domain": min_dom,
                    "train_seed": TRAIN_SEEDS[d], "rr_perm_seed": 3000 + d,
                    "master_seed": args.master_seed,
                    "subject_composition": comp,
                    "target_ids": ids, "target_ids_sha256": sha(sorted(ids))}
            json.dump(meta, open(os.path.join(args.out_dir, f"{name}.meta.json"), "w"), indent=2)
            draws_meta.append(meta)
            print(f"{name}: 64 ex ({maj_ct} {maj_dom}/{min_ct} {min_dom}), "
                  f"train_seed={TRAIN_SEEDS[d]} rr_seed={3000+d} subjects={len(comp)}")

    # overlap matrix (example-ID Jaccard) — expect 0 off-diagonal
    keys = [f"{dr}_draw{d}" for dr in DIRECTIONS for d in range(5)]
    setmap = {f"{dr}_draw{d}": all_ids[(dr, d)] for dr in DIRECTIONS for d in range(5)}
    mat = [[round(len(setmap[a] & setmap[b]) / len(setmap[a] | setmap[b]), 4) for b in keys] for a in keys]
    max_off = max(mat[i][j] for i in range(len(keys)) for j in range(len(keys)) if i != j)
    with open(os.path.join(args.out_dir, "overlap_matrix.csv"), "w") as f:
        f.write("," + ",".join(keys) + "\n")
        for i, a in enumerate(keys):
            f.write(a + "," + ",".join(str(x) for x in mat[i]) + "\n")
    print(f"\noverlap matrix -> {args.out_dir}/overlap_matrix.csv  max off-diagonal Jaccard = {max_off}")
    assert max_off == 0.0, f"draws NOT globally disjoint! max off-diagonal {max_off}"
    # total unique ids
    allset = set().union(*setmap.values())
    print(f"total target examples = {sum(len(s) for s in setmap.values())}, unique = {len(allset)} "
          f"(disjoint={'YES' if len(allset)==640 else 'NO'})")
    json.dump({"draws": [m["name"] for m in draws_meta], "master_seed": args.master_seed,
               "max_offdiag_jaccard": max_off, "total_unique": len(allset)},
              open(os.path.join(args.out_dir, "draws_index.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
