#!/usr/bin/env python3
"""Build skewed multi-mode MMLU target sets (80/20) from MMLU dev csvs.
Majority/minority = STEM vs Humanities (official Hendrycks grouping).
Format matches data/mmlu_target.jsonl (sharegpt, "...about {subject}\n\n{Q}\nA..\nAnswer:" / " {letter}").
No test leakage: T is drawn from dev only.
"""
import csv, json, random, os, glob

DEV = "/jizhicfs/karonhe/less_data_zip/data/eval/mmlu/dev"
OUT = "/jizhicfs/karonhe/DataFlex_fa/data"

STEM = ["abstract_algebra","anatomy","astronomy","college_biology","college_chemistry",
"college_computer_science","college_mathematics","college_physics","computer_security",
"conceptual_physics","electrical_engineering","elementary_mathematics","high_school_biology",
"high_school_chemistry","high_school_computer_science","high_school_mathematics",
"high_school_physics","high_school_statistics","machine_learning"]
HUM = ["formal_logic","high_school_european_history","high_school_us_history",
"high_school_world_history","international_law","jurisprudence","logical_fallacies",
"moral_disputes","moral_scenarios","philosophy","prehistory","professional_law","world_religions"]


def subj_pretty(s):
    return s.replace("_", " ")


def load_subject(subj):
    """Return list of (question, [A,B,C,D], answer_letter) for a subject's dev csv."""
    p = os.path.join(DEV, f"{subj}_dev.csv")
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 6:
                continue
            q = r[0]; opts = r[1:5]; ans = r[5].strip()
            rows.append((q, opts, ans, subj))
    return rows


def make_example(q, opts, ans, subj, idx):
    prompt = (f"The following are multiple choice questions (with answers) about {subj_pretty(subj)}.\n\n"
              f"{q}\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer:")
    return {"dataset": "mmlu_target", "id": f"mmlu_{subj}_{idx}",
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": f" {ans}"}]}


def sample_group(subjects, n, seed):
    """Sample n examples spread across the given subjects (round-robin then random)."""
    pool = []
    for s in subjects:
        pool.extend(load_subject(s))
    random.Random(seed).shuffle(pool)
    return pool[:n]


def build(name, maj_subjects, min_subjects, n_maj, n_min, seed):
    maj = sample_group(maj_subjects, n_maj, seed)
    mino = sample_group(min_subjects, n_min, seed + 1)
    allex = maj + mino
    random.Random(seed + 2).shuffle(allex)
    out = os.path.join(OUT, f"{name}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for i, (q, opts, ans, subj) in enumerate(allex):
            f.write(json.dumps(make_example(q, opts, ans, subj, i), ensure_ascii=False) + "\n")
    print(f"{name}: {len(maj)} majority + {len(mino)} minority = {len(allex)} -> {out}")
    return out


if __name__ == "__main__":
    SEED = 42
    # Both T at exactly 80/20, total=80 (64 maj + 16 min). Capped so both the
    # STEM-major and HUM-major sets fit within dev (HUM max = 13*5 = 65).
    # T_stem80 = 64 STEM (majority) + 16 HUM (minority)
    build("mmlu_target_stem80", STEM, HUM, 64, 16, SEED)
    # T_hum80  = 64 HUM (majority) + 16 STEM (minority)
    build("mmlu_target_hum80", HUM, STEM, 64, 16, SEED)
