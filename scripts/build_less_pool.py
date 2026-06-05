#!/usr/bin/env python3
"""
Build the LESS-paper 270K Tulu-V2 candidate pool as data/less_train_all.jsonl
(sharegpt format, one JSON record per line with a `messages` field).

The four princeton-nlp/less_data processed JSONL files are ALREADY in the
sharegpt `messages` schema, so this is a lossless concatenation (keeps
multi-turn structure, plus the original dataset/id fields). This matches the
`less_train_all` entry registered in data/dataset_info.json on the `fa` branch.

Source (unzip princeton-nlp/less_data -> less-data.zip):
    {LESS_ROOT}/flan_v2/flan_v2_data.jsonl   (100000)
    {LESS_ROOT}/cot/cot_data.jsonl           (100000)
    {LESS_ROOT}/dolly/dolly_data.jsonl       ( 15011)
    {LESS_ROOT}/oasst1/oasst1_data.jsonl     ( 55668)
Total ~270,679 examples.
"""
import json
import sys
from pathlib import Path

LESS_ROOT = Path("/jizhicfs/karonhe/less_data_zip/data/train/processed")
SOURCES = [
    ("flan_v2", LESS_ROOT / "flan_v2/flan_v2_data.jsonl"),
    ("cot",     LESS_ROOT / "cot/cot_data.jsonl"),
    ("dolly",   LESS_ROOT / "dolly/dolly_data.jsonl"),
    ("oasst1",  LESS_ROOT / "oasst1/oasst1_data.jsonl"),
]
OUTPUT_PATH = Path("/jizhicfs/karonhe/DataFlex/data/less_train_all.jsonl")


def main():
    counts = {}
    total = 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for name, path in SOURCES:
            if not path.exists():
                print(f"[error] missing source: {path}", file=sys.stderr)
                sys.exit(1)
            kept = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)  # validate
                    msgs = rec.get("messages")
                    if not msgs or len(msgs) < 2:
                        continue
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
            counts[name] = kept
            total += kept
            print(f"  [{name}] kept={kept}", flush=True)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"[done] {total} examples -> {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"[breakdown] {counts}")


if __name__ == "__main__":
    main()
