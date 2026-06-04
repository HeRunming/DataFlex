#!/usr/bin/env python3
"""
Build the LESS-paper 270K Tulu-V2 mix dataset by concatenating the four official
LESS-processed JSONL files (princeton-nlp/less_data) into one alpaca-format JSON.

Output: /jizhicfs/karonhe/DataFlex/data/tulu2_270k.json

Source files:
    /jizhicfs/karonhe/less_data_zip/data/train/processed/flan_v2/flan_v2_data.jsonl  (100K)
    /jizhicfs/karonhe/less_data_zip/data/train/processed/cot/cot_data.jsonl          (100K)
    /jizhicfs/karonhe/less_data_zip/data/train/processed/dolly/dolly_data.jsonl       (15K)
    /jizhicfs/karonhe/less_data_zip/data/train/processed/oasst1/oasst1_data.jsonl    (55.7K)

Total: ~270K examples, exactly matching LESS paper.
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

OUTPUT_PATH = Path("/jizhicfs/karonhe/DataFlex/data/tulu2_270k.json")


def messages_to_alpaca(messages):
    """LESS messages format: [{"role": "user", "content": ...}, {"role": "assistant", ...}].
    Convert to alpaca: prompt = concat all turns before final assistant; output = final assistant."""
    if not messages or len(messages) < 2:
        return None
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return None
    response = (messages[last_assistant_idx].get("content") or "").strip()
    if not response:
        return None
    parts = []
    for m in messages[:last_assistant_idx]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Previous answer]\n{content}")
        elif role == "system":
            parts.append(f"[System]\n{content}")
    prompt = "\n\n".join(parts).strip()
    if not prompt:
        return None
    return {"instruction": prompt, "input": "", "output": response}


def main():
    out = []
    src_counts = {}
    skipped_counts = {}

    for src_name, src_path in SOURCES:
        if not src_path.exists():
            print(f"[error] source missing: {src_path}", file=sys.stderr)
            sys.exit(1)
        n_kept, n_skipped = 0, 0
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue
                msgs = rec.get("messages")
                converted = messages_to_alpaca(msgs)
                if converted is None:
                    n_skipped += 1
                    continue
                converted["source"] = src_name
                out.append(converted)
                n_kept += 1
        src_counts[src_name] = n_kept
        skipped_counts[src_name] = n_skipped
        print(f"  [{src_name}] kept={n_kept}, skipped={n_skipped}", flush=True)

    print(f"\n[total] kept={len(out)}, breakdown={src_counts}", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing {OUTPUT_PATH} ...", flush=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"[done] {len(out)} examples, {size_mb:.1f} MB", flush=True)
    print(f"[done] sample[0]:")
    s = json.dumps(out[0], ensure_ascii=False)
    print(f"  {s[:400]}{'...' if len(s) > 400 else ''}")


if __name__ == "__main__":
    main()
