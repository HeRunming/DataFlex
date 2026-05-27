#!/usr/bin/env python3
"""Compute sentence embeddings for sharegpt-format jsonl datasets.

Used to prepare candidate + target embeddings for MMD-Emb-RBF selector.
"""
import json
import os
import sys
import numpy as np
from pathlib import Path
import argparse

def load_sharegpt_text(path: str):
    """Load sharegpt jsonl, concat all message contents into a single string per example."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages", [])
            text = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs)
            texts.append(text)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="sharegpt jsonl path")
    ap.add_argument("--output", required=True, help=".npy output path")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5",
                    help="sentence-transformers model name or local path")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_chars", type=int, default=4000,
                    help="truncate each text to this many characters before embedding")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[embed] Loading texts from {args.input}", flush=True)
    texts = load_sharegpt_text(args.input)
    if args.max_chars > 0:
        texts = [t[: args.max_chars] for t in texts]
    print(f"[embed] {len(texts)} texts loaded; example length = {len(texts[0])}", flush=True)

    print(f"[embed] Loading sentence-transformers model: {args.model}", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)

    print(f"[embed] Encoding...", flush=True)
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embs = embs.astype(np.float32)
    print(f"[embed] Output shape: {embs.shape}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, embs)
    print(f"[embed] Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
