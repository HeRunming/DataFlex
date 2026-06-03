#!/usr/bin/env python3
"""Build MMLU few-shot target set for LESS-style data selection.

Uses cais/mmlu dev split (285 examples = 5 per subject x 57 subjects).
Each example is formatted as the standard MMLU multiple-choice prompt
(matching lm-eval's `mmlu` presentation) so that the gradient/embedding
of the target reflects the actual downstream task.

Output: sharegpt jsonl with user/assistant messages.
"""
import json
import os
from datasets import load_dataset

LETTERS = ["A", "B", "C", "D"]
OUT = "/jizhicfs/karonhe/DataFlex/data/mmlu_target.jsonl"


def format_question(subject: str, question: str, choices, answer_idx: int) -> tuple:
    subj_readable = subject.replace("_", " ")
    user = (
        f"The following are multiple choice questions (with answers) about {subj_readable}.\n\n"
        f"{question.strip()}\n"
    )
    for i, ch in enumerate(choices):
        user += f"{LETTERS[i]}. {ch}\n"
    user += "Answer:"
    assistant = f" {LETTERS[answer_idx]}"
    return user, assistant


def main():
    ds = load_dataset("cais/mmlu", "all", split="dev")
    rows = []
    for ex in ds:
        subject = ex["subject"]
        question = ex["question"]
        choices = ex["choices"]
        # choices may be a stringified list in some cache versions
        if isinstance(choices, str):
            choices = json.loads(choices.replace("'", '"'))
        answer = ex["answer"]
        answer_idx = int(answer)
        user, assistant = format_question(subject, question, choices, answer_idx)
        rows.append({
            "dataset": "mmlu_target",
            "id": f"mmlu_{subject}_{len(rows)}",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} MMLU target examples to {OUT}")
    print("Example user:\n", rows[0]["messages"][0]["content"])
    print("Example assistant:", repr(rows[0]["messages"][1]["content"]))


if __name__ == "__main__":
    main()
