#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
 # TODO(student): use regex with DOTALL, strip whitespace, and return a string.
    #raise NotImplementedError("TODO: implement extract_answer")
    match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return text.strip()

    # TODO(student): normalize case and robustly parse A-H from short or verbose output.
def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""

    answer_text = extract_answer(text).strip()

    if re.fullmatch(r"[A-Ha-h]", answer_text):
        return answer_text.upper()

    match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([A-H])\b",
        answer_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-H])\b", answer_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


    # TODO(student): read every prediction *.json line-by-line and index by item["id"].
def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""

    predictions = {}

    for file_path in output_dir.glob("*.json"):
        if file_path.name == "result.json":
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)
                predictions[str(item["id"])] = item

    return predictions


    # TODO(student): compare predictions against every GT item, count missing predictions
    # as incorrect, and report Completed plus Total/subcategory accuracy statistics.
def score_predictions(
    preds: dict[str, dict[str, Any]],
    gt_list: list[dict[str, Any]],
) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""

    results = {}

    total = len(gt_list)
    completed = 0
    correct = 0

    subcat_total = defaultdict(int)
    subcat_correct = defaultdict(int)

    for gt in gt_list:
        item_id = str(gt["id"])
        gt_letter = str(gt["answer"]).strip().upper()
        question_type = gt.get("question_type", "unknown")

        subcat_total[question_type] += 1

        pred_item = preds.get(item_id)

        if pred_item is not None:
            completed += 1
            pred_text = str(pred_item.get("pred", ""))
            pred_letter = extract_letter(pred_text)
        else:
            pred_text = ""
            pred_letter = ""

        acc = int(pred_letter == gt_letter)

        if acc:
            correct += 1
            subcat_correct[question_type] += 1

        results[item_id] = {
            "gt_letter": gt_letter,
            "pred_letter": pred_letter,
            "acc": acc,
            "question_type": question_type,
        }

    lines = []

    completion_pct = (completed / total * 100) if total else 0.0
    accuracy_pct = (correct / total * 100) if total else 0.0

    lines.append(
        f"Completed: {completed}/{total} = {completion_pct:.2f}%"
    )
    lines.append(
        f"Total: {correct}/{total} = {accuracy_pct:.2f}%"
    )

    for question_type in sorted(subcat_total):
        cat_total = subcat_total[question_type]
        cat_correct = subcat_correct[question_type]
        cat_pct = (cat_correct / cat_total * 100) if cat_total else 0.0

        lines.append(
            f"{question_type}: {cat_correct}/{cat_total} = {cat_pct:.2f}%"
        )

    csv_text = "\n".join(lines) + "\n"

    return results, csv_text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
