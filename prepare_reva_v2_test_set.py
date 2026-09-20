"""Convert ReVA_V2 test_set.json to a flat list for Qwen inference.

Student task: complete every block marked TODO(student).
"""

import json
import os
from collections import Counter
from typing import Any, Iterator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REVA_ROOT = os.environ.get("REVA_ROOT", os.path.join(PROJECT_ROOT, "data", "reva_test"))
REVA_JSON = os.environ.get("REVA_JSON", os.path.join(REVA_ROOT, "test_set.json"))
OUTPUT_JSON = os.environ.get("REVA_PREPARED_JSON", os.path.join(os.path.dirname(__file__), "reva_v2_test_set.json"))


def get_video_duration(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception as exc:
        print(f"  Warning: could not read duration for {video_path}: {exc}")
        return 0.0



def to_abs_path(file_path: str) -> str:
    """Convert an annotation video path to an absolute path under REVA_ROOT."""
# TODO(student): remove optional dataset prefixes and join with REVA_ROOT.
    #raise NotImplementedError("TODO: implement to_abs_path")
    normalized = file_path

    prefixes = [
        "#dataset/ReVA_V2/",
        "ReVA_V2/",
        "#dataset/",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    return os.path.join(REVA_ROOT, normalized)



def to_rel_stem(file_path: str) -> str:
    """Convert an annotation video path to a relative stem without extension."""
 # TODO(student): normalize the same prefixes as to_abs_path and strip extension.
    normalized = file_path

    prefixes = [
        "#dataset/ReVA_V2/",
        "ReVA_V2/",
        "#dataset/",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    stem, _ = os.path.splitext(normalized)
    return stem


def get_qa_id(qa: dict, video_item: dict, subcategory: str, question_idx: int) -> str:
    """Return a stable QA id for evaluation."""
 # TODO(student): prefer qa_id, then global_index, otherwise synthesize from video stem.
    if qa.get("qa_id") is not None:
        return str(qa["qa_id"])

    if qa.get("global_index") is not None:
        return f"REVA-G{int(qa['global_index']):06d}"

    video_stem = to_rel_stem(video_item["file_path"])
    return f"{video_stem}_{subcategory}_{question_idx}"


def iter_flat_items(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield flat Qwen-evaluation items from nested ReVA test annotations."""
 # TODO(student): iterate videos, skip missing/unreadable videos, flatten MCQs.
    for video_key, video_item in data["videos"].items():
        file_path = video_item["file_path"]

        video_path = to_abs_path(file_path)

        if not os.path.exists(video_path):
            continue

        duration = get_video_duration(video_path)

        if duration <= 0:
            continue

        video_id = to_rel_stem(file_path)

        for category, subcategories in video_item.get("mcq", {}).items():
            for subcategory, qa_list in subcategories.items():
                for question_idx, qa in enumerate(qa_list):
                    qa_id = get_qa_id(
                        qa,
                        video_item,
                        subcategory,
                        question_idx,
                    )

                    options_text = "\n".join(
                        f"{label}. {qa['options'][label]}"
                        for label in sorted(qa["options"])
                    )

                    question_text = (
                        f"{qa['question']}\n"
                        f"{options_text}"
                    )

                    yield {
                        "id": qa_id,
                        "video_id": video_id,
                        "duration": duration,
                        "question": question_text,
                        "answer": qa["correct_answer"],
                        "question_type": subcategory,
                    }

def main():
    with open(REVA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    subcats = Counter()
    for item in iter_flat_items(data):
        items.append(item)
        subcats[item.get("question_type", "unknown")] += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(items)} QA items to {OUTPUT_JSON}")
    print("\nSubcategory distribution:")
    for subcat, count in sorted(subcats.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {subcat}: {count}")


if __name__ == "__main__":
    main()
