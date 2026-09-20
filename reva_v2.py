import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()



    # TODO(student): read question_file, flatten instances, and assign stable IDs
    # using qa_id, then global_index, then video/subcategory/question index.

def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""

    with open(question_file, encoding="utf-8") as f:
        data = json.load(f)

    instances = []

    for video_id, video_item in data["videos"].items():
        file_path = video_item["file_path"]
        dataset_name = video_item.get("dataset_name", "")

        for category, subcategories in video_item.get("mcq", {}).items():
            for subcategory, qa_list in subcategories.items():
                for question_idx, qa in enumerate(qa_list):

                    if qa.get("qa_id") is not None:
                        qa_id = str(qa["qa_id"])

                    elif qa.get("global_index") is not None:
                        qa_id = f"REVA-G{int(qa['global_index']):06d}"

                    else:
                        dataset_tag = dataset_name or str(video_id)
                        subcat_tag = subcategory[:3].upper()
                        qa_id = f"REVA-{dataset_tag}-{subcat_tag}-{question_idx:04d}"

                    instances.append({
                        "qa_id": qa_id,
                        "video_id": video_id,
                        "video_path": file_path,
                        "dataset_name": dataset_name,
                        "category": category,
                        "subcategory": subcategory,
                        "question": qa["question"],
                        "options": qa["options"],
                        "correct_answer": qa["correct_answer"],
                        "reasoning": qa.get("reasoning", ""),
                        "example": qa.get("example", ""),
                    })

    return instances


    # TODO(student): try raw_path, dataset_root/raw_path, and paths with dataset_prefix removed.
def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""

    candidates = []

    # 1. origianl path
    candidates.append(raw_path)

    # 2. dataset_root + origianl path
    candidates.append(os.path.join(dataset_root, raw_path))

    # 3. delete dataset_prefix，ex"#dataset"
    normalized = raw_path

    if dataset_prefix and normalized.startswith(dataset_prefix):
        normalized = normalized[len(dataset_prefix):].lstrip("/\\")

    candidates.append(os.path.join(dataset_root, normalized))

    # 4. delete ReVA_V2/
    if normalized.startswith("ReVA_V2/"):
        normalized = normalized[len("ReVA_V2/"):]

    candidates.append(os.path.join(dataset_root, normalized))

    # find the first true existed file
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not resolve video path: {raw_path}")


    # TODO(student): include question, sorted options, and one-letter instruction.

def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""

    lines = [question]

    for label in sorted(options.keys()):
        lines.append(f"{label}. {options[label]}")

    lines.append(
        "Answer with only the option letter from the given choices."
    )

    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str]) -> str | None:
    """Parse a choice letter from VILA raw response."""

    text = response.strip()
    upper_text = text.upper()

    # 1. 如果回答本身就是一个选项字母
    if re.fullmatch(r"[A-H]", upper_text):
        if upper_text in options:
            return upper_text

    # 2. 解析类似 "The answer is B" / "Answer: B"
    match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([A-H])\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        letter = match.group(1).upper()
        if letter in options:
            return letter

    # 3. 尝试匹配选项文本
    lower_text = text.lower()

    for label, option_text in options.items():
        option_lower = str(option_text).strip().lower()

        if lower_text == option_lower:
            return label.upper()

        if option_lower and option_lower in lower_text:
            return label.upper()

    return None

def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""

    num_questions = len(records)
    num_answered = sum(1 for r in records if r.get("pred_letter") is not None)
    num_correct = sum(1 for r in records if r.get("is_correct"))

    accuracy = (
        num_correct / num_questions
        if num_questions > 0
        else 0.0
    )

    by_category = {}
    by_subcategory = {}

    category_groups = {}
    subcategory_groups = {}

    for record in records:
        category = record.get("category", "unknown")
        subcategory = record.get("subcategory", "unknown")

        category_groups.setdefault(category, []).append(record)
        subcategory_groups.setdefault(subcategory, []).append(record)

    for category, group in category_groups.items():
        total = len(group)
        answered = sum(1 for r in group if r.get("pred_letter") is not None)
        correct = sum(1 for r in group if r.get("is_correct"))

        by_category[category] = {
            "num_questions": total,
            "num_answered": answered,
            "num_correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    for subcategory, group in subcategory_groups.items():
        total = len(group)
        answered = sum(1 for r in group if r.get("pred_letter") is not None)
        correct = sum(1 for r in group if r.get("is_correct"))

        by_subcategory[subcategory] = {
            "num_questions": total,
            "num_answered": answered,
            "num_correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    return {
        "num_questions": num_questions,
        "num_answered": num_answered,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "by_category": by_category,
        "by_subcategory": by_subcategory,
    }

def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
