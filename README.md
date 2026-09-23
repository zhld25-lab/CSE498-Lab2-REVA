# CSE498 Lab 2: ReVA VQA

This repository contains my submission for **CSE398/CSE498 Lab 2: ReVA VQA**.

## Contents

- `build_qwen_train_data.py` — converts ReVA annotations into Qwen training format.
- `prepare_reva_v2_test_set.py` — prepares the ReVA evaluation set.
- `score_reva_predictions.py` — parses predictions and computes evaluation accuracy.
- `reva_v2.py` — VILA/ReVA evaluation utilities.
- `model_comparison.csv` — comparison of the evaluated models.
- `qwen_base_result.csv` — Qwen3-VL baseline evaluation result.
- `qwen_finetuned_result.csv` — LoRA fine-tuned Qwen3-VL evaluation result.
- `vila_metrics.json` — VILA baseline evaluation metrics.
- `example_analysis.md` — short qualitative analysis of three representative ReVA examples.
- `REPORT.md` — detailed experiment report covering the workflow, smoke test, models, results, limitations, and future improvements.

## Evaluation Results

| Model | Accuracy | Questions | Answered |
|---|---:|---:|---:|
| Qwen3-VL Base | 66.00% | 50 | 50 |
| Qwen3-VL Fine-tuned | 66.00% | 50 | 50 |
| VILA1.5-3B Base | 56.00% | 50 | 46 |

The final comparison uses the same fixed **50-question subset** sampled from the official ReVA test set available on the MAGIC server.

For Qwen evaluation, the RTX 2080 Ti memory limit required a low-memory configuration with `MAX_FRAMES=2`, `MAX_PIXELS=28224`, and `MAX_MODEL_LEN=2048`. Both Qwen Base and Qwen Fine-tuned were evaluated with the same configuration. VILA used its own evaluation pipeline with `NUM_VIDEO_FRAMES=4`.

## Key Observations

- Qwen3-VL Base completed all 50 questions and achieved **33/50 = 66.00%**.
- Qwen3-VL Fine-tuned also completed all 50 questions and achieved **33/50 = 66.00%**.
- VILA1.5-3B produced valid parsed answers for **46/50** questions and achieved **28/50 = 56.00%** overall accuracy.
- The lightweight LoRA fine-tuning run did not produce a measurable accuracy improvement over the Qwen base model on this 50-question subset.
- Temporal Grounding remained challenging for Qwen: **1/6 = 16.67%** for both Base and Fine-tuned.
- General Understanding was strong for all evaluated models: Qwen **6/6** and VILA **6/6**.

## Notes

- All unit tests passed: **7 passed**.
- Qwen fine-tuning used LoRA.
- The ReVA dataset and model checkpoints are not included in this repository.
- The evaluation results in this repository now correspond to the final **50-question** comparison rather than the earlier 10-question smoke test.
