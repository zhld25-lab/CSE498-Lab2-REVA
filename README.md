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
- `example_analysis.md` — short analysis of three representative evaluation examples.

## Evaluation Results

| Model | Accuracy | Questions | Answered |
|---|---:|---:|---:|
| Qwen3-VL Base | 70.00% | 10 | 10 |
| Qwen3-VL Fine-tuned | 70.00% | 10 | 10 |
| VILA1.5-3B Base | 60.00% | 10 | 10 |

The results above were obtained on a 10-question subset sampled from the official ReVA test set available on the MAGIC server.

Because of the available GPU memory on the RTX 2080 Ti, Qwen evaluation used a low-memory configuration with 2 sampled video frames.

## Notes

- All unit tests passed: **7 passed**.
- Qwen fine-tuning used LoRA.
- The three-example qualitative analysis is provided in `example_analysis.md`.
- The ReVA dataset and model checkpoints are not included in this repository.
