# CSE498 Lab 2: ReVA VQA

This repository contains my submission for **CSE398/CSE498 Lab 2: ReVA VQA**.

## Contents

- `build_qwen_train_data.py` — converts ReVA annotations into Qwen training format.
- `prepare_reva_v2_test_set.py` — prepares the ReVA evaluation set.
- `score_reva_predictions.py` — parses predictions and computes evaluation accuracy.
- `reva_v2.py` — VILA/ReVA evaluation utilities.
- `model_comparison.csv` — comparison of the evaluated models.
- `qwen_base_result.csv` — Qwen3-VL baseline evaluation result.
- `qwen_finetuned_result.csv` — revised LoRA fine-tuned Qwen3-VL evaluation result.
- `vila_metrics.json` — VILA baseline evaluation metrics.
- `example_analysis.md` — qualitative analysis of representative ReVA examples.
- `REPORT.md` — detailed experiment report.

## Final Evaluation Results

| Model | Accuracy | Correct | Questions | Answered |
|---|---:|---:|---:|---:|
| Qwen3-VL Base | 66.00% | 33/50 | 50 | 50 |
| **Qwen3-VL Fine-tuned V2** | **72.00%** | **36/50** | 50 | 50 |
| VILA1.5-3B Base | 56.00% | 28/50 | 50 | 46 |

The final comparison uses the same fixed **50-question ReVA subset** for Qwen Base and Qwen Fine-tuned V2. Qwen evaluation used the same low-memory inference configuration for both models:

```text
MAX_FRAMES=2
MAX_PIXELS=28224
MAX_MODEL_LEN=2048
BACKEND=transformers
```

VILA used its own evaluation pipeline with `NUM_VIDEO_FRAMES=4`.

## Revised Fine-tuning Experiment

The first LoRA run produced no measurable gain because the generated training file accidentally contained only **2 training examples**. The training data pipeline was corrected before the revised experiment.

The revised LoRA run used:

- **1,000 ReVA QA training examples**
- **72 unique videos**
- **3 epochs**
- **375 optimizer steps**
- **2 × RTX 2080 Ti GPUs**
- learning rate: **1e-5**
- LoRA rank: **16**
- LoRA alpha: **32**
- LoRA dropout: **0.05**
- training video frames: **2**

The final adapter was saved at `checkpoint-375`.

## Key Results

- Qwen3-VL Base: **33/50 = 66.00%**
- Qwen3-VL Fine-tuned V2: **36/50 = 72.00%**
- Absolute improvement after revised fine-tuning: **+6 percentage points**
- VILA1.5-3B: **28/50 = 56.00%**

The largest subcategory improvement was **Temporal Grounding**, which increased from **1/6 = 16.67%** for the base model to **4/6 = 66.67%** after fine-tuning.

Other changes included:

- Geometric Relation: **2/3 → 3/3**
- Object and Land Cover Recognition: **7/14 → 8/14**
- Change Detection: **4/4 → 3/4**
- Hypothetical Reasoning: **3/3 → 2/3**

The revised experiment therefore shows a measurable overall improvement, while also showing that gains are not uniform across every subcategory.

## Notes

- All provided unit tests passed: **7 passed**.
- Fine-tuning used LoRA rather than full-parameter training.
- The ReVA dataset and model checkpoints are not included in this repository.
- The final reported Qwen comparison uses the same fixed 50-question test subset and the same inference settings.
