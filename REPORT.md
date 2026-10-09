# CSE498 Lab 2 Report: ReVA Video Question Answering

## 1. Objective

This lab evaluates vision-language models on the ReVA video question-answering task and studies the effect of LoRA fine-tuning. The workflow includes data preparation, unit testing, smoke testing, Qwen baseline evaluation, LoRA fine-tuning, fine-tuned Qwen evaluation, VILA evaluation, and result analysis.

## 2. Models

Three model settings were compared:

- **Qwen3-VL Base:** `Qwen/Qwen3-VL-4B-Instruct`
- **Qwen3-VL Fine-tuned V2:** the same Qwen3-VL backbone plus a revised LoRA adapter
- **VILA1.5-3B:** `Efficient-Large-Model/VILA1.5-3b`

The controlled comparison of interest is Qwen Base vs. Qwen Fine-tuned V2.

## 3. ReVA Data and Implementation

ReVA is a video multiple-choice QA benchmark. Each item contains a video, question, answer choices, and a ground-truth answer.

The main implementation tasks were:

- converting ReVA annotations into Qwen training format;
- preparing the evaluation set and resolving video paths;
- extracting predicted answer letters;
- computing overall and subcategory accuracy;
- implementing VILA-side ReVA evaluation utilities.

All provided unit tests passed: **7 passed**.

## 4. Smoke Test

A 10-question smoke test was first used to verify the end-to-end pipeline:

```text
data loading
→ video processing
→ model loading
→ inference
→ answer parsing
→ scoring
```

The final reported comparison uses a fixed 50-question subset.

## 5. Fine-tuning

LoRA was used instead of full-parameter fine-tuning.

### 5.1 Initial Run

The initial LoRA evaluation achieved the same score as the base model, **33/50 = 66.00%**. Investigation showed that the generated training file accidentally contained only **2 training examples**. The adapter had been generated and loaded correctly, but two samples were insufficient to produce a meaningful adaptation.

### 5.2 Revised Run

The training data pipeline was corrected and regenerated with **1,000 ReVA QA examples** from **72 unique videos**.

The revised training configuration was:

```text
training examples = 1000
epochs = 3
optimizer steps = 375
GPUs = 2 x RTX 2080 Ti
batch size per GPU = 1
gradient accumulation = 4
learning rate = 1e-5
LoRA rank = 16
LoRA alpha = 32
LoRA dropout = 0.05
video frames = 2
```

The revised run completed normally and saved the final LoRA adapter at `checkpoint-375`.

## 6. Evaluation Setup

Because the available RTX 2080 Ti has about 11 GB of VRAM, Qwen evaluation used:

```text
MAX_FRAMES=2
MAX_PIXELS=28224
MAX_MODEL_LEN=2048
BACKEND=transformers
```

Qwen Base and Qwen Fine-tuned V2 used the same fixed 50 questions and the same inference settings. The fine-tuned evaluation log confirmed that the `checkpoint-375` LoRA adapter was loaded successfully.

VILA used its own pipeline with:

```text
NUM_VIDEO_FRAMES=4
```

Therefore, Qwen Base vs. Fine-tuned V2 is the controlled before/after comparison. Qwen vs. VILA is a practical cross-model comparison with different visual budgets.

## 7. Final Results

| Model | Correct | Answered | Questions | Accuracy |
|---|---:|---:|---:|---:|
| Qwen3-VL Base | 33 | 50 | 50 | **66.00%** |
| **Qwen3-VL Fine-tuned V2** | **36** | **50** | **50** | **72.00%** |
| VILA1.5-3B | 28 | 46 | 50 | **56.00%** |

The revised LoRA model improved from **66.00% to 72.00%**, an absolute gain of **6 percentage points** and three additional correct answers.

VILA produced valid parsed answers for 46 of 50 questions. Its accuracy is calculated over all 50 questions.

## 8. Qwen Base vs. Fine-tuned V2 by Subcategory

| Subcategory | Base | Fine-tuned V2 | Change |
|---|---:|---:|---:|
| Causation Reasoning | 1/2 (50.00%) | 1/2 (50.00%) | 0 |
| Change Detection | 4/4 (100.00%) | 3/4 (75.00%) | -1 correct |
| Consequence Reasoning | 1/1 (100.00%) | 1/1 (100.00%) | 0 |
| General Understanding | 6/6 (100.00%) | 6/6 (100.00%) | 0 |
| Geometric Relation | 2/3 (66.67%) | 3/3 (100.00%) | +1 correct |
| Hypothetical Reasoning | 3/3 (100.00%) | 2/3 (66.67%) | -1 correct |
| Object and Land Cover Recognition | 7/14 (50.00%) | 8/14 (57.14%) | +1 correct |
| Perspective and Viewpoint | 2/3 (66.67%) | 2/3 (66.67%) | 0 |
| Structural Layout | 3/5 (60.00%) | 3/5 (60.00%) | 0 |
| Temporal Grounding | 1/6 (16.67%) | 4/6 (66.67%) | **+3 correct** |
| Trend and Pattern | 3/3 (100.00%) | 3/3 (100.00%) | 0 |

The largest improvement was in **Temporal Grounding**, where accuracy increased from **16.67% to 66.67%**.

## 9. Discussion

The revised experiment demonstrates that the LoRA pipeline can produce a measurable gain when trained on a meaningful amount of task-specific data.

The initial no-gain result was traced to a data-pipeline issue rather than an adapter-loading failure: only two training samples had been included. After correcting the data preparation step and training on 1,000 examples for three epochs, accuracy increased by six percentage points.

The improvement was not uniform across all categories. Temporal Grounding improved substantially, while Change Detection and Hypothetical Reasoning each lost one correct answer. This suggests that LoRA adaptation changed the model's decision behavior rather than simply improving every category.

Because the evaluation set contains only 50 questions, this result should be described as a **measurable improvement**, not as a statistically significant improvement without additional statistical testing.

## 10. Limitations

The main limitations are:

- only 50 questions in the final comparison;
- only two sampled video frames for Qwen evaluation;
- the revised result comes from one training configuration;
- different frame settings for Qwen and VILA;
- four VILA outputs were not parsed into valid answer choices;
- no separate validation split was used for hyperparameter or checkpoint selection.

## 11. Future Work

Future experiments should:

- create a held-out validation split for checkpoint and hyperparameter selection;
- evaluate on a larger fixed subset or the full test set;
- test additional LoRA learning rates and ranks;
- compare earlier checkpoints using validation performance rather than test performance;
- use more video frames if GPU memory permits;
- run a per-question Base vs. Fine-tuned error-transition analysis;
- repeat training with multiple random seeds to measure result stability.

## 12. Conclusion

This lab implemented the full ReVA VQA workflow:

```text
prepare data
→ run tests
→ smoke test
→ Qwen Base evaluation
→ LoRA fine-tuning
→ Qwen Fine-tuned evaluation
→ VILA evaluation
→ model comparison
→ error analysis
```

Final results:

```text
Qwen3-VL Base:          66.00%
Qwen3-VL Fine-tuned V2: 72.00%
VILA1.5-3B:             56.00%
```

After correcting the training-data issue, LoRA fine-tuning improved Qwen3-VL from **33/50 to 36/50**, with the largest gain occurring in Temporal Grounding.
