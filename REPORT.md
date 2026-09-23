# CSE498 Lab 2 Report: ReVA Video Question Answering

## 1. Objective

This lab evaluates vision-language models on the ReVA video question-answering task and studies the effect of LoRA fine-tuning. The workflow includes data preparation, unit testing, smoke testing, Qwen baseline evaluation, LoRA fine-tuning, fine-tuned Qwen evaluation, VILA evaluation, and qualitative error analysis.

## 2. Models

Three model settings were compared:

- **Qwen3-VL Base:** `Qwen/Qwen3-VL-4B-Instruct`
- **Qwen3-VL Fine-tuned:** the same Qwen3-VL backbone plus a trained LoRA adapter
- **VILA1.5-3B:** `Efficient-Large-Model/VILA1.5-3b`

The Base-vs.-Fine-tuned comparison measures the effect of LoRA adaptation. The Qwen-vs.-VILA comparison provides a cross-model baseline.

## 3. ReVA Data and Implementation

ReVA is a video multiple-choice QA benchmark. Each item contains a video, question, answer choices, and a ground-truth answer.

The main implementation tasks were:

- converting nested ReVA annotations into Qwen training format;
- preparing the evaluation set and resolving video paths;
- extracting predicted answer letters;
- computing overall and subcategory accuracy;
- implementing VILA-side ReVA evaluation utilities.

All provided unit tests passed: **7 passed**.

## 4. Smoke Test

Before the larger evaluation, a 10-question **smoke test** was used to verify the complete pipeline:

```text
data loading
→ video processing
→ model loading
→ inference
→ answer parsing
→ scoring
```

The smoke test was for pipeline validation, not for the final performance estimate. After the workflow was confirmed, the evaluation was expanded to a fixed 50-question subset. With 10 questions, one question changes accuracy by 10 percentage points; with 50 questions, one question changes it by only 2 points.

## 5. Fine-tuning

LoRA was used instead of full-parameter fine-tuning. Most original Qwen parameters remained frozen while a small number of low-rank adapter parameters were trained.

Training used memory-saving techniques including DeepSpeed ZeRO-3 and CPU offload. During fine-tuned evaluation, the inference log explicitly showed that the LoRA adapter checkpoint was loaded, confirming that the fine-tuned model was actually evaluated.

## 6. Evaluation Setup

Because the available RTX 2080 Ti has about 11 GB of VRAM, Qwen evaluation used a low-memory configuration:

```text
MAX_FRAMES=2
MAX_PIXELS=28224
MAX_MODEL_LEN=2048
BACKEND=transformers
```

Both Qwen Base and Qwen Fine-tuned used exactly the same 50 questions and the same evaluation settings.

VILA used its own pipeline with:

```text
NUM_VIDEO_FRAMES=4
```

Therefore, Qwen Base vs. Fine-tuned is a controlled comparison, while Qwen vs. VILA is a practical cross-model comparison rather than a perfectly matched visual-budget comparison.

## 7. Final Results

| Model | Correct | Answered | Questions | Accuracy |
|---|---:|---:|---:|---:|
| Qwen3-VL Base | 33 | 50 | 50 | **66.00%** |
| Qwen3-VL Fine-tuned | 33 | 50 | 50 | **66.00%** |
| VILA1.5-3B | 28 | 46 | 50 | **56.00%** |

VILA produced valid parsed answers for 46 of 50 questions. Its accuracy is calculated over all 50 questions.

## 8. Qwen Subcategory Results

| Subcategory | Result | Accuracy |
|---|---:|---:|
| Causation Reasoning | 1/2 | 50.00% |
| Change Detection | 4/4 | 100.00% |
| Consequence Reasoning | 1/1 | 100.00% |
| General Understanding | 6/6 | 100.00% |
| Geometric Relation | 2/3 | 66.67% |
| Hypothetical Reasoning | 3/3 | 100.00% |
| Object and Land Cover Recognition | 7/14 | 50.00% |
| Perspective and Viewpoint | 2/3 | 66.67% |
| Structural Layout | 3/5 | 60.00% |
| Temporal Grounding | 1/6 | 16.67% |
| Trend and Pattern | 3/3 | 100.00% |

The weakest Qwen subcategory was **Temporal Grounding (1/6 = 16.67%)**.

## 9. Discussion

### Why did fine-tuning not improve accuracy?

Qwen Base and Qwen Fine-tuned both achieved 66.00%. This does not mean that the fine-tuning pipeline failed. The adapter was trained, saved, and successfully loaded during inference.

The most likely reasons are:

1. the LoRA run was relatively lightweight;
2. Qwen evaluation used only two sampled video frames;
3. there was a train-test visual-input mismatch;
4. the base model was already strong on several categories.

The appropriate conclusion is:

> The LoRA pipeline worked correctly, but this particular lightweight training configuration did not produce a measurable accuracy improvement on the final 50-question subset.

### Why was Temporal Grounding difficult?

Temporal Grounding requires identifying when an event starts and ends. With only two sampled frames, the model may miss short events, event boundaries, or changes occurring between sampled frames. This likely contributed to the low score.

## 10. Representative Examples

- **QA-000006, Change Detection:** all three models were correct.
- **QA-000044, Object and Land Cover Recognition:** both Qwen models were correct, while VILA was incorrect.
- **QA-000010, Temporal Grounding:** all three models were incorrect.

These examples show that large scene-level changes can be recognized under sparse sampling, while temporal interval questions are much more sensitive to limited frame coverage.

## 11. Limitations

The main limitations are:

- only 50 questions in the final comparison;
- only two video frames for Qwen evaluation;
- train-test visual-input mismatch;
- lightweight LoRA training;
- different frame settings for Qwen and VILA;
- four VILA outputs were not parsed into valid answer choices.

## 12. Future Work

Future experiments should:

- use more ReVA training examples;
- train for more optimization steps or epochs;
- tune LoRA learning rate and rank;
- save and compare multiple checkpoints;
- use more video frames if GPU memory allows;
- make training and evaluation visual settings consistent;
- evaluate a larger fixed subset or the full test set;
- compare per-question Base and Fine-tuned predictions.

## 13. Conclusion

This lab successfully implemented the full ReVA VQA workflow:

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
Qwen3-VL Base:       66.00%
Qwen3-VL Fine-tuned: 66.00%
VILA1.5-3B:          56.00%
```

The key lesson is that successful fine-tuning does not automatically guarantee higher accuracy. Training strength, frame sampling, GPU memory, evaluation size, and task difficulty all affect the final result.
