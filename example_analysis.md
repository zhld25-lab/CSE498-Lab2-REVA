# Example Analysis

## Example 1 — Change Detection

**QA ID:** QA-000006

**Question:**  
How does the visible area of the parking lot change throughout the video?

**Ground Truth:**  
A. The visible area expands as the camera moves upward, revealing more of the surrounding landscape.

**Qwen Base Prediction:**  
A — Correct

**Qwen Fine-tuned Prediction:**  
A — Correct

**VILA Prediction:**  
A — Correct

**Analysis:**  
All three models answered this example correctly. The main visual change is large and scene-level: as the camera moves upward, the field of view expands and more of the parking lot and surrounding landscape becomes visible. This type of global change can be recognized even with sparse video sampling.

## Example 2 — Object and Land Cover Recognition

**QA ID:** QA-000044

**Question:**  
Where is the electric shuttle bus located in the scene?

**Ground Truth:**  
B. On the right side of the street, moving toward the camera.

**Qwen Base Prediction:**  
B — Correct

**Qwen Fine-tuned Prediction:**  
B — Correct

**VILA Prediction:**  
D — Incorrect

**Analysis:**  
The two Qwen models correctly identified both the spatial location and motion direction of the shuttle bus, while VILA selected an incorrect option. This illustrates a model-level difference on object and spatial understanding. In the final 50-question evaluation, both Qwen models achieved 7/14 (50.00%) on Object and Land Cover Recognition, while VILA also achieved 7/14 (50.00%) overall in this subcategory but produced only 12 valid parsed answers out of the 14 questions.

## Example 3 — Temporal Grounding

**QA ID:** QA-000010

**Question:**  
How long does the group of people remain visible in the parking lot during the video?

**Ground Truth:**  
C. From 0.0s to 5.0s (duration: 5.0s)

**Qwen Base Prediction:**  
B — Incorrect

**Qwen Fine-tuned Prediction:**  
B — Incorrect

**VILA Prediction:**  
B — Incorrect

**Analysis:**  
This temporal example is difficult because the model must identify both the beginning and end of the visible interval rather than only recognize the scene. Sparse temporal information makes this type of question particularly challenging. In the final 50-question evaluation, Temporal Grounding was one of the weakest Qwen categories: both Qwen Base and Qwen Fine-tuned achieved 1/6 (16.67%). VILA achieved 2/6 (33.33%) on Temporal Grounding.

## Overall Observation

On the final **50-question ReVA subset**, Qwen Base achieved **33/50 = 66.00%**, Qwen Fine-tuned also achieved **33/50 = 66.00%**, and VILA achieved **28/50 = 56.00%**. Qwen Base and Fine-tuned completed all 50 questions, while VILA produced valid parsed answers for 46 of 50 questions.

The lightweight LoRA fine-tuning run did not produce a measurable improvement in overall accuracy on this evaluation set. The Qwen models performed especially well on General Understanding (6/6), Change Detection (4/4), Hypothetical Reasoning (3/3), and Trend and Pattern (3/3), while Temporal Grounding was the most difficult category at 1/6.

The Qwen evaluation used a low-memory configuration with two sampled video frames because of the RTX 2080 Ti memory limit. This is an important experimental limitation, particularly for temporal questions that depend on observing changes across the full video.
