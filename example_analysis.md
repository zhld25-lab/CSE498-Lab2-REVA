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
All three models answered this question correctly. The main visual change is relatively large and easy to observe: as the camera moves upward, the field of view expands and more of the parking lot and surrounding area becomes visible. This type of global scene-level change can still be captured reasonably well even when only a small number of video frames are sampled. The fine-tuned Qwen model produced the same prediction as the baseline model on this example.


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
The two Qwen models correctly identified both the spatial location and movement direction of the electric shuttle bus, while VILA selected an incorrect option. This example shows a difference between the models on object-level spatial understanding. In this 10-question subset, Qwen performed better than VILA on Object and Land Cover Recognition overall. Fine-tuning did not change the Qwen prediction for this example because both the baseline and fine-tuned models already selected the correct answer.


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
All three models failed on this temporal grounding question. The task requires identifying both when the group first becomes visible and when it disappears, which depends on fine-grained temporal information rather than only general scene understanding. In this experiment, only two video frames were used during evaluation because of GPU memory limitations. This sparse temporal sampling likely made exact duration estimation more difficult. The result is consistent with the overall evaluation, where both Qwen models and VILA achieved 0% accuracy on the two Temporal Grounding questions.


## Overall Observation

On the 10-question official ReVA subset, Qwen Base achieved 70% accuracy, Qwen Fine-tuned also achieved 70%, and VILA achieved 60%. The fine-tuned Qwen model produced the same predictions as the baseline Qwen model on all ten evaluated questions, so no measurable improvement from fine-tuning was observed on this small subset.

The models performed well on General Understanding, Change Detection, Structural Layout, and Trend and Pattern questions, while Temporal Grounding was the most difficult category. All three models answered both Temporal Grounding questions incorrectly. Because the evaluation subset is small and only two video frames were used to fit the models within the available GPU memory, these results should be interpreted as a limited comparison rather than a comprehensive evaluation of model performance.
