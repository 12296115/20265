# DistilBERT Model Development and Evaluation

## Overview

A transformer-based approach using DistilBERT was implemented to evaluate whether contextual language representations could improve phishing email classification compared with the traditional machine-learning baselines.

## Tokenisation

Email text was tokenised using the DistilBERT tokenizer.

The input configuration used:

- Maximum sequence length: 512 tokens
- Input IDs
- Attention masks
- Binary classification labels

Sample validation confirmed that the tokenised input IDs and attention masks were generated with the expected dimensions of 512 tokens.

## Dataset Preparation

The prepared dataset was divided into:

- Training samples: 14,016
- Testing samples: 3,505

The labels were encoded as:

- Safe Email = 0
- Phishing Email = 1

PyTorch datasets and dataloaders were created using:

- `input_ids`
- `attention_mask`
- `labels`

The training batch size was 8.

## GPU Configuration

Initial testing identified that the original environment did not have CUDA-enabled GPU support. The runtime was subsequently configured with GPU acceleration.

The final training environment reported:

- PyTorch: 2.11.0+cu128
- CUDA available: True
- GPU: Tesla T4
- GPU memory: approximately 14.56 GB

This enabled practical fine-tuning of the DistilBERT model.

## Training Configuration

The model was trained using the following configuration:

| Parameter | Value |
|---|---:|
| Learning rate | 2 × 10⁻⁵ |
| Epochs | 2 |
| Batch size | 8 |
| Weight decay | 0.01 |
| Maximum gradient norm | 1.0 |
| Optimizer | AdamW |
| Scheduler | Linear warmup + linear decay |
| Warmup steps | 350 |
| Total training steps | 3,504 |

## Training Results

### Epoch 1

| Metric | Training | Testing |
|---|---:|---:|
| Loss | 0.1491 | 0.0450 |
| Accuracy | 94.53% | 98.86% |
| Precision | 95.28% | 98.85% |
| Recall | 89.82% | 98.09% |
| F1-Score | 92.47% | 98.47% |

### Epoch 2

| Metric | Training | Testing |
|---|---:|---:|
| Loss | 0.0259 | 0.0571 |
| Accuracy | 99.39% | 98.80% |
| Precision | 98.97% | 98.47% |
| Recall | 99.39% | 98.32% |
| F1-Score | 99.18% | 98.39% |

The model completed two training epochs in approximately 8.98 minutes.

## Final DistilBERT Performance

The final evaluation produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 98.80% |
| Precision | 98.47% |
| Recall | 98.32% |
| F1-Score | 98.39% |
| ROC-AUC | 0.9989 |

## Classification Results

The classification report was:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Safe Email | 99.00% | 99.09% | 99.04% | 2,196 |
| Phishing Email | 98.47% | 98.32% | 98.39% | 1,309 |
| **Accuracy** | | | **98.80%** | **3,505** |

The macro-average F1-score was 98.72%, while the weighted-average F1-score was 98.80%.

## Confusion Matrix

The final confusion matrix was:

```text
[[2176   20]
 [  22 1287]]

## This represents:

True negatives: 2,176
False positives: 20
False negatives: 22
True positives: 1,287
There were therefore 42 incorrect predictions among 3,505 test samples.

**ROC-AUC**
The final DistilBERT model achieved a ROC-AUC of 0.9989. This indicates a very strong ability to distinguish between Safe Email and Phishing Email across classification thresholds.

Overall Result
DistilBERT achieved the highest accuracy and F1-score among the evaluated models. Its ROC-AUC of 0.9989 was equal to the SVM baseline.
The results therefore provide evidence that the transformer-based approach is highly effective for the current phishing email classification task while also providing a basis for comparing its additional computational requirements against the traditional machine-learning approaches.
