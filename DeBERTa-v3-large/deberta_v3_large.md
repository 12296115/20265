# DeBERTa-v3-large Model Development

## Overview

Following the initial transformer experimentation, DeBERTa-v3 was investigated as an advanced transformer architecture for phishing and social-engineering email classification.

The selected model was:

`microsoft/deberta-v3-large`

The objective was to evaluate the model's ability to classify safe and phishing email content and determine its suitability for subsequent system development.

## Model Configuration

- Model: DeBERTa-v3-large
- Maximum sequence length: 512 tokens
- Training epochs: 3
- Learning rate: 1e-5
- Optimiser: AdamW
- Weight decay: 0.01
- Random seed: 42
- GPU: NVIDIA A100-SXM4-40GB
- Best model selection: Validation F1

## Development Process

The prepared and validated email dataset was tokenised using the DeBERTa tokenizer. Email text was padded or truncated to a maximum sequence length of 512 tokens.

The model was fine-tuned using the training dataset and evaluated using the test dataset. Accuracy, precision, recall and F1-score were used to assess classification performance.

## Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 99.3721% |
| Precision | 98.9346% |
| Recall | 99.3884% |
| F1-score | 99.1609% |
| Test Loss | 0.04341 |

The final evaluation produced 7 false positives and 4 false negatives.

## Confusion Matrix

| Actual / Predicted | Safe | Phishing |
|---|---:|---:|
| Safe | 1091 | 7 |
| Phishing | 4 | 650 |

## Outcome

DeBERTa-v3-large demonstrated the strongest observed performance among the transformer architectures investigated during the project and was selected as the final transformer architecture for subsequent system development.

The selected model subsequently provided the foundation for further development of the gated hybrid architecture and integrated phishing detection prototype.
