# Testing, Evaluation and Final Refinement

## Overview

Following model development and system integration, testing and evaluation were conducted to assess the performance and functionality of the phishing email detection prototype.

The evaluation considered both machine-learning performance and system-level functionality.

## Model Evaluation

The final DeBERTa-v3-large model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive and false-negative analysis

The final test results were:

| Metric | Result |
|---|---:|
| Accuracy | 99.3721% |
| Precision | 98.9346% |
| Recall | 99.3884% |
| F1-score | 99.1609% |

The evaluation produced:

- False Positives: 7
- False Negatives: 4

## Hybrid Model Evaluation

The gated hybrid DeBERTa architecture was also evaluated.

| Metric | Standard DeBERTa-v3-large | Gated Hybrid DeBERTa |
|---|---:|---:|
| Accuracy | 99.3721% | 99.3721% |
| Precision | 98.9346% | 99.2343% |
| Recall | 99.3884% | 99.0826% |
| F1-score | 99.1609% | 99.1584% |
| False Positives | 7 | 5 |
| False Negatives | 4 | 6 |

The results showed that the hybrid architecture reduced false positives and increased precision, but the standard DeBERTa-v3-large model achieved higher recall and slightly higher F1-score.

The results were therefore considered during the model refinement process rather than assuming that the hybrid architecture was universally superior.

## Error Analysis

Classification errors were examined to identify potential limitations of the model.

The analysis considered factors including:

- Prediction confidence
- Email length
- Token length
- False-positive cases
- False-negative cases

The analysis also considered the effect of the 512-token sequence limit used by the transformer models.

## System Testing

Testing was conducted across the major system components, including:

- Model prediction
- FastAPI communication
- SHAP explanation generation
- Streamlit interface
- Browser-extension communication

The purpose was to identify integration problems and verify that the individual components operated together as an end-to-end prototype.

## Refinement

Testing and evaluation were used to refine the system architecture and implementation.

The development progressed from standalone model experimentation to:

1. Transformer model selection
2. DeBERTa-v3-large implementation
3. Gated hybrid model investigation
4. SHAP explainability
5. FastAPI integration
6. Streamlit interface
7. Browser-extension integration

## Outcome

The project has progressed from experimental machine-learning models to an integrated and explainable phishing detection prototype.

Remaining work focuses on final validation, refinement, documentation and preparation of the final demonstration and report.
