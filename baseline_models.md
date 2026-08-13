# Baseline Machine-Learning Models

## Purpose

Traditional machine-learning classifiers were implemented as baseline models for the phishing email detection project. These models provide reference performance against which the transformer-based DistilBERT model can subsequently be evaluated.

## Text Representation

Email text was converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF). The resulting representation was used as the input for the traditional machine-learning classifiers.

The same feature representation and testing dataset were used for both baseline models to support a consistent comparison.

## Logistic Regression

Logistic Regression was implemented as the first traditional machine-learning baseline.

### Performance

| Metric | Result |
|---|---:|
| Accuracy | 98.18% |
| Precision | 98.60% |
| Recall | 96.49% |
| F1-Score | 97.53% |
| ROC-AUC | 0.9981 |

The results demonstrate strong classification performance, with particularly high precision and ROC-AUC. The lower recall compared with the subsequent models indicates that some phishing emails were not identified by the classifier.

## Support Vector Machine

Support Vector Machine (SVM) was implemented as the second traditional machine-learning baseline.

### Performance

| Metric | Result |
|---|---:|
| Accuracy | 98.72% |
| Precision | 98.62% |
| Recall | 97.94% |
| F1-Score | 98.28% |
| ROC-AUC | 0.9989 |

SVM achieved higher accuracy, recall and F1-score than Logistic Regression while maintaining similarly high precision. Its ROC-AUC of 0.9989 also indicates strong separation between Safe Email and Phishing Email.

## Baseline Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.18% | 98.60% | 96.49% | 97.53% | 0.9981 |
| SVM | 98.72% | 98.62% | 97.94% | 98.28% | 0.9989 |

SVM provided stronger overall baseline performance than Logistic Regression and therefore established a more competitive reference point for comparison with the DistilBERT model.

## Role in the Project

The baseline models provide a quantitative reference for assessing whether the additional computational and architectural complexity of a transformer-based approach produces meaningful performance improvements.

The baseline experiments were implemented and evaluated in Google Colab, with the resulting performance metrics documented as part of the project's technical evidence.
