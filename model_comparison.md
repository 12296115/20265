# Model Performance Comparison

## Purpose

Three classification approaches were evaluated for the phishing email detection task:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. DistilBERT

The traditional machine-learning models were used as baseline approaches, while DistilBERT was evaluated as the deep learning transformer-based approach.

The models were compared using Accuracy, Precision, Recall, F1-Score and ROC-AUC.

## Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.18% | 98.60% | 96.49% | 97.53% | 99.81% |
| SVM | 98.72% | 98.62% | 97.94% | 98.28% | 99.89% |
| DistilBERT | **98.80%** | 98.47% | **98.32%** | **98.39%** | 99.89% |

## Accuracy

DistilBERT achieved the highest accuracy at 98.80%, followed by SVM at 98.72% and Logistic Regression at 98.18%.

The improvement from Logistic Regression to DistilBERT was 0.62 percentage points, while the improvement over SVM was 0.08 percentage points.

## Precision

Logistic Regression achieved the highest precision at 98.60%, followed closely by SVM at 98.62% and DistilBERT at 98.47%.

The small differences indicate that all three approaches were highly effective at limiting false-positive predictions.

## Recall

DistilBERT achieved the highest recall at 98.32%, followed by SVM at 97.94% and Logistic Regression at 96.49%.

Recall is particularly important for phishing detection because false negatives represent phishing emails that are incorrectly classified as safe.

The higher recall achieved by DistilBERT indicates that it identified a larger proportion of phishing emails than the two traditional baseline models.

## F1-Score

DistilBERT achieved the highest F1-score at 98.39%, followed by SVM at 98.28% and Logistic Regression at 97.53%.

The F1-score provides a combined measure of precision and recall and therefore provides a useful overall comparison of classification performance.

## ROC-AUC

SVM and DistilBERT both achieved a ROC-AUC of 99.89%, while Logistic Regression achieved 99.81%.

The results indicate that all three models provided excellent discrimination between Safe Email and Phishing Email.

## Overall Comparison

The comparison demonstrates that all three approaches performed strongly on the evaluated dataset.

DistilBERT produced:

- The highest accuracy.
- The highest recall.
- The highest F1-score.
- ROC-AUC equal to the SVM baseline.
- Precision slightly below the two traditional baseline models.

SVM provided the strongest traditional-machine-learning baseline, while DistilBERT provided the strongest overall performance based on accuracy, recall and F1-score.

## Interpretation

The performance results suggest that the transformer-based approach provides a modest but measurable improvement over the traditional baseline models.

The difference between SVM and DistilBERT is relatively small, particularly in accuracy and ROC-AUC. Therefore, the results do not suggest that DistilBERT completely outperforms traditional machine-learning methods across every metric.

However, the improvement in recall and F1-score is relevant to phishing detection because the ability to identify phishing emails while maintaining high precision is an important consideration.

## Evidence

The comparison results were generated from the model evaluation performed in Google Colab. The evidence includes the final performance table and the corresponding performance comparison visualisation.

The results provide the quantitative basis for selecting DistilBERT as the primary deep learning model for the project.
