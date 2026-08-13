# DistilBERT Error Analysis and Model Interpretation

## Purpose

Following model evaluation, an error analysis was conducted to investigate incorrect predictions and identify patterns in the classification errors.

The analysis focused on:

- False positives
- False negatives
- Prediction confidence
- Email length
- Token length
- The effect of the 512-token input limit

## Overall Classification Errors

The final DistilBERT model produced 42 incorrect predictions from 3,505 test emails.

| Error Type | Count |
|---|---:|
| False Positive | 20 |
| False Negative | 22 |
| **Total** | **42** |

The relatively balanced number of false positives and false negatives indicates that the model did not exhibit a strong bias toward either class in its incorrect predictions.

## False Positives

There were 20 false-positive predictions.

A false positive occurred when a Safe Email was incorrectly classified as a Phishing Email.

Examples included legitimate messages containing terminology or structures that may resemble suspicious communication patterns. For example, some messages contained references to domains, technical systems, order confirmations, or other content that could potentially resemble phishing-related language.

The presence of such examples demonstrates that legitimate emails can contain features that overlap with characteristics commonly associated with phishing.

## False Negatives

There were 22 false-negative predictions.

A false negative occurred when a Phishing Email was incorrectly classified as a Safe Email.

Examples included messages with relatively ordinary or contextual language, including:

- Industrial Linux news
- Conference invitations
- Requests for information
- Technical discussions
- Promotional messages
- Other messages that did not contain sufficiently strong phishing indicators

The false-negative examples are particularly important because they represent phishing emails that could potentially pass through the detection system without being identified.

## Prediction Confidence

The probability associated with each prediction was examined to identify highly confident errors and borderline classifications.

The analysis identified:

- 31 highly confident incorrect predictions
- 73.81% of all errors were highly confident
- 6 borderline errors had probabilities between 0.30 and 0.70
- 14.29% of all errors were borderline

These results indicate that most incorrect predictions were not simply cases where the model was close to the classification boundary. Some errors were made with relatively high confidence.

This suggests that future improvement should focus not only on classification thresholds but also on improving the model's representation of difficult or ambiguous email patterns.

## False-Negative Confidence

The false-negative predictions showed a wide range of phishing probabilities.

The hardest false negatives had very low predicted phishing probabilities. Examples included:

| Test Index | Phishing Probability | Text Length |
|---:|---:|---:|
| 1068 | 0.000111 | 762 |
| 997 | 0.000118 | 6,773 |
| 858 | 0.000275 | 614 |
| 3482 | 0.000381 | 1,398 |
| 368 | 0.000485 | 99 |
| 2447 | 0.000663 | 421 |
| 118 | 0.000700 | 1,974 |
| 274 | 0.000792 | 239 |
| 3419 | 0.000806 | 77 |
| 2813 | 0.001190 | 189 |

These examples demonstrate that some phishing emails were assigned extremely low phishing probabilities despite belonging to the phishing class.

## Email Length Analysis

Email length was analysed to determine whether incorrect predictions were associated with unusually short or long messages.

### Prediction Correctness

| Prediction | Count | Mean Length | Median Length |
|---|---:|---:|---:|
| Correct | 3,463 | 1,875.24 | 916 |
| Incorrect | 42 | 2,186.83 | 782 |

The mean length of incorrectly classified emails was higher than that of correctly classified emails. However, the median length of incorrect predictions was lower than the median for correct predictions.

This indicates that email length alone does not provide a simple explanation for classification errors.

### Actual Class and Prediction Correctness

For Safe Email:

| Prediction | Count | Mean Length |
|---|---:|---:|
| Correct | 2,176 | 1,986.57 |
| Incorrect | 20 | 3,054.05 |

For Phishing Email:

| Prediction | Count | Mean Length |
|---|---:|---:|
| Correct | 1,287 | 1,687.01 |
| Incorrect | 22 | 1,398.45 |

The results suggest that incorrectly classified Safe Emails were, on average, longer than correctly classified Safe Emails, whereas incorrectly classified Phishing Emails were somewhat shorter than correctly classified Phishing Emails.

## Long Email Analysis

Emails were also divided into two groups using 2,000 characters as the analysis threshold.

There were:

- 3,505 total test emails
- 844 emails longer than 2,000 characters
- 2,661 emails at or below 2,000 characters

Prediction accuracy for emails longer than 2,000 characters was:

**98.8152%**

Prediction accuracy for emails at or below 2,000 characters was:

**98.7974%**

The difference is very small, indicating that character length above 2,000 characters did not substantially affect classification accuracy in this test set.

## DistilBERT Token-Length Analysis

Because DistilBERT was configured with a maximum sequence length of 512 tokens, the actual token lengths were also examined.

The analysis reported:

| Statistic | Token Length |
|---|---:|
| Mean | 486.86 |
| Median | 245 |
| 75th percentile | 485 |
| Maximum | 89,982 |

The maximum value represents an exceptionally long input before the model's sequence-length constraint is applied.

The analysis identified:

- 2,679 emails not exceeding 512 tokens
- 826 emails exceeding 512 tokens

Prediction accuracy was:

| Input Length | Accuracy |
|---|---:|
| ≤ 512 tokens | 98.8429% |
| > 512 tokens | 98.6683% |

The difference between the two groups was approximately 0.17 percentage points.

Therefore, although truncation is a potential limitation of the 512-token configuration, the observed difference in accuracy between the two groups was relatively small in this test set.

## Interpretation

The error analysis demonstrates that the model achieved very high overall performance but still produced a small number of difficult classification errors.

Several observations are important:

1. False negatives slightly outnumbered false positives.
2. Some false negatives received extremely low phishing probabilities.
3. Many incorrect predictions were made with high confidence.
4. Email character length alone did not clearly explain prediction errors.
5. Emails exceeding 512 tokens had slightly lower accuracy than shorter inputs.
6. The 512-token limitation therefore represents a potential area for future investigation, particularly for unusually long emails.

These findings provide a basis for future model improvement and demonstrate that evaluation was not limited to aggregate performance metrics.

## Limitations Identified

The analysis identified several areas that could be investigated in future work:

- Improving detection of difficult false-negative examples.
- Investigating highly confident incorrect predictions.
- Examining whether longer contextual windows improve classification.
- Comparing alternative transformer architectures.
- Investigating explainability techniques to determine which words or phrases influence predictions.
- Evaluating the model on additional and potentially more diverse phishing datasets.

## Evidence

The analysis was performed using the final DistilBERT predictions generated during the Google Colab evaluation process. The evidence includes confusion-matrix results, error tables, prediction probabilities, email-length statistics and token-length analysis.
