# Experimental Configuration

## Development Environment

The machine-learning experiments were conducted using Google Colab with GPU acceleration enabled for DistilBERT training.

### Hardware and Software

| Component | Configuration |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU Memory | Approximately 14.56 GB |
| PyTorch | 2.11.0+cu128 |
| CUDA | Available |
| Transformer framework | Hugging Face Transformers |
| Dataset framework | Hugging Face Datasets |
| Evaluation framework | Scikit-learn / custom evaluation functions |

## Dataset

The final cleaned dataset contained:

- 17,521 unique email records
- 10,978 Safe Emails
- 6,543 Phishing Emails
- No missing values
- No duplicate email texts

The dataset was divided into:

- Training set: 14,016 samples
- Testing set: 3,505 samples

The label mapping was:

```text
Safe Email      = 0
Phishing Email  = 1


he train/test split maintained approximately the same class proportions in both subsets.

DistilBERT Configuration

The transformer model used a maximum sequence length of 512 tokens.

The training configuration was:

Parameter	Value
Batch size	8
Learning rate	2 × 10⁻⁵
Epochs	2
Weight decay	0.01
Maximum gradient norm	1.0
Optimizer	AdamW
Warmup steps	350
Total training steps	3,504
Scheduler	Linear warmup + linear decay
Evaluation Metrics

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC

A confusion matrix and classification report were also generated for the final DistilBERT evaluation.

Reproducibility

The experimental configuration was recorded to support reproducibility of the modelling process.

The DistilBERT model and tokenizer were also saved as project artefacts after training.
