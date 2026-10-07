# DeBERTa-v3-large Gated-Hybrid Phishing Detection Model

This directory contains the implementation and testing files for the
DeBERTa-v3-large gated-hybrid phishing email detection model developed
as part of the NLP-Based Phishing and Social Engineering Email Detection project.

The implementation combines contextual representations from
DeBERTa-v3-large with engineered phishing-related and structural
features through a learned gated-fusion architecture.

## Files

- `custom_deberta_model.py` – implementation of the custom gated-hybrid
  DeBERTa-v3-large classification architecture.
- `predict.py` – inference and prediction functionality.
- `test_custom_deberta.py` – testing of the custom gated-hybrid model.
- `test_deberta.py` – testing of the DeBERTa model implementation.
- `checkpoint_keys.txt` – checkpoint/model key information used during
  implementation and verification.
