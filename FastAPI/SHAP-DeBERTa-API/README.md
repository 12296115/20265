# SHAP DeBERTa FastAPI Backend

This directory contains the FastAPI backend used by the DeBERTa-v3-large
gated-hybrid phishing detection system.

The API connects the user-facing interfaces to the classification and
SHAP explainability pipeline. It provides endpoints for prediction,
explanation and service health checking.

## Main File

- `api.py` – FastAPI application providing prediction, SHAP explanation
  and health-check endpoints.

## Related Components

- DeBERTa-v3-large gated-hybrid classifier
- SHAP explainability
- Streamlit interface
- SHAP browser extension
