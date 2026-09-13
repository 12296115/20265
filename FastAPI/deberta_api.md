# FastAPI Backend Integration

## Overview

The trained DeBERTa-v3-large model was integrated into a FastAPI backend to provide a REST-based prediction service for the phishing email detection prototype.

The backend provides an interface between the trained machine-learning model and the user-facing applications.

## System Architecture

The general workflow is:

User Input
→ Streamlit / Browser Extension
→ FastAPI API
→ DeBERTa-v3-large
→ Prediction
→ API Response

For explainability:

User Input
→ FastAPI `/explain`
→ Model Prediction
→ SHAP Analysis
→ Explanation Response

## Prediction Endpoint

The `/predict` endpoint accepts email text and returns the model's classification result.

The prediction process uses the DeBERTa-based phishing detection model and obtains supplementary contextual/security indicators where implemented.

These supplementary indicators provide additional information about the email but do not independently override the model prediction.

## Explainability Endpoint

The `/explain` endpoint performs the prediction and generates SHAP-based explanatory information.

This allows the system to provide both:

- Classification result
- Explanation of the model prediction

## Model Integration

The backend integrates the project's DeBERTa-v3-large model and supports the enhanced model architecture developed during the experimentation stage.

The API separates the machine-learning model from the user interface, allowing multiple interfaces to communicate with the same prediction service.

## Purpose

Using FastAPI provides a modular system architecture in which:

- The machine-learning model is hosted by the backend.
- User interfaces communicate through API requests.
- Prediction and explanation functionality are exposed through separate endpoints.
- The same backend can support the Streamlit interface and browser extension.

## Outcome

The FastAPI integration transformed the trained model from an experimental notebook component into a reusable prediction service suitable for integration with the project's user-facing prototype.
