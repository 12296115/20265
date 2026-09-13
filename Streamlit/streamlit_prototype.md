# Streamlit Prototype

## Overview

A Streamlit-based user interface was developed to provide a user-friendly front end for the phishing and social-engineering email detection system.

The Streamlit application communicates with the FastAPI backend rather than loading and executing the machine-learning model directly in the frontend.

## System Workflow

The user-facing workflow is:

Email Text
→ Streamlit Interface
→ FastAPI Backend
→ DeBERTa-v3-large
→ Classification Result
→ Streamlit Display

For explanations:

Email Text
→ FastAPI `/explain`
→ SHAP Analysis
→ Explanation
→ Streamlit Display

## Main Functions

The Streamlit prototype allows users to:

- Enter or provide email content for analysis.
- Submit the email to the backend prediction service.
- Receive the phishing/safe classification.
- View the prediction result.
- Request and view model explanations.
- Interpret the classification using the SHAP-based explanation.

## Backend Communication

The frontend communicates with the FastAPI REST API through HTTP requests.

This separation provides a modular architecture where:

- The frontend handles user interaction and presentation.
- FastAPI manages API requests and model inference.
- The DeBERTa-based model performs classification.
- SHAP provides explainability.

## Design Considerations

Separating the frontend from the machine-learning backend makes the prototype easier to maintain and allows the same prediction service to support additional interfaces.

This architecture also supports the project's later browser-extension integration.

## Outcome

The Streamlit interface transformed the model and backend components into an accessible user-facing prototype.

It provides an end-to-end workflow from email input to phishing classification and explainability.
