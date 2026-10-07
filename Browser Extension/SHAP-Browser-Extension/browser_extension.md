# Browser Extension – Additional Stretch Goal

## Overview

As an additional stretch goal, a browser extension was developed to demonstrate how the phishing email detection system could be accessed through a practical browser-based workflow.

The extension communicates with the project's FastAPI backend rather than running the machine-learning model directly within the extension.

## Architecture

The general workflow is:

Browser Extension
→ FastAPI Backend
→ DeBERTa-based Detection Model
→ Prediction
→ Browser Extension

Where explanation functionality is requested:

Browser Extension
→ FastAPI `/explain`
→ SHAP
→ Explanation Response
→ Browser Extension

## Purpose

The purpose of the extension was to investigate how the developed phishing detection system could be extended beyond the Streamlit prototype.

The extension provides an additional interface for interacting with the detection service while maintaining separation between the user interface, API and machine-learning model.

## Integration

The browser extension was integrated with the FastAPI backend through API requests.

This allows the extension to make use of the same detection service used by the main prototype.

## Scope

The browser extension was developed as an additional stretch goal and was not part of the original core project requirements.

Its development demonstrates additional initiative in exploring the practical application of the phishing detection system.

## Outcome

The extension extended the project beyond the primary Streamlit interface and demonstrated how the developed DeBERTa-based detection service could support an additional user-facing interface.
