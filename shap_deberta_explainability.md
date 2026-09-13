# SHAP Explainability for DeBERTa

## Overview

Explainable Artificial Intelligence (XAI) was investigated to improve the interpretability of phishing email classification results.

Two approaches, LIME and SHAP, were considered during the investigation. SHAP was selected for implementation because it provides feature contribution values that can be used to identify which parts of an input contributed to the model's prediction.

## SHAP Integration

SHAP was integrated with the DeBERTa-based phishing detection system to provide token-level explanations for individual email predictions.

The explanation workflow is:

Email Input
→ DeBERTa-v3-large
→ Classification
→ SHAP Analysis
→ Token-Level Contributions
→ User-Readable Explanation

The explanation identifies tokens or sections of the email that contribute towards the classification outcome.

## Purpose

The purpose of the SHAP implementation is to improve transparency and help users understand why an email has been classified as safe or phishing.

This is particularly relevant to phishing detection because suspicious emails may contain linguistic indicators such as urgency, requests for credentials, financial requests or calls to action.

## System Integration

SHAP was integrated into the backend explanation workflow. The system provides a separate explanation process in addition to the prediction process.

The API includes an `/explain` endpoint that obtains the model prediction and generates SHAP-based explanatory information.

## Outcome

The SHAP implementation extended the project from a classification-only system to an explainable phishing detection prototype.

The resulting explanations provide additional information about the model's decision and support the project's objective of developing an interpretable phishing detection system.
