# SHAP Browser Extension

This directory contains the browser-extension implementation developed
for the SHAP-based version of the NLP-Based Phishing and Social
Engineering Email Detection system.

The browser extension provides browser-integrated phishing analysis by
extracting the currently displayed email from the controlled email
testing environment and communicating with the project's FastAPI
backend.

The extension provides two interaction mechanisms:

1. **Analyse Current Email** – analysis initiated through the browser
   extension popup.
2. **Scan Email for Phishing** – analysis initiated through the button
   injected into the controlled email interface.

Both interaction mechanisms use the same browser-extension integration
and FastAPI inference backend. They are alternative interaction methods
within the same browser-extension implementation rather than separate
detection systems.

## Main Files

- `manifest.json` – Chrome Manifest V3 extension configuration.
- `content.js` – email extraction, page integration and communication
  functionality.
- `popup.html` – browser-extension popup interface.
- `popup.css` – popup styling.
- `popup.js` – popup interaction and analysis functionality.

## Testing

The `test-environment` directory contains controlled HTML-based email
testing environments used for end-to-end browser-extension testing.

## Explainability

The associated phishing-detection system uses SHAP-based explainability.
SHAP implementation and explainability-specific code are maintained in
the project's `XAI` directory.
