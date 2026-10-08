# NLP-Based Phishing and Social-Engineering Email Detection

## 1. Project Overview

This project is an NLP-based system that detects phishing and social-engineering emails. It accepts the text of an email, processes the content and classifies it as either:

- **Phishing**
- **Legitimate**

The system also returns a confidence score and a LIME explanation showing which words influenced the prediction. A Flask API provides access to the model, while a Chrome extension allows users to analyse email content through a simple interface.

This implementation forms the backend and browser-integration components of the university capstone project.

---

## 2. Main Features

The implemented system provides the following functions:

- Accepts email text through a REST API
- Validates empty or invalid requests
- Normalises email text before analysis
- Extracts 12 additional NLP-based features
- Uses a saved transformer-based phishing detection model
- Classifies messages as phishing or legitimate
- Returns a model confidence score
- Generates a local explanation using LIME
- Identifies words that support or oppose the prediction
- Provides Gmail and manual-text analysis through a Chrome extension
- Keeps model processing on the local computer

---

## 3. Backend Architecture

The backend follows this processing sequence:

1. The user enters email text manually or opens an email in Gmail.
2. The Chrome extension collects the available email content.
3. The extension sends the text as JSON to the Flask API.
4. The Flask API validates the request.
5. The predictor normalises the email text.
6. Twelve NLP features are calculated and standardised.
7. The email text is tokenised for the transformer model.
8. The saved hybrid model calculates class probabilities.
9. The class with the highest probability becomes the final prediction.
10. LIME generates an explanation for the prediction.
11. The API returns the verdict, score and explanation as JSON.
12. The Chrome extension displays the result to the user.

The saved model is loaded once when the Flask application starts. This prevents the large model from being loaded again for every API request.

---

## 4. Main Project Files

```text
phishing-project/
├── README.md
├── requirements.txt
├── app.py
├── predictor.py
├── check_model.py
├── model/
│   ├── config.json
│   ├── model_config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── feature_mean.npy
│   ├── feature_std.npy
│   ├── feature_names.json
│   └── test_results.json
└── chrome_extension/
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    └── icons/
    
