import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./deberta_v3_large_phishing_model"
MAX_LENGTH = 512


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DeBERTa-v3-large Standalone Model Test")
print("=" * 70)


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel directory not found:\n{os.path.abspath(MODEL_PATH)}"
    )

print(f"\nModel directory:")
print(os.path.abspath(MODEL_PATH))


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU for this test.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "-" * 70)
print("Loading tokenizer...")
print("-" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "-" * 70)
print("Loading DeBERTa-v3-large model...")
print("-" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Model type:              {model.config.model_type}")
print(f"Number of labels:        {model.config.num_labels}")
print(f"Hidden size:             {model.config.hidden_size}")
print(f"Hidden layers:           {model.config.num_hidden_layers}")
print(f"Attention heads:         {model.config.num_attention_heads}")
print(f"Intermediate size:       {model.config.intermediate_size}")
print(f"Maximum sequence length: {model.config.max_position_embeddings}")
print(f"Vocabulary size:         {model.config.vocab_size}")

print("\nLabel mapping:")

if hasattr(model.config, "id2label"):
    print(model.config.id2label)
else:
    print("No id2label mapping found.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_email(email_text):
    """
    Classify one email using the fine-tuned DeBERTa model.
    """

    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    safe_probability = probabilities[0].item()
    phishing_probability = probabilities[1].item()

    return (
        predicted_class,
        safe_probability,
        phishing_probability
    )


# ============================================================
# TEST 1 — SAFE EMAIL
# ============================================================

safe_email = """
Subject: University Library Opening Hours

Dear Student,

The university library will operate from 8:00 AM to 8:00 PM
during the examination period.

Please check the university website for the current opening
hours and available study spaces.

Regards,
University Library
"""


print("\n" + "=" * 70)
print("TEST 1 — SAFE EMAIL")
print("=" * 70)

prediction, safe_prob, phishing_prob = predict_email(
    safe_email
)

if prediction == 0:
    prediction_text = "Safe Email"
else:
    prediction_text = "Phishing Email"

confidence = max(
    safe_prob,
    phishing_prob
)

print(f"\nPrediction:         {prediction_text}")
print(f"Confidence:         {confidence * 100:.2f}%")
print(f"Safe probability:   {safe_prob * 100:.2f}%")
print(f"Phishing probability: {phishing_prob * 100:.2f}%")


# ============================================================
# TEST 2 — SYNTHETIC SUSPICIOUS EMAIL
# ============================================================

suspicious_email = """
Subject: Urgent Account Verification Required

Dear User,

Your account requires immediate verification.

Please review your account information through the
official organisation website before continuing.

If you did not request this notification, contact the
organisation using a verified contact method.

Regards,
Account Security Team
"""


print("\n" + "=" * 70)
print("TEST 2 — SYNTHETIC SUSPICIOUS EMAIL")
print("=" * 70)

prediction, safe_prob, phishing_prob = predict_email(
    suspicious_email
)

if prediction == 0:
    prediction_text = "Safe Email"
else:
    prediction_text = "Phishing Email"

confidence = max(
    safe_prob,
    phishing_prob
)

print(f"\nPrediction:         {prediction_text}")
print(f"Confidence:         {confidence * 100:.2f}%")
print(f"Safe probability:   {safe_prob * 100:.2f}%")
print(f"Phishing probability: {phishing_prob * 100:.2f}%")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STANDALONE MODEL TEST COMPLETED")
print("=" * 70)