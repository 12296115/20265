import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("./deberta_v3_large_phishing_model")

MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
FEATURE_MEAN_PATH = MODEL_DIR / "feature_mean.npy"
FEATURE_STD_PATH = MODEL_DIR / "feature_std.npy"
CHECKPOINT_PATH = MODEL_DIR / "pytorch_model.bin"


# ============================================================
# LOAD CONFIGURATION
# ============================================================

with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)

with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)

FEATURE_MEAN = np.load(FEATURE_MEAN_PATH)
FEATURE_STD = np.load(FEATURE_STD_PATH)


MODEL_NAME = MODEL_CONFIG["model_name"]
NUM_FEATURES = MODEL_CONFIG["num_features"]
NUM_LABELS = MODEL_CONFIG["num_labels"]
MAX_LENGTH = MODEL_CONFIG["max_length"]


print("=" * 70)
print("CUSTOM DEBERTA PHISHING MODEL TEST")
print("=" * 70)

print(f"Model:              {MODEL_NAME}")
print(f"Number of features: {NUM_FEATURES}")
print(f"Number of labels:   {NUM_LABELS}")
print(f"Maximum length:     {MAX_LENGTH}")
print(f"Architecture:       {MODEL_CONFIG['architecture']}")
print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

URGENCY_TERMS = [
    "urgent",
    "immediately",
    "immediate",
    "asap",
    "action required",
    "act now",
    "respond now",
    "important",
    "critical",
    "warning",
    "deadline",
]

CREDENTIAL_TERMS = [
    "password",
    "username",
    "login",
    "log in",
    "sign in",
    "credential",
    "credentials",
    "verify your account",
    "verification",
    "authentication",
    "security code",
    "otp",
]

THREAT_TERMS = [
    "suspend",
    "suspended",
    "blocked",
    "block",
    "terminate",
    "terminated",
    "deactivate",
    "deactivated",
    "locked",
    "lock",
    "warning",
    "penalty",
    "legal action",
]

FINANCIAL_TERMS = [
    "payment",
    "invoice",
    "bank",
    "banking",
    "account",
    "credit card",
    "debit card",
    "refund",
    "transaction",
    "money",
    "transfer",
    "billing",
    "fee",
]

CTA_TERMS = [
    "click",
    "click here",
    "open",
    "download",
    "verify",
    "confirm",
    "update",
    "submit",
    "review",
    "visit",
    "access",
    "login",
    "sign in",
]


def count_terms(text, terms):
    """
    Count occurrences of the supplied terms.

    Matching is case-insensitive.
    """
    text_lower = text.lower()

    count = 0

    for term in terms:
        if " " in term:
            count += text_lower.count(term.lower())
        else:
            count += len(
                re.findall(
                    rf"\b{re.escape(term.lower())}\b",
                    text_lower
                )
            )

    return count


def extract_nlp_features(text):
    """
    Extract the 12 engineered NLP features used by the
    GatedHybridPhishingModel.
    """

    text = str(text)

    words = text.split()

    # --------------------------------------------------------
    # 1. Urgency terms
    # --------------------------------------------------------
    urgency_term_count = count_terms(
        text,
        URGENCY_TERMS
    )

    # --------------------------------------------------------
    # 2. Credential terms
    # --------------------------------------------------------
    credential_term_count = count_terms(
        text,
        CREDENTIAL_TERMS
    )

    # --------------------------------------------------------
    # 3. Threat terms
    # --------------------------------------------------------
    threat_term_count = count_terms(
        text,
        THREAT_TERMS
    )

    # --------------------------------------------------------
    # 4. Financial terms
    # --------------------------------------------------------
    financial_term_count = count_terms(
        text,
        FINANCIAL_TERMS
    )

    # --------------------------------------------------------
    # 5. Call-to-action terms
    # --------------------------------------------------------
    cta_term_count = count_terms(
        text,
        CTA_TERMS
    )

    # --------------------------------------------------------
    # 6. URL count
    # --------------------------------------------------------
    url_count = len(
        re.findall(
            r"(?:https?://|www\.)\S+",
            text,
            flags=re.IGNORECASE
        )
    )

    # --------------------------------------------------------
    # 7. Email count
    # --------------------------------------------------------
    email_count = len(
        re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text
        )
    )

    # --------------------------------------------------------
    # 8. Phone count
    # --------------------------------------------------------
    phone_count = len(
        re.findall(
            r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)",
            text
        )
    )

    # --------------------------------------------------------
    # 9. Exclamation count
    # --------------------------------------------------------
    exclamation_count = text.count("!")

    # --------------------------------------------------------
    # 10. Question count
    # --------------------------------------------------------
    question_count = text.count("?")

    # --------------------------------------------------------
    # 11. Uppercase word ratio
    # --------------------------------------------------------
    if len(words) > 0:
        uppercase_words = sum(
            1 for word in words
            if len(re.sub(r"[^A-Za-z]", "", word)) > 1
            and re.sub(r"[^A-Za-z]", "", word).isupper()
        )

        uppercase_word_ratio = (
            uppercase_words / len(words)
        )
    else:
        uppercase_word_ratio = 0.0

    # --------------------------------------------------------
    # 12. Log text length
    # --------------------------------------------------------
    log_text_length = np.log1p(len(text))

    features = np.array(
        [
            urgency_term_count,
            credential_term_count,
            threat_term_count,
            financial_term_count,
            cta_term_count,
            url_count,
            email_count,
            phone_count,
            exclamation_count,
            question_count,
            uppercase_word_ratio,
            log_text_length,
        ],
        dtype=np.float32
    )

    return features


# ============================================================
# CUSTOM MODEL
# ============================================================

class GatedHybridPhishingModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_features=12,
        num_labels=2
    ):
        super().__init__()

        # ----------------------------------------------------
        # Pretrained DeBERTa backbone
        # ----------------------------------------------------
        self.transformer = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        hidden_size = self.transformer.config.hidden_size

        # ----------------------------------------------------
        # Context projection
        # ----------------------------------------------------
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ----------------------------------------------------
        # NLP feature projection
        # ----------------------------------------------------
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ----------------------------------------------------
        # Gated fusion
        # ----------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------
        self.classifier = nn.Linear(
            256,
            num_labels
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        nlp_features=None,
        labels=None
    ):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS representation
        contextual_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        context = self.context_projection(
            contextual_embedding
        )

        features = self.feature_projection(
            nlp_features.float()
        )

        # Concatenate context + explicit NLP features
        gate_input = torch.cat(
            [context, features],
            dim=1
        )

        gate = self.gate(
            gate_input
        )

        # Gated fusion
        fused = (
            gate * context
            + (1 - gate) * features
        )

        fused = self.fusion(
            fused
        )

        logits = self.classifier(
            fused
        )

        loss = None

        if labels is not None:
            loss = self.loss_fn(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True
)

print("Tokenizer loaded.")
print()


print("Creating model architecture...")

model = GatedHybridPhishingModel(
    MODEL_NAME,
    num_features=NUM_FEATURES,
    num_labels=NUM_LABELS
)

print("Architecture created.")
print()


print("Loading trained checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint,
    strict=True
)

model.eval()

print("Checkpoint loaded successfully.")
print()

print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
print()


# ============================================================
# TEST PREDICTION
# ============================================================

def predict_email(text):

    # Extract features
    features = extract_nlp_features(text)

    # Standardise using training statistics
    features = (
        features - FEATURE_MEAN
    ) / (
        FEATURE_STD + 1e-8
    )

    features_tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0)

    # Tokenize
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():

        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            nlp_features=features_tensor
        )

        probabilities = torch.softmax(
            output["logits"],
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

    return {
        "prediction": prediction,
        "safe_probability": float(probabilities[0]),
        "phishing_probability": float(probabilities[1]),
        "features": features
    }


# ============================================================
# RUN TEST
# ============================================================

test_email = """
Hello,

This is a routine notification from the university IT service desk.
Your account information has been reviewed and no action is required.
If you have questions, please contact the IT help desk through the
usual university support channels.

Regards,
University IT Support
"""

result = predict_email(test_email)

print("=" * 70)
print("TEST PREDICTION")
print("=" * 70)

print(
    "Prediction:",
    "PHISHING" if result["prediction"] == 1 else "SAFE"
)

print(
    f"Safe probability:     {result['safe_probability']:.6f}"
)

print(
    f"Phishing probability: {result['phishing_probability']:.6f}"
)

print()
print("Standardised NLP features:")

for name, value in zip(
    FEATURE_NAMES,
    result["features"]
):
    print(
        f"{name:25s}: {value:.6f}"
    )

print()
print("=" * 70)
print("MODEL TEST COMPLETE")
print("=" * 70)