import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-large"

MODEL_PATH = Path("./deberta_v3_large_phishing_model")

MAX_LENGTH = 512
NUM_FEATURES = 12
NUM_LABELS = 2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# NLP FEATURE VOCABULARIES
# ============================================================

URGENCY_WORDS = {
    "urgent",
    "urgently",
    "immediately",
    "immediate",
    "now",
    "today",
    "asap",
    "quickly",
    "deadline",
    "expire",
    "expired",
    "final",
    "action"
}

CREDENTIAL_WORDS = {
    "password",
    "passwd",
    "username",
    "login",
    "credential",
    "credentials",
    "verify",
    "verification",
    "authenticate",
    "authentication",
    "account"
}

THREAT_WORDS = {
    "suspend",
    "suspended",
    "terminate",
    "terminated",
    "blocked",
    "block",
    "close",
    "closed",
    "penalty",
    "fraud",
    "unauthorized",
    "warning",
    "security"
}

FINANCIAL_WORDS = {
    "payment",
    "pay",
    "invoice",
    "money",
    "bank",
    "transfer",
    "transaction",
    "refund",
    "credit",
    "debit",
    "fee",
    "account",
    "billing"
}

CTA_WORDS = {
    "click",
    "clicking",
    "visit",
    "open",
    "download",
    "confirm",
    "verify",
    "submit",
    "update",
    "activate",
    "login"
}


# ============================================================
# NLP FEATURE EXTRACTION
# ============================================================

def count_terms(text, vocabulary):
    """
    Count how many words in the text belong to
    the supplied vocabulary.
    """

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    return sum(
        word in vocabulary
        for word in words
    )


def extract_nlp_features(text):
    """
    Extract the exact 12 NLP features used during
    training of the DeBERTa hybrid model.
    """

    text = str(text)

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    word_count = max(
        len(words),
        1
    )

    # --------------------------------------------------------
    # Uppercase word ratio
    # --------------------------------------------------------

    uppercase_words = [
        word
        for word in words
        if len(word) > 1 and word.isupper()
    ]

    uppercase_ratio = (
        len(uppercase_words) / word_count
    )

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE
        )
    )

    # --------------------------------------------------------
    # Email addresses
    # --------------------------------------------------------

    email_count = len(
        re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text
        )
    )

    # --------------------------------------------------------
    # Phone numbers
    # --------------------------------------------------------

    phone_count = len(
        re.findall(
            r"(?:\+?\d[\d\s().-]{7,}\d)",
            text
        )
    )

    # --------------------------------------------------------
    # EXACT 12 FEATURES
    # --------------------------------------------------------

    features = [
        count_terms(text, URGENCY_WORDS),       # 1
        count_terms(text, CREDENTIAL_WORDS),    # 2
        count_terms(text, THREAT_WORDS),        # 3
        count_terms(text, FINANCIAL_WORDS),     # 4
        count_terms(text, CTA_WORDS),           # 5
        url_count,                              # 6
        email_count,                            # 7
        phone_count,                            # 8
        text.count("!"),                        # 9
        text.count("?"),                        # 10
        uppercase_ratio,                        # 11
        np.log1p(len(text))                     # 12
    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# DEBERTA GATED HYBRID MODEL
# ============================================================

class GatedHybridPhishingModel(nn.Module):
    """
    Exact architecture used in Proposed_model_draft2.

    Architecture:

        DeBERTa-v3-large
                |
        contextual embedding
                |
        256-dimensional projection
                |
                |------\
                |       \
                |        Gated Fusion
                |       /
        12 NLP features
                |
        256-dimensional projection
                |
        Fusion layer
                |
        Classifier
                |
        Safe / Phishing
    """

    def __init__(
        self,
        model_name,
        num_features=12,
        num_labels=2
    ):
        super().__init__()

        # ----------------------------------------------------
        # DeBERTa backbone
        # ----------------------------------------------------

        self.transformer = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        hidden_size = (
            self.transformer.config.hidden_size
        )

        # ----------------------------------------------------
        # Context projection
        # ----------------------------------------------------

        self.context_projection = nn.Sequential(
            nn.Linear(
                hidden_size,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ----------------------------------------------------
        # NLP feature projection
        # ----------------------------------------------------

        self.feature_projection = nn.Sequential(
            nn.Linear(
                num_features,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ----------------------------------------------------
        # Gated fusion
        # ----------------------------------------------------

        self.gate = nn.Sequential(
            nn.Linear(
                512,
                256
            ),
            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # Fusion layer
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(
                256,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            256,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        nlp_features
    ):

        # ----------------------------------------------------
        # Transformer representation
        # ----------------------------------------------------

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        contextual_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        # ----------------------------------------------------
        # Project transformer features
        # ----------------------------------------------------

        context = self.context_projection(
            contextual_embedding
        )

        # ----------------------------------------------------
        # Project handcrafted NLP features
        # ----------------------------------------------------

        features = self.feature_projection(
            nlp_features
        )

        # ----------------------------------------------------
        # Create gate
        # ----------------------------------------------------

        gate_input = torch.cat(
            [
                context,
                features
            ],
            dim=1
        )

        gate = self.gate(
            gate_input
        )

        # ----------------------------------------------------
        # Gated fusion
        # ----------------------------------------------------

        fused = (
            gate * context
            + (1 - gate) * features
        )

        # ----------------------------------------------------
        # Fusion transformation
        # ----------------------------------------------------

        fused = self.fusion(
            fused
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading DeBERTa phishing detection model")
print("=" * 60)

print()
print("Model:", MODEL_NAME)
print("Checkpoint:", MODEL_PATH)
print("Device:", device)


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model directory not found: {MODEL_PATH}"
    )


checkpoint_path = (
    MODEL_PATH / "pytorch_model.bin"
)

mean_path = (
    MODEL_PATH / "feature_mean.npy"
)

std_path = (
    MODEL_PATH / "feature_std.npy"
)


if not checkpoint_path.exists():

    raise FileNotFoundError(
        f"Model checkpoint not found: {checkpoint_path}"
    )


if not mean_path.exists():

    raise FileNotFoundError(
        f"Feature mean not found: {mean_path}"
    )


if not std_path.exists():

    raise FileNotFoundError(
        f"Feature standard deviation not found: {std_path}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("Loading tokenizer from trained model folder...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD FEATURE NORMALISATION
# ============================================================

print()
print("Loading feature normalisation statistics...")

feature_mean = np.load(
    mean_path
)

feature_std = np.load(
    std_path
).copy()

# Avoid division by zero
feature_std[
    feature_std == 0
] = 1.0


print(
    "Feature mean shape:",
    feature_mean.shape
)

print(
    "Feature std shape:",
    feature_std.shape
)


# ============================================================
# VALIDATE FEATURE DIMENSIONS
# ============================================================

if len(feature_mean) != NUM_FEATURES:

    raise ValueError(
        f"Expected {NUM_FEATURES} feature means, "
        f"but found {len(feature_mean)}."
    )


if len(feature_std) != NUM_FEATURES:

    raise ValueError(
        f"Expected {NUM_FEATURES} feature standard deviations, "
        f"but found {len(feature_std)}."
    )


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating model architecture...")

model = GatedHybridPhishingModel(
    model_name=MODEL_NAME,
    num_features=NUM_FEATURES,
    num_labels=NUM_LABELS
)


print("Architecture created successfully.")


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

print()
print("Loading trained checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)


# ============================================================
# HANDLE CHECKPOINT FORMAT
# ============================================================

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

missing_keys, unexpected_keys = (
    model.load_state_dict(
        state_dict,
        strict=False
    )
)


if missing_keys:

    print()
    print("WARNING - Missing model keys:")

    for key in missing_keys:
        print(" ", key)


if unexpected_keys:

    print()
    print("WARNING - Unexpected model keys:")

    for key in unexpected_keys:
        print(" ", key)


model.to(device)

model.eval()


print()
print("Checkpoint loaded successfully.")

print(
    "Missing keys:",
    len(missing_keys)
)

print(
    "Unexpected keys:",
    len(unexpected_keys)
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_email(email_text):
    """
    Predict whether an email is Safe or Phishing.

    Returns:
        dict containing:
        - label
        - prediction
        - confidence
        - safe_probability
        - phishing_probability
    """

    # --------------------------------------------------------
    # Normalise input
    # --------------------------------------------------------

    if email_text is None:

        email_text = ""

    email_text = str(
        email_text
    ).strip()


    # --------------------------------------------------------
    # Tokenise email
    # --------------------------------------------------------

    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # Extract 12 NLP features
    # --------------------------------------------------------

    raw_features = extract_nlp_features(
        email_text
    )


    # --------------------------------------------------------
    # Standardise NLP features
    #
    # IMPORTANT:
    # We use the training-set mean and standard deviation.
    # --------------------------------------------------------

    standardised_features = (
        raw_features - feature_mean
    ) / feature_std


    nlp_features = torch.tensor(
        standardised_features,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            nlp_features=nlp_features
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )


    # --------------------------------------------------------
    # Get prediction
    # --------------------------------------------------------

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()


    # Class 0 = Safe
    safe_probability = (
        probabilities[0][0].item()
    )

    # Class 1 = Phishing
    phishing_probability = (
        probabilities[0][1].item()
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        probabilities[0][prediction].item()
    )


    # --------------------------------------------------------
    # Convert class number to label
    # --------------------------------------------------------

    if prediction == 1:

        label = "Phishing Email"

    else:

        label = "Safe Email"


    return {
        "label": label,
        "prediction": prediction,
        "confidence": confidence,
        "safe_probability": safe_probability,
        "phishing_probability": phishing_probability
    }


# ============================================================
# SIMPLE MODEL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TESTING MODEL")
    print("=" * 60)


    test_email = """
    Dear students,

    This is a reminder that your final project report
    is due on Friday.

    Please submit your report through the university
    student portal before 5:00 PM.

    If you have any questions, please contact the help desk.

    Regards,
    University Help Desk
    """


    result = predict_email(
        test_email
    )


    print()
    print("Prediction:", result["label"])

    print(
        "Confidence:",
        f"{result['confidence']:.2%}"
    )

    print(
        "Safe probability:",
        f"{result['safe_probability']:.2%}"
    )

    print(
        "Phishing probability:",
        f"{result['phishing_probability']:.2%}"
    )


    print()
    print("=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)