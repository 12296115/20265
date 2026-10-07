import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-large"

MODEL_DIR = Path("deberta_v3_large_phishing_model")
TEST_FILE = Path("phishing_test.csv")

MAX_LENGTH = 512
NUM_FEATURES = 12
NUM_LABELS = 2

DEVICE = torch.device("cpu")

LABEL_MAP = {
    "Safe Email": 0,
    "Phishing Email": 1,
    "safe": 0,
    "phishing": 1,
    "Safe": 0,
    "Phishing": 1,
}


# ============================================================
# NLP FEATURE VOCABULARIES
# ============================================================

URGENCY_WORDS = {
    "urgent", "urgently", "immediately", "immediate",
    "now", "today", "asap", "quickly", "deadline",
    "expire", "expired", "final", "action"
}

CREDENTIAL_WORDS = {
    "password", "passwd", "username", "login",
    "credential", "credentials", "verify", "verification",
    "authenticate", "authentication", "account"
}

THREAT_WORDS = {
    "suspend", "suspended", "terminate", "terminated",
    "blocked", "block", "close", "closed", "penalty",
    "fraud", "unauthorized", "warning", "security"
}

FINANCIAL_WORDS = {
    "payment", "pay", "invoice", "money", "bank",
    "transfer", "transaction", "refund", "credit",
    "debit", "fee", "account", "billing"
}

CTA_WORDS = {
    "click", "clicking", "visit", "open", "download",
    "confirm", "verify", "submit", "update",
    "activate", "login"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def count_terms(text, vocabulary):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return sum(word in vocabulary for word in words)


def extract_nlp_features(text):
    text = str(text)

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    word_count = max(len(words), 1)

    uppercase_words = [
        word for word in words
        if len(word) > 1 and word.isupper()
    ]

    uppercase_ratio = len(uppercase_words) / word_count

    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE
        )
    )

    email_count = len(
        re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text
        )
    )

    phone_count = len(
        re.findall(
            r"(?:\+?\d[\d\s().-]{7,}\d)",
            text
        )
    )

    features = [
        count_terms(text, URGENCY_WORDS),
        count_terms(text, CREDENTIAL_WORDS),
        count_terms(text, THREAT_WORDS),
        count_terms(text, FINANCIAL_WORDS),
        count_terms(text, CTA_WORDS),
        url_count,
        email_count,
        phone_count,
        text.count("!"),
        text.count("?"),
        uppercase_ratio,
        np.log1p(len(text))
    ]

    return np.array(features, dtype=np.float32)


# ============================================================
# DEBERTA GATED HYBRID MODEL
# ============================================================

class GatedHybridPhishingModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_features=12,
        num_labels=2
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        hidden_size = self.transformer.config.hidden_size

        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

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

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        contextual_embedding = outputs.last_hidden_state[:, 0, :]

        context = self.context_projection(
            contextual_embedding
        )

        features = self.feature_projection(
            nlp_features
        )

        gate_input = torch.cat(
            [context, features],
            dim=1
        )

        gate = self.gate(
            gate_input
        )

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

        return {
            "logits": logits
        }


# ============================================================
# LABEL LOADING
# ============================================================

def convert_label(value):

    if isinstance(value, str):

        value_clean = value.strip()

        if value_clean in LABEL_MAP:
            return LABEL_MAP[value_clean]

        lower_value = value_clean.lower()

        if lower_value in LABEL_MAP:
            return LABEL_MAP[lower_value]

        if lower_value in {
            "0",
            "safe email",
            "safe"
        }:
            return 0

        if lower_value in {
            "1",
            "phishing email",
            "phishing"
        }:
            return 1

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        return int(value)

    raise ValueError(
        f"Unknown label value: {value}"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("ORIGINAL TEST SET EVALUATION")
    print("=" * 70)

    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {MODEL_DIR}")
    print(f"Test file: {TEST_FILE}")
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print()
    print("Loading original test dataset...")

    test_df = pd.read_csv(TEST_FILE)

    print(f"Test records: {len(test_df):,}")
    print(
        f"Columns: {list(test_df.columns)}"
    )

    # --------------------------------------------------------
    # Identify text and label columns
    # --------------------------------------------------------

    possible_text_columns = [
        "processed_text",
        "Email Text",
        "email_text",
        "text",
        "Text"
    ]

    possible_label_columns = [
        "label",
        "Label",
        "Email Type",
        "email_type",
        "Email_Type"
    ]

    text_column = None
    label_column = None

    for column in possible_text_columns:
        if column in test_df.columns:
            text_column = column
            break

    for column in possible_label_columns:
        if column in test_df.columns:
            label_column = column
            break

    if text_column is None:
        raise ValueError(
            "Could not identify the email-text column."
        )

    if label_column is None:
        raise ValueError(
            "Could not identify the label column."
        )

    print()
    print(f"Text column: {text_column}")
    print(f"Label column: {label_column}")

    # --------------------------------------------------------
    # Convert labels
    # --------------------------------------------------------

    y_true = np.array(
        [
            convert_label(value)
            for value in test_df[label_column]
        ],
        dtype=np.int64
    )

    print()
    print("Test label distribution:")
    print(
        pd.Series(y_true)
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "Safe Email",
                1: "Phishing Email"
            }
        )
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    mean_file = MODEL_DIR / "feature_mean.npy"
    std_file = MODEL_DIR / "feature_std.npy"

    if not mean_file.exists():
        raise FileNotFoundError(
            f"Missing feature mean file: {mean_file}"
        )

    if not std_file.exists():
        raise FileNotFoundError(
            f"Missing feature std file: {std_file}"
        )

    feature_mean = np.load(mean_file)
    feature_std = np.load(std_file)

    feature_std = feature_std.copy()
    feature_std[feature_std == 0] = 1.0

    print()
    print(
        f"Feature mean shape: {feature_mean.shape}"
    )

    print(
        f"Feature std shape: {feature_std.shape}"
    )

    # --------------------------------------------------------
    # Extract NLP features
    # --------------------------------------------------------

    print()
    print("Extracting 12 NLP features...")

    raw_features = np.vstack(
        test_df[text_column]
        .fillna("")
        .astype(str)
        .apply(extract_nlp_features)
    )

    print(
        f"Raw feature shape: {raw_features.shape}"
    )

    # --------------------------------------------------------
    # Standardise using training statistics
    # --------------------------------------------------------

    test_features = (
        raw_features - feature_mean
    ) / feature_std

    print(
        f"Standardised feature shape: {test_features.shape}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating model architecture...")

    model = GatedHybridPhishingModel(
        model_name=MODEL_NAME,
        num_features=NUM_FEATURES,
        num_labels=NUM_LABELS
    )

    # --------------------------------------------------------
    # Load trained checkpoint
    # --------------------------------------------------------

    checkpoint_file = MODEL_DIR / "pytorch_model.bin"

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_file}"
        )

    print()
    print("Loading trained checkpoint...")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Checkpoint loaded.")

    print(
        f"Missing keys: {len(missing_keys)}"
    )

    print(
        f"Unexpected keys: {len(unexpected_keys)}"
    )

    if missing_keys:
        print("Missing:")
        for key in missing_keys:
            print(f"  {key}")

    if unexpected_keys:
        print("Unexpected:")
        for key in unexpected_keys:
            print(f"  {key}")

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("Running predictions...")
    print(
        "This may take some time because the local environment "
        "is CPU-only."
    )

    predictions = []
    probabilities = []

    batch_size = 4

    total = len(test_df)

    with torch.no_grad():

        for start in range(
            0,
            total,
            batch_size
        ):

            end = min(
                start + batch_size,
                total
            )

            batch_texts = (
                test_df[text_column]
                .iloc[start:end]
                .fillna("")
                .astype(str)
                .tolist()
            )

            batch_features = torch.tensor(
                test_features[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encoded[
                "input_ids"
            ].to(DEVICE)

            attention_mask = encoded[
                "attention_mask"
            ].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                nlp_features=batch_features
            )

            logits = outputs["logits"]

            probs = torch.softmax(
                logits,
                dim=1
            )

            batch_predictions = torch.argmax(
                probs,
                dim=1
            )

            predictions.extend(
                batch_predictions.cpu().numpy().tolist()
            )

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

            processed = end

            if (
                processed % 100 == 0
                or processed == total
            ):
                print(
                    f"Processed {processed:,}/{total:,}"
                )

    y_pred = np.array(
        predictions,
        dtype=np.int64
    )

    probabilities = np.array(
        probabilities
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Safe Email",
            "Phishing Email"
        ],
        zero_division=0
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)

    print()
    print(f"Test samples : {len(y_true):,}")
    print(f"Accuracy     : {accuracy:.6f} ({accuracy * 100:.4f}%)")
    print(f"Precision    : {precision:.6f} ({precision * 100:.4f}%)")
    print(f"Recall       : {recall:.6f} ({recall * 100:.4f}%)")
    print(f"F1 Score     : {f1:.6f} ({f1 * 100:.4f}%)")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(report)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_output = test_df.copy()

    prediction_output["true_label"] = y_true

    prediction_output["predicted_label"] = y_pred

    prediction_output["true_class"] = [
        "Safe Email" if value == 0
        else "Phishing Email"
        for value in y_true
    ]

    prediction_output["predicted_class"] = [
        "Safe Email" if value == 0
        else "Phishing Email"
        for value in y_pred
    ]

    prediction_output["safe_probability"] = (
        probabilities[:, 0]
    )

    prediction_output["phishing_probability"] = (
        probabilities[:, 1]
    )

    prediction_output.to_csv(
        "original_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    results = {
        "model": MODEL_NAME,
        "checkpoint": str(MODEL_DIR),
        "test_file": str(TEST_FILE),
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "feature_count": NUM_FEATURES,
        "max_length": MAX_LENGTH,
        "architecture": (
            "DeBERTa-v3-large + 12 NLP features "
            "+ gated fusion"
        )
    }

    with open(
        "original_test_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print("Saved:")
    print("  original_test_predictions.csv")
    print("  original_test_results.json")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()