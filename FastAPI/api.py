# ============================================================
# UNIVERSITY PHISHING EMAIL DETECTION SYSTEM
# FastAPI Backend
#
# Primary classifier:
#   DeBERTa-v3-large + 12 NLP features + gated fusion
#
# Additional components:
#   - Context-aware rule-based phishing indicators
#   - SHAP Explainable AI
#   - REST API for Streamlit and browser extension
#
# IMPORTANT:
# The trained DeBERTa model architecture and checkpoint are
# preserved. The expanded indicator system is supplementary
# and does NOT alter the model prediction.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import re
import traceback

import numpy as np
import torch
import torch.nn as nn
import shap

from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./deberta_v3_large_phishing_model"

MODEL_NAME = "microsoft/deberta-v3-large"

MAX_LENGTH = 512

NUM_FEATURES = 12

NUM_LABELS = 2

CONTEXT_DIMENSION = 1024

PROJECTION_DIMENSION = 256

FUSION_DIMENSION = 256


LABELS = [
    "Safe Email",
    "Phishing Email"
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="University Phishing Email Detection API",

    description=(
        "Phishing and social-engineering email detection "
        "using DeBERTa-v3-large, engineered NLP features, "
        "gated fusion, context-aware security indicators, "
        "and SHAP Explainable AI."
    ),

    version="2.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class EmailRequest(BaseModel):

    sender: str = ""

    subject: str = ""

    body: str = ""


# ============================================================
# EXACT DRAFT 2 MODEL ARCHITECTURE
# ============================================================
#
# This matches the saved trained architecture:
#
# DeBERTa-v3-large
#        |
#        | CLS contextual embedding
#        v
# Context projection 1024 -> 256
#
# 12 engineered NLP features
#        |
#        v
# Feature projection 12 -> 256
#
# Context + features
#        |
#        v
#       Gate
#        |
#        v
# Gated fusion
#        |
#        v
# Fusion layer
#        |
#        v
# Classifier
#
# ============================================================


class GatedHybridPhishingModel(nn.Module):

    def __init__(
        self,
        model_name=MODEL_NAME,
        num_features=NUM_FEATURES,
        num_labels=NUM_LABELS
    ):

        super().__init__()


        # --------------------------------------------------------
        # DeBERTa-v3-large backbone
        # --------------------------------------------------------

        self.transformer = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )


        hidden_size = (
            self.transformer.config.hidden_size
        )


        # --------------------------------------------------------
        # Context projection
        # --------------------------------------------------------

        self.context_projection = nn.Sequential(

            nn.Linear(
                hidden_size,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            )

        )


        # --------------------------------------------------------
        # Engineered NLP feature projection
        # --------------------------------------------------------

        self.feature_projection = nn.Sequential(

            nn.Linear(
                num_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            )

        )


        # --------------------------------------------------------
        # Gating mechanism
        # --------------------------------------------------------

        self.gate = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.Sigmoid()

        )


        # --------------------------------------------------------
        # Fusion layer
        # --------------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            )

        )


        # --------------------------------------------------------
        # Final classifier
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Transformer representation
        # --------------------------------------------------------

        outputs = self.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask

        )


        # CLS / first-token representation

        contextual_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )


        # --------------------------------------------------------
        # Project contextual representation
        # --------------------------------------------------------

        context = (
            self.context_projection(
                contextual_embedding
            )
        )


        # --------------------------------------------------------
        # Project engineered NLP features
        # --------------------------------------------------------

        features = (
            self.feature_projection(
                nlp_features
            )
        )


        # --------------------------------------------------------
        # Calculate gate
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Gated fusion
        #
        # gate * context
        # +
        # (1 - gate) * features
        # --------------------------------------------------------

        fused = (
            gate * context
            +
            (1.0 - gate) * features
        )


        # --------------------------------------------------------
        # Fusion layer
        # --------------------------------------------------------

        fused = self.fusion(
            fused
        )


        # --------------------------------------------------------
        # Classification
        # --------------------------------------------------------

        logits = self.classifier(
            fused
        )


        return logits


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)

print(
    "LOADING UNIVERSITY PHISHING EMAIL DETECTION SYSTEM"
)

print("=" * 70)


if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(

        f"\nModel folder not found:\n"
        f"{os.path.abspath(MODEL_PATH)}\n\n"
        f"Expected folder:\n"
        f"{MODEL_PATH}\n\n"
        f"Make sure api.py is located in the project "
        f"folder containing the trained model."

    )


print(
    "\nLoading tokenizer from trained model folder..."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)


print(
    "Tokenizer loaded successfully."
)


# ============================================================
# LOAD FEATURE NORMALISATION PARAMETERS
# ============================================================

print(
    "\nLoading feature normalisation statistics..."
)


FEATURE_MEAN_PATH = os.path.join(
    MODEL_PATH,
    "feature_mean.npy"
)


FEATURE_STD_PATH = os.path.join(
    MODEL_PATH,
    "feature_std.npy"
)


if not os.path.exists(
    FEATURE_MEAN_PATH
):

    raise FileNotFoundError(
        f"Missing feature mean file:\n"
        f"{FEATURE_MEAN_PATH}"
    )


if not os.path.exists(
    FEATURE_STD_PATH
):

    raise FileNotFoundError(
        f"Missing feature standard deviation file:\n"
        f"{FEATURE_STD_PATH}"
    )


feature_mean = np.load(
    FEATURE_MEAN_PATH
).astype(
    np.float32
)


feature_std = np.load(
    FEATURE_STD_PATH
).astype(
    np.float32
)


if feature_mean.shape != (
    NUM_FEATURES,
):

    raise ValueError(

        "Unexpected feature_mean shape: "
        f"{feature_mean.shape}. "
        f"Expected ({NUM_FEATURES},)."

    )


if feature_std.shape != (
    NUM_FEATURES,
):

    raise ValueError(

        "Unexpected feature_std shape: "
        f"{feature_std.shape}. "
        f"Expected ({NUM_FEATURES},)."

    )


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
# DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print(
    "\nDevice:",
    device
)


# ============================================================
# CREATE MODEL ARCHITECTURE
# ============================================================

print(
    "\nCreating DeBERTa gated hybrid architecture..."
)


model = GatedHybridPhishingModel(
    model_name=MODEL_NAME,
    num_features=NUM_FEATURES,
    num_labels=NUM_LABELS
)


print(
    "Architecture created successfully."
)


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

CHECKPOINT_PATH = os.path.join(
    MODEL_PATH,
    "pytorch_model.bin"
)


if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(

        f"Missing trained checkpoint:\n"
        f"{CHECKPOINT_PATH}"

    )


print(
    "\nLoading trained checkpoint..."
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)


# ------------------------------------------------------------
# Support either:
#
# 1. Plain state_dict
#
# 2. Dictionary containing state_dict/model_state_dict
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
):

    if "state_dict" in checkpoint:

        state_dict = (
            checkpoint["state_dict"]
        )

    elif "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

    else:

        state_dict = checkpoint

else:

    raise ValueError(
        "Unsupported checkpoint format."
    )


# ------------------------------------------------------------
# Remove DataParallel prefix if present
# ------------------------------------------------------------

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        clean_key = key[
            len("module.") :
        ]

    else:

        clean_key = key


    clean_state_dict[
        clean_key
    ] = value


# ------------------------------------------------------------
# Load state dictionary strictly
# ------------------------------------------------------------

missing_keys, unexpected_keys = (
    model.load_state_dict(
        clean_state_dict,
        strict=False
    )
)


print(
    "Missing keys:",
    len(missing_keys)
)


print(
    "Unexpected keys:",
    len(unexpected_keys)
)


if missing_keys:

    print(
        "Missing key names:"
    )

    for key in missing_keys:

        print(
            "  ",
            key
        )


if unexpected_keys:

    print(
        "Unexpected key names:"
    )

    for key in unexpected_keys:

        print(
            "  ",
            key
        )


if missing_keys or unexpected_keys:

    raise RuntimeError(

        "The trained checkpoint does not exactly match "
        "the expected GatedHybridPhishingModel architecture. "
        "The model was NOT started."

    )


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model.to(
    device
)


model.eval()


print(
    "\nTrained checkpoint loaded successfully."
)


print(
    "Model ready."
)


# ============================================================
# FEATURE VOCABULARIES
#
# These are kept compatible with the 12 engineered features
# used during training.
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
    "expires",
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
    "suspension",
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
# HELPER: COUNT VOCABULARY TERMS
# ============================================================

def count_terms(
    text,
    vocabulary
):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )


    return sum(

        word in vocabulary

        for word in words

    )


# ============================================================
# EXACT 12 TRAINING FEATURES
# ============================================================

def extract_nlp_features(
    text
):

    text = str(
        text
    )


    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )


    word_count = max(
        len(words),
        1
    )


    uppercase_words = [

        word

        for word in words

        if (
            len(word) > 1
            and word.isupper()
        )

    ]


    uppercase_ratio = (
        len(uppercase_words)
        /
        word_count
    )


    url_count = len(

        re.findall(

            r"https?://\S+|www\.\S+",

            text,

            flags=re.IGNORECASE

        )

    )


    email_count = len(

        re.findall(

            r"\b[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}\b",

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

        count_terms(
            text,
            URGENCY_WORDS
        ),

        count_terms(
            text,
            CREDENTIAL_WORDS
        ),

        count_terms(
            text,
            THREAT_WORDS
        ),

        count_terms(
            text,
            FINANCIAL_WORDS
        ),

        count_terms(
            text,
            CTA_WORDS
        ),

        url_count,

        email_count,

        phone_count,

        text.count(
            "!"
        ),

        text.count(
            "?"
        ),

        uppercase_ratio,

        np.log1p(
            len(text)
        )

    ]


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# NORMALISE ENGINEERED FEATURES
#
# IMPORTANT:
# The training-set mean and standard deviation are used.
# They are NEVER recalculated for incoming emails.
# ============================================================

def normalise_features(
    raw_features
):

    raw_features = np.asarray(
        raw_features,
        dtype=np.float32
    )


    if raw_features.ndim == 1:

        raw_features = (
            raw_features.reshape(
                1,
                -1
            )
        )


    if raw_features.shape[1] != NUM_FEATURES:

        raise ValueError(

            "Unexpected NLP feature count: "
            f"{raw_features.shape[1]}. "
            f"Expected {NUM_FEATURES}."

        )


    return (

        (
            raw_features
            -
            feature_mean
        )
        /
        feature_std

    ).astype(
        np.float32
    )


# ============================================================
# BUILD EMAIL TEXT
# ============================================================

def build_email_text(
    subject="",
    body=""
):

    subject = str(
        subject or ""
    ).strip()


    body = str(
        body or ""
    ).strip()


    if subject and body:

        return (

            f"Subject: {subject}\n\n"
            f"{body}"

        )


    if subject:

        return (
            f"Subject: {subject}"
        )


    return body


# ============================================================
# NORMALISE SHAP TEXT INPUT
# ============================================================

def normalise_texts(
    texts
):

    if isinstance(
        texts,
        str
    ):

        return [
            texts
        ]


    if isinstance(
        texts,
        np.ndarray
    ):

        texts = texts.tolist()


    if isinstance(
        texts,
        tuple
    ):

        texts = list(
            texts
        )


    if isinstance(
        texts,
        list
    ):

        result = []


        for item in texts:

            if item is None:

                result.append(
                    ""
                )


            elif isinstance(
                item,
                str
            ):

                result.append(
                    item
                )


            elif isinstance(
                item,
                np.ndarray
            ):

                item = item.tolist()


                result.append(

                    " ".join(
                        str(x)
                        for x in item
                    )

                )


            elif isinstance(
                item,
                (list, tuple)
            ):

                result.append(

                    " ".join(
                        str(x)
                        for x in item
                    )

                )


            else:

                result.append(
                    str(item)
                )


        return result


    return [
        str(texts)
    ]


# ============================================================
# MODEL PREDICTION
#
# This is the PRIMARY classification function.
# ============================================================

def predict_email(
    email_text
):

    # --------------------------------------------------------
    # Tokenise
    # --------------------------------------------------------

    inputs = tokenizer(

        email_text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=MAX_LENGTH

    )


    # --------------------------------------------------------
    # Extract and normalise 12 NLP features
    # --------------------------------------------------------

    raw_features = (
        extract_nlp_features(
            email_text
        )
    )


    normalised_features = (
        normalise_features(
            raw_features
        )
    )


    nlp_features = torch.tensor(
        normalised_features,
        dtype=torch.float32
    ).to(
        device
    )


    # --------------------------------------------------------
    # Move token inputs to device
    # --------------------------------------------------------

    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(

            input_ids=inputs[
                "input_ids"
            ],

            attention_mask=inputs[
                "attention_mask"
            ],

            nlp_features=nlp_features

        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


    # --------------------------------------------------------
    # Probability values
    # --------------------------------------------------------

    safe_probability = float(
        probabilities[0].item()
    )


    phishing_probability = float(
        probabilities[1].item()
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_index = int(

        torch.argmax(
            probabilities
        ).item()

    )


    prediction = LABELS[
        prediction_index
    ]


    confidence = float(

        probabilities[
            prediction_index
        ].item()

    )


    return {

        "prediction_index":
            prediction_index,

        "prediction":
            prediction,

        "confidence":
            confidence,

        "confidence_percentage":
            round(
                confidence * 100,
                2
            ),

        "safe_probability":
            safe_probability,

        "safe_probability_percentage":
            round(
                safe_probability * 100,
                2
            ),

        "phishing_probability":
            phishing_probability,

        "phishing_probability_percentage":
            round(
                phishing_probability * 100,
                2
            )

    }


# ============================================================
# CONTEXT-AWARE SECURITY INDICATORS
#
# IMPORTANT:
#
# These indicators are NOT the trained model.
#
# They provide human-readable supporting evidence.
#
# A URL by itself is NOT treated as a phishing indicator.
#
# The system looks for contextual combinations such as:
#
#   - request + credentials
#   - urgency + account action
#   - financial request
#   - personal information request
#   - suspicious CTA
#   - reward/prize solicitation
#   - authority/impersonation language
#
# ============================================================


def detect_phishing_indicators(
    subject="",
    body="",
    sender=""
):

    subject = str(
        subject or ""
    ).strip()


    body = str(
        body or ""
    ).strip()


    sender = str(
        sender or ""
    ).strip()


    # --------------------------------------------------------
    # Combined text
    # --------------------------------------------------------

    text = (
        f"{subject}\n{body}"
    ).lower()


    indicators = []


    # ========================================================
    # 1. URGENCY / THREATENING LANGUAGE
    # ========================================================

    urgency_patterns = [

        r"\burgent\b",

        r"\burgently\b",

        r"\bimmediately\b",

        r"\bact\s+now\b",

        r"\baction\s+required\b",

        r"\brespond\s+(?:now|immediately|today)\b",

        r"\bwithin\s+\d+\s+(?:hour|hours|day|days)\b",

        r"\b(?:account|access|service|membership)\s+"
        r"(?:will\s+be|has\s+been|is)\s+"
        r"(?:suspended|blocked|terminated|closed)\b",

        r"\bfinal\s+warning\b",

        r"\blast\s+warning\b",

        r"\bfinal\s+notice\b",

        r"\bexpires?\s+(?:today|soon|shortly)\b",

        r"\bavoid\s+(?:suspension|termination|closure)\b",

        r"\bfailure\s+to\s+respond\b",

        r"\bdeadline\b",

        r"\bpenalty\b"

    ]


    urgency_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in urgency_patterns

    )


    if urgency_detected:

        indicators.append(
            "Urgent or threatening language detected"
        )


    # ========================================================
    # 2. CREDENTIAL / ACCOUNT INFORMATION REQUEST
    #
    # IMPORTANT:
    # Merely mentioning "account" is NOT enough.
    # There must be request/verification context.
    # ========================================================

    credential_patterns = [

        r"\b(?:enter|provide|submit|send|share|confirm|"
        r"verify|update|reset)\b.{0,80}\b"
        r"(?:password|passwd|username|credential|"
        r"login|security\s+code|verification\s+code|otp)\b",

        r"\b(?:password|passwd|username|credential|"
        r"login|security\s+code|verification\s+code|otp)\b"
        r".{0,80}\b(?:enter|provide|submit|send|share|"
        r"confirm|verify|update|reset)\b",

        r"\bverify\s+your\s+account\b",

        r"\bconfirm\s+your\s+(?:account|identity|credentials)\b",

        r"\bconfirm\s+your\s+password\b",

        r"\bprovide\s+your\s+(?:login|credentials|"
        r"account\s+(?:details|information))\b",

        r"\b(?:login|log\s+in)\s+to\s+(?:verify|confirm|"
        r"secure|unlock|restore)\b",

        r"\baccount\s+(?:verification|authentication)"
        r"\s+(?:required|needed)\b"

    ]


    credential_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in credential_patterns

    )


    if credential_detected:

        indicators.append(
            "Possible request for credentials or account information"
        )


    # ========================================================
    # 3. FINANCIAL / PAYMENT REQUEST
    # ========================================================

    financial_patterns = [

        r"\b(?:send|transfer|pay|wire|deposit)\b"
        r".{0,80}\b(?:money|funds|payment|cash)\b",

        r"\b(?:payment|invoice|refund|transaction|"
        r"bank\s+details|bank\s+account|credit\s+card|"
        r"debit\s+card|financial\s+information)\b"
        r".{0,80}\b(?:provide|confirm|verify|update|"
        r"submit|send|pay|enter)\b",

        r"\b(?:provide|enter|submit|send|confirm|update)\b"
        r".{0,80}\b(?:bank|credit\s+card|debit\s+card|"
        r"payment|financial\s+information)\b",

        r"\baccount\s+details\b"
        r".{0,80}\b(?:payment|bank|transfer|money)\b",

        r"\b(?:bank|payment|invoice|refund|billing)\b"
        r".{0,80}\b(?:required|needed|urgent|immediately)\b",

        r"\b(?:free|easy|quick)\s+(?:loan|credit|"
        r"debt\s+consolidation)\b",

        r"\bdebt\s+(?:relief|consolidation|reduction)\b",

        r"\b(?:save|reduce)\b.{0,60}\b(?:debt|loan|"
        r"credit|monthly\s+payments)\b",

        r"\b(?:loan|credit|debt)\b.{0,60}\b"
        r"(?:apply|submit|qualify|approval|quote)\b"

    ]


    financial_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in financial_patterns

    )


    if financial_detected:

        indicators.append(
            "Possible financial or payment-related request"
        )


    # ========================================================
    # 4. PERSONAL INFORMATION REQUEST
    # ========================================================

    personal_information_patterns = [

        r"\b(?:provide|enter|submit|send|share|confirm|"
        r"update)\b.{0,80}\b(?:full\s+name|name|"
        r"address|street\s+address|home\s+address|"
        r"date\s+of\s+birth|dob|phone\s+number|"
        r"mobile\s+number|contact\s+details|"
        r"personal\s+information|identity)\b",

        r"\b(?:full\s+name|street\s+address|home\s+address|"
        r"date\s+of\s+birth|dob|phone\s+number|"
        r"personal\s+information|identity)\b"
        r".{0,80}\b(?:required|needed|provide|submit|"
        r"enter|send|share)\b",

        r"\bfill\s+out\b.{0,80}\b(?:name|address|"
        r"phone|personal|information|details)\b",

        r"\bpersonal\s+details\b.{0,80}\b"
        r"(?:submit|provide|confirm|update|enter)\b",

        r"\bidentity\s+(?:verification|confirmation)\b"
        r".{0,80}\b(?:required|needed|submit|provide)\b"

    ]


    personal_information_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in personal_information_patterns

    )


    if personal_information_detected:

        indicators.append(
            "Request for personal information detected"
        )


    # ========================================================
    # 5. SUSPICIOUS CALL-TO-ACTION
    #
    # A URL alone is NOT sufficient.
    # ========================================================

    cta_patterns = [

        r"\bclick\s+(?:here|below|the\s+link)\b",

        r"\bclick\s+to\s+(?:verify|confirm|update|"
        r"activate|unlock|continue|claim)\b",

        r"\bfollow\s+(?:this|the)\s+link\b",

        r"\bvisit\s+(?:this|the)\s+(?:link|page|website)\b",

        r"\b(?:open|download)\b.{0,50}\b"
        r"(?:attachment|document|file|link)\b",

        r"\b(?:verify|confirm|update|activate|unlock)\b"
        r".{0,60}\b(?:here|below|link|page|form)\b",

        r"\bsubmit\s+(?:the|your)\s+(?:form|information|details)\b",

        r"\bcomplete\s+(?:the|your)\s+(?:form|verification|"
        r"registration|application)\b",

        r"\brespond\s+(?:using|through|via)\b",

        r"\b(?:login|log\s+in)\b.{0,60}\b"
        r"(?:here|below|link|page)\b"

    ]


    cta_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in cta_patterns

    )


    if cta_detected:

        indicators.append(
            "Suspicious call-to-action language detected"
        )


    # ========================================================
    # 6. REWARD / PRIZE / UNEXPECTED BENEFIT
    # ========================================================

    reward_patterns = [

        r"\bcongratulations\b",

        r"\byou\s+(?:have\s+)?won\b",

        r"\byou\s+(?:have\s+)?been\s+selected\b",

        r"\bselected\s+to\s+(?:receive|claim|get)\b",

        r"\bclaim\s+your\s+(?:prize|reward|bonus)\b",

        r"\b(?:prize|reward|bonus|gift|cash)\b"
        r".{0,80}\b(?:claim|collect|receive|selected)\b",

        r"\bfree\s+(?:gift|money|reward|prize)\b",

        r"\b(?:exclusive|special)\s+offer\b",

        r"\blimited\s+time\s+offer\b"

    ]


    reward_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in reward_patterns

    )


    if reward_detected:

        indicators.append(
            "Unexpected prize, reward or promotional offer detected"
        )


    # ========================================================
    # 7. ACCOUNT / SECURITY ACTION
    # ========================================================

    account_security_patterns = [

        r"\b(?:unlock|restore|reactivate)\b"
        r".{0,50}\b(?:account|access|profile)\b",

        r"\b(?:account|profile|access)\b"
        r".{0,60}\b(?:locked|blocked|suspended|"
        r"disabled|restricted)\b",

        r"\bsecurity\s+(?:alert|notice|notification)\b"
        r".{0,80}\b(?:verify|confirm|update|login)\b",

        r"\bunusual\s+(?:activity|sign[- ]?in|login)\b"
        r".{0,80}\b(?:verify|confirm|secure|login)\b",

        r"\bunauthorized\s+(?:activity|access|login)\b"
        r".{0,80}\b(?:verify|confirm|secure)\b"

    ]


    account_security_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in account_security_patterns

    )


    if account_security_detected:

        indicators.append(
            "Account or security action requested"
        )


    # ========================================================
    # 8. AUTHORITY / IMPERSONATION LANGUAGE
    # ========================================================

    authority_patterns = [

        r"\b(?:it|this)\s+is\s+(?:the|your)\s+"
        r"(?:bank|university|it\s+department|security\s+team|"
        r"support\s+team|administrator)\b",

        r"\b(?:university|bank|security|support|"
        r"administrator|it\s+department)\s+"
        r"(?:requires?|requested?|asks?|needs?)\b"
        r".{0,80}\b(?:verify|confirm|provide|submit|update)\b",

        r"\b(?:official|authorized)\s+(?:request|notice|message)\b"
        r".{0,80}\b(?:verify|confirm|provide|submit)\b",

        r"\bsecurity\s+team\b"
        r".{0,80}\b(?:password|credentials|login|verify)\b"

    ]


    authority_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in authority_patterns

    )


    if authority_detected:

        indicators.append(
            "Possible impersonation or authority-based request detected"
        )


    # ========================================================
    # 9. SECRECY / PRESSURE TO AVOID NORMAL CHANNELS
    # ========================================================

    secrecy_patterns = [

        r"\bdo\s+not\s+(?:tell|inform|contact|notify)\b",

        r"\bkeep\s+this\s+(?:private|confidential|secret)\b",

        r"\bdo\s+not\s+share\b"
        r".{0,60}\b(?:message|request|information)\b",

        r"\bavoid\s+contacting\b"
        r".{0,60}\b(?:support|bank|administrator|office)\b"

    ]


    secrecy_detected = any(

        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for pattern in secrecy_patterns

    )


    if secrecy_detected:

        indicators.append(
            "Pressure to keep the request confidential or avoid normal channels detected"
        )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    indicators = list(
        dict.fromkeys(
            indicators
        )
    )


    return indicators


# ============================================================
# INDICATOR MESSAGE
# ============================================================

def build_indicator_message(
    prediction,
    indicators
):

    if prediction == "Phishing Email":

        if indicators:

            return (
                "Rule-based indicators matched this message "
                "and provide supporting evidence for the "
                "model's phishing classification."
            )


        return (
            "The model detected phishing based on "
            "contextual and semantic patterns. "
            "No predefined rule-based indicators matched "
            "this message."
        )


    # --------------------------------------------------------
    # Safe email
    # --------------------------------------------------------

    if indicators:

        return (
            "The email was classified as Safe Email, "
            "although some general security-related "
            "characteristics were detected."
        )


    return (
        "No common phishing indicators detected."
    )


# ============================================================
# SHAP PREDICTION FUNCTION
#
# SHAP passes masked text to this function.
#
# For every masked text:
#   1. Tokenise it
#   2. Recalculate the 12 NLP features
#   3. Apply the TRAINING normalisation statistics
#   4. Run the same hybrid model
#
# This keeps SHAP aligned with the actual trained model.
# ============================================================

def shap_predict(
    texts
):

    texts = normalise_texts(
        texts
    )


    if len(texts) == 0:

        return np.empty(
            (
                0,
                NUM_LABELS
            ),
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Tokenisation
    # --------------------------------------------------------

    inputs = tokenizer(

        texts,

        return_tensors="pt",

        padding=True,

        truncation=True,

        max_length=MAX_LENGTH

    )


    # --------------------------------------------------------
    # NLP features
    # --------------------------------------------------------

    raw_features = np.vstack(

        [

            extract_nlp_features(
                text
            )

            for text in texts

        ]

    ).astype(
        np.float32
    )


    normalised_features = (
        normalise_features(
            raw_features
        )
    )


    nlp_features = torch.tensor(
        normalised_features,
        dtype=torch.float32
    ).to(
        device
    )


    # --------------------------------------------------------
    # Move inputs to device
    # --------------------------------------------------------

    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(

            input_ids=inputs[
                "input_ids"
            ],

            attention_mask=inputs[
                "attention_mask"
            ],

            nlp_features=nlp_features

        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )


    return probabilities.cpu().numpy()


# ============================================================
# INITIALISE SHAP
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "INITIALISING SHAP EXPLAINER"
)

print(
    "=" * 70
)


explainer = None


try:

    text_masker = shap.maskers.Text(
        tokenizer
    )


    explainer = shap.Explainer(

        shap_predict,

        text_masker,

        output_names=LABELS

    )


    print(
        "SHAP explainer initialised successfully."
    )


except Exception as error:

    explainer = None


    print(
        "SHAP INITIALISATION ERROR:"
    )


    print(
        str(error)
    )


# ============================================================
# SHAP TOKEN EXTRACTION
# ============================================================

def extract_shap_tokens(
    shap_values
):

    data = shap_values.data


    data = np.array(
        data,
        dtype=object
    )


    print(
        "Raw SHAP data shape:",
        data.shape
    )


    if data.ndim >= 2:

        data = data[0]


    data = np.array(
        data,
        dtype=object
    ).flatten()


    tokens = []


    for token in data:

        if token is None:

            continue


        token = str(
            token
        ).strip()


        if token:

            tokens.append(
                token
            )


    return tokens


# ============================================================
# SHAP CONTRIBUTION EXTRACTION
# ============================================================

def extract_shap_contributions(
    shap_values,
    prediction_index
):

    values = np.array(
        shap_values.values
    )


    print(
        "SHAP values shape:",
        values.shape
    )


    # --------------------------------------------------------
    # Shape:
    # (samples, tokens, classes)
    # --------------------------------------------------------

    if values.ndim == 3:

        contributions = values[
            0,
            :,
            prediction_index
        ]


        return np.array(
            contributions,
            dtype=float
        ).flatten()


    # --------------------------------------------------------
    # Shape:
    # (tokens, classes)
    # --------------------------------------------------------

    if values.ndim == 2:

        contributions = values[
            :,
            prediction_index
        ]


        return np.array(
            contributions,
            dtype=float
        ).flatten()


    # --------------------------------------------------------
    # Shape:
    # (tokens,)
    # --------------------------------------------------------

    if values.ndim == 1:

        return np.array(
            values,
            dtype=float
        ).flatten()


    raise ValueError(

        "Unsupported SHAP values shape: "
        f"{values.shape}"

    )


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "University Phishing Email Detection API",

        "model":
            "DeBERTa-v3-large",

        "architecture":
            "DeBERTa-v3-large + 12 NLP features + gated fusion",

        "explainable_ai":
            "SHAP",

        "indicator_system":
            "Context-aware rule-based phishing indicators",

        "version":
            "2.0"

    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "healthy",

        "model_loaded":
            True,

        "shap_available":
            explainer is not None,

        "model":
            "DeBERTa-v3-large",

        "architecture":
            "DeBERTa-v3-large + 12 NLP features + gated fusion",

        "device":
            str(device),

        "num_features":
            NUM_FEATURES,

        "max_length":
            MAX_LENGTH

    }


# ============================================================
# MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model":
            "DeBERTa-v3-large",

        "model_name":
            MODEL_NAME,

        "architecture":
            "DeBERTa-v3-large + 12 NLP features + gated fusion",

        "num_features":
            NUM_FEATURES,

        "max_length":
            MAX_LENGTH,

        "device":
            str(device),

        "shap_available":
            explainer is not None,

        "primary_classifier":
            "DeBERTa-v3-large gated hybrid model",

        "supporting_analysis":
            "Context-aware rule-based phishing indicators",

        "explainability":
            "SHAP"

    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: EmailRequest
):

    try:

        # ----------------------------------------------------
        # Build combined model input
        # ----------------------------------------------------

        email_text = build_email_text(

            request.subject,

            request.body

        )


        if not email_text.strip():

            return {

                "success":
                    False,

                "error":
                    "Email subject or body is required"

            }


        # ----------------------------------------------------
        # PRIMARY DEBERTA PREDICTION
        # ----------------------------------------------------

        result = predict_email(
            email_text
        )


        # ----------------------------------------------------
        # CONTEXT-AWARE INDICATORS
        # ----------------------------------------------------

        indicators = detect_phishing_indicators(

            subject=request.subject,

            body=request.body,

            sender=request.sender

        )


        # ----------------------------------------------------
        # EXPLANATION MESSAGE
        # ----------------------------------------------------

        indicator_message = (
            build_indicator_message(

                result["prediction"],

                indicators

            )
        )


        # ----------------------------------------------------
        # RETURN API RESPONSE
        # ----------------------------------------------------

        return {

            "success":
                True,

            "sender":
                request.sender,

            "subject":
                request.subject,

            "prediction_index":
                result["prediction_index"],

            "prediction":
                result["prediction"],

            "confidence":
                result["confidence"],

            "confidence_percentage":
                result["confidence_percentage"],

            "safe_probability":
                result["safe_probability"],

            "safe_probability_percentage":
                result[
                    "safe_probability_percentage"
                ],

            "phishing_probability":
                result["phishing_probability"],

            "phishing_probability_percentage":
                result[
                    "phishing_probability_percentage"
                ],

            # Primary field used by extension
            "indicators":
                indicators,

            # Explicit name for clarity
            "rule_based_indicators":
                indicators,

            "indicator_message":
                indicator_message

        }


    except Exception as error:

        print(
            "\nPREDICTION ERROR"
        )


        traceback.print_exc()


        return {

            "success":
                False,

            "error":
                f"Prediction failed: {str(error)}"

        }


# ============================================================
# EXPLAIN ENDPOINT
# ============================================================

@app.post("/explain")
def explain(
    request: EmailRequest
):

    # --------------------------------------------------------
    # SHAP availability
    # --------------------------------------------------------

    if explainer is None:

        return {

            "success":
                False,

            "message":
                "SHAP explainer is not available"

        }


    # --------------------------------------------------------
    # Build email text
    # --------------------------------------------------------

    email_text = build_email_text(

        request.subject,

        request.body

    )


    if not email_text.strip():

        return {

            "success":
                False,

            "message":
                "Email subject or body is required"

        }


    try:

        print()
        print(
            "=" * 70
        )

        print(
            "GENERATING SHAP EXPLANATION"
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # Get model prediction
        # ----------------------------------------------------

        prediction_result = predict_email(
            email_text
        )


        prediction_index = (
            prediction_result[
                "prediction_index"
            ]
        )


        prediction_label = (
            prediction_result[
                "prediction"
            ]
        )


        print(
            "Predicted class:",
            prediction_label
        )


        print(
            "Prediction index:",
            prediction_index
        )


        # ----------------------------------------------------
        # Generate SHAP values
        # ----------------------------------------------------

        print(
            "Generating SHAP values..."
        )


        shap_values = explainer(

            [email_text]

        )


        print(
            "SHAP explanation generated successfully."
        )


        # ----------------------------------------------------
        # Extract tokens
        # ----------------------------------------------------

        tokens = extract_shap_tokens(
            shap_values
        )


        print(
            "Number of tokens:",
            len(tokens)
        )


        # ----------------------------------------------------
        # Extract contributions
        # ----------------------------------------------------

        contributions = (
            extract_shap_contributions(

                shap_values,

                prediction_index

            )
        )


        print(
            "Number of contributions:",
            len(contributions)
        )


        # ----------------------------------------------------
        # Match token/contribution lengths
        # ----------------------------------------------------

        usable_length = min(

            len(tokens),

            len(contributions)

        )


        print(
            "Usable SHAP values:",
            usable_length
        )


        if usable_length == 0:

            return {

                "success":
                    False,

                "message":
                    "Unable to extract SHAP contributions"

            }


        # ----------------------------------------------------
        # Build factors
        # ----------------------------------------------------

        factors = []


        for i in range(
            usable_length
        ):

            token = str(
                tokens[i]
            ).strip()


            contribution = float(
                contributions[i]
            )


            # Skip empty tokens
            if not token:

                continue


            # Skip special tokens
            if token in [

                "[CLS]",
                "[SEP]",
                "[PAD]",
                "[MASK]"

            ]:

                continue


            # Skip numerical noise
            if abs(
                contribution
            ) < 0.00000001:

                continue


            factors.append({

                "token":
                    token,

                "contribution":
                    round(
                        contribution,
                        6
                    )

            })


        print(
            "Meaningful factors:",
            len(factors)
        )


        # ----------------------------------------------------
        # Supporting factors
        #
        # Positive contribution supports the predicted class.
        # ----------------------------------------------------

        supporting_factors = sorted(

            [

                factor

                for factor in factors

                if factor[
                    "contribution"
                ] > 0

            ],

            key=lambda x:
                abs(
                    x["contribution"]
                ),

            reverse=True

        )[:10]


        # ----------------------------------------------------
        # Opposing factors
        # ----------------------------------------------------

        opposing_factors = sorted(

            [

                factor

                for factor in factors

                if factor[
                    "contribution"
                ] < 0

            ],

            key=lambda x:
                abs(
                    x["contribution"]
                ),

            reverse=True

        )[:10]


        # ----------------------------------------------------
        # Strongest factors overall
        # ----------------------------------------------------

        strongest_factors = sorted(

            factors,

            key=lambda x:
                abs(
                    x["contribution"]
                ),

            reverse=True

        )[:10]


        # ----------------------------------------------------
        # Security indicators
        # ----------------------------------------------------

        indicators = (
            detect_phishing_indicators(

                subject=request.subject,

                body=request.body,

                sender=request.sender

            )
        )


        indicator_message = (
            build_indicator_message(

                prediction_label,

                indicators

            )
        )


        # ----------------------------------------------------
        # CONSOLE RESULTS
        # ----------------------------------------------------

        print()

        print(
            "=" * 70
        )

        print(
            f"EXPLANATION FOR: {prediction_label}"
        )

        print(
            "=" * 70
        )


        print()

        print(
            "TOP SUPPORTING FACTORS"
        )


        for factor in supporting_factors:

            print(

                factor["token"],

                "→",

                factor["contribution"]

            )


        print()

        print(
            "TOP OPPOSING FACTORS"
        )


        for factor in opposing_factors:

            print(

                factor["token"],

                "→",

                factor["contribution"]

            )


        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

        return {

            "success":
                True,

            "prediction":
                prediction_label,

            "prediction_index":
                prediction_index,

            "supporting_factors":
                supporting_factors,

            "opposing_factors":
                opposing_factors,

            "strongest_factors":
                strongest_factors,

            "total_factors":
                len(factors),

            "indicators":
                indicators,

            "rule_based_indicators":
                indicators,

            "indicator_message":
                indicator_message,

            "message":
                "SHAP explanation generated successfully"

        }


    except Exception as error:

        print()

        print(
            "=" * 70
        )

        print(
            "SHAP EXPLANATION ERROR"
        )

        print(
            "=" * 70
        )


        traceback.print_exc()


        return {

            "success":
                False,

            "message":
                f"Unable to generate SHAP explanation: {str(error)}"

        }


# ============================================================
# RUN API
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000

    )