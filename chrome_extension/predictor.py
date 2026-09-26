from pathlib import Path
import html
import json
import re
import unicodedata
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
BATCH_SIZE = 8
LIME_NUM_SAMPLES = 50
LIME_NUM_FEATURES = 8

MAX_LENGTH = 512

# Word lists and extract_nlp_features are copied verbatim from
# NLP Models/Proposed_draft4_5.ipynb so features match training exactly.
URGENCY_TERMS = [
    "urgent", "urgently", "immediately", "immediate", "now",
    "today", "asap", "quickly", "deadline", "expire",
    "expired", "final", "action"
]

CREDENTIAL_TERMS = [
    "password", "passwd", "username", "login", "credential",
    "credentials", "verify", "verification", "authenticate",
    "authentication", "account"
]

THREAT_TERMS = [
    "suspend", "suspended", "terminate", "terminated", "blocked",
    "block", "close", "closed", "penalty", "fraud",
    "unauthorized", "warning", "security"
]

FINANCIAL_TERMS = [
    "payment", "pay", "invoice", "money", "bank", "transfer",
    "transaction", "refund", "credit", "debit", "fee",
    "account", "billing"
]

CTA_TERMS = [
    "click", "clicking", "visit", "open", "download", "confirm",
    "verify", "submit", "update", "activate", "login"
]


def preprocess_email(text):
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        char for char in text
        if char in "\n\t"
        or not unicodedata.category(char).startswith("C")
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


#Set feature extraction function
def extract_nlp_features(text):

    text = str(text)

    words = re.findall(
        r"\b\w+\b",
        text.lower()
    )

    # 1. Urgency term count
    urgency_count = sum(
        word in URGENCY_TERMS
        for word in words
    )

    # 2. Credential-related term count
    credential_count = sum(
        word in CREDENTIAL_TERMS
        for word in words
    )

    # 3. Threat/authority term count
    threat_count = sum(
        word in THREAT_TERMS
        for word in words
    )

    # 4. Financial/payment term count
    financial_count = sum(
        word in FINANCIAL_TERMS
        for word in words
    )

    # 5. Call-to-action term count
    cta_count = sum(
        word in CTA_TERMS
        for word in words
    )

    # 6. URL count
    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE
        )
    )

    # 7. Email address count
    email_count = len(
        re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text
        )
    )

    # 8. Phone number count
    phone_count = len(
        re.findall(
            r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
            text
        )
    )

    # 9. Exclamation count
    exclamation_count = text.count("!")

    # 10. Question count
    question_count = text.count("?")

    # 11. Uppercase-word ratio
    all_words = re.findall(
        r"\b[A-Za-z]+\b",
        text
    )

    if len(all_words) > 0:
        uppercase_ratio = sum(
            word.isupper() and len(word) > 1
            for word in all_words
        ) / len(all_words)
    else:
        uppercase_ratio = 0.0

    # 12. Log-transformed text length
    log_text_length = np.log1p(len(text))

    return [
        urgency_count,
        credential_count,
        threat_count,
        financial_count,
        cta_count,
        url_count,
        email_count,
        phone_count,
        exclamation_count,
        question_count,
        uppercase_ratio,
        log_text_length
    ]


class GatedHybridPhishingModel(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()

        self.transformer = AutoModel.from_config(backbone_config)

        self.context_projection = nn.Sequential(
            nn.Linear(backbone_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, nlp_features):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        context = self.context_projection(
            outputs.last_hidden_state[:, 0, :]
        )
        features = self.feature_projection(nlp_features.float())
        gate = self.gate(torch.cat([context, features], dim=1))
        fused = gate * context + (1 - gate) * features
        return self.classifier(self.fusion(fused))

    def forward_with_gate(self, input_ids, attention_mask, nlp_features):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        context = self.context_projection(
            outputs.last_hidden_state[:, 0, :]
        )

        def fuse(normalised_features):
            features = self.feature_projection(normalised_features.float())
            gate = self.gate(torch.cat([context, features], dim=1))
            fused = gate * context + (1 - gate) * features
            return self.classifier(self.fusion(fused)), features, gate

        logits, features, gate = fuse(nlp_features)
        # Zero after normalisation == every feature at its training mean.
        mean_feature_logits, _, _ = fuse(torch.zeros_like(nlp_features))

        return {
            "logits": logits,
            "text_only_logits": self.classifier(self.fusion(context)),
            "features_only_logits": self.classifier(self.fusion(features)),
            "mean_feature_logits": mean_feature_logits,
            "gate_mean": gate.mean(dim=1),
        }


class PhishingPredictor:
    def __init__(self):
        self.settings = json.loads(
            (MODEL_DIR / "model_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.mean = np.load(
            MODEL_DIR / "feature_mean.npy", allow_pickle=False
        )
        self.std = np.load(
            MODEL_DIR / "feature_std.npy", allow_pickle=False
        )

        if self.mean.shape != (12,) or self.std.shape != (12,):
            raise ValueError("Expected 12 feature normalisation values.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Invalid feature normalisation values.")
        if (self.std <= 0).any():
            raise ValueError("Feature standard deviations must be positive.")

        self.feature_names = json.loads(
            (MODEL_DIR / "feature_names.json").read_text(encoding="utf-8")
        )
        if len(self.feature_names) != 12:
            raise ValueError("Expected 12 feature names.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        config = AutoConfig.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )

        print("Loading saved model...")
        self.model = GatedHybridPhishingModel(config)

        weights = torch.load(
            MODEL_DIR / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" in weights:
            weights = weights["model_state_dict"]
        elif "state_dict" in weights:
            weights = weights["state_dict"]

        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        print("Saved model loaded successfully.")

    def _run_batched(self, texts):
        # Texts are assumed already cleaned; LIME perturbations may be empty.
        keys = (
            "logits", "text_only_logits", "features_only_logits",
            "mean_feature_logits", "gate_mean",
        )
        collected = {key: [] for key in keys}
        raw_features = []

        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                batch = list(texts[start:start + BATCH_SIZE])
                features = np.asarray(
                    [extract_nlp_features(t) for t in batch], dtype=np.float32
                )
                normalised = (features - self.mean) / self.std

                tokens = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                outputs = self.model.forward_with_gate(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    nlp_features=torch.tensor(normalised, dtype=torch.float32),
                )
                for key in keys:
                    collected[key].append(outputs[key].double().numpy())
                raw_features.append(features)

        result = {key: np.concatenate(values) for key, values in collected.items()}
        result["raw_features"] = np.concatenate(raw_features)
        return result

    @staticmethod
    def _softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def _clean_or_raise(text):
        cleaned = preprocess_email(text)
        if not cleaned:
            raise ValueError("Please provide non-empty email text.")
        return cleaned

    def predict(self, text):
        cleaned = self._clean_or_raise(text)
        outputs = self._run_batched([cleaned])

        logits = outputs["logits"][0]
        probabilities = self._softmax(outputs["logits"])[0]
        text_only = self._softmax(outputs["text_only_logits"])[0]
        features_only = self._softmax(outputs["features_only_logits"])[0]
        mean_features = self._softmax(outputs["mean_feature_logits"])[0]

        def log_odds(row):
            return round(float(row[1] - row[0]), 4)

        label = int(probabilities.argmax())

        return {
            "label_id": label,
            "verdict": "phishing" if label == 1 else "legitimate",
            "score": round(float(probabilities[label]), 4),
            "class_scores": [float(probabilities[0]), float(probabilities[1])],
            "gate_analysis": {
                "gate_mean": round(float(outputs["gate_mean"][0]), 4),
                "phishing_probability": {
                    "combined": float(probabilities[1]),
                    "text_only": float(text_only[1]),
                    "features_only": float(features_only[1]),
                    "features_at_training_mean": float(mean_features[1]),
                },
                "phishing_log_odds": {
                    "combined": round(float(logits[1] - logits[0]), 4),
                    "text_only": log_odds(outputs["text_only_logits"][0]),
                    "features_only": log_odds(
                        outputs["features_only_logits"][0]
                    ),
                    "features_at_training_mean": log_odds(
                        outputs["mean_feature_logits"][0]
                    ),
                },
                # combined minus the counterfactual, in probability units.
                "feature_branch_change": {
                    "vs_text_only": float(probabilities[1] - text_only[1]),
                    "vs_features_at_training_mean": float(
                        probabilities[1] - mean_features[1]
                    ),
                },
                "features": [
                    {"name": name, "value": round(float(value), 4)}
                    for name, value in zip(
                        self.feature_names, outputs["raw_features"][0]
                    )
                ],
                "note": (
                    "gate_mean is the average share of weight on the text "
                    "branch (1 = text only, 0 = features only). text_only "
                    "and features_only force the gate to 1 and 0. "
                    "features_at_training_mean sets all 12 normalised "
                    "features to 0."
                ),
            },
        }

    def predict_probabilities(self, texts):
        cleaned = [preprocess_email(text) for text in texts]
        return self._softmax(self._run_batched(cleaned)["logits"])

    def predict_log_odds(self, texts):
        # LIME fits column 1: phishing logit minus legitimate logit.
        cleaned = [preprocess_email(text) for text in texts]
        logits = self._run_batched(cleaned)["logits"]
        log_odds = logits[:, 1] - logits[:, 0]
        return np.column_stack([-log_odds, log_odds])

    def explain(self, text):
        cleaned = self._clean_or_raise(text)

        explainer = LimeTextExplainer(
            class_names=["legitimate", "phishing"],
            random_state=42,
        )
        lime_result = explainer.explain_instance(
            cleaned,
            self.predict_log_odds,
            labels=(1,),
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )

        return {
            "method": "LIME",
            "target": "phishing_log_odds",
            "class_explained": "phishing",
            "weight_label": "log-odds contribution",
            "terms": [
                {"word": word, "weight": round(float(weight), 6)}
                for word, weight in lime_result.as_list(label=1)
            ],
            "num_samples": LIME_NUM_SAMPLES,
            "local_fit_r2": round(float(lime_result.score), 4),
            "note": (
                "Approximate local explanation of the phishing log-odds. "
                "Positive weights push towards phishing; negative weights "
                "push towards legitimate."
            ),
        }

    def predict_with_explanation(self, text):
        result = self.predict(text)
        result["explanation"] = self.explain(text)
        return result
    # def predict_with_explanation(self, text, max_words=30, top_k=5):
    #     cleaned = preprocess_email(text)
    #     baseline = self.predict(cleaned)

    #     label = baseline["label_id"]
    #     baseline_score = baseline["class_scores"][label]

    #     # Check individual word occurrences in the beginning of the email.
    #     matches = list(re.finditer(r"\b[\w]+\b", cleaned))
    #     checked = matches[:max_words]

    #     contributions = []

    #     for match in checked:
    #         # Remove this occurrence, then recalculate all model inputs.
    #         modified = (
    #             cleaned[:match.start()]
    #             + " "
    #             + cleaned[match.end():]
    #         )

    #         if not modified.strip():
    #             continue

    #         changed = self.predict(modified)
    #         changed_score = changed["class_scores"][label]
    #         drop = baseline_score - changed_score

    #         if drop > 0:
    #             contributions.append({
    #                 "word": match.group(),
    #                 "start": match.start(),
    #                 "end": match.end(),
    #                 "score_drop_percentage_points": round(drop * 100, 6),
    #             })

    #     contributions.sort(
    #         key=lambda item: item["score_drop_percentage_points"],
    #         reverse=True,
    #     )

    #     strongest = contributions[:top_k]

    #     if strongest:
    #         summary = (
    #             "Removing these word occurrences reduced the model's "
    #             f"score for the {baseline['verdict']} prediction."
    #         )
    #     else:
    #         summary = (
    #             "No supporting word occurrences were identified "
    #             "among the words checked."
    #         )

    #     return {
    #         "label_id": label,
    #         "verdict": baseline["verdict"],
    #         "score": baseline["score"],
    #         "explanation": {
    #             "method": "word-removal sensitivity",
    #             "summary": summary,
    #             "influential_words": strongest,
    #             "words_checked": len(checked),
    #             "total_words": len(matches),
    #             "scope": (
    #                 "Approximate local explanation. Checks only the first "
    #                 f"{max_words} words and recalculates the NLP features "
    #                 "after each removal. Positions refer to cleaned text."
    #             ),
    #         },
    #     }    

    #     return {
    #         "label_id": label,
    #         "verdict": "phishing" if label == 1 else "legitimate",
    #         "score": round(float(probabilities[label].item()), 4),
    #         "class_scores": [
    #             float(probabilities[0].item()),
    #             float(probabilities[1].item()),
    #         ],
    #     }
      
        


if __name__ == "__main__":
    predictor = PhishingPredictor()
    email = input("\nPaste a short email and press Enter:\n")
    print(json.dumps(predictor.predict(email), indent=2))
