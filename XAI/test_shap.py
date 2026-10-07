import pandas as pd
import torch
import shap

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./distilbert_phishing_model"

DATASET_PATH = "./Phishing_Email_Cleaned.csv"

TEXT_COLUMN = "Email Text"

LABEL_COLUMN = "Email Type"

TOP_WORDS = 15


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

print("=" * 70)
print("LOADING CLEANED DATASET")
print("=" * 70)


df = pd.read_csv(
    DATASET_PATH
)


print("Dataset loaded successfully.")

print(
    "Number of rows:",
    len(df)
)


print("\nDataset columns:")

for column in df.columns:

    print(
        "-",
        column
    )


# ============================================================
# DISPLAY LABEL DISTRIBUTION
# ============================================================

print("\nAvailable email labels:\n")

print(
    df[LABEL_COLUMN]
    .value_counts()
)


# ============================================================
# FIND A PHISHING EMAIL
# ============================================================

phishing_emails = df[
    df[LABEL_COLUMN]
    .astype(str)
    .str.lower()
    .str.contains(
        "phishing",
        na=False
    )
]


if phishing_emails.empty:

    raise ValueError(
        "No phishing emails were found in the dataset."
    )


# Select first phishing email
test_row = phishing_emails.iloc[0]


test_email = str(
    test_row[TEXT_COLUMN]
)


actual_label = str(
    test_row[LABEL_COLUMN]
)


# ============================================================
# DISPLAY SELECTED EMAIL
# ============================================================

print("\n" + "=" * 70)
print("SELECTED DATASET EMAIL")
print("=" * 70)


print("\nActual Label:")

print(
    actual_label
)


print("\nEmail Preview:\n")

print(
    test_email[:1000]
)


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING DISTILBERT PHISHING DETECTION MODEL")
print("=" * 70)


# Load tokenizer

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH
)


# Load trained model

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


model.to(
    device
)


model.eval()


print(
    "Model loaded successfully"
)


print(
    "Device:",
    device
)


print(
    "Number of labels:",
    model.config.num_labels
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_proba(texts):

    # SHAP may provide one string
    if isinstance(
        texts,
        str
    ):

        texts = [texts]


    # Convert to normal Python list
    texts = list(
        texts
    )


    # Tokenize emails

    inputs = tokenizer(

        texts,

        return_tensors="pt",

        padding=True,

        truncation=True,

        max_length=512

    )


    # Move tensors to CPU/GPU

    inputs = {

        key: value.to(
            device
        )

        for key, value
        in inputs.items()

    }


    # Disable gradient calculations

    with torch.no_grad():

        outputs = model(
            **inputs
        )


        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )


    return probabilities.cpu().numpy()


# ============================================================
# RUN NORMAL MODEL PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("MODEL PREDICTION")
print("=" * 70)


probabilities = predict_proba(

    [test_email]

)[0]


safe_probability = probabilities[0]

phishing_probability = probabilities[1]


prediction = probabilities.argmax()


# ============================================================
# CONVERT PREDICTION TO LABEL
# ============================================================

if prediction == 0:

    predicted_label = "Safe Email"

else:

    predicted_label = "Phishing Email"


# ============================================================
# DISPLAY PROBABILITIES
# ============================================================

print()


print(

    f"Safe Email Probability: "
    f"{safe_probability * 100:.2f}%"

)


print(

    f"Phishing Email Probability: "
    f"{phishing_probability * 100:.2f}%"

)


print()


print(

    "Model Prediction:",

    predicted_label

)


print(

    "Actual Dataset Label:",

    actual_label

)


# ============================================================
# CHECK PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPARISON")
print("=" * 70)


normalized_actual_label = actual_label.lower()


if (

    "phishing"
    in normalized_actual_label

    and prediction == 1

):

    print(

        "SUCCESS: Model correctly identified "
        "the phishing email."

    )


elif (

    "safe"
    in normalized_actual_label

    and prediction == 0

):

    print(

        "SUCCESS: Model correctly identified "
        "the safe email."

    )


else:

    print(

        "WARNING: Model prediction does not "
        "match the dataset label."

    )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING SHAP EXPLAINER")
print("=" * 70)


explainer = shap.Explainer(

    predict_proba,

    tokenizer

)


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

print("\nGenerating SHAP explanation...")

print(

    "This may take some time because the "
    "model is running on CPU."

)


shap_values = explainer(

    [test_email]

)


# ============================================================
# DISPLAY SHAP INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLANATION GENERATED SUCCESSFULLY")
print("=" * 70)


print()


print(

    "SHAP values shape:",

    shap_values.values.shape

)


print()


print(

    "Explanation generated for:",

    predicted_label

)


# ============================================================
# EXTRACT TOKENS AND SHAP VALUES
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING SHAP EXPLANATION")
print("=" * 70)


# Extract original SHAP tokens

tokens = shap_values.data[0]


# Extract SHAP values for predicted class

values = shap_values.values[

    0,

    :,

    prediction

]


# ============================================================
# CLEAN TOKEN IMPORTANCE
# ============================================================

token_importance = []


for token, value in zip(

    tokens,

    values

):

    token = str(
        token
    ).strip()


    value = float(
        value
    )


    # Skip empty tokens

    if not token:

        continue


    # Skip punctuation-only tokens

    if all(

        not character.isalnum()

        for character in token

    ):

        continue


    # Skip extremely short fragments

    if len(token) < 2:

        continue


    # Store cleaned token

    token_importance.append(

        {

            "token": token,

            "shap_value": value,

            "absolute_value": abs(value)

        }

    )


# ============================================================
# SORT TOKENS BY IMPORTANCE
# ============================================================

token_importance.sort(

    key=lambda item:

        item["absolute_value"],

    reverse=True

)


# ============================================================
# REMOVE DUPLICATE TOKENS
# ============================================================

unique_tokens = []


seen_tokens = set()


for item in token_importance:


    normalized_token = (

        item["token"]

        .lower()

    )


    # Skip duplicates

    if normalized_token in seen_tokens:

        continue


    seen_tokens.add(

        normalized_token

    )


    unique_tokens.append(

        item

    )


# ============================================================
# FACTORS SUPPORTING PREDICTION
# ============================================================

print("\n" + "=" * 70)

print(

    f"TOP FACTORS SUPPORTING "
    f"{predicted_label.upper()}"

)

print("=" * 70)


supporting_factors = []


for item in unique_tokens:


    if item["shap_value"] > 0:


        supporting_factors.append(

            item

        )


        if len(

            supporting_factors

        ) >= TOP_WORDS:

            break


if supporting_factors:


    for index, item in enumerate(

        supporting_factors,

        start=1

    ):


        print(

            f"{index}. "

            f"{item['token']}"

            f" | Contribution: "

            f"{item['shap_value']:.6f}"

        )


else:

    print(

        "No strong positive factors found."

    )


# ============================================================
# FACTORS WORKING AGAINST PREDICTION
# ============================================================

print("\n" + "=" * 70)

print(

    f"TOP FACTORS WORKING AGAINST "
    f"{predicted_label.upper()}"

)

print("=" * 70)


opposing_factors = []


for item in unique_tokens:


    if item["shap_value"] < 0:


        opposing_factors.append(

            item

        )


        if len(

            opposing_factors

        ) >= TOP_WORDS:

            break


if opposing_factors:


    for index, item in enumerate(

        opposing_factors,

        start=1

    ):


        print(

            f"{index}. "

            f"{item['token']}"

            f" | Contribution: "

            f"{item['shap_value']:.6f}"

        )


else:

    print(

        "No strong negative factors found."

    )


# ============================================================
# SAVE RESULTS TO TEXT FILE
# ============================================================

OUTPUT_FILE = "shap_explanation.txt"


with open(

    OUTPUT_FILE,

    "w",

    encoding="utf-8"

) as file:


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    file.write(

        "DISTILBERT PHISHING EMAIL "
        "SHAP EXPLANATION\n"

    )


    file.write(

        "=" * 70

        + "\n\n"

    )


    # --------------------------------------------------------
    # Prediction Information
    # --------------------------------------------------------

    file.write(

        f"Actual Dataset Label: "
        f"{actual_label}\n"

    )


    file.write(

        f"Model Prediction: "
        f"{predicted_label}\n"

    )


    file.write(

        f"Safe Probability: "
        f"{safe_probability * 100:.2f}%\n"

    )


    file.write(

        f"Phishing Probability: "
        f"{phishing_probability * 100:.2f}%\n\n"

    )


    # --------------------------------------------------------
    # Supporting Factors
    # --------------------------------------------------------

    file.write(

        f"TOP FACTORS SUPPORTING "
        f"{predicted_label.upper()}\n"

    )


    file.write(

        "-" * 70

        + "\n"

    )


    if supporting_factors:


        for index, item in enumerate(

            supporting_factors,

            start=1

        ):


            file.write(

                f"{index}. "

                f"{item['token']}"

                f" | Contribution: "

                f"{item['shap_value']:.6f}\n"

            )


    else:


        file.write(

            "No strong positive factors found.\n"

        )


    # --------------------------------------------------------
    # Opposing Factors
    # --------------------------------------------------------

    file.write(

        f"\nTOP FACTORS WORKING AGAINST "
        f"{predicted_label.upper()}\n"

    )


    file.write(

        "-" * 70

        + "\n"

    )


    if opposing_factors:


        for index, item in enumerate(

            opposing_factors,

            start=1

        ):


            file.write(

                f"{index}. "

                f"{item['token']}"

                f" | Contribution: "

                f"{item['shap_value']:.6f}\n"

            )


    else:


        file.write(

            "No strong negative factors found.\n"

        )


# ============================================================
# RESULTS SAVED
# ============================================================

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)


print(

    f"SHAP explanation saved to: "

    f"{OUTPUT_FILE}"

)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SHAP TEST COMPLETE")
print("=" * 70)