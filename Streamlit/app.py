# ============================================================
# UNIVERSITY EMAIL SECURITY
# Streamlit Frontend
#
# Backend:
#   FastAPI
#
# Model:
#   DeBERTa-v3-large + 12 NLP features + gated fusion
#
# Explainability:
#   SHAP
#
# IMPORTANT:
# Streamlit communicates with FastAPI.
# The model is NOT loaded directly by Streamlit.
# ============================================================

import re
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

HEALTH_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"
EXPLAIN_ENDPOINT = f"{API_URL}/explain"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="University Email Security",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "email_data" not in st.session_state:
    st.session_state.email_data = None

if "shap_result" not in st.session_state:
    st.session_state.shap_result = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #123b7a;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .section-note {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }

    .indicator-item {
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #f59e0b;
        background: #fff7ed;
        border-radius: 6px;
    }

    .characteristic-item {
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #3b82f6;
        background: #eff6ff;
        border-radius: 6px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API HEALTH CHECK
# ============================================================

def check_api_health():

    try:

        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=10
        )

        response.raise_for_status()

        return response.json(), None

    except requests.exceptions.ConnectionError:

        return None, (
            "Could not connect to the FastAPI backend. "
            "Make sure api.py is running on "
            "http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:

        return None, (
            "The FastAPI health check timed out."
        )

    except requests.exceptions.RequestException as error:

        return None, (
            f"API health-check error: {error}"
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_email(
    sender,
    subject,
    body
):

    payload = {
        "sender": sender,
        "subject": subject,
        "body": body
    }

    try:

        response = requests.post(
            PREDICT_ENDPOINT,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        return response.json(), None

    except requests.exceptions.ConnectionError:

        return None, (
            "Could not connect to the prediction API. "
            "Please make sure api.py is running."
        )

    except requests.exceptions.Timeout:

        return None, (
            "The prediction request timed out. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:

        return None, (
            f"Prediction API error: {error}"
        )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    sender,
    subject,
    body
):

    payload = {
        "sender": sender,
        "subject": subject,
        "body": body
    }

    try:

        response = requests.post(
            EXPLAIN_ENDPOINT,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        return response.json(), None

    except requests.exceptions.ConnectionError:

        return None, (
            "Could not connect to the Explainable AI API. "
            "Please make sure api.py is running."
        )

    except requests.exceptions.Timeout:

        return None, (
            "The SHAP explanation timed out. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:

        return None, (
            f"SHAP API error: {error}"
        )


# ============================================================
# EXTRACT SECURITY INDICATORS
# ============================================================

def get_indicators(result):

    """
    Get indicators returned by the revised FastAPI backend.

    Priority:

        1. rule_based_indicators
        2. indicators

    This ensures the Streamlit application uses the same
    context-aware indicator system as the browser extension.
    """

    indicators = result.get(
        "rule_based_indicators",
        None
    )

    if indicators is None:

        indicators = result.get(
            "indicators",
            []
        )

    if not isinstance(indicators, list):

        return []

    cleaned = []

    for indicator in indicators:

        if indicator is None:
            continue

        indicator = str(indicator).strip()

        if not indicator:
            continue

        if indicator not in cleaned:

            cleaned.append(indicator)

    return cleaned


# ============================================================
# EMAIL CHARACTERISTICS
# ============================================================

def get_email_characteristics(
    subject,
    body
):

    """
    Detect neutral structural characteristics.

    IMPORTANT:
    A URL, email address, or phone number by itself is NOT
    considered a phishing indicator.
    """

    combined_text = (
        f"{subject}\n{body}"
    )

    characteristics = []

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        combined_text,
        flags=re.IGNORECASE
    )

    if urls:

        characteristics.append(
            f"🔗 {len(urls)} URL/link"
            f"{'s' if len(urls) != 1 else ''} present"
        )

    # --------------------------------------------------------
    # Email addresses
    # --------------------------------------------------------

    email_addresses = re.findall(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}\b",
        combined_text
    )

    if email_addresses:

        characteristics.append(
            f"✉️ {len(email_addresses)} email address"
            f"{'es' if len(email_addresses) != 1 else ''} present"
        )

    # --------------------------------------------------------
    # Phone numbers
    # --------------------------------------------------------

    phone_numbers = re.findall(
        r"(?:\+?\d[\d\s().-]{7,}\d)",
        combined_text
    )

    if phone_numbers:

        characteristics.append(
            f"☎️ {len(phone_numbers)} phone number"
            f"{'s' if len(phone_numbers) != 1 else ''} present"
        )

    # --------------------------------------------------------
    # Text length
    # --------------------------------------------------------

    if len(body) > 1000:

        characteristics.append(
            "📝 Long-form email content detected"
        )

    elif len(body) > 300:

        characteristics.append(
            "📝 Moderate-length email content detected"
        )

    # --------------------------------------------------------
    # Uppercase emphasis
    # --------------------------------------------------------

    words = re.findall(
        r"\b[A-Za-z]+\b",
        body
    )

    if words:

        uppercase_words = [
            word
            for word in words
            if len(word) > 1 and word.isupper()
        ]

        uppercase_ratio = (
            len(uppercase_words) /
            max(len(words), 1)
        )

        if uppercase_ratio >= 0.15:

            characteristics.append(
                "🔠 Elevated uppercase-word usage detected"
            )

    return characteristics


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Status"
    )

    health, health_error = check_api_health()

    if health_error:

        st.error(
            "🔴 API Offline"
        )

        st.caption(
            health_error
        )

    else:

        if health.get(
            "model_loaded",
            False
        ):

            st.success(
                "🟢 API Online"
            )

        else:

            st.warning(
                "🟡 API Online — Model not loaded"
            )

        st.write(
            f"**Model:** "
            f"{health.get('model', 'Unknown')}"
        )

        st.write(
            f"**Architecture:** "
            f"{health.get('architecture', 'Unknown')}"
        )

        st.write(
            f"**Device:** "
            f"{health.get('device', 'Unknown')}"
        )

        st.write(
            f"**NLP Features:** "
            f"{health.get('num_features', 'Unknown')}"
        )

        st.write(
            f"**Maximum Length:** "
            f"{health.get('max_length', 'Unknown')}"
        )

        shap_status = health.get(
            "shap_available",
            False
        )

        st.write(
            f"**SHAP:** "
            f"{'Available' if shap_status else 'Unavailable'}"
        )

    st.divider()

    st.caption(
        "The Streamlit interface communicates with the "
        "FastAPI backend for prediction, security indicators "
        "and SHAP explanations."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ University Email Security</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Phishing Email Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Analyse an email using the DeBERTa-v3-large "
    "phishing detection model."
)


# ============================================================
# EMAIL INPUT SECTION
# ============================================================

st.write(
    "## 📧 Email Analysis"
)


sender_email = st.text_input(
    "Sender Email Address",
    placeholder="sender@example.com"
)


subject = st.text_input(
    "Email Subject",
    placeholder="Enter the email subject..."
)


email_text = st.text_area(
    "Email Body",
    height=260,
    placeholder="Paste the email body here..."
)


# ============================================================
# ANALYSE BUTTON
# ============================================================

if st.button(
    "🔍 Analyse Email",
    type="primary",
    use_container_width=True
):

    if not email_text.strip():

        st.warning(
            "Please enter the email body before analysing."
        )

    else:

        # Clear previous SHAP result
        st.session_state.shap_result = None

        with st.spinner(
            "Analysing email with DeBERTa-v3-large..."
        ):

            result, error = predict_email(
                sender_email,
                subject,
                email_text
            )

        if error:

            st.error(
                error
            )

        else:

            st.session_state.analysis_result = result

            st.session_state.email_data = {
                "sender": sender_email,
                "subject": subject,
                "body": email_text
            }


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.analysis_result is not None:

    result = st.session_state.analysis_result

    email_data = st.session_state.email_data

    # ========================================================
    # DETECTION RESULT
    # ========================================================

    st.divider()

    st.write(
        "## 🎯 Detection Result"
    )

    prediction = result.get(
        "prediction",
        "Unknown"
    )

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    # --------------------------------------------------------
    # Phishing result
    # --------------------------------------------------------

    if prediction == "Phishing Email":

        st.error(
            f"⚠️ Potential Phishing Email Detected\n\n"
            f"Confidence: {confidence:.2%}"
        )

    # --------------------------------------------------------
    # Safe result
    # --------------------------------------------------------

    elif prediction == "Safe Email":

        st.success(
            f"✅ Safe Email\n\n"
            f"Confidence: {confidence:.2%}"
        )

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    else:

        st.warning(
            f"Model returned: {prediction}"
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.write(
        "### 📊 Probability Breakdown"
    )

    safe_probability = float(
        result.get(
            "safe_probability",
            0
        )
    )

    phishing_probability = float(
        result.get(
            "phishing_probability",
            0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Safe Probability",
            f"{safe_probability:.2%}"
        )

    with col2:

        st.metric(
            "Phishing Probability",
            f"{phishing_probability:.2%}"
        )


    # ========================================================
    # CONFIDENCE MESSAGE
    # ========================================================

    if confidence < 0.60:

        st.warning(
            "⚠️ The model has relatively low confidence "
            "in this classification. Manual review is recommended."
        )

    elif confidence < 0.80:

        st.info(
            "ℹ️ The model has moderate confidence. "
            "Review the email carefully before taking action."
        )

    else:

        st.caption(
            "The model produced a high-confidence classification."
        )


    # ========================================================
    # EMAIL DETAILS
    # ========================================================

    st.write(
        "## 📋 Email Details"
    )

    st.write(
        f"**Sender:** "
        f"{email_data.get('sender') or 'Not provided'}"
    )

    st.write(
        f"**Subject:** "
        f"{email_data.get('subject') or 'Not provided'}"
    )


    # ========================================================
    # SECURITY INDICATORS
    # ========================================================

    st.write(
        "## 🛡️ Security Indicators"
    )

    st.markdown(
        '<div class="section-note">'
        'These indicators provide additional context about '
        'patterns detected in the email. They supplement the '
        'DeBERTa-v3-large classification and are not the model '
        'prediction itself.'
        '</div>',
        unsafe_allow_html=True
    )

    indicators = get_indicators(
        result
    )


    # --------------------------------------------------------
    # Actual indicators found
    # --------------------------------------------------------

    if indicators:

        for indicator in indicators:

            st.warning(
                f"⚠️ {indicator}"
            )


    # --------------------------------------------------------
    # No rule indicators + phishing
    #
    # IMPORTANT FALLBACK
    # --------------------------------------------------------

    elif prediction == "Phishing Email":

        fallback_message = result.get(
            "indicator_message",
            "The model detected phishing based on "
            "contextual and semantic patterns. "
            "No predefined rule-based indicators "
            "matched this message."
        )

        st.warning(
            f"⚠️ Model-based detection\n\n"
            f"{fallback_message}"
        )


    # --------------------------------------------------------
    # No indicators + safe
    # --------------------------------------------------------

    else:

        st.success(
            "✅ No common phishing indicators detected."
        )


    # ========================================================
    # EMAIL CHARACTERISTICS
    # ========================================================

    st.write(
        "## 📌 Email Characteristics"
    )

    st.markdown(
        '<div class="section-note">'
        'These are neutral structural characteristics of the '
        'message. Their presence alone does not mean that an '
        'email is phishing.'
        '</div>',
        unsafe_allow_html=True
    )

    characteristics = get_email_characteristics(

        email_data.get(
            "subject",
            ""
        ),

        email_data.get(
            "body",
            ""
        )

    )


    if characteristics:

        for characteristic in characteristics:

            st.info(
                characteristic
            )

    else:

        st.caption(
            "No notable structural characteristics detected."
        )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.divider()

    st.write(
        "## 🧠 Explainable AI Analysis"
    )

    st.write(
        "SHAP Explainable AI identifies the words and tokens "
        "that contributed most strongly to the model's decision."
    )


    # --------------------------------------------------------
    # Explain button
    # --------------------------------------------------------

    if st.button(
        "🧠 Explain AI Decision",
        use_container_width=True
    ):

        with st.spinner(
            "Generating SHAP explanation... "
            "This may take some time because the model "
            "is running on CPU."
        ):

            explanation, error = get_shap_explanation(

                email_data.get(
                    "sender",
                    ""
                ),

                email_data.get(
                    "subject",
                    ""
                ),

                email_data.get(
                    "body",
                    ""
                )

            )


        if error:

            st.error(
                error
            )

        elif not explanation.get(
            "success",
            True
        ):

            st.error(
                explanation.get(
                    "message",
                    "Unable to generate SHAP explanation."
                )
            )

        else:

            st.session_state.shap_result = explanation

            st.success(
                "SHAP explanation generated successfully."
            )


    # ========================================================
    # DISPLAY SHAP EXPLANATION
    # ========================================================

    if st.session_state.shap_result is not None:

        shap_data = st.session_state.shap_result

        st.write(
            "### 🔎 SHAP Explanation"
        )

        shap_prediction = shap_data.get(
            "prediction",
            prediction
        )

        st.write(
            f"Explanation for: **{shap_prediction}**"
        )


        # ====================================================
        # SUPPORTING FACTORS
        # ====================================================

        supporting_factors = shap_data.get(
            "supporting_factors",
            []
        )

        st.write(
            "#### 🔴 Factors Supporting the Prediction"
        )

        if supporting_factors:

            for index, factor in enumerate(
                supporting_factors,
                start=1
            ):

                token = factor.get(
                    "token",
                    "Unknown"
                )

                contribution = float(
                    factor.get(
                        "contribution",
                        0
                    )
                )

                st.write(
                    f"**{index}. {token}**  \n"
                    f"Contribution: `{contribution:.6f}`"
                )

        else:

            st.info(
                "No significant supporting factors were returned."
            )


        # ====================================================
        # OPPOSING FACTORS
        # ====================================================

        opposing_factors = shap_data.get(
            "opposing_factors",
            []
        )

        st.write(
            "#### 🟢 Factors Working Against the Prediction"
        )

        if opposing_factors:

            for index, factor in enumerate(
                opposing_factors,
                start=1
            ):

                token = factor.get(
                    "token",
                    "Unknown"
                )

                contribution = float(
                    factor.get(
                        "contribution",
                        0
                    )
                )

                st.write(
                    f"**{index}. {token}**  \n"
                    f"Contribution: `{contribution:.6f}`"
                )

        else:

            st.info(
                "No significant opposing factors were returned."
            )


        # ====================================================
        # STRONGEST FACTORS
        # ====================================================

        strongest_factors = shap_data.get(
            "strongest_factors",
            []
        )

        if strongest_factors:

            st.write(
                "#### 🔍 Strongest SHAP Factors"
            )

            for index, factor in enumerate(
                strongest_factors,
                start=1
            ):

                token = factor.get(
                    "token",
                    "Unknown"
                )

                contribution = float(
                    factor.get(
                        "contribution",
                        0
                    )
                )

                if contribution > 0:

                    direction = (
                        "supports the phishing prediction"
                    )

                else:

                    direction = (
                        "works against the phishing prediction"
                    )

                st.write(
                    f"**{index}. {token}** — "
                    f"`{contribution:.6f}` "
                    f"({direction})"
                )


        # ====================================================
        # SHAP INDICATORS
        # ====================================================

        shap_indicators = (
            shap_data.get(
                "rule_based_indicators",
                None
            )
        )

        if shap_indicators is None:

            shap_indicators = (
                shap_data.get(
                    "indicators",
                    []
                )
            )

        if isinstance(
            shap_indicators,
            list
        ) and shap_indicators:

            st.write(
                "#### 🛡️ Security Indicators Confirmed"
            )

            for indicator in shap_indicators:

                st.warning(
                    f"⚠️ {indicator}"
                )


        # ====================================================
        # SHAP EXPLANATION NOTE
        # ====================================================

        st.caption(
            "Positive SHAP contributions support the phishing "
            "classification, while negative contributions "
            "work against it. SHAP explanations provide "
            "interpretability and do not change the model's "
            "prediction."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'University Phishing Email Detection System | '
    'DeBERTa-v3-large + NLP + FastAPI + SHAP Explainable AI'
    '</div>',
    unsafe_allow_html=True
)