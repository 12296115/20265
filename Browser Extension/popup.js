// ============================================================
// UNIVERSITY PHISHING EMAIL DETECTOR
// popup.js
//
// DeBERTa-v3-large + 12 NLP features + gated fusion
// + SHAP Explainable AI
//
// The DeBERTa model is the primary classifier.
// Rule-based indicators are supplementary explanations.
//
// UI behaviour:
//
// 1. Safe + no indicators
//       -> Phishing Indicators
//       -> "No common phishing indicators detected"
//
// 2. Safe + characteristics
//       -> Email Characteristics
//       -> Phishing Indicators
//
// 3. Phishing + rule indicators
//       -> Phishing Indicators
//
// 4. Phishing + no rule indicators
//       -> Detection Explanation
//       -> Model-based fallback message
//
// A phishing prediction MUST NEVER display the green
// "No common phishing indicators detected" message.
// ============================================================


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE_URL =
    "http://127.0.0.1:8000";

const PREDICT_ENDPOINT =
    `${API_BASE_URL}/predict`;


// ============================================================
// DOM ELEMENTS
// ============================================================

const analyseButton =
    document.getElementById(
        "analyseButton"
    );

const loading =
    document.getElementById(
        "loading"
    );

const errorBox =
    document.getElementById(
        "error"
    );

const result =
    document.getElementById(
        "result"
    );


// ============================================================
// ANALYSE BUTTON
// ============================================================

if (analyseButton) {

    analyseButton.addEventListener(
        "click",
        analyseCurrentEmail
    );

}


// ============================================================
// MAIN ANALYSIS FUNCTION
// ============================================================

async function analyseCurrentEmail() {

    hideError();

    if (result) {

        result.classList.add(
            "hidden"
        );

    }


    if (analyseButton) {

        analyseButton.disabled =
            true;

    }


    if (loading) {

        loading.classList.remove(
            "hidden"
        );

    }


    try {

        // ========================================================
        // GET ACTIVE BROWSER TAB
        // ========================================================

        const tabs =
            await chrome.tabs.query({

                active: true,

                currentWindow: true

            });


        const tab =
            tabs[0];


        if (
            !tab ||
            !tab.id
        ) {

            throw new Error(
                "Unable to access the current browser tab."
            );

        }


        // ========================================================
        // GET EMAIL FROM content.js
        // ========================================================

        const email =
            await chrome.tabs.sendMessage(

                tab.id,

                {
                    action:
                        "getEmail"
                }

            );


        console.log(
            "EMAIL RECEIVED BY POPUP:",
            email
        );


        if (
            !email ||
            !email.success ||
            !email.body
        ) {

            throw new Error(

                email?.error ||

                "Could not detect an email on this page."

            );

        }


        // ========================================================
        // SEND EMAIL TO FASTAPI
        // ========================================================

        const response =
            await fetch(

                PREDICT_ENDPOINT,

                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            sender:
                                email.sender || "",

                            subject:
                                email.subject || "",

                            body:
                                email.body || ""

                        })

                }

            );


        // ========================================================
        // HTTP ERROR
        // ========================================================

        if (
            !response.ok
        ) {

            let errorMessage =
                "API request failed. Make sure api.py is running.";


            try {

                const errorData =
                    await response.json();


                if (
                    errorData &&
                    errorData.error
                ) {

                    errorMessage =
                        errorData.error;

                }

            }

            catch (
                ignored
            ) {

                // Keep default error message.

            }


            throw new Error(
                errorMessage
            );

        }


        // ========================================================
        // READ API RESPONSE
        // ========================================================

        const data =
            await response.json();


        console.log(
            "API RESULT:",
            data
        );


        if (
            data.success === false
        ) {

            throw new Error(

                data.error ||

                data.message ||

                "The API could not analyse this email."

            );

        }


        // ========================================================
        // DISPLAY RESULT
        // ========================================================

        displayResult(
            data,
            email
        );

    }


    catch (
        error
    ) {

        console.error(
            "Email analysis error:",
            error
        );


        showError(

            error?.message ||

            "An unexpected error occurred."

        );

    }


    finally {

        if (analyseButton) {

            analyseButton.disabled =
                false;

        }


        if (loading) {

            loading.classList.add(
                "hidden"
            );

        }

    }

}


// ============================================================
// DISPLAY COMPLETE RESULT
// ============================================================

function displayResult(
    data,
    email
) {

    if (result) {

        result.classList.remove(
            "hidden"
        );

    }


    // ========================================================
    // DOM ELEMENTS
    // ========================================================

    const predictionElement =
        document.getElementById(
            "prediction"
        );


    const confidenceElement =
        document.getElementById(
            "confidence"
        );


    const predictionBox =
        document.getElementById(
            "predictionBox"
        );


    const predictionIcon =
        document.getElementById(
            "predictionIcon"
        );


    // ========================================================
    // PREDICTION
    // ========================================================

    const prediction =
        String(
            data.prediction ||
            "Unknown"
        ).trim();


    if (
        predictionElement
    ) {

        predictionElement.textContent =
            prediction;

    }


    // ========================================================
    // CONFIDENCE
    // ========================================================

    const confidence =
        getNumericValue(
            data.confidence_percentage
        );


    if (
        confidenceElement
    ) {

        if (
            Number.isFinite(
                confidence
            )
        ) {

            confidenceElement.textContent =
                `Confidence: ${confidence.toFixed(2)}%`;

        }

        else {

            confidenceElement.textContent =
                "Confidence: unavailable";

        }

    }


    // ========================================================
    // PREDICTION BOX STYLE
    // ========================================================

    if (
        predictionBox
    ) {

        predictionBox.classList.remove(

            "prediction-safe",

            "prediction-phishing"

        );

    }


    if (
        predictionIcon
    ) {

        predictionIcon.textContent =
            "";

    }


    if (
        prediction ===
        "Phishing Email"
    ) {

        if (
            predictionBox
        ) {

            predictionBox.classList.add(
                "prediction-phishing"
            );

        }


        if (
            predictionIcon
        ) {

            predictionIcon.textContent =
                "⚠️";

        }

    }

    else {

        if (
            predictionBox
        ) {

            predictionBox.classList.add(
                "prediction-safe"
            );

        }


        if (
            predictionIcon
        ) {

            predictionIcon.textContent =
                "✅";

        }

    }


    // ========================================================
    // PROBABILITIES
    // ========================================================

    displayProbabilities(
        data
    );


    // ========================================================
    // EMAIL DETAILS
    // ========================================================

    displayEmailDetails(
        email
    );


    // ========================================================
    // EMAIL CHARACTERISTICS
    // ========================================================

    displayEmailCharacteristics(
        email
    );


    // ========================================================
    // INDICATORS / DETECTION EXPLANATION
    // ========================================================

    displayIndicators(
        data
    );

}


// ============================================================
// DISPLAY PROBABILITIES
//
// The displayed percentages are normalised so:
//
// Safe + Phishing = exactly 100.00%
//
// This prevents cases such as:
//
// Safe      99.98%
// Phishing   2.00%
//
// which incorrectly total 101.98% due to rounding.
// ============================================================

function displayProbabilities(
    data
) {

    const safeElement =
        document.getElementById(
            "safeProbability"
        );


    const phishingElement =
        document.getElementById(
            "phishingProbability"
        );


    let safePercentage =
        getNumericValue(
            data.safe_probability_percentage
        );


    let phishingPercentage =
        getNumericValue(
            data.phishing_probability_percentage
        );


    // ========================================================
    // FALLBACK: CALCULATE FROM RAW PROBABILITIES
    // ========================================================

    if (
        !Number.isFinite(
            safePercentage
        )
    ) {

        const rawSafe =
            getNumericValue(
                data.safe_probability
            );


        if (
            Number.isFinite(
                rawSafe
            )
        ) {

            safePercentage =
                rawSafe <= 1
                    ? rawSafe * 100
                    : rawSafe;

        }

    }


    if (
        !Number.isFinite(
            phishingPercentage
        )
    ) {

        const rawPhishing =
            getNumericValue(
                data.phishing_probability
            );


        if (
            Number.isFinite(
                rawPhishing
            )
        ) {

            phishingPercentage =
                rawPhishing <= 1
                    ? rawPhishing * 100
                    : rawPhishing;

        }

    }


    // ========================================================
    // NORMALISE
    // ========================================================

    if (
        Number.isFinite(
            safePercentage
        )
    ) {

        safePercentage =
            clamp(
                safePercentage,
                0,
                100
            );


        // ----------------------------------------------------
        // Use Safe as the primary displayed value.
        // Calculate Phishing as the exact remainder.
        // ----------------------------------------------------

        safePercentage =
            Number(
                safePercentage.toFixed(2)
            );


        phishingPercentage =
            Number(
                (
                    100 -
                    safePercentage
                ).toFixed(2)
            );

    }

    else if (
        Number.isFinite(
            phishingPercentage
        )
    ) {

        phishingPercentage =
            clamp(
                phishingPercentage,
                0,
                100
            );


        phishingPercentage =
            Number(
                phishingPercentage.toFixed(2)
            );


        safePercentage =
            Number(
                (
                    100 -
                    phishingPercentage
                ).toFixed(2)
            );

    }


    // ========================================================
    // DISPLAY SAFE
    // ========================================================

    if (
        safeElement
    ) {

        if (
            Number.isFinite(
                safePercentage
            )
        ) {

            safeElement.textContent =
                `${safePercentage.toFixed(2)}%`;

        }

        else {

            safeElement.textContent =
                "Unavailable";

        }

    }


    // ========================================================
    // DISPLAY PHISHING
    // ========================================================

    if (
        phishingElement
    ) {

        if (
            Number.isFinite(
                phishingPercentage
            )
        ) {

            phishingElement.textContent =
                `${phishingPercentage.toFixed(2)}%`;

        }

        else {

            phishingElement.textContent =
                "Unavailable";

        }

    }

}


// ============================================================
// DISPLAY EMAIL DETAILS
// ============================================================

function displayEmailDetails(
    email
) {

    const senderElement =
        document.getElementById(
            "sender"
        );


    const subjectElement =
        document.getElementById(
            "subject"
        );


    if (
        senderElement
    ) {

        senderElement.textContent =
            email.sender ||
            "Not detected";

    }


    if (
        subjectElement
    ) {

        subjectElement.textContent =
            email.subject ||
            "Not detected";

    }

}


// ============================================================
// EMAIL CHARACTERISTICS
//
// These are intentionally NEUTRAL characteristics.
//
// A URL, email address or phone number is not automatically
// treated as a phishing indicator.
//
// Example:
//
// Email Characteristics
// 🔗 URL or link present
// ✉️ Email address present
//
// These are separate from actual phishing indicators.
// ============================================================

function displayEmailCharacteristics(
    email
) {

    const section =
        document.getElementById(
            "characteristicsSection"
        );


    const container =
        document.getElementById(
            "characteristics"
        );


    if (
        !section ||
        !container
    ) {

        return;

    }


    container.innerHTML =
        "";


    const body =
        String(
            email?.body ||
            ""
        );


    const sender =
        String(
            email?.sender ||
            ""
        );


    const subject =
        String(
            email?.subject ||
            ""
        );


    const completeText =
        `${sender} ${subject} ${body}`;


    const characteristics =
        [];


    // ========================================================
    // URL
    // ========================================================

    const urlMatches =
        completeText.match(
            /(?:https?:\/\/|www\.)[^\s<>"']+/gi
        );


    if (
        urlMatches &&
        urlMatches.length > 0
    ) {

        characteristics.push({

            icon:
                "🔗",

            text:
                `${urlMatches.length} URL/link${urlMatches.length === 1 ? "" : "s"} present`

        });

    }


    // ========================================================
    // EMAIL ADDRESS
    // ========================================================

    const emailMatches =
        completeText.match(

            /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g

        );


    if (
        emailMatches &&
        emailMatches.length > 0
    ) {

        characteristics.push({

            icon:
                "✉️",

            text:
                `${emailMatches.length} email address${emailMatches.length === 1 ? "" : "es"} present`

        });

    }


    // ========================================================
    // PHONE NUMBER
    // ========================================================

    const phoneMatches =
        completeText.match(

            /(?:\+?\d[\d\s().-]{7,}\d)/g

        );


    if (
        phoneMatches &&
        phoneMatches.length > 0
    ) {

        characteristics.push({

            icon:
                "📞",

            text:
                `${phoneMatches.length} phone number${phoneMatches.length === 1 ? "" : "s"} present`

        });

    }


    // ========================================================
    // EXCLAMATION MARKS
    //
    // This is a neutral characteristic, not automatically
    // a phishing indicator.
    // ========================================================

    const exclamationCount =
        (
            completeText.match(
                /!/g
            ) ||
            []
        ).length;


    if (
        exclamationCount > 0
    ) {

        characteristics.push({

            icon:
                "❗",

            text:
                `${exclamationCount} exclamation mark${exclamationCount === 1 ? "" : "s"} present`

        });

    }


    // ========================================================
    // IF NOTHING TO DISPLAY
    // ========================================================

    if (
        characteristics.length === 0
    ) {

        section.classList.add(
            "hidden"
        );

        return;

    }


    // ========================================================
    // CREATE CHARACTERISTIC ITEMS
    // ========================================================

    characteristics.forEach(
        function (
            characteristic
        ) {

            const div =
                document.createElement(
                    "div"
                );


            div.className =
                "characteristic";


            div.textContent =
                `${characteristic.icon} ${characteristic.text}`;


            container.appendChild(
                div
            );

        }
    );


    section.classList.remove(
        "hidden"
    );

}


// ============================================================
// DISPLAY INDICATORS
//
// IMPORTANT:
//
// This is the key UI logic.
//
// PHISHING + INDICATORS:
//     Heading = "Phishing Indicators"
//
// PHISHING + NO INDICATORS:
//     Heading = "Detection Explanation"
//
// SAFE + NO INDICATORS:
//     Heading = "Phishing Indicators"
//     Green confirmation message
//
// Therefore a phishing prediction can NEVER display:
//
//     ✅ No common phishing indicators detected
// ============================================================

function displayIndicators(
    data
) {

    const heading =
        document.getElementById(
            "indicatorHeading"
        );


    const container =
        document.getElementById(
            "indicators"
        );


    if (
        !container
    ) {

        return;

    }


    // ========================================================
    // CLEAR OLD CONTENT
    // ========================================================

    container.innerHTML =
        "";


    // ========================================================
    // GET PREDICTION
    // ========================================================

    const prediction =
        String(
            data?.prediction ||
            ""
        ).trim();


    const isPhishing =
        prediction ===
        "Phishing Email";


    // ========================================================
    // GET INDICATORS FROM API
    // ========================================================

    let indicators =
        [];


    if (
        Array.isArray(
            data?.indicators
        )
    ) {

        indicators =
            data.indicators
                .filter(
                    function (
                        indicator
                    ) {

                        return (

                            indicator !==
                            null &&

                            indicator !==
                            undefined &&

                            String(
                                indicator
                            ).trim() !== ""

                        );

                    }
                )
                .map(
                    function (
                        indicator
                    ) {

                        return String(
                            indicator
                        ).trim();

                    }
                );

    }


    // ========================================================
    // FALLBACK API FIELD
    //
    // Supports a revised API response using
    // rule_based_indicators.
    // ========================================================

    if (
        indicators.length === 0
        &&
        Array.isArray(
            data?.rule_based_indicators
        )
    ) {

        indicators =
            data.rule_based_indicators
                .filter(
                    function (
                        indicator
                    ) {

                        return (

                            indicator !==
                            null &&

                            indicator !==
                            undefined &&

                            String(
                                indicator
                            ).trim() !== ""

                        );

                    }
                )
                .map(
                    function (
                        indicator
                    ) {

                        return String(
                            indicator
                        ).trim();

                    }
                );

    }


    // ========================================================
    // REMOVE DUPLICATES
    // ========================================================

    indicators =
        [
            ...new Set(
                indicators
            )
        ];


    // ========================================================
    // PHISHING EMAIL
    // ========================================================

    if (
        isPhishing
    ) {

        // ----------------------------------------------------
        // CASE 1:
        // Actual rule-based indicators exist.
        // ----------------------------------------------------

        if (
            indicators.length > 0
        ) {

            if (
                heading
            ) {

                heading.textContent =
                    "Phishing Indicators";

            }


            indicators.forEach(
                function (
                    indicator
                ) {

                    addIndicator(
                        container,
                        indicator
                    );

                }
            );


            return;

        }


        // ----------------------------------------------------
        // CASE 2:
        // DeBERTa detected phishing but no predefined
        // rule-based indicator matched.
        // ----------------------------------------------------

        if (
            heading
        ) {

            heading.textContent =
                "Detection Explanation";

        }


        const modelExplanation =
            data?.indicator_message &&
            String(
                data.indicator_message
            ).trim() !== ""

                ? String(
                    data.indicator_message
                ).trim()

                :

                "The model detected phishing based on contextual and semantic patterns. No predefined rule-based indicators matched this message.";


        addModelDetectionExplanation(

            container,

            modelExplanation

        );


        return;

    }


    // ========================================================
    // SAFE EMAIL
    // ========================================================

    if (
        heading
    ) {

        heading.textContent =
            "Phishing Indicators";

    }


    // --------------------------------------------------------
    // Safe email with rule-based indicators
    // --------------------------------------------------------

    if (
        indicators.length > 0
    ) {

        indicators.forEach(
            function (
                indicator
            ) {

                addIndicator(
                    container,
                    indicator
                );

            }
        );


        return;

    }


    // --------------------------------------------------------
    // Safe email with no indicators
    //
    // ONLY HERE do we show the green confirmation.
    // --------------------------------------------------------

    const noIndicators =
        document.createElement(
            "div"
        );


    noIndicators.className =
        "no-indicators";


    noIndicators.textContent =
        "✅ No common phishing indicators detected";


    container.appendChild(
        noIndicators
    );

}


// ============================================================
// ADD STANDARD PHISHING INDICATOR
// ============================================================

function addIndicator(
    container,
    indicator
) {

    const div =
        document.createElement(
            "div"
        );


    div.className =
        "indicator";


    const text =
        String(
            indicator
        ).trim();


    // Avoid duplicate warning icons.
    if (
        text.startsWith(
            "⚠️"
        )
    ) {

        div.textContent =
            text;

    }

    else {

        div.textContent =
            `⚠️ ${text}`;

    }


    container.appendChild(
        div
    );

}


// ============================================================
// ADD MODEL-BASED DETECTION EXPLANATION
// ============================================================

function addModelDetectionExplanation(
    container,
    message
) {

    const div =
        document.createElement(
            "div"
        );


    div.className =
        "model-detection";


    div.textContent =
        `⚠️ ${message}`;


    container.appendChild(
        div
    );

}


// ============================================================
// NUMERIC VALUE HELPER
// ============================================================

function getNumericValue(
    value
) {

    if (
        value ===
        null ||
        value ===
        undefined ||
        value ===
        ""
    ) {

        return NaN;

    }


    const number =
        Number(
            value
        );


    return Number.isFinite(
        number
    )
        ? number
        : NaN;

}


// ============================================================
// CLAMP VALUE
// ============================================================

function clamp(
    value,
    minimum,
    maximum
) {

    return Math.min(

        maximum,

        Math.max(
            minimum,
            value
        )

    );

}


// ============================================================
// SHOW ERROR
// ============================================================

function showError(
    message
) {

    if (
        !errorBox
    ) {

        return;

    }


    errorBox.textContent =
        message;


    errorBox.classList.remove(
        "hidden"
    );

}


// ============================================================
// HIDE ERROR
// ============================================================

function hideError() {

    if (
        !errorBox
    ) {

        return;

    }


    errorBox.classList.add(
        "hidden"
    );


    errorBox.textContent =
        "";

}