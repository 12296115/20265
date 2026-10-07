// ============================================================
// UNIVERSITY PHISHING EMAIL DETECTOR
// popup.js
//
// DeBERTa-v3-large + 12 engineered NLP features + gated fusion
//
// IMPORTANT:
//
// 1. DeBERTa-v3-large is the primary classifier.
// 2. Rule-based indicators are supplementary explanations.
// 3. Indicators do NOT override or determine the prediction.
// 4. Every phishing prediction receives at least one
//    human-readable phishing indicator.
// 5. SHAP is NOT executed from this popup.
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

        // ====================================================
        // GET ACTIVE BROWSER TAB
        // ====================================================

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


        // ====================================================
        // GET EMAIL FROM content.js
        // ====================================================

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


        // ====================================================
        // SEND EMAIL TO FASTAPI
        // ====================================================

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


        // ====================================================
        // HTTP ERROR
        // ====================================================

        if (!response.ok) {

            let errorMessage =
                "API request failed. Make sure api.py is running.";


            try {

                const errorData =
                    await response.json();


                errorMessage =

                    errorData?.error ||

                    errorData?.message ||

                    errorData?.detail ||

                    errorMessage;

            }

            catch (ignored) {

                // Keep default error message.

            }


            throw new Error(
                errorMessage
            );

        }


        // ====================================================
        // READ API RESPONSE
        // ====================================================

        const data =
            await response.json();


        console.log(
            "API RESULT:",
            data
        );


        if (
            data?.success === false
        ) {

            throw new Error(

                data?.error ||

                data?.message ||

                "The API could not analyse this email."

            );

        }


        // ====================================================
        // DISPLAY RESULT
        // ====================================================

        displayResult(
            data,
            email
        );

    }


    catch (error) {

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
            data?.prediction ||
            "Unknown"
        ).trim();


    if (predictionElement) {

        predictionElement.textContent =
            prediction;

    }


    // ========================================================
    // CONFIDENCE
    // ========================================================

    let confidence =
        getNumericValue(
            data?.confidence_percentage
        );


    if (
        !Number.isFinite(
            confidence
        )
    ) {

        confidence =
            getNumericValue(
                data?.confidence
            );


        if (
            Number.isFinite(
                confidence
            ) &&
            confidence <= 1
        ) {

            confidence *= 100;

        }

    }


    if (confidenceElement) {

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
    // RESET PREDICTION STYLE
    // ========================================================

    if (predictionBox) {

        predictionBox.classList.remove(

            "prediction-safe",

            "prediction-phishing"

        );

    }


    if (predictionIcon) {

        predictionIcon.textContent =
            "";

    }


    // ========================================================
    // PHISHING / SAFE STYLE
    // ========================================================

    if (
        prediction ===
        "Phishing Email"
    ) {

        if (predictionBox) {

            predictionBox.classList.add(
                "prediction-phishing"
            );

        }


        if (predictionIcon) {

            predictionIcon.textContent =
                "⚠️";

        }

    }

    else {

        if (predictionBox) {

            predictionBox.classList.add(
                "prediction-safe"
            );

        }


        if (predictionIcon) {

            predictionIcon.textContent =
                "✅";

        }

    }


    // ========================================================
    // RESULT SECTIONS
    // ========================================================

    displayProbabilities(
        data
    );


    displayEmailDetails(
        email
    );


    displayEmailCharacteristics(
        email
    );


    displayIndicators(
        data,
        email
    );

}


// ============================================================
// DISPLAY PROBABILITIES
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
            data?.safe_probability_percentage
        );


    let phishingPercentage =
        getNumericValue(
            data?.phishing_probability_percentage
        );


    // ========================================================
    // FALLBACK: RAW SAFE PROBABILITY
    // ========================================================

    if (
        !Number.isFinite(
            safePercentage
        )
    ) {

        const rawSafe =
            getNumericValue(
                data?.safe_probability
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


    // ========================================================
    // FALLBACK: RAW PHISHING PROBABILITY
    // ========================================================

    if (
        !Number.isFinite(
            phishingPercentage
        )
    ) {

        const rawPhishing =
            getNumericValue(
                data?.phishing_probability
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
    // NORMALISE DISPLAY VALUES
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

    if (safeElement) {

        safeElement.textContent =

            Number.isFinite(
                safePercentage
            )

                ? `${safePercentage.toFixed(2)}%`

                : "Unavailable";

    }


    // ========================================================
    // DISPLAY PHISHING
    // ========================================================

    if (phishingElement) {

        phishingElement.textContent =

            Number.isFinite(
                phishingPercentage
            )

                ? `${phishingPercentage.toFixed(2)}%`

                : "Unavailable";

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


    if (senderElement) {

        senderElement.textContent =

            email?.sender ||

            "Not detected";

    }


    if (subjectElement) {

        subjectElement.textContent =

            email?.subject ||

            "Not detected";

    }

}


// ============================================================
// EMAIL CHARACTERISTICS
//
// These are NEUTRAL characteristics.
//
// URLs, email addresses, phone numbers and punctuation are
// not automatically considered phishing.
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


    const completeText =

        `${email?.sender || ""} ` +

        `${email?.subject || ""} ` +

        `${email?.body || ""}`;


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

                `${urlMatches.length} URL/link` +

                `${urlMatches.length === 1 ? "" : "s"} present`

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

                `${emailMatches.length} email address` +

                `${emailMatches.length === 1 ? "" : "es"} present`

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

                `${phoneMatches.length} phone number` +

                `${phoneMatches.length === 1 ? "" : "s"} present`

        });

    }


    // ========================================================
    // EXCLAMATION MARKS
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

                `${exclamationCount} exclamation mark` +

                `${exclamationCount === 1 ? "" : "s"} present`

        });

    }


    // ========================================================
    // QUESTION MARKS
    // ========================================================

    const questionCount =
        (
            completeText.match(
                /\?/g
            ) ||
            []
        ).length;


    if (
        questionCount > 0
    ) {

        characteristics.push({

            icon:
                "❓",

            text:

                `${questionCount} question mark` +

                `${questionCount === 1 ? "" : "s"} present`

        });

    }


    // ========================================================
    // NOTHING TO DISPLAY
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
    // DISPLAY CHARACTERISTICS
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

                `${characteristic.icon} ` +

                `${characteristic.text}`;


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
// GET API PHISHING INDICATORS
// ============================================================

function getApiIndicators(
    data
) {

    let indicators =
        [];


    // ========================================================
    // PRIMARY API FIELD
    // ========================================================

    if (
        Array.isArray(
            data?.indicators
        )
    ) {

        indicators =
            data.indicators;

    }


    // ========================================================
    // FALLBACK API FIELD
    // ========================================================

    if (
        indicators.length === 0 &&
        Array.isArray(
            data?.rule_based_indicators
        )
    ) {

        indicators =
            data.rule_based_indicators;

    }


    // ========================================================
    // CLEAN VALUES
    // ========================================================

    indicators =
        indicators

            .filter(

                function (
                    indicator
                ) {

                    return (

                        indicator !== null &&

                        indicator !== undefined &&

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


    return removeDuplicateIndicators(
        indicators
    );

}


// ============================================================
// DETECT SUPPLEMENTARY PHISHING INDICATORS
//
// These provide human-readable descriptions of suspicious
// patterns present in the email.
//
// They do NOT make or override the model prediction.
// ============================================================

function detectPhishingIndicators(
    email
) {

    const sender =
        String(
            email?.sender || ""
        );


    const subject =
        String(
            email?.subject || ""
        );


    const body =
        String(
            email?.body || ""
        );


    const text =
        `${subject} ${body}`;


    const lowerText =
        text.toLowerCase();


    const indicators =
        [];


    // ========================================================
    // 1. URGENCY / PRESSURE
    // ========================================================

    const urgencyPattern =
        /\b(urgent|urgently|immediately|immediate|act now|action required|action needed|as soon as possible|asap|within 24 hours|within 48 hours|within one hour|deadline|expires|expired|expiring|final warning|final notice|last chance|limited time|time sensitive|do not delay|respond immediately|attention required|important notice|take action|quickly|hurry)\b/i;


    if (
        urgencyPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Urgency or pressure language detected"
        );

    }


    // ========================================================
    // 2. CREDENTIAL / LOGIN / ACCOUNT VERIFICATION
    // ========================================================

    const credentialPattern =
        /\b(password|passcode|login|log in|signin|sign in|username|credential|credentials|authentication|authenticate|verification code|security code|verify your account|verify account|account verification|confirm your account|confirm account|update your account|update account|security verification|identity verification|validate your account|validate account|reactivate your account|reactivate account)\b/i;


    if (
        credentialPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Credential or account-verification language detected"
        );

    }


    // ========================================================
    // 3. THREAT / NEGATIVE CONSEQUENCE
    // ========================================================

    const threatPattern =
        /\b(account.{0,30}(closed|locked|suspended|disabled|terminated|restricted)|suspend(ed|ing)?|deactivat(e|ed|ion)|terminat(e|ed|ion)|legal action|penalty|access.{0,20}(blocked|revoked|removed|restricted)|security breach|unauthori[sz]ed access|compromised account|failure to respond|fail to respond|service interruption|lose access|account restriction|account closure)\b/i;


    if (
        threatPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Threat or negative-consequence language detected"
        );

    }


    // ========================================================
    // 4. FINANCIAL / PAYMENT
    // ========================================================

    const financialPattern =
        /\b(payment|invoice|billing|bill|bank|banking|credit card|debit card|card details|refund|transaction|transfer|wire transfer|outstanding balance|amount due|overdue|pay now|payment required|payment failed|financial|tax refund|tax payment|prize money|cash|debt|loan|mortgage|investment|crypto|cryptocurrency|bitcoin|wallet|funds|deposit|withdrawal)\b/i;


    if (
        financialPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Financial or payment-related language detected"
        );

    }


    // ========================================================
    // 5. CALL-TO-ACTION
    // ========================================================

    const ctaPattern =
        /\b(click here|click below|click the link|click on the link|follow the link|open the link|visit the link|verify now|confirm now|sign in now|login now|log in now|update now|download now|download the attachment|open attachment|open the attachment|view attachment|view the attachment|reply now|respond now|claim now|apply now|complete verification|continue here|access now|review now|check now|register now|submit now|activate now|renew now|unsubscribe here)\b/i;


    if (
        ctaPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Suspicious call-to-action language detected"
        );

    }


    // ========================================================
    // 6. URL + ACTION LANGUAGE
    //
    // URL alone remains a neutral characteristic.
    // ========================================================

    const containsUrl =
        /(?:https?:\/\/|www\.)[^\s<>"']+/i.test(
            text
        );


    const actionLanguagePattern =
        /\b(click|visit|open|follow|verify|confirm|login|log in|sign in|update|access|review|continue|download|activate|claim|register|submit|check)\b/i;


    if (
        containsUrl &&
        actionLanguagePattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Action-oriented message contains an external link"
        );

    }


    // ========================================================
    // 7. SENSITIVE INFORMATION REQUEST
    // ========================================================

    const sensitivePattern =
        /\b(card number|credit card number|debit card number|bank details|bank account|account number|routing number|security code|cvv|cvc|pin number|pin code|social security|social security number|personal information|personal details|date of birth|identity verification|verification details|security answer|security question)\b/i;


    if (
        sensitivePattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Request for sensitive or financial information detected"
        );

    }


    // ========================================================
    // 8. REWARD / PRIZE / ENTICEMENT
    // ========================================================

    const rewardPattern =
        /\b(congratulations|winner|you have won|you've won|selected to receive|selected to win|claim your|claim prize|free gift|free prize|exclusive reward|special offer|limited offer|guaranteed|bonus|free money|cash prize|reward waiting|gift card|lottery|jackpot|sweepstakes|lucky winner)\b/i;


    if (
        rewardPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Reward, prize, or strong enticement language detected"
        );

    }


    // ========================================================
    // 9. UNSOLICITED / SUSPICIOUS PROMOTION
    // ========================================================

    const promotionPattern =
        /\b(unsolicited|advertiser|advertisement|promotional email|promotional offer|remove from this advertiser|future mailings|mailing list|adult site|adult content|sex site|special promotion|exclusive deal|exclusive offer|marketing offer|free offer)\b/i;


    if (
        promotionPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Unsolicited or suspicious promotional language detected"
        );

    }


    // ========================================================
    // 10. ATTACHMENT-RELATED ACTION
    // ========================================================

    const attachmentPattern =
        /\b(attachment|attached file|attached document|attached invoice|attached receipt|attached form|attached statement)\b/i;


    const attachmentActionPattern =
        /\b(open|download|review|view|enable|complete|sign|fill|read|check)\b/i;


    if (
        attachmentPattern.test(
            lowerText
        ) &&
        attachmentActionPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Message encourages interaction with an attachment"
        );

    }


    // ========================================================
    // 11. SECURITY ALERT / ACCOUNT PROBLEM
    // ========================================================

    const securityAlertPattern =
        /\b(security alert|security warning|security notice|unusual activity|suspicious activity|unusual login|suspicious login|login attempt|sign-in attempt|account compromised|account security|security issue|security problem|security incident|account alert)\b/i;


    if (
        securityAlertPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Security-alert or account-problem language detected"
        );

    }


    // ========================================================
    // 12. DELIVERY / PARCEL / SHIPPING
    // ========================================================

    const deliveryPattern =
        /\b(parcel|package|delivery|shipment|shipping|courier|delivery attempt|missed delivery|tracking number|shipping fee|delivery fee|reschedule delivery|package waiting|parcel waiting)\b/i;


    if (
        deliveryPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Delivery or parcel-related lure detected"
        );

    }


    // ========================================================
    // 13. EMPLOYMENT / RECRUITMENT LURE
    // ========================================================

    const employmentPattern =
        /\b(job offer|employment offer|work from home|remote job|remote position|job opportunity|career opportunity|hiring|recruitment|recruiter|salary offer|weekly income|earn money|easy income|part.time job)\b/i;


    if (
        employmentPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Employment or income-related lure detected"
        );

    }


    // ========================================================
    // 14. CHARITY / DONATION
    // ========================================================

    const charityPattern =
        /\b(donation|donate now|charity|charitable|fundraising|fundraiser|humanitarian aid|financial assistance|support our cause)\b/i;


    if (
        charityPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Donation or charity-related solicitation detected"
        );

    }


    // ========================================================
    // 15. SUBSCRIPTION / ACCOUNT RENEWAL
    // ========================================================

    const renewalPattern =
        /\b(subscription|membership|renewal|renew your|subscription expired|membership expired|automatic renewal|renew now|service renewal)\b/i;


    if (
        renewalPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Subscription or account-renewal language detected"
        );

    }


    // ========================================================
    // 16. IMPERSONATION / AUTHORITY LANGUAGE
    // ========================================================

    const authorityPattern =
        /\b(IT department|IT support|administrator|system administrator|security team|support team|help desk|helpdesk|account team|compliance team|payroll department|human resources|HR department|bank security|customer support)\b/i;


    if (
        authorityPattern.test(
            text
        )
    ) {

        indicators.push(
            "Authority or trusted-service impersonation language detected"
        );

    }


    // ========================================================
    // 17. PAYMENT / INVOICE + ACTION
    // ========================================================

    const invoicePattern =
        /\b(invoice|payment|receipt|billing|statement|amount due|outstanding balance)\b/i;


    if (
        invoicePattern.test(
            lowerText
        ) &&
        (
            actionLanguagePattern.test(
                lowerText
            ) ||
            attachmentPattern.test(
                lowerText
            )
        )
    ) {

        indicators.push(
            "Financial lure combined with a requested action detected"
        );

    }


    // ========================================================
    // 18. PASSWORD / ACCOUNT + URGENCY
    // ========================================================

    if (
        credentialPattern.test(
            lowerText
        ) &&
        urgencyPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Urgent account or credential request detected"
        );

    }


    // ========================================================
    // 19. THREAT + ACTION REQUEST
    // ========================================================

    if (
        threatPattern.test(
            lowerText
        ) &&
        actionLanguagePattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Threatening language combined with a requested action detected"
        );

    }


    // ========================================================
    // 20. LINK + ACCOUNT LANGUAGE
    // ========================================================

    if (
        containsUrl &&
        credentialPattern.test(
            lowerText
        )
    ) {

        indicators.push(
            "Account-related request includes an external link"
        );

    }


    // ========================================================
    // REMOVE DUPLICATES
    // ========================================================

    return removeDuplicateIndicators(
        indicators
    );

}


// ============================================================
// DISPLAY PHISHING INDICATORS
// ============================================================

function displayIndicators(
    data,
    email
) {

    const heading =
        document.getElementById(
            "indicatorHeading"
        );


    const container =
        document.getElementById(
            "indicators"
        );


    if (!container) {

        return;

    }


    // ========================================================
    // ALWAYS KEEP THIS HEADING
    // ========================================================

    if (heading) {

        heading.textContent =
            "Phishing Indicators";

    }


    // ========================================================
    // CLEAR OLD CONTENT
    // ========================================================

    container.innerHTML =
        "";


    // ========================================================
    // PREDICTION
    // ========================================================

    const prediction =
        String(
            data?.prediction || ""
        ).trim();


    const isPhishing =
        prediction ===
        "Phishing Email";


    // ========================================================
    // API INDICATORS
    // ========================================================

    const apiIndicators =
        getApiIndicators(
            data
        );


    // ========================================================
    // SUPPLEMENTARY INDICATORS
    // ========================================================

    const detectedIndicators =
        detectPhishingIndicators(
            email
        );


    // ========================================================
    // PHISHING EMAIL
    // ========================================================

    if (isPhishing) {

        // ----------------------------------------------------
        // COMBINE API + LOCAL EXPLANATORY INDICATORS
        // ----------------------------------------------------

        let indicators =
            removeDuplicateIndicators([

                ...apiIndicators,

                ...detectedIndicators

            ]);


        // ----------------------------------------------------
        // DISPLAY SPECIFIC INDICATORS
        // ----------------------------------------------------

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


        // ----------------------------------------------------
        // ALWAYS DISPLAY MODEL-BASED INDICATOR
        //
        // This means EVERY phishing prediction has an
        // explanation even if no predefined pattern matches.
        //
        // This does not claim that a keyword caused the
        // prediction. It accurately identifies that the
        // classifier found contextual/semantic evidence.
        // ----------------------------------------------------

        addIndicator(

            container,

            "Contextual and semantic phishing patterns detected by the NLP model"

        );


        return;

    }


    // ========================================================
    // SAFE EMAIL
    //
    // Do not turn local keyword matches into phishing warnings
    // when the model predicted Safe.
    //
    // Only explicit API indicators are shown.
    // ========================================================

    if (
        apiIndicators.length > 0
    ) {

        apiIndicators.forEach(

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


    // ========================================================
    // SAFE + NO INDICATORS
    // ========================================================

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
// REMOVE DUPLICATE INDICATORS
// ============================================================

function removeDuplicateIndicators(
    indicators
) {

    const seen =
        new Set();


    const result =
        [];


    indicators.forEach(

        function (
            indicator
        ) {

            const cleaned =
                String(
                    indicator || ""
                )
                    .replace(
                        /^⚠️\s*/,
                        ""
                    )
                    .trim();


            if (!cleaned) {

                return;

            }


            const key =
                cleaned.toLowerCase();


            if (
                seen.has(
                    key
                )
            ) {

                return;

            }


            seen.add(
                key
            );


            result.push(
                cleaned
            );

        }

    );


    return result;

}


// ============================================================
// ADD PHISHING INDICATOR
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
        )
            .replace(
                /^⚠️\s*/,
                ""
            )
            .trim();


    div.textContent =
        `⚠️ ${text}`;


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
        value === null ||
        value === undefined ||
        value === ""
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

    if (!errorBox) {

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

    if (!errorBox) {

        return;

    }


    errorBox.classList.add(
        "hidden"
    );


    errorBox.textContent =
        "";

}