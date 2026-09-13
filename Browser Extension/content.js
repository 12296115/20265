// University Phishing Email Detector
// Revised content.js

const API_BASE_URL = "http://127.0.0.1:8000";
const PREDICT_URL = `${API_BASE_URL}/predict`;
const EXPLAIN_URL = `${API_BASE_URL}/explain`;

const BUTTON_ID = "university-phishing-scan-button";
const RESULT_ID = "university-phishing-result";
const ERROR_ID = "university-phishing-error";

if (!window.__universityPhishingDetectorLoaded) {
    window.__universityPhishingDetectorLoaded = true;
    initialiseDetector();
}

function initialiseDetector() {
    if (!isDatasetTestEnvironment()) return;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", createScanButton);
    } else {
        createScanButton();
    }
}

function isDatasetTestEnvironment() {
    const local =
        location.hostname === "localhost" ||
        location.hostname === "127.0.0.1";

    if (!local) return false;

    return [...document.querySelectorAll("h1,h2,h3")]
        .some(el =>
            normaliseText(el.textContent).toLowerCase() === "university email"
        );
}

function createScanButton() {
    if (document.getElementById(BUTTON_ID)) return;

    const button = document.createElement("button");
    button.id = BUTTON_ID;
    button.textContent = "🛡️ Scan Email for Phishing";

    Object.assign(button.style, {
        position: "fixed",
        top: "20px",
        right: "20px",
        zIndex: "2147483647",
        padding: "10px 16px",
        border: "none",
        borderRadius: "8px",
        background: "#2563eb",
        color: "#fff",
        fontFamily: "Arial, sans-serif",
        fontSize: "14px",
        fontWeight: "bold",
        cursor: "pointer",
        boxShadow: "0 3px 10px rgba(0,0,0,.2)"
    });

    button.addEventListener("click", analyseCurrentEmail);
    document.body.appendChild(button);
}

async function analyseCurrentEmail() {
    const button = document.getElementById(BUTTON_ID);

    try {
        setButtonLoading(button, true, "⏳ Analysing Email...");
        removeElement(ERROR_ID);

        const email = extractEmail();

        if (!email.body || !email.body.trim()) {
            throw new Error("Could not detect the selected email body.");
        }

        const response = await fetch(PREDICT_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                sender: email.sender || "",
                subject: email.subject || "",
                body: email.body
            })
        });

        if (!response.ok) {
            throw new Error(
                await readApiError(response) ||
                "Prediction API request failed. Make sure api.py is running."
            );
        }

        displayResult(await response.json(), email);

    } catch (error) {
        console.error(error);
        showError(error.message || "Unable to analyse this email.");
    } finally {
        setButtonLoading(button, false);
    }
}

function setButtonLoading(button, loading, text = "") {
    if (!button) return;

    button.disabled = loading;
    button.textContent = loading
        ? text
        : "🛡️ Scan Email for Phishing";

    button.style.background = loading
        ? "#6b7280"
        : "#2563eb";

    button.style.cursor = loading
        ? "wait"
        : "pointer";
}

function extractEmail() {
    const card = getVisibleEmailCard();

    if (!card) {
        return {
            sender: "",
            subject: "",
            body: ""
        };
    }

    return {
        sender: extractLabelValue(card, "From:"),
        subject: extractLabelValue(card, "Subject:"),
        body: cleanEmailBody(
            extractBodyFromCard(card)
        )
    };
}

function getVisibleEmailCard() {
    const heading = [...document.querySelectorAll("h1,h2,h3")]
        .find(el =>
            isVisible(el) &&
            normaliseText(el.textContent).toLowerCase() ===
            "university email"
        );

    if (!heading) return null;

    let element = heading.parentElement;

    for (let i = 0; i < 8 && element; i++) {
        const text = normaliseText(element.innerText);

        if (
            text.length > 100 &&
            /from:/i.test(text) &&
            /subject:/i.test(text)
        ) {
            return element;
        }

        element = element.parentElement;
    }

    return heading.parentElement;
}

function extractLabelValue(container, label) {
    for (const element of container.querySelectorAll(
        "p,div,span,strong,b"
    )) {
        if (!isVisible(element)) continue;

        const text = normaliseText(
            element.innerText ||
            element.textContent
        );

        if (
            text
                .toLowerCase()
                .startsWith(label.toLowerCase()) &&
            text.length > label.length
        ) {
            return text
                .slice(label.length)
                .trim();
        }
    }

    return "";
}

function extractBodyFromCard(card) {
    const clone = card.cloneNode(true);

    clone.querySelectorAll(
        "h1,h2,h3,.email-header,.header,[class*='ground'],[id*='ground']"
    ).forEach(el => el.remove());

    return removeMetadata(
        clone.innerText ||
        clone.textContent ||
        ""
    );
}

function removeMetadata(text) {
    return String(text || "")
        .replace(/\r/g, "")
        .split("\n")
        .map(line => line.trim())
        .filter(line =>
            line &&
            !/^University Email$/i.test(line) &&
            !/^From:/i.test(line) &&
            !/^Subject:/i.test(line) &&
            !/^Dataset ground truth:/i.test(line)
        )
        .join("\n");
}

function cleanEmailBody(text) {
    return removeMetadata(text).trim();
}

function normaliseText(text) {
    return String(text || "")
        .replace(/\s+/g, " ")
        .trim();
}

function isVisible(element) {
    if (!element) return false;

    const style = getComputedStyle(element);

    return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        element.getClientRects().length > 0
    );
}

function displayResult(data, email) {
    removeElement(RESULT_ID);

    const isPhishing = String(
        data.prediction || ""
    )
        .toLowerCase()
        .includes("phishing");

    const container = document.createElement("div");
    container.id = RESULT_ID;

    Object.assign(container.style, {
        position: "fixed",
        top: "78px",
        right: "20px",
        width: "360px",
        maxHeight: "calc(100vh - 100px)",
        overflowY: "auto",
        zIndex: "2147483647",
        background: "#fff",
        border: isPhishing
            ? "1px solid #dc2626"
            : "1px solid #16a34a",
        borderRadius: "10px",
        boxShadow: "0 8px 25px rgba(0,0,0,.2)",
        padding: "16px",
        fontFamily: "Arial, sans-serif",
        color: "#172b4d"
    });

    const header = document.createElement("div");

    Object.assign(header.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center"
    });

    const title = document.createElement("strong");

    title.textContent = isPhishing
        ? "⚠️ Potential Phishing Email"
        : "✅ Safe Email";

    title.style.color = isPhishing
        ? "#b91c1c"
        : "#15803d";

    title.style.fontSize = "16px";

    const close = document.createElement("button");

    close.textContent = "×";

    Object.assign(close.style, {
        border: "none",
        background: "transparent",
        fontSize: "20px",
        cursor: "pointer",
        color: "#6b7280"
    });

    close.onclick = () => container.remove();

    header.append(title, close);
    container.appendChild(header);

    appendText(
        container,
        `Confidence: ${formatPercentage(
            data.confidence_percentage ??
            data.confidence
        )}`,
        {
            margin: "10px 0 5px",
            fontSize: "13px"
        }
    );

    appendText(
        container,
        `Safe: ${formatPercentage(
            data.safe_probability_percentage ??
            data.safe_probability
        )} | ` +
        `Phishing: ${formatPercentage(
            data.phishing_probability_percentage ??
            data.phishing_probability
        )}`,
        {
            margin: "4px 0",
            fontSize: "12px"
        }
    );

    const indicators = Array.isArray(data.indicators)
        ? data.indicators
        : [];

    if (indicators.length) {
        appendText(
            container,
            "Detected Indicators:",
            {
                marginTop: "10px",
                fontSize: "13px",
                fontWeight: "bold"
            }
        );

        indicators.forEach(indicator => {
            appendText(
                container,
                `⚠️ ${indicator}`,
                {
                    marginTop: "6px",
                    padding: "7px",
                    background: "#fef3c7",
                    color: "#92400e",
                    borderRadius: "5px",
                    fontSize: "11px"
                }
            );
        });
    }

    const details = document.createElement("div");

    Object.assign(details.style, {
        marginTop: "12px",
        paddingTop: "10px",
        borderTop: "1px solid #e5e7eb",
        fontSize: "11px",
        lineHeight: "1.6"
    });

    appendDetail(
        details,
        "Sender",
        email.sender || "Not detected"
    );

    appendDetail(
        details,
        "Subject",
        email.subject || "Not detected"
    );

    container.appendChild(details);

    // Explainable AI
    const explainSection = document.createElement("div");

    Object.assign(explainSection.style, {
        marginTop: "14px",
        paddingTop: "12px",
        borderTop: "1px solid #e5e7eb"
    });

    appendText(
        explainSection,
        "🧠 Explainable AI Analysis",
        {
            fontSize: "14px",
            fontWeight: "bold",
            marginBottom: "5px",
            color: "#374151"
        }
    );

    appendText(
        explainSection,
        "Identify the words and tokens that influenced the model's prediction.",
        {
            fontSize: "11px",
            color: "#6b7280",
            marginBottom: "10px"
        }
    );

    const output = document.createElement("div");
    output.style.marginTop = "10px";

    const explainButton = document.createElement("button");

    explainButton.textContent = "🧠 Explain Decision";

    Object.assign(explainButton.style, {
        background: "#fff",
        color: "#374151",
        border: "1px solid #cbd5e1",
        borderRadius: "6px",
        padding: "8px 12px",
        fontSize: "12px",
        fontWeight: "bold",
        cursor: "pointer"
    });

    explainButton.onclick = () =>
        generateExplanation(
            explainButton,
            output,
            email
        );

    explainSection.append(
        explainButton,
        output
    );

    container.appendChild(explainSection);

    appendText(
        container,
        "Powered by DeBERTa-v3-large + SHAP Explainable AI",
        {
            marginTop: "14px",
            paddingTop: "10px",
            borderTop: "1px solid #e5e7eb",
            fontSize: "10px",
            color: "#9ca3af"
        }
    );

    document.body.appendChild(container);
}

async function generateExplanation(
    button,
    output,
    email
) {
    output.innerHTML = "";

    try {
        button.disabled = true;

        button.textContent =
            "⏳ Generating Explanation...";

        button.style.background = "#e5e7eb";
        button.style.cursor = "wait";

        const response = await fetch(
            EXPLAIN_URL,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    sender: email.sender || "",
                    subject: email.subject || "",
                    body: email.body || ""
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                await readApiError(response) ||
                "Unable to generate the SHAP explanation."
            );
        }

        displayExplanation(
            output,
            await response.json()
        );

    } catch (error) {
        console.error(
            "SHAP explanation error:",
            error
        );

        appendText(
            output,
            `Unable to generate SHAP explanation: ${
                error.message ||
                "Unknown error"
            }`,
            {
                padding: "9px",
                background: "#fee2e2",
                color: "#991b1b",
                borderRadius: "6px",
                fontSize: "11px",
                lineHeight: "1.4"
            }
        );

    } finally {
        button.disabled = false;

        button.textContent =
            "🧠 Explain Decision";

        button.style.background = "#fff";
        button.style.cursor = "pointer";
    }
}

function displayExplanation(
    output,
    data
) {
    output.innerHTML = "";

    appendText(
        output,
        "✅ SHAP explanation generated successfully.",
        {
            padding: "8px",
            background: "#dcfce7",
            color: "#166534",
            borderRadius: "6px",
            fontSize: "11px",
            marginBottom: "10px"
        }
    );

    appendText(
        output,
        `Explanation for: ${
            data.prediction ||
            data.label ||
            "Model Prediction"
        }`,
        {
            fontSize: "13px",
            fontWeight: "bold",
            marginBottom: "10px"
        }
    );

    const supporting = getFactorArray(
        data,
        [
            "supporting_factors",
            "top_supporting_factors",
            "factors_supporting",
            "positive_factors"
        ]
    );

    const opposing = getFactorArray(
        data,
        [
            "opposing_factors",
            "top_opposing_factors",
            "factors_against",
            "negative_factors"
        ]
    );

    appendFactorSection(
        output,
        "🔴 Factors Supporting the Prediction",
        supporting,
        "#fee2e2",
        "#991b1b",
        "No strong supporting factors were identified."
    );

    appendFactorSection(
        output,
        "🟢 Factors Working Against the Prediction",
        opposing,
        "#dcfce7",
        "#166534",
        "No strong opposing factors were identified."
    );

    appendText(
        output,
        "SHAP contributions show how individual words and tokens influenced the model's prediction.",
        {
            marginTop: "10px",
            fontSize: "10px",
            color: "#6b7280",
            lineHeight: "1.4"
        }
    );
}

function getFactorArray(
    data,
    keys
) {
    for (const key of keys) {
        if (Array.isArray(data[key])) {
            return data[key];
        }
    }

    return [];
}

function appendFactorSection(
    output,
    title,
    factors,
    background,
    colour,
    emptyMessage
) {
    const section = document.createElement("div");

    section.style.marginTop = "12px";

    appendText(
        section,
        title,
        {
            fontSize: "12px",
            fontWeight: "bold",
            color: colour,
            marginBottom: "7px"
        }
    );

    if (!factors.length) {
        appendText(
            section,
            emptyMessage,
            {
                padding: "8px",
                background,
                color: colour,
                borderRadius: "6px",
                fontSize: "10px"
            }
        );

    } else {
        const list = document.createElement("ol");

        Object.assign(list.style, {
            margin: "0",
            paddingLeft: "20px",
            fontSize: "11px",
            lineHeight: "1.8"
        });

        factors
            .slice(0, 10)
            .forEach(factor => {
                const token = getFactorToken(factor);
                const contribution =
                    getFactorContribution(factor);

                const item =
                    document.createElement("li");

                item.textContent =
                    contribution === null
                        ? token
                        : `${token} → Contribution: ${
                            contribution.toFixed(6)
                        }`;

                list.appendChild(item);
            });

        section.appendChild(list);
    }

    output.appendChild(section);
}

function getFactorToken(factor) {
    if (typeof factor === "string") {
        return factor;
    }

    if (!factor || typeof factor !== "object") {
        return "Unknown token";
    }

    return String(
        factor.token ??
        factor.word ??
        factor.feature ??
        factor.text ??
        "Unknown token"
    );
}

function getFactorContribution(factor) {
    if (!factor || typeof factor !== "object") {
        return null;
    }

    const number = Number(
        factor.contribution ??
        factor.shap_value ??
        factor.value
    );

    return Number.isFinite(number)
        ? number
        : null;
}

function appendText(
    parent,
    text,
    styles = {}
) {
    const element =
        document.createElement("div");

    element.textContent = text;

    Object.assign(
        element.style,
        styles
    );

    parent.appendChild(element);

    return element;
}

function appendDetail(
    parent,
    label,
    value
) {
    const row =
        document.createElement("div");

    const strong =
        document.createElement("strong");

    strong.textContent =
        `${label}: `;

    const text =
        document.createElement("span");

    text.textContent = value;

    row.append(
        strong,
        text
    );

    parent.appendChild(row);
}

function formatPercentage(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "N/A";
    }

    return `${
        (number <= 1
            ? number * 100
            : number
        ).toFixed(2)
    }%`;
}

function removeElement(id) {
    const element =
        document.getElementById(id);

    if (element) {
        element.remove();
    }
}

async function readApiError(response) {
    try {
        const data =
            await response.json();

        return (
            data.detail ||
            data.message ||
            data.error ||
            ""
        );

    } catch {
        return "";
    }
}

function showError(message) {
    removeElement(ERROR_ID);

    const error =
        document.createElement("div");

    error.id = ERROR_ID;

    error.textContent =
        `⚠️ ${message}`;

    Object.assign(error.style, {
        position: "fixed",
        top: "78px",
        right: "20px",
        width: "340px",
        zIndex: "2147483647",
        background: "#fee2e2",
        color: "#991b1b",
        border: "1px solid #dc2626",
        borderRadius: "8px",
        padding: "14px",
        fontFamily: "Arial, sans-serif",
        fontSize: "13px",
        boxShadow: "0 5px 15px rgba(0,0,0,.15)"
    });

    document.body.appendChild(error);

    setTimeout(() => {
        if (document.body.contains(error)) {
            error.remove();
        }
    }, 6000);
}

// Popup communication remains compatible with popup.js.
chrome.runtime.onMessage.addListener(
    (
        request,
        sender,
        sendResponse
    ) => {
        if (request.action !== "getEmail") {
            return;
        }

        try {
            const email =
                extractEmail();

            if (
                !email.body ||
                !email.body.trim()
            ) {
                sendResponse({
                    success: false,
                    error:
                        "Could not detect an email on this page."
                });

                return true;
            }

            sendResponse({
                success: true,
                sender: email.sender,
                subject: email.subject,
                body: email.body
            });

        } catch (error) {
            sendResponse({
                success: false,
                error:
                    error.message ||
                    "Email extraction failed."
            });
        }

        return true;
    }
);