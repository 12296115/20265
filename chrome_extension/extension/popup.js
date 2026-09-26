"use strict";

const API_BASE = "http://127.0.0.1:5000/api";
const MAX_CHARS = 30000;
const TIMEOUT_MS = 10 * 60 * 1000; // Per request; LIME is slow on a laptop CPU.
const MIN_RELATIVE_WEIGHT = 0.05;

const ui = {
  read: document.getElementById("read-email"),
  clear: document.getElementById("clear-email"),
  analyse: document.getElementById("analyse-email"),
  text: document.getElementById("email-text"),
  status: document.getElementById("status"),
  result: document.getElementById("result"),
  verdict: document.getElementById("verdict"),
  score: document.getElementById("score"),
  gate: document.getElementById("gate"),
  gateWeight: document.getElementById("gate-weight"),
  gateRows: document.getElementById("gate-rows"),
  gateSummary: document.getElementById("gate-summary"),
  featureRows: document.getElementById("feature-rows"),
  explanation: document.getElementById("explanation"),
  note: document.getElementById("explanation-note"),
  terms: document.getElementById("terms"),
};

function setStatus(message, kind = "") {
  ui.status.textContent = message;
  ui.status.className = `status ${kind}`.trim();
}

function setBusy(busy, label = "Analysing…") {
  ui.read.disabled = busy;
  ui.clear.disabled = busy;
  ui.analyse.disabled = busy;
  ui.analyse.textContent = busy ? label : "Analyse email";
}

function clearResult() {
  ui.result.hidden = true;
  ui.gate.hidden = true;
  ui.explanation.hidden = true;
  ui.gateRows.replaceChildren();
  ui.featureRows.replaceChildren();
  ui.terms.replaceChildren();
}

// Keeps saturated probabilities (e.g. 99.9996%) distinguishable from 100%.
function formatPercent(p) {
  const value = Number(p) * 100;
  const decimals = value >= 99.9 || value <= 0.1 ? 4 : 1;
  return `${value.toFixed(decimals)}%`;
}

function formatSigned(value, decimals = 2) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}`;
}

function tableRow(cells) {
  const row = document.createElement("tr");
  cells.forEach(([text, numeric]) => {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (numeric) cell.className = "num";
    row.append(cell);
  });
  return row;
}

// This function runs inside the Gmail tab when the user clicks Read.
// Gmail may change its page structure, so manual paste is also supported.
function readVisibleGmailMessage() {
  const main = document.querySelector('[role="main"]') || document;
  const subject = main.querySelector("h2.hP")?.innerText?.trim() || "";

  const bodies = [...main.querySelectorAll(".a3s")].filter(
    element => element.getClientRects().length > 0 && element.innerText.trim()
  );

  const body = bodies.at(-1)?.innerText?.trim() || "";

  if (!body) {
    return {
      error: "No open Gmail message body was found. Open a message, or paste text manually.",
    };
  }

  return {
    text: [subject, body].filter(Boolean).join("\n\n"),
    messageCount: bodies.length,
  };
}

ui.read.addEventListener("click", async () => {
  clearResult();
  setStatus("Reading the open email…");

  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });

    if (!tab?.id || !/^https:\/\/mail\.google\.com\/mail\//.test(tab.url || "")) {
      throw new Error(
        "Open a Gmail message in the active tab, or paste email text manually."
      );
    }

    const injected = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: readVisibleGmailMessage,
    });

    const message = injected?.[0]?.result;

    if (!message || message.error) {
      throw new Error(message?.error || "Unable to read the open message.");
    }

    if (message.text.length > MAX_CHARS) {
      throw new Error(
        "Email is too long; paste a shorter de-identified sample."
      );
    }

    ui.text.value = message.text;

    setStatus(
      message.messageCount > 1
        ? "Loaded the last visible message in this conversation. Check the text before analysing."
        : "Open message loaded. Check the text before analysing.",
      "success"
    );
  } catch (error) {
    setStatus(error.message || "Could not read Gmail.", "error");
  }
});

ui.clear.addEventListener("click", () => {
  ui.text.value = "";
  clearResult();
  forgetResult();
  setStatus("Text cleared.");
  ui.text.focus();
});

function showVerdict(data) {
  if (
    !["phishing", "legitimate"].includes(data?.verdict) ||
    !Number.isFinite(Number(data.score))
  ) {
    throw new Error("The API returned an incomplete result.");
  }

  ui.result.hidden = false;

  const phishing = data.verdict === "phishing";
  ui.verdict.textContent = phishing
    ? "Potential phishing"
    : "Predicted legitimate";
  ui.verdict.className = `verdict ${data.verdict}`;
  ui.score.textContent = formatPercent(data.score);
}

function describeChange(label, fromProb, toProb) {
  const deltaPoints = (toProb - fromProb) * 100;
  const range = `(${formatPercent(fromProb)} → ${formatPercent(toProb)})`;

  if (Math.abs(deltaPoints) < 0.01) {
    return `${label} changed the phishing probability by less than 0.01 percentage points ${range}.`;
  }
  const direction = deltaPoints > 0 ? "raised" : "lowered";
  return (
    `${label} ${direction} the phishing probability by ` +
    `${Math.abs(deltaPoints).toFixed(2)} percentage points ${range}.`
  );
}

function showGate(gate) {
  const probs = gate?.phishing_probability;
  const logOdds = gate?.phishing_log_odds;

  if (!probs || !logOdds || !Number.isFinite(Number(gate.gate_mean))) {
    return;
  }

  ui.gate.hidden = false;

  const textShare = Number(gate.gate_mean) * 100;
  ui.gateWeight.textContent =
    `Mean gate: ${textShare.toFixed(1)}% of the fused representation comes from the ` +
    `text (DeBERTa) branch and ${(100 - textShare).toFixed(1)}% from the 12 hand-crafted features.`;

  const rows = [
    ["Combined (model)", "combined"],
    ["Text only (gate = 1)", "text_only"],
    ["Features only (gate = 0)", "features_only"],
    ["Features at training mean", "features_at_training_mean"],
  ];
  ui.gateRows.replaceChildren(
    ...rows
      .filter(([, key]) => Number.isFinite(Number(probs[key])))
      .map(([label, key]) => tableRow([
        [label],
        [formatPercent(probs[key]), true],
        [Number(logOdds[key]).toFixed(2), true],
      ]))
  );

  const lines = [
    describeChange(
      "Adding the feature branch to the text branch",
      Number(probs.text_only),
      Number(probs.combined),
    ),
  ];
  if (Number.isFinite(Number(probs.features_at_training_mean))) {
    lines.push(describeChange(
      "Using this email's 12 feature values instead of the training averages",
      Number(probs.features_at_training_mean),
      Number(probs.combined),
    ));
  }
  ui.gateSummary.textContent = lines.join(" ");

  ui.featureRows.replaceChildren(
    ...(gate.features || []).map(feature =>
      tableRow([[feature.name], [String(feature.value), true]])
    )
  );
}

function showExplanation(explanation) {
  ui.explanation.hidden = false;
  ui.terms.replaceChildren();

  if (explanation?.method !== "LIME" || !Array.isArray(explanation.terms)) {
    ui.note.textContent = "A LIME explanation was not returned.";
    return;
  }

  const terms = explanation.terms
    .map(term => ({ word: term.word, weight: Number(term.weight) }))
    .filter(term => typeof term.word === "string" && Number.isFinite(term.weight));

  const strongest = Math.max(0, ...terms.map(term => Math.abs(term.weight)));
  const shown = strongest > 0
    ? terms.filter(term => Math.abs(term.weight) >= MIN_RELATIVE_WEIGHT * strongest)
    : [];
  const hidden = terms.length - shown.length;

  ui.note.textContent =
    `LIME (${explanation.num_samples} samples). Values are each word's ` +
    `${explanation.weight_label || "log-odds contribution"} to the phishing score, not percentages. ` +
    "Red bars push towards phishing, green towards legitimate. " +
    "Bars are scaled to the strongest term" +
    (hidden > 0 ? `; ${hidden} term(s) under 5% of it are hidden.` : ".");

  if (!shown.length) {
    const item = document.createElement("li");
    item.textContent = "No influential terms were returned for this message.";
    ui.terms.append(item);
    return;
  }

  for (const term of shown) {
    const direction = term.weight >= 0 ? "toward-phishing" : "toward-legitimate";

    const row = document.createElement("li");
    row.className = "bar-row";

    const word = document.createElement("span");
    word.className = "term";
    word.textContent = term.word;
    word.title = term.word;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill ${direction}`;
    fill.style.width = `${(Math.abs(term.weight) / strongest) * 100}%`;
    track.append(fill);

    const value = document.createElement("span");
    value.className = `bar-value ${direction}`;
    value.textContent = formatSigned(term.weight, 3);

    row.append(word, track, value);
    ui.terms.append(row);
  }
}

async function postJson(path, text) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `API error ${response.status}`);
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

function describeError(error, fallback) {
  if (error.name === "AbortError") {
    return "The request took longer than 10 minutes. Check whether Flask is still running.";
  }
  if (error.message === "Failed to fetch") {
    return "Cannot reach the local API. Start app.py and check the extension's localhost permission.";
  }
  return error.message || fallback;
}

// Only model outputs are kept (not the email text); session storage is
// in-memory and cleared when the browser closes.
const STORAGE_KEY = "lastResult";

function saveResult(result) {
  return chrome.storage.session.set({ [STORAGE_KEY]: result });
}

function forgetResult() {
  return chrome.storage.session.remove(STORAGE_KEY);
}

async function restoreLastResult() {
  const { [STORAGE_KEY]: last } = await chrome.storage.session.get(STORAGE_KEY);
  if (!last?.prediction) return;

  try {
    showVerdict(last.prediction);
    showGate(last.prediction.gate_analysis);
  } catch {
    return;
  }

  const time = new Date(last.savedAt).toLocaleTimeString();
  if (last.explanation) {
    showExplanation(last.explanation);
    setStatus(`Showing the last result from ${time}.`, "success");
  } else {
    ui.explanation.hidden = false;
    ui.note.textContent =
      "The LIME explanation did not finish (the popup was closed). Analyse again to compute it.";
    setStatus(`Showing the last verdict from ${time}.`);
  }
}

restoreLastResult();

ui.analyse.addEventListener("click", async () => {
  const text = ui.text.value.trim();
  clearResult();

  if (!text) {
    setStatus("Open a Gmail message or paste email text first.", "error");
    return;
  }

  if (text.length > MAX_CHARS) {
    setStatus("Text is too long for this demo.", "error");
    return;
  }

  setBusy(true, "Analysing…");
  setStatus("Getting the verdict…");
  await forgetResult();

  try {
    let prediction;
    try {
      prediction = await postJson("predict", text);
      showVerdict(prediction);
      showGate(prediction.gate_analysis);
      await saveResult({ prediction, savedAt: Date.now() });
    } catch (error) {
      setStatus(describeError(error, "Analysis failed."), "error");
      return;
    }

    setBusy(true, "Explaining…");
    setStatus("Verdict ready. Computing the LIME explanation (this can take a few minutes on CPU)…");
    ui.explanation.hidden = false;
    ui.note.textContent = "Computing LIME explanation…";

    try {
      const data = await postJson("explain", text);
      showExplanation(data.explanation);
      await saveResult({ prediction, explanation: data.explanation, savedAt: Date.now() });
      setStatus("Analysis complete.", "success");
    } catch (error) {
      ui.note.textContent = "The LIME explanation could not be computed.";
      setStatus(describeError(error, "Explanation failed."), "error");
    }
  } finally {
    setBusy(false);
  }
});
