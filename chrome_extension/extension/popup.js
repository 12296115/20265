"use strict";

const API_BASE = "http://127.0.0.1:5000/api";
const MAX_CHARS = 30000;
const TIMEOUT_MS = 10 * 60 * 1000; // Per request; LIME is slow on a laptop CPU.
const MIN_RELATIVE_WEIGHT = 0.05;
const TOP_TERMS = 5;
const NO_DRIVER_THRESHOLD = 0.1; // log-odds

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
  fitBadge: document.getElementById("fit-badge"),
  noDriver: document.getElementById("no-driver"),
  termLists: document.getElementById("term-lists"),
  termsPhishing: document.getElementById("terms-phishing"),
  termsLegitimate: document.getElementById("terms-legitimate"),
  deep: document.getElementById("deep"),
  deepButton: document.getElementById("deep-explain"),
  igBadge: document.getElementById("ig-badge"),
  igNote: document.getElementById("ig-note"),
  igDetails: document.getElementById("ig-details"),
  igTruncated: document.getElementById("ig-truncated"),
  igLists: document.getElementById("ig-lists"),
  igPhishing: document.getElementById("ig-phishing"),
  igLegitimate: document.getElementById("ig-legitimate"),
};

// Text that produced the current verdict; never persisted.
let analysedText = null;
let busy = false;
let lastResult = null;

function setStatus(message, kind = "") {
  ui.status.textContent = message;
  ui.status.className = `status ${kind}`.trim();
}

function setBusy(isBusy, label = "Analysing…") {
  busy = isBusy;
  ui.read.disabled = isBusy;
  ui.clear.disabled = isBusy;
  ui.analyse.disabled = isBusy;
  ui.analyse.textContent = isBusy ? label : "Analyse email";
  updateDeepButton();
}

function updateDeepButton() {
  ui.deepButton.disabled = busy || analysedText === null;
}

function clearResult() {
  ui.result.hidden = true;
  ui.gate.hidden = true;
  ui.explanation.hidden = true;
  ui.gateRows.replaceChildren();
  ui.featureRows.replaceChildren();
  resetExplanation();
  ui.deep.hidden = true;
  resetDeepExplanation();
  analysedText = null;
  updateDeepButton();
}

function resetDeepExplanation() {
  ui.igBadge.hidden = true;
  ui.igDetails.hidden = true;
  ui.igTruncated.hidden = true;
  ui.igNote.textContent = "";
  ui.igLists.hidden = true;
  ui.igPhishing.replaceChildren();
  ui.igLegitimate.replaceChildren();
}

function resetExplanation() {
  ui.fitBadge.hidden = true;
  ui.noDriver.hidden = true;
  ui.termLists.hidden = true;
  ui.termsPhishing.replaceChildren();
  ui.termsLegitimate.replaceChildren();
}

function formatPercent(p) {
  return `${(Number(p) * 100).toFixed(2)}%`;
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

  const phishingProb = Number(data.class_scores?.[1]);
  ui.score.textContent = Number.isFinite(phishingProb)
    ? formatPercent(phishingProb)
    : "–";
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

  const combined = Number(logOdds.combined);
  const textOnly = Number(logOdds.text_only);
  const atMean = Number(logOdds.features_at_training_mean);

  if ([combined, textOnly, atMean].every(Number.isFinite)) {
    const offset = textOnly - combined;
    ui.gateSummary.textContent =
      "The feature branch acts as a fixed offset: it " +
      `${offset >= 0 ? "lowers" : "raises"} the score by ${Math.abs(offset).toFixed(2)} log-odds ` +
      `(text only minus combined = ${formatSigned(offset)}). ` +
      "This email's own feature values changed the score by " +
      `only ${formatSigned(combined - atMean, 3)} log-odds ` +
      `(combined ${combined.toFixed(2)} vs. features at training mean ${atMean.toFixed(2)}).`;
  } else {
    ui.gateSummary.textContent = "";
  }

  ui.featureRows.replaceChildren(
    ...(gate.features || []).map(feature =>
      tableRow([[feature.name], [String(feature.value), true]])
    )
  );
}

function showFitBadge(r2) {
  if (!Number.isFinite(r2)) return;
  const [label, kind] =
    r2 >= 0.5 ? ["Good fit", "good"]
    : r2 >= 0.2 ? ["Weak fit", "weak"]
    : ["Unreliable", "unreliable"];
  ui.fitBadge.textContent = `${label} (R² ${r2.toFixed(2)})`;
  ui.fitBadge.className = `badge ${kind}`;
  ui.fitBadge.title = "R² of LIME's local linear fit to the model's log-odds.";
  ui.fitBadge.hidden = false;
}

function fillTermList(list, terms, strongest) {
  if (!terms.length) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = "No notable terms.";
    list.append(item);
    return;
  }
  for (const term of terms) {
    list.append(termRow(term, strongest));
  }
}

function showExplanation(explanation) {
  ui.explanation.hidden = false;
  resetExplanation();

  if (explanation?.method !== "LIME" || !Array.isArray(explanation.terms)) {
    ui.note.textContent = "A LIME explanation was not returned.";
    return;
  }

  showFitBadge(Number(explanation.local_fit_r2));

  const terms = explanation.terms
    .map(term => ({ word: term.word, weight: Number(term.weight) }))
    .filter(term => typeof term.word === "string" && Number.isFinite(term.weight));

  const strongest = Math.max(0, ...terms.map(term => Math.abs(term.weight)));
  const shown = strongest > 0
    ? terms.filter(term => Math.abs(term.weight) >= MIN_RELATIVE_WEIGHT * strongest)
    : [];
  const byStrength = (a, b) => Math.abs(b.weight) - Math.abs(a.weight);
  const towardsPhishing = shown.filter(t => t.weight > 0).sort(byStrength).slice(0, TOP_TERMS);
  const towardsLegitimate = shown.filter(t => t.weight < 0).sort(byStrength).slice(0, TOP_TERMS);

  ui.note.textContent =
    `LIME (${explanation.num_samples} samples). Values are each word's ` +
    `${explanation.weight_label || "log-odds contribution"} to the phishing score, not percentages. ` +
    `Top ${TOP_TERMS} per direction; bars are scaled to the strongest term and terms under 5% of it are hidden.`;

  ui.noDriver.hidden = strongest > NO_DRIVER_THRESHOLD;
  ui.termLists.hidden = false;
  fillTermList(ui.termsPhishing, towardsPhishing, strongest);
  fillTermList(ui.termsLegitimate, towardsLegitimate, strongest);
}

function showIgBadge(relativeDelta) {
  const rel = relativeDelta === null ? NaN : Number(relativeDelta);
  const [label, kind] =
    !Number.isFinite(rel) || rel > 0.6 ? ["Unreliable", "unreliable"]
    : rel >= 0.1 ? ["Use with caution", "weak"]
    : ["Reliable", "good"];
  ui.igBadge.textContent = Number.isFinite(rel)
    ? `${label} (Δ ${(rel * 100).toFixed(0)}%)`
    : label;
  ui.igBadge.className = `badge ${kind}`;
  ui.igBadge.title =
    "Relative convergence delta: |delta| / |F(input) − F(baseline)|. Lower is better.";
  ui.igBadge.hidden = false;
}

function showDeepExplanation(deep) {
  ui.deep.hidden = false;
  resetDeepExplanation();

  if (deep?.method !== "Integrated Gradients" || !Array.isArray(deep.words)) {
    ui.igNote.textContent = "An Integrated Gradients result was not returned.";
    return;
  }

  showIgBadge(deep.relative_delta);

  const words = deep.words
    .map(item => ({ word: item.word, weight: Number(item.score) }))
    .filter(item => typeof item.word === "string" && Number.isFinite(item.weight));
  const strongest = Math.max(0, ...words.map(item => Math.abs(item.weight)));
  const byStrength = (a, b) => Math.abs(b.weight) - Math.abs(a.weight);

  const fInput = Number(deep.f_input);
  const fBaseline = Number(deep.f_baseline);
  if (Number.isFinite(fInput) && Number.isFinite(fBaseline)) {
    ui.igDetails.textContent =
      `F(input) ${fInput.toFixed(2)} · F(baseline) ${fBaseline.toFixed(2)} log-odds · ` +
      `baseline: ${deep.baseline || "pad"} tokens · ${deep.n_steps} steps`;
    ui.igDetails.hidden = false;
  }

  ui.igNote.textContent =
    "The badge checks whether the word scores add up to the model's output. " +
    "If it says Unreliable, treat the words as rough hints only.";

  if (deep.truncated) {
    ui.igTruncated.textContent = `Explained the first ${deep.max_tokens || 256} tokens only`;
    ui.igTruncated.hidden = false;
  }

  ui.igLists.hidden = false;
  if (strongest > 0) {
    fillTermList(ui.igPhishing, words.filter(w => w.weight > 0).sort(byStrength), strongest);
    fillTermList(ui.igLegitimate, words.filter(w => w.weight < 0).sort(byStrength), strongest);
  } else {
    fillTermList(ui.igPhishing, [], 1);
    fillTermList(ui.igLegitimate, [], 1);
  }
}

function termRow(term, strongest) {
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
  return row;
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

function saveResult(patch) {
  lastResult = { ...(lastResult || {}), ...patch, savedAt: Date.now() };
  return chrome.storage.session.set({ [STORAGE_KEY]: lastResult });
}

function forgetResult() {
  lastResult = null;
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
  lastResult = last;

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

  if (last.deepExplanation) {
    showDeepExplanation(last.deepExplanation);
  } else {
    ui.deep.hidden = false;
    ui.igNote.textContent =
      "The email text is not stored, so analyse the email again to run the deep explanation.";
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
      analysedText = text;
      ui.deep.hidden = false;
      await saveResult({ prediction });
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
      await saveResult({ explanation: data.explanation });
      setStatus("Analysis complete.", "success");
    } catch (error) {
      ui.note.textContent = "The LIME explanation could not be computed.";
      setStatus(describeError(error, "Explanation failed."), "error");
    }
  } finally {
    setBusy(false);
  }
});

ui.deepButton.addEventListener("click", async () => {
  if (analysedText === null) return;

  setBusy(true, "Explaining…");
  resetDeepExplanation();
  ui.igNote.textContent = "This can take up to 5 minutes on CPU.";
  setStatus("Running Integrated Gradients…");

  try {
    const data = await postJson("deep-explain", analysedText);
    showDeepExplanation(data.deep_explanation);
    await saveResult({ deepExplanation: data.deep_explanation });
    setStatus("Deep explanation complete.", "success");
  } catch (error) {
    ui.igNote.textContent = "The deep explanation could not be computed.";
    setStatus(describeError(error, "Deep explanation failed."), "error");
  } finally {
    setBusy(false);
  }
});
