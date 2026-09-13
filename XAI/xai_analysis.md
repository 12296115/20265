# Explainability Analysis of the DeBERTa Gated Hybrid Model

## Purpose

Following the development and evaluation of the DeBERTa-v3-large Gated Hybrid model (DeBERTa-v3-large transformer branch + 12 handcrafted phishing-indicator features fused via a learned gate), an explainability analysis was conducted to investigate *why* the model makes its predictions, rather than only *how well* it performs.

The analysis focused on:

- Token-level attribution for individual predictions using SHAP (SHapley Additive exPlanations)
- The model's known error cases (false positives and false negatives)
- Whether the transformer branch, the handcrafted NLP feature branch, or their fusion was responsible for specific misclassifications
- Patterns in what the model associates with "Phishing Email" versus "Safe Email"

This analysis builds directly on the error cases identified during model evaluation and provides a mechanistic explanation for the small number of misclassifications the model produced.

## Model Under Analysis

The DeBERTa Gated Hybrid model combines two branches:

1. **Contextual branch** — a DeBERTa-v3-large transformer encodes the email text, and the first-token ([CLS]-equivalent) representation is projected to a 256-dimensional context vector.
2. **Feature branch** — 12 handcrafted phishing-oriented features (urgency term count, credential term count, threat term count, financial term count, call-to-action term count, URL count, email address count, phone number count, exclamation count, question count, uppercase-word ratio, and log-transformed text length) are projected to the same 256-dimensional space.

A learned gating mechanism combines the two branches:

```text
gate = sigmoid(Linear(concat(context, features)))
fused = gate * context + (1 - gate) * features
```

The fused representation is passed through a final fusion layer and a linear classifier to produce the phishing/safe prediction.

## Test Set Performance

The model was evaluated on a held-out test set of 1,752 emails (1,098 Safe Email, 654 Phishing Email):

| Metric | Result |
|---|---:|
| Accuracy | 99.49% |
| Precision | 99.69% |
| Recall | 98.93% |
| F1-Score | 99.31% |

The confusion matrix was:

```text
[[1096    2]
 [   7  647]]
```

This produced 9 total misclassifications: 2 false positives and 7 false negatives. This is a substantially smaller error set than the traditional DistilBERT baseline (42 errors), consistent with the Gated Hybrid model's higher overall performance.

## Methodology

SHAP's `PartitionExplainer` was applied to the model's phishing-probability output using a text masker, allowing individual word/token contributions to be attributed to each prediction. For each explained email, SHAP values were computed against the phishing class (class 1), producing a visualisation in which:

- **Red** tokens push the prediction toward "Phishing Email"
- **Blue** tokens push the prediction toward "Safe Email"

Four representative error cases were selected from the 9 total misclassifications to cover a range of error types: two false negatives with contrasting textual characteristics, and two false positives with contrasting characteristics.

## Case Analysis

### Case 1 (False Negative, test index 1150) — Phishing Disguised as Routine Technical Instruction

**True label:** Phishing Email. **Predicted:** Safe Email (phishing probability 0.00096).

This email explicitly contains multiple classic phishing indicators: references to "identity theft," a time-limited "90 day trial," instructions to "activate privacy service," and step-by-step instructions to click and install software. Despite this, SHAP attribution shows the overwhelming majority of the email's tokens pushing toward "Safe Email." Only a small fragment near the opening of the email contributed toward the phishing class, and this was outweighed by the remainder of the text.

The explanation is that the email is written in a calm, procedural, customer-support tone — describing where to click within an existing, named security application ("mcafeecenter," "securitycenter") rather than using manipulative or alarming language. The model appears to associate phishing primarily with an urgent or manipulative *tone*, rather than with the literal presence of security- or action-related vocabulary. Because this email's phrasing resembles legitimate software instructions, the transformer branch overrode any signal that may have come from the handcrafted feature branch (which would have registered non-zero urgency, credential, and call-to-action term counts).

### Case 2 (False Negative, test index 99) — Phishing Disguised as Ordinary Commercial Correspondence

**True label:** Phishing Email. **Predicted:** Safe Email (phishing probability 0.00035).

This is a long email formatted as a straightforward laptop-computer price list from a legitimate-sounding supplier. SHAP attribution shows almost no red tokens anywhere in the body of the email; the few red fragments occur on incidental phrases such as the opening line and scattered product/warranty details, with negligible influence on the outcome.

This case demonstrates a scenario in which the phishing intent is not recoverable from surface-level lexical or structural cues at all. There is no urgency language, no credential request, and no threatening language — the email is indistinguishable in tone and structure from a genuine business communication. None of the 12 handcrafted features would be expected to activate meaningfully for this email, and the transformer branch found no contextual basis for suspicion either. This represents the limiting case for both branches of the model: when a phishing email is authored to closely mimic ordinary business correspondence, neither branch of the architecture has a signal to detect it, and it is possible this example approaches the boundary of what is detectable from message content alone.

### Case 3 (False Positive, test index 1470) — Legitimate Marketing Email Resembling Phishing Structure

**True label:** Safe Email. **Predicted:** Phishing Email (phishing probability 0.9995).

This is a legitimate product marketing email (Compaq-branded scanner promotion) that was confidently misclassified as phishing. SHAP attribution shows heavy red weighting across much of the email, concentrated on terms such as "digitize," "scanners," "click away," tracking-style URLs, and the closing "remove your name from our mailing list."

This case illustrates the inverse of Cases 1 and 2: here, the model's association between phishing and certain structural/lexical conventions is well-founded in general, but those same conventions are also standard practice in legitimate commercial email marketing — promotional calls-to-action, tracking links, and unsubscribe mechanisms. The overlap between legitimate marketing conventions and phishing conventions is a genuine ambiguity in the underlying task, not simply a model error.

### Case 4 (False Positive, test index 717) — Very Short Text with Insufficient Context

**True label:** Safe Email. **Predicted:** Phishing Email (phishing probability 0.9888).

This email consists of a single short sentence (15 words) announcing an executive ranking, with a link to a legitimate news site. SHAP attribution shows the phrase "list -" and the bare URL "http://" driving the prediction toward phishing, while the remaining tokens ("steve," "com/", "the," "index") pushed toward safe but were insufficient to change the outcome.

With so little text available, the model has very few tokens from which to build contextual understanding. The presence of a bare URL combined with minimal surrounding context appears sufficient to bias the model toward a phishing classification by default. This highlights a structural limitation: transformer-based classifiers generally require sufficient context to disambiguate intent, and very short emails do not provide this, regardless of architecture.

## Synthesis

Across all four cases, the model's errors are not random but cluster around a consistent underlying pattern: **the model relies substantially on tone and structural convention rather than deep semantic understanding of intent.**

| Case | Type | Test Index | Underlying Cause |
|---|---|---:|---|
| 1 | False Negative | 1150 | Calm, procedural tone overrides literal phishing vocabulary |
| 2 | False Negative | 99 | Phishing disguised as ordinary commercial correspondence; no lexical or contextual cues present |
| 3 | False Positive | 1470 | Legitimate marketing conventions structurally overlap with phishing conventions |
| 4 | False Positive | 717 | Insufficient context in very short text; bare URL biases toward phishing |

This finding is consistent with, and extends, the error analysis conducted for the DistilBERT baseline model, which similarly found that many errors involved emails with "relatively ordinary or contextual language" (false negatives) or legitimate emails containing "domains, technical systems, order confirmations" resembling phishing structure (false positives). The explainability analysis presented here provides a mechanistic, token-level account of why those patterns occur, rather than only describing that they occur.

## Limitations

- SHAP's `PartitionExplainer` was applied with a text-based masker, which perturbs the input at the token/word level; it does not directly attribute importance to the handcrafted NLP feature branch. A complementary feature-level analysis (e.g., examining the 12 handcrafted feature values and the learned gate weight for each case) would be needed to determine, for each prediction, how much the feature branch versus the transformer branch contributed to the final decision.
- Attribution was performed on only 4 of the 9 total error cases; the remaining 5 errors were not individually explained.
- SHAP values reflect correlational, not necessarily causal, token importance, and results can be sensitive to the choice of masker and perturbation strategy.
- Explanations were generated on the `processed_text` field, consistent with the field used during model training; results may differ if applied to the raw, unprocessed email text.

## Future Work

- Extend the SHAP analysis to the remaining 5 error cases for full coverage of the model's misclassifications.
- Inspect the learned gate values for each error case to quantify the relative contribution of the transformer branch versus the handcrafted feature branch.
- Apply LIME as a cross-validation method against the SHAP attributions to confirm that both methods agree on the same influential tokens.
- Extend the explainability analysis to a sample of correctly classified emails, to establish a baseline for what the model considers a "typical" confident correct prediction, against which the error cases can be more precisely contrasted.

## Evidence

The analysis was performed using the trained DeBERTa Gated Hybrid model, tokenizer, and feature normalisation statistics loaded from the saved model artefacts, evaluated on the project's `phishing_test.csv` split (1,752 samples). SHAP attributions were computed using the `shap.Explainer` API with a `PartitionExplainer` backend and a text masker, applied to the model's phishing-class probability output.
