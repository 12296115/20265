"""Compare PhishingPredictor.predict() on fixed samples against a saved baseline.

First run (or --update) writes baseline_predictions.json; later runs compare.
"""
import json
import sys
from pathlib import Path

from predictor import PhishingPredictor

BASELINE_PATH = Path(__file__).resolve().parent / "baseline_predictions.json"

SAMPLES = {
    "phishing": (
        "URGENT: Your bank account has been suspended due to unauthorized "
        "activity. Click here http://secure-verify-login.com to verify your "
        "password and username immediately or your account will be closed!"
    ),
    "work": (
        "Hi Sarah, thanks for sending the draft report. I've added my comments "
        "on section 3 and moved our review meeting to Thursday at 2pm. "
        "Let me know if that time still works for you. Best, Tom"
    ),
    "marketing": (
        "Autumn Sale is here! Get 30% off all jackets and boots this weekend "
        "only. Browse the new collection at www.example-outdoor-store.com. "
        "You are receiving this email because you subscribed to our newsletter."
    ),
}


def run_predictions():
    predictor = PhishingPredictor()
    results = {}
    for name, text in SAMPLES.items():
        prediction = predictor.predict(text)
        results[name] = {
            "verdict": prediction["verdict"],
            "class_scores": prediction["class_scores"],
        }
    return results


def main():
    results = run_predictions()

    if "--update" in sys.argv or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Baseline written to {BASELINE_PATH.name}:")
        print(json.dumps(results, indent=2))
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    all_match = True

    for name, current in results.items():
        before = baseline[name]
        same_verdict = before["verdict"] == current["verdict"]
        all_match &= same_verdict
        delta = current["class_scores"][1] - before["class_scores"][1]
        print(
            f"{name:10s} verdict {before['verdict']:>10s} -> "
            f"{current['verdict']:<10s} {'OK' if same_verdict else 'CHANGED'} | "
            f"P(phishing) {before['class_scores'][1]:.6f} -> "
            f"{current['class_scores'][1]:.6f} (delta {delta:+.6f})"
        )

    print("All verdicts identical." if all_match else "VERDICT MISMATCH.")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
