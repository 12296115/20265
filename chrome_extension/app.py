from flask import Flask, jsonify, request
from predictor import PhishingPredictor

app = Flask(__name__)

# Load once when the server starts.
predictor = PhishingPredictor()


# @app.get("/")
# def home():
#     return jsonify({"message": "Phishing API is running"})
@app.get("/")
def home():
    return "Phishing API is running"


def run_on_request_text(method):
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Provide non-empty email text"}), 400

    try:
        result = method(text)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        # Email text is never logged or stored; log only the error type.
        app.logger.error("Request failed: %s", type(error).__name__)
        return jsonify({"error": "Internal server error"}), 500

    return jsonify(result)


@app.post("/api/predict")
def predict():
    return run_on_request_text(predictor.predict)


@app.post("/api/explain")
def explain():
    return run_on_request_text(
        lambda text: {"explanation": predictor.explain(text)}
    )


@app.post("/api/analyse")
def analyse():
    return run_on_request_text(predictor.predict_with_explanation)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)