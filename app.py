from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, request

from db import init_db, seed_data
from reporting import save_evaluation_reports
from training import evaluate_saved_model, predict_sentiment, save_model_artifacts, set_cached_models, train_model

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    payload = request.get_json(silent=True)

    if isinstance(payload, list):
        tweets = payload
    elif isinstance(payload, dict) and "tweets" in payload:
        tweets = payload["tweets"]
    else:
        return jsonify({"error": "Le corps JSON doit contenir une clé 'tweets' ou une liste directe de chaînes"}), 400

    if not isinstance(tweets, list):
        return jsonify({"error": "'tweets' doit être une liste"}), 400
    if not tweets:
        return jsonify({"error": "La liste de tweets ne peut pas être vide"}), 400
    if not all(isinstance(item, str) for item in tweets):
        return jsonify({"error": "Tous les éléments doivent être des chaînes de caractères"}), 400

    try:
        result = predict_sentiment(tweets)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/train", methods=["POST"])
def train_endpoint() -> Any:
    try:
        positive_model, negative_model, vectorizer = train_model()
        save_model_artifacts(positive_model, negative_model, vectorizer)
        set_cached_models(positive_model, negative_model, vectorizer)
        return jsonify({"message": "Modèle entraîné avec succès"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/report", methods=["GET"])
def report() -> Any:
    try:
        report_payload = evaluate_saved_model()
        pdf_path, json_path = save_evaluation_reports(report_payload)

        return jsonify({
            "positive_confusion_matrix": report_payload["positive_confusion_matrix"],
            "negative_confusion_matrix": report_payload["negative_confusion_matrix"],
            "classification_report": report_payload["classification_report"],
            "report_path": pdf_path,
            "report_json_path": json_path,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    init_db()
    seed_data()
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.getenv("PORT", "5001"))
    app.run(debug=debug, host="0.0.0.0", port=port)
