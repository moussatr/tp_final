from __future__ import annotations

import os
from typing import Any

import mysql.connector
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

app = Flask(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}

MODEL: LogisticRegression | None = None
VECTORIZER: TfidfVectorizer | None = None


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tweets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            text TEXT NOT NULL,
            positive TINYINT NOT NULL DEFAULT 0,
            negative TINYINT NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    cursor.close()
    conn.close()


def seed_data() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tweets")
    count = cursor.fetchone()[0]
    if count > 0:
        cursor.close()
        conn.close()
        return

    samples = [
        ("J'adore ce produit, il est incroyable !", 1, 0),
        ("C'est vraiment nul et très décevant.", 0, 1),
        ("Excellent service, merci beaucoup !", 1, 0),
        ("Je déteste cette expérience, c'est horrible.", 0, 1),
        ("Super contenu, très utile et clair.", 1, 0),
        ("Mauvaise qualité, je suis très mécontent.", 0, 1),
        ("Bravo, c'est parfait et très agréable.", 1, 0),
        ("Terrible expérience, je ne recommande pas.", 0, 1),
    ]
    cursor.executemany(
        "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
        samples,
    )
    conn.commit()
    cursor.close()
    conn.close()


def train_model() -> tuple[LogisticRegression, TfidfVectorizer]:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT text, positive, negative FROM tweets")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise ValueError("Aucune donnée disponible pour entraîner le modèle")

    texts = [row["text"] for row in rows]
    labels = [1 if row["positive"] == 1 else 0 for row in rows]

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, vectorizer


def predict_sentiment(tweets: list[str]) -> dict[str, float]:
    global MODEL, VECTORIZER
    if MODEL is None or VECTORIZER is None:
        MODEL, VECTORIZER = train_model()

    vectors = VECTORIZER.transform(tweets)
    probs = MODEL.predict_proba(vectors)
    result: dict[str, float] = {}
    for tweet, proba in zip(tweets, probs):
        score = round(float(proba[1] - proba[0]), 3)
        result[tweet] = score
    return result


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "tweets" not in data:
        return jsonify({"error": "Le corps JSON doit contenir une clé 'tweets' avec une liste de chaînes"}), 400

    tweets = data["tweets"]
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
        global MODEL, VECTORIZER
        MODEL, VECTORIZER = train_model()
        return jsonify({"message": "Modèle entraîné avec succès"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/report", methods=["GET"])
def report() -> Any:
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT text, positive, negative FROM tweets")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        texts = [row["text"] for row in rows]
        labels = [1 if row["positive"] == 1 else 0 for row in rows]
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        positive_conf = confusion_matrix(y_test, y_pred, labels=[0, 1])
        return jsonify({
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": positive_conf.tolist(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    init_db()
    seed_data()
    app.run(debug=True, host="0.0.0.0", port=5000)
