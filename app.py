from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mysql.connector
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model.joblib"))
REPORT_PATH = os.getenv("REPORT_PATH", str(Path(__file__).resolve().parent / "reports" / "sentiment_evaluation_report.pdf"))

MODEL_POSITIVE: LogisticRegression | None = None
MODEL_NEGATIVE: LogisticRegression | None = None
VECTORIZER: TfidfVectorizer | None = None


def get_db_connection(include_database: bool = True) -> mysql.connector.MySQLConnection:
    config = DB_CONFIG.copy()
    if not include_database:
        config.pop("database", None)
    return mysql.connector.connect(**config)


def init_db() -> None:
    conn = get_db_connection(include_database=False)
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS socialmetrics")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS socialmetrics.tweets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                positive TINYINT NOT NULL DEFAULT 0,
                negative TINYINT NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def seed_data() -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tweets")
        count = cursor.fetchone()[0]
        if count > 0:
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
            ("Je suis ravi de cette amélioration, c'est génial.", 1, 0),
            ("Ça ne marche pas du tout, c'est catastrophique.", 0, 1),
        ]
        cursor.executemany(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            samples,
        )
        conn.commit()
    finally:
        conn.close()


def load_training_data() -> tuple[list[str], list[int], list[int]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT text, positive, negative FROM tweets")
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError("Aucune donnée disponible pour entraîner le modèle")

    texts = [row["text"] for row in rows]
    positive_labels = [1 if row["positive"] == 1 else 0 for row in rows]
    negative_labels = [1 if row["negative"] == 1 else 0 for row in rows]
    return texts, positive_labels, negative_labels


def train_model() -> tuple[LogisticRegression, LogisticRegression, TfidfVectorizer]:
    texts, positive_labels, negative_labels = load_training_data()
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train_pos, y_test_pos, y_train_neg, y_test_neg = train_test_split(
        X,
        positive_labels,
        negative_labels,
        test_size=0.25,
        random_state=42,
        stratify=positive_labels,
    )

    positive_model = LogisticRegression(max_iter=1000)
    negative_model = LogisticRegression(max_iter=1000)
    positive_model.fit(X_train, y_train_pos)
    negative_model.fit(X_train, y_train_neg)

    return positive_model, negative_model, vectorizer


def save_model_artifacts(
    positive_model: LogisticRegression,
    negative_model: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "positive_model": positive_model,
            "negative_model": negative_model,
            "vectorizer": vectorizer,
        },
        MODEL_PATH,
    )


def load_model_artifacts() -> tuple[LogisticRegression, LogisticRegression, TfidfVectorizer]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Aucun modèle sauvegardé n'a été trouvé")

    payload = joblib.load(MODEL_PATH)
    return (
        payload["positive_model"],
        payload["negative_model"],
        payload["vectorizer"],
    )


def predict_sentiment(tweets: list[str]) -> dict[str, float]:
    global MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER

    if MODEL_POSITIVE is None or MODEL_NEGATIVE is None or VECTORIZER is None:
        try:
            MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER = load_model_artifacts()
        except FileNotFoundError:
            MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER = train_model()
            save_model_artifacts(MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER)

    vectors = VECTORIZER.transform(tweets)
    positive_probs = MODEL_POSITIVE.predict_proba(vectors)
    negative_probs = MODEL_NEGATIVE.predict_proba(vectors)

    result: dict[str, float] = {}
    for tweet, positive_proba, negative_proba in zip(tweets, positive_probs, negative_probs):
        positive_score = float(positive_proba[1])
        negative_score = float(negative_proba[1])
        score = round(max(-1.0, min(1.0, positive_score - negative_score)), 3)
        result[tweet] = score
    return result


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_pdf_report(output_path: str, report_payload: dict[str, Any]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    classification_text = json.dumps(report_payload["classification_report"], ensure_ascii=False, indent=2)
    lines = [
        "Rapport d'évaluation du modèle de sentiment",
        "",
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        "Matrice de confusion - classe positive",
        str(report_payload["positive_confusion_matrix"]),
        "",
        "Matrice de confusion - classe negative",
        str(report_payload["negative_confusion_matrix"]),
        "",
        "Précision, rappel et F1-score",
        classification_text,
        "",
        "Observations :",
        "- Le modèle est sensible au vocabulaire positif/negatif présent dans les données annotées.",
        "- Les erreurs fréquentes proviennent des phrases ambiguës ou du sarcasme.",
        "- Les biais potentiels sont liés à une distribution inégale des exemples positifs et négatifs.",
        "",
        "Recommandations :",
        "- Ajouter des exemples annotés plus variés et plus équilibrés.",
        "- Introduire des caractéristiques lexicales plus riches et des n-grammes.",
        "- Réentraîner le modèle régulièrement avec des données récentes.",
    ]

    stream = "BT\n/F1 10 Tf\n50 760 Td\n"
    for index, line in enumerate(lines):
        if index > 0:
            stream += "0 -14 Td\n"
        stream += f"({_escape_pdf_text(line)}) Tj\n"
    stream += "ET\n"

    stream_bytes = stream.encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )
    Path(output_path).write_bytes(pdf)


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
        global MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER
        MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER = train_model()
        save_model_artifacts(MODEL_POSITIVE, MODEL_NEGATIVE, VECTORIZER)
        return jsonify({"message": "Modèle entraîné avec succès"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/report", methods=["GET"])
def report() -> Any:
    try:
        texts, positive_labels, negative_labels = load_training_data()
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(texts)

        X_train, X_test, y_train_pos, y_test_pos, y_train_neg, y_test_neg = train_test_split(
            X,
            positive_labels,
            negative_labels,
            test_size=0.25,
            random_state=42,
            stratify=positive_labels,
        )

        positive_model = LogisticRegression(max_iter=1000)
        negative_model = LogisticRegression(max_iter=1000)
        positive_model.fit(X_train, y_train_pos)
        negative_model.fit(X_train, y_train_neg)

        y_pred_pos = positive_model.predict(X_test)
        y_pred_neg = negative_model.predict(X_test)

        positive_conf = confusion_matrix(y_test_pos, y_pred_pos, labels=[0, 1]).tolist()
        negative_conf = confusion_matrix(y_test_neg, y_pred_neg, labels=[0, 1]).tolist()
        positive_report = classification_report(y_test_pos, y_pred_pos, output_dict=True, zero_division=0)
        negative_report = classification_report(y_test_neg, y_pred_neg, output_dict=True, zero_division=0)

        report_payload = {
            "positive_confusion_matrix": positive_conf,
            "negative_confusion_matrix": negative_conf,
            "classification_report": {
                "positive": positive_report,
                "negative": negative_report,
            },
        }
        generate_pdf_report(REPORT_PATH, report_payload)

        return jsonify({
            "positive_confusion_matrix": positive_conf,
            "negative_confusion_matrix": negative_conf,
            "classification_report": report_payload["classification_report"],
            "report_path": REPORT_PATH,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    init_db()
    seed_data()
    app.run(debug=True, host="0.0.0.0", port=5000)
