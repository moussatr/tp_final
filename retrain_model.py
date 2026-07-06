import json
import os
from pathlib import Path

import joblib
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model.joblib"))
REPORT_PATH = os.getenv("REPORT_PATH", str(Path(__file__).resolve().parent / "reports" / "sentiment_evaluation_report.json"))


def load_data() -> tuple[list[str], list[int], list[int]]:
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT text, positive, negative FROM tweets")
        rows = cursor.fetchall()
    finally:
        conn.close()

    texts = [row["text"] for row in rows]
    positive_labels = [1 if row["positive"] == 1 else 0 for row in rows]
    negative_labels = [1 if row["negative"] == 1 else 0 for row in rows]
    return texts, positive_labels, negative_labels


def retrain_and_save() -> None:
    texts, positive_labels, negative_labels = load_data()
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

    payload = {
        "positive_model": positive_model,
        "negative_model": negative_model,
        "vectorizer": vectorizer,
    }
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

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
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Modèle réentraîné et sauvegardé dans {MODEL_PATH}")
    print(f"Rapport d'évaluation enregistré dans {REPORT_PATH}")


if __name__ == "__main__":
    retrain_and_save()
