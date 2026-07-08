from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from db import load_training_data

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model.joblib"))

TEST_SIZE = 0.25
RANDOM_STATE = 42

_cached_positive_model: LogisticRegression | None = None
_cached_negative_model: LogisticRegression | None = None
_cached_vectorizer: TfidfVectorizer | None = None


def _split_training_data(
    texts: list[str],
    positive_labels: list[int],
    negative_labels: list[int],
) -> tuple[list[str], list[str], list[int], list[int], list[int], list[int]]:
    return train_test_split(
        texts,
        positive_labels,
        negative_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=positive_labels,
    )


def train_model() -> tuple[LogisticRegression, LogisticRegression, TfidfVectorizer]:
    texts, positive_labels, negative_labels = load_training_data()
    train_texts, _, y_train_pos, _, y_train_neg, _ = _split_training_data(
        texts, positive_labels, negative_labels
    )

    vectorizer = TfidfVectorizer(max_features=500)
    X_train = vectorizer.fit_transform(train_texts)

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


def set_cached_models(
    positive_model: LogisticRegression,
    negative_model: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    global _cached_positive_model, _cached_negative_model, _cached_vectorizer
    _cached_positive_model = positive_model
    _cached_negative_model = negative_model
    _cached_vectorizer = vectorizer


def _ensure_models_loaded() -> tuple[LogisticRegression, LogisticRegression, TfidfVectorizer]:
    global _cached_positive_model, _cached_negative_model, _cached_vectorizer

    if _cached_positive_model is None or _cached_negative_model is None or _cached_vectorizer is None:
        try:
            positive_model, negative_model, vectorizer = load_model_artifacts()
        except FileNotFoundError:
            positive_model, negative_model, vectorizer = train_model()
            save_model_artifacts(positive_model, negative_model, vectorizer)
        set_cached_models(positive_model, negative_model, vectorizer)

    assert _cached_positive_model is not None
    assert _cached_negative_model is not None
    assert _cached_vectorizer is not None
    return _cached_positive_model, _cached_negative_model, _cached_vectorizer


def predict_sentiment(tweets: list[str]) -> dict[str, float]:
    positive_model, negative_model, vectorizer = _ensure_models_loaded()
    vectors = vectorizer.transform(tweets)
    positive_probs = positive_model.predict_proba(vectors)
    negative_probs = negative_model.predict_proba(vectors)

    result: dict[str, float] = {}
    for tweet, positive_proba, negative_proba in zip(tweets, positive_probs, negative_probs):
        positive_score = float(positive_proba[1])
        negative_score = float(negative_proba[1])
        score = round(max(-1.0, min(1.0, positive_score - negative_score)), 3)
        result[tweet] = score
    return result


def evaluate_saved_model() -> dict[str, Any]:
    texts, positive_labels, negative_labels = load_training_data()
    _, test_texts, _, y_test_pos, _, y_test_neg = _split_training_data(
        texts, positive_labels, negative_labels
    )

    positive_model, negative_model, vectorizer = load_model_artifacts()
    X_test = vectorizer.transform(test_texts)

    y_pred_pos = positive_model.predict(X_test)
    y_pred_neg = negative_model.predict(X_test)

    positive_conf = confusion_matrix(y_test_pos, y_pred_pos, labels=[0, 1]).tolist()
    negative_conf = confusion_matrix(y_test_neg, y_pred_neg, labels=[0, 1]).tolist()
    positive_report = classification_report(y_test_pos, y_pred_pos, output_dict=True, zero_division=0)
    negative_report = classification_report(y_test_neg, y_pred_neg, output_dict=True, zero_division=0)

    return {
        "positive_confusion_matrix": positive_conf,
        "negative_confusion_matrix": negative_conf,
        "classification_report": {
            "positive": positive_report,
            "negative": negative_report,
        },
    }
