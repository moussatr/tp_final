import os
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}


def load_data() -> tuple[list[str], list[int]]:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT text, positive, negative FROM tweets")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    texts = [row["text"] for row in rows]
    labels = [1 if row["positive"] == 1 else 0 for row in rows]
    return texts, labels


def retrain_and_save() -> None:
    texts, labels = load_data()
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump((model, vectorizer), "model.joblib")
    print("Modèle réentraîné et sauvegardé dans model.joblib")


if __name__ == "__main__":
    retrain_and_save()
