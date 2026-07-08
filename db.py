from __future__ import annotations

import os

import mysql.connector

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}

SAMPLE_TWEETS = [
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
                negative TINYINT NOT NULL DEFAULT 0,
                CHECK (NOT (positive = 1 AND negative = 1))
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

        cursor.executemany(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            SAMPLE_TWEETS,
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
