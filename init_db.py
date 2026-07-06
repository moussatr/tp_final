import os
import mysql.connector

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}


def init_db() -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE DATABASE IF NOT EXISTS socialmetrics
        """
    )
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
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Base de données et table tweets initialisées.")
