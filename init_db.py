from db import init_db, seed_data

if __name__ == "__main__":
    init_db()
    seed_data()
    print("Base de données initialisée et données d'exemple insérées.")
