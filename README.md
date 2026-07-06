# API d'analyse de sentiments

## Installation

1. Créer un environnement virtuel :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Installer les dépendances :
   ```bash
   pip install flask mysql-connector-python scikit-learn joblib
   ```
3. Démarrer MySQL et créer la base :
   ```bash
   python3 init_db.py
   ```
4. Lancer l'API :
   ```bash
   python3 app.py
   ```

## Utilisation de l'API

### Analyse de sentiment
```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweets": ["J'adore ce produit", "C'est vraiment nul"]}'
```

### Réentraînement du modèle
```bash
python3 retrain_model.py
```

## Réentraînement automatisé

Ajouter une tâche cron :
```bash
0 3 * * 1 cd /chemin/vers/le/projet && /chemin/vers/.venv/bin/python retrain_model.py >> cron.log 2>&1
```
