# API d'analyse de sentiments

Cette application Flask expose une API pour analyser le sentiment de tweets à partir d'un modèle de régression logistique entraîné sur une base MySQL.

## Installation

1. Créer un environnement virtuel :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Démarrer MySQL et initialiser la base de données (création de la table + données d'exemple) :
   ```bash
   python3 init_db.py
   ```
4. Réentraîner et sauvegarder le modèle initial :
   ```bash
   python3 retrain_model.py
   ```
5. Lancer l'API :
   ```bash
   python3 app.py
   ```

## Variables d'environnement

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `DB_HOST` | Hôte MySQL | `localhost` |
| `DB_PORT` | Port MySQL | `3306` |
| `DB_USER` | Utilisateur MySQL | `root` |
| `DB_PASSWORD` | Mot de passe MySQL | *(vide)* |
| `DB_NAME` | Nom de la base | `socialmetrics` |
| `MODEL_PATH` | Chemin du modèle sauvegardé | `model.joblib` |
| `REPORT_PATH` | Chemin du rapport PDF | `reports/sentiment_evaluation_report.pdf` |
| `REPORT_JSON_PATH` | Chemin du rapport JSON | `reports/sentiment_evaluation_report.json` |
| `FLASK_DEBUG` | Mode debug Flask (`true` / `false`) | `true` |
| `PORT` | Port de l'API Flask | `5001` |

## Endpoints disponibles

### Santé de l'API
```bash
curl http://127.0.0.1:5001/health
```

### Analyse de sentiment

Le corps JSON peut être un objet avec une clé `tweets` ou une liste directe de chaînes.

```bash
curl -X POST http://127.0.0.1:5001/analyze \
  -H "Content-Type: application/json" \
  -d "{\"tweets\": [\"J'adore ce produit\", \"C'est vraiment nul\", \"Excellent service\"]}"
```

Réponse attendue :
```json
{
  "J'adore ce produit": 0.812,
  "C'est vraiment nul": -0.751,
  "Excellent service": 0.745
}
```

### Réentraînement du modèle
```bash
curl -X POST http://127.0.0.1:5001/train
```

### Rapport d'évaluation

Évalue le modèle sauvegardé sur le jeu de test (25 % des données) et génère les rapports PDF et JSON.

```bash
curl http://127.0.0.1:5001/report
```

Les rapports sont générés dans le dossier `reports/`.

## Réentraînement automatisé

Un script d'automatisation est fourni dans `scripts/retrain.sh`. Il réentraîne le modèle et génère les rapports PDF et JSON.

```bash
chmod +x scripts/retrain.sh
./scripts/retrain.sh
```

Exemple de cron hebdomadaire :
```bash
0 3 * * 1 cd /chemin/vers/le/projet && /chemin/vers/.venv/bin/python retrain_model.py >> cron.log 2>&1
```
