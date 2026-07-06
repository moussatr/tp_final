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
3. Démarrer MySQL et initialiser la base de données :
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

## Endpoints disponibles

### Santé de l'API
```bash
curl http://127.0.0.1:5000/health
```

### Analyse de sentiment
```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweets": ["J'adore ce produit", "C'est vraiment nul", "Excellent service"]}'
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
curl -X POST http://127.0.0.1:5000/train
```

### Rapport d'évaluation
```bash
curl http://127.0.0.1:5000/report
```

Le rapport PDF est généré dans le dossier reports/.

## Réentraînement automatisé

Un script d'automatisation est fourni dans scripts/retrain.sh.

```bash
chmod +x scripts/retrain.sh
./scripts/retrain.sh
```

Exemple de cron hebdomadaire :
```bash
0 3 * * 1 cd /chemin/vers/le/projet && /chemin/vers/.venv/bin/python retrain_model.py >> cron.log 2>&1
```
