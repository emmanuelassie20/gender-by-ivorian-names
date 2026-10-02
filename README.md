# Reconnaissance du genre à partir de noms ivoiriens

Modèle CNN + BiLSTM + Multi-Head Attention, exporté en ONNX, servi avec Gradio.

- `train_model.ipynb` : analyse exploratoire, modèles classiques, modèle v4 et export ONNX
- `optimisation.ipynb` : modèle v5, ablations, recherche d'hyperparamètres (Optuna), suivi MLflow
- `boosting.ipynb` : LightGBM et XGBoost (caractéristiques construites, empilement)
- `diagnostic.ipynb` : entrées, courbe d'apprentissage, précision selon le seuil de confiance
- `evaluation_finale.ipynb` : évaluation unique sur le test et choix du modèle de production
- `registre.ipynb` : registre MLflow des modèles, alias `production` / `challenger`, publication
- `adaptateurs.py` : interface commune aux modèles (régression logistique, v4, v5)
- `app_gradio.py` : interface de démonstration, sert le modèle portant l'alias `production`

Le jeu de données et les poids du modèle ne sont pas publiés.

## Installation (uv)

Application seule :

```bash
uv sync
uv run python app_gradio.py
```

Entraînement (TensorFlow, scikit-learn, Optuna) :

```bash
uv sync --group train
```

Sous WSL, utiliser un environnement séparé de celui de Windows :

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/gender-by-ivorian-names
uv sync --group train
```
