# Reconnaissance du genre à partir de noms ivoiriens

Modèle CNN + BiLSTM + Multi-Head Attention, exporté en ONNX, servi avec Gradio.

- `train_model.ipynb` : entraînement et évaluation
- `app_gradio.py` : interface de démonstration

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
