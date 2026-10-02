# Reconnaissance du genre à partir de noms ivoiriens

Prédiction du genre (Masculin / Féminin) à partir d'un nom complet ivoirien, dans un ordre
NOM / PRÉNOMS quelconque, avec interprétabilité par mot.

Modèle en production : **régression logistique** sur n-grammes de caractères (1 à 6) et mots
entiers, retenue par l'évaluation finale sur le jeu de test (accuracy 0,9557, IC 95 %
[0,950 ; 0,962]). Aucun autre modèle testé (CNN + BiLSTM + MHA, modèle par mot, LightGBM,
XGBoost, ensembles) n'est significativement meilleur au test de McNemar. Sous 70 % de
confiance, l'application signale la prédiction comme « à vérifier ».

Le jeu de données et les poids des modèles ne sont pas publiés.

## Contenu

| Fichier | Rôle |
|---|---|
| `train_model.ipynb` | analyse exploratoire, modèles classiques, CNN + BiLSTM + MHA (v4) |
| `optimisation.ipynb` | modèle par mot (v5), ablations, recherche Optuna, entraînements finaux |
| `boosting.ipynb` | LightGBM et XGBoost (caractéristiques construites, empilement) |
| `diagnostic.ipynb` | choix des entrées, courbe d'apprentissage, précision selon le seuil |
| `evaluation_finale.ipynb` | évaluation unique sur le test, choix du modèle de production |
| `registre.ipynb` | registre MLflow, alias `production` / `challenger`, publication |
| `figures_memoire.ipynb` | tableaux et figures du mémoire |
| `adaptateurs.py` | interface commune aux modèles (régression logistique, v4, v5) |
| `app_gradio.py` | application, sert le modèle portant l'alias `production` |

## Fichiers locaux nécessaires (non versionnés)

- `dataset_nettoye.csv` : colonnes `NOM`, `PRENOMS`, `GENRE` (M / F).
- `split_reference.csv` : découpage train / validation / test figé, lu par tous les notebooks.
  Chaque ligne porte une empreinte du nom : un notebook s'arrête si le fichier ne correspond
  plus au dataset.

## Ordre d'exécution

1. `optimisation.ipynb` et `boosting.ipynb` : réglages choisis sur la validation.
2. Entraînements finaux : `optimisation.ipynb` avec `RUN_FINAL=True` (v5, puis v4 avec
   `FINAL_USE_BEST=False`), `boosting.ipynb` avec `RUN_TEST=True`.
3. `evaluation_finale.ipynb` : une seule fois.
4. `registre.ipynb` : enregistrement, alias, export de `production/` (et publication avec
   `PUBLIER=True`).
5. `figures_memoire.ipynb`.

Les notebooks peuvent être lancés sans être ouverts, avec papermill :

```bash
papermill optimisation.ipynb runs/optimisation.ipynb -p RUN_FINAL True
```

## Installation (uv)

Application seule :

```bash
uv sync
```

Entraînement (TensorFlow, scikit-learn, Optuna, LightGBM, XGBoost, MLflow, papermill) :

```bash
uv sync --group train
```

Sous WSL (GPU), utiliser un environnement séparé de celui de Windows :

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/gender-by-ivorian-names
uv sync --group train
```

Remarques :

- cuDNN est figé en 9.3 : les versions récentes plantent sur les GPU Pascal (Quadro P1000).
- Sous WSL, LightGBM et XGBoost sont limités à 4 threads (blocage au-delà).
- Le registre exporte les modèles sur le processeur : sur GPU, les LSTM ne sont pas
  convertibles en ONNX.

## Lancer l'application

En local, après `registre.ipynb` (dossier `production/`) :

```bash
uv run python app_gradio.py
```

Variables d'environnement :

| Variable | Rôle |
|---|---|
| `HF_MODEL_REPO` | dépôt Hugging Face privé d'où télécharger `production/` |
| `HF_TOKEN` | token en lecture sur ce dépôt |
| `APP_PASSWORD`, `APP_USER` | accès protégé (utilisateur `jury` par défaut) |
| `SEUIL_INCERTAIN` | seuil de confiance du mode « à vérifier » (0,70 par défaut) |
| `PAQUET_DIR` | dossier du modèle à servir (`production` par défaut) |
| `PORT` | port imposé par l'hébergeur |

## Déploiement

- Render (`render.yaml`) déploie la branche `main` : installation avec `uv sync --frozen`,
  lancement de `app_gradio.py`.
- L'application télécharge le dossier `production/` du dépôt Hugging Face privé au démarrage.
- Changer de modèle en production : déplacer l'alias `production` dans `registre.ipynb`,
  republier, puis redémarrer le service Render. Le code de l'application ne change pas.

## Suivi des expériences (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir http://localhost:5000 (onglet **Models** pour le registre `genre-noms-ivoiriens`).
