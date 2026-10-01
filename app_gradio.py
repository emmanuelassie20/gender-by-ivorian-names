"""
app_gradio.py - Interface Gradio reconnaissance de genre - INP-HB STIC v4

Nouveautés v4 :
  - Inférence via ONNX Runtime (plus léger, pas de dépendance TensorFlow)
  - Compatible avec le modèle CNN + BiLSTM + MHA exporté par train_model.py
  - Batch masking conservé pour l'interprétabilité mot / caractère
  - Fallback automatique sur le modèle Keras si ONNX non disponible
"""

import re, pickle, unicodedata
import numpy as np
import os, datetime
import pandas as pd
import gradio as gr

# ═══════════════════════════════
# CHARGEMENT
# ═══════════════════════════════
ONNX_MODEL_PATH  = "best_gender_model.onnx"
KERAS_MODEL_PATH = "best_gender_model.keras"
CHAR_INDEX_PATH  = "char_to_index.pkl"
MAX_LEN_PATH     = "max_len.pkl"
CORRECTIONS_FILE = "corrections_dataset.csv"

# En ligne (Hugging Face Spaces), les fichiers du modèle ne sont pas dans le
# dépôt : ils sont téléchargés depuis un dépôt modèle PRIVÉ (secrets
# HF_MODEL_REPO et HF_TOKEN). En local, les fichiers présents sont utilisés.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")

print(">>> Chargement des ressources...")

if HF_MODEL_REPO:
    from huggingface_hub import hf_hub_download
    for _fname in (ONNX_MODEL_PATH, CHAR_INDEX_PATH, MAX_LEN_PATH):
        if not os.path.exists(_fname):
            hf_hub_download(HF_MODEL_REPO, _fname, local_dir=".",
                            token=os.environ.get("HF_TOKEN"))
            print(f">>> Téléchargé depuis {HF_MODEL_REPO} : {_fname}")

# Chargement vocabulaire et longueur max
try:
    with open(CHAR_INDEX_PATH, "rb") as f:
        char_to_index = pickle.load(f)
    with open(MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)
    print(f">>> Vocab ({len(char_to_index)} chars) | max_len={max_len}")
except FileNotFoundError as e:
    print(f">>> ERREUR : {e}\nLancez d'abord train_model.py.")
    raise SystemExit(1)

# ── Chargement du modèle : ONNX en priorité, Keras en fallback ──
USE_ONNX = False
_keras_model = None
_ort_session = None

try:
    import onnxruntime as ort
    _ort_session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=['CPUExecutionProvider']
    )
    _ort_input_name = _ort_session.get_inputs()[0].name
    USE_ONNX = True
    print(f">>> ONNX Runtime chargé  ✓  ({ONNX_MODEL_PATH})")

except (ImportError, FileNotFoundError) as e:
    print(f">>> ONNX non disponible ({e}) - fallback Keras...")
    try:
        import tensorflow as tf
        _keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print(f">>> Modèle Keras chargé  ✓  ({KERAS_MODEL_PATH})")
    except FileNotFoundError as e2:
        print(f">>> ERREUR : {e2}\nLancez d'abord train_model.py.")
        raise SystemExit(1)


# ═══════════════════════════════
# UTILITAIRES
# ═══════════════════════════════
def clean_text(text: str) -> str:
    """Normalise un nom propre : minuscules, ASCII, espaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r"['\-]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def text_to_sequence(text: str) -> list:
    return [char_to_index.get(c, char_to_index['<UNK>']) for c in text]


def _pad(seqs: list) -> np.ndarray:
    """
    Pad une liste de séquences → array int32 (requis par ONNX).
    Équivalent NumPy de pad_sequences(padding='post', truncating='pre'),
    pour ne pas dépendre de TensorFlow en production.
    """
    X = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, seq in enumerate(seqs):
        seq = seq[-max_len:]
        X[i, :len(seq)] = seq
    return X


def _predict_batch(X: np.ndarray) -> np.ndarray:
    """
    Inférence batch unifiée : ONNX ou Keras selon disponibilité.
    Retourne un array 1D de probabilités P(Masculin).
    """
    if USE_ONNX:
        return _ort_session.run(None, {_ort_input_name: X})[0].flatten()
    else:
        return _keras_model.predict(X, verbose=0).flatten()


def predict_proba(text: str) -> float:
    """Retourne P(Masculin) pour un texte nettoyé."""
    seq = text_to_sequence(text)
    X   = _pad([seq])
    return float(_predict_batch(X)[0])


# ═══════════════════════════════
# INTERPRÉTABILITÉ PAR MOT
# ═══════════════════════════════
def word_importance(cleaned: str, original_prob: float) -> list:
    """
    Importance de chaque MOT par masquage batch.
    impact(w_i) = |P(M|complet) − P(M|sans w_i)|

    Retourne : [(mot, score), ...]
    """
    words = cleaned.split()
    if not words:
        return []

    masked_texts = []
    for i in range(len(words)):
        remaining = [w for j, w in enumerate(words) if j != i]
        masked_texts.append(' '.join(remaining) if remaining else '')

    seqs  = [text_to_sequence(t) for t in masked_texts]
    X     = _pad(seqs)
    probs = _predict_batch(X)

    return [(w, float(abs(original_prob - probs[i])))
            for i, w in enumerate(words)]


# ═══════════════════════════════
# INTERPRÉTABILITÉ PAR CARACTÈRE
# ═══════════════════════════════
def char_importance(cleaned: str, original_prob: float) -> list:
    """
    Importance de chaque CARACTÈRE par masquage batch.
    Retourne : [(char, score), ...]
    """
    chars = list(cleaned[:max_len])
    if not chars:
        return []

    base_seq = text_to_sequence(cleaned)
    masked_seqs = []
    for i in range(len(chars)):
        masked = base_seq.copy()
        masked[i] = 0   # masque le caractère par le token PAD
        masked_seqs.append(masked)

    X     = _pad(masked_seqs)
    probs = _predict_batch(X)

    return [(c, float(abs(original_prob - probs[i])))
            for i, c in enumerate(chars)]


# ═══════════════════════════════
# RENDU HTML
# ═══════════════════════════════
def render_word_html(word_scores: list, genre: str) -> str:
    """Mots colorés proportionnellement à leur impact sur la prédiction."""
    if not word_scores:
        return ""

    max_imp = max(s for _, s in word_scores) or 1.0
    color   = "220,80,50" if genre == "Masculin" else "50,100,220"

    html = (
        "<div style='margin-top:8px; font-family:monospace;'>"
        "<p style='color:#555; margin-bottom:6px;'>"
        "<strong>Importance par mot</strong> - "
        "couleur intense = mot déterminant pour la prédiction</p>"
        "<div style='font-size:18px; line-height:2.6;'>"
    )
    for word, score in word_scores:
        alpha = 0.12 + 0.82 * (score / max_imp)
        pct   = f"{score / max_imp * 100:.0f} %"
        html += (
            f"<span title='influence : {pct}' "
            f"style='background:rgba({color},{alpha:.2f}); "
            f"border:1px solid rgba({color},{min(alpha+0.2,1):.2f}); "
            f"padding:4px 10px; margin:3px; border-radius:5px; "
            f"color:{'white' if alpha > 0.5 else '#222'}; "
            f"font-weight:{'bold' if alpha > 0.5 else 'normal'};'>"
            f"{word}</span> "
        )
    html += "</div></div>"
    return html


def render_char_html(char_scores: list) -> str:
    """Caractères colorés selon leur impact (détail fin)."""
    if not char_scores:
        return ""

    max_imp = max(s for _, s in char_scores) or 1.0
    html = (
        "<div style='margin-top:12px; font-family:monospace;'>"
        "<p style='color:#555; margin-bottom:6px;'>"
        "<strong>Détail : importance par caractère</strong></p>"
        "<div style='font-size:16px; line-height:2.4; word-wrap:break-word;'>"
    )
    for char, score in char_scores:
        alpha = 0.10 + 0.85 * (score / max_imp)
        bg    = f"rgba(120,80,200,{alpha:.2f})"
        html += (
            f"<span title='{score:.4f}' "
            f"style='background:{bg}; padding:3px 5px; margin:1px; "
            f"border-radius:3px; "
            f"color:{'white' if alpha > 0.45 else '#222'};'>"
            f"{'&nbsp;' if char == ' ' else char}</span>"
        )
    html += "</div></div>"
    return html


# ═══════════════════════════════
# PRÉDICTION PRINCIPALE
# ═══════════════════════════════
def predict_single_input(full_name_input: str):
    """
    Pipeline complet :
    saisie → nettoyage → prédiction ONNX (ou Keras) → importance → HTML

    Accepte NOM PRÉNOMS ou PRÉNOMS NOM indifféremment.
    """
    if not full_name_input or not full_name_input.strip():
        return "Veuillez entrer un nom.", "-", "", "", None

    cleaned = clean_text(full_name_input)
    if not cleaned:
        return "Aucun caractère valide après nettoyage.", "-", "", "", None

    # ── Prédiction ──────────────────────────────────────────────
    prob_m     = predict_proba(cleaned)
    genre_pred = "Masculin" if prob_m > 0.5 else "Féminin"
    confidence = prob_m if prob_m > 0.5 else 1 - prob_m
    conf_str   = f"{confidence:.2%}"

    # ── Importance par mot ───────────────────────────────────────
    w_scores = word_importance(cleaned, prob_m)

    # ── Importance par caractère ─────────────────────────────────
    c_scores = char_importance(cleaned, prob_m)

    # ── Rendu HTML ───────────────────────────────────────────────
    html_word = render_word_html(w_scores, genre_pred)
    html_char = render_char_html(c_scores)

    top_mot = max(w_scores, key=lambda x: x[1])[0] if w_scores else "-"
    engine  = "ONNX ⚡" if USE_ONNX else "Keras 🔁"
    cleaned_info = (
        f" **Nom traité :** `{cleaned}`  \n"
        f" **Mot le + influent :** `{top_mot}`  \n"
        f" **Moteur d'inférence :** {engine}"
    )

    html_full = html_word + "<hr style='margin:10px 0; opacity:0.3'/>" + html_char

    return genre_pred, conf_str, cleaned_info, html_full, None


# ═══════════════════════════════
# CORRECTION
# ═══════════════════════════════
def save_correction(nom_complet, prediction_modele,
                    correction_utilisateur, confiance):
    if not correction_utilisateur:
        return "⚠ Sélectionnez le genre correct avant d'enregistrer."
    row = {
        'nom_complet':       nom_complet,
        'genre_reel':        correction_utilisateur,
        'prediction_modele': prediction_modele,
        'confiance_modele':  confiance,
        'date_correction':   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df_new = pd.DataFrame([row])
    header = not os.path.exists(CORRECTIONS_FILE)
    df_new.to_csv(CORRECTIONS_FILE, mode='a', header=header, index=False)
    return (
        f"✅ Correction enregistrée - "
        f"'{nom_complet}' marqué comme '{correction_utilisateur}'."
    )


# ═══════════════════════════════
# INTERFACE GRADIO
# ═══════════════════════════════
engine_label = "ONNX ⚡" if USE_ONNX else "Keras 🔁"

with gr.Blocks(theme=gr.themes.Soft(),
               title="Reconnaissance Genre") as demo:

    gr.Markdown("# 🇨🇮 Reconnaissance de Genre - Noms Locaux Ivoiriens")
    gr.Markdown(
        f"Saisissez le **nom complet** (NOM PRÉNOMS ou PRÉNOMS NOM) - "
        f"moteur : **{engine_label}**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_name = gr.Textbox(
                label="Nom complet",
                placeholder="Ex : MARIE KOUASSI  ou  KOUASSI MARIE",
                lines=1
            )
            btn_predict = gr.Button("🔍 Prédire", variant="primary")

        with gr.Column(scale=1):
            out_genre   = gr.Textbox(label="Genre Prédit")
            out_conf    = gr.Textbox(label="Confiance")
            out_cleaned = gr.Markdown(label="Nom traité / mot clé")

    gr.Markdown("### 📊 Analyse d'importance")
    out_html = gr.HTML(label="Interprétabilité")

    with gr.Accordion("🔧 Corriger une mauvaise prédiction", open=False):
        gr.Markdown(
            "Si le genre prédit est incorrect, sélectionnez le genre réel "
            "puis enregistrez. Ces données serviront au fine-tuning futur."
        )
        with gr.Row():
            radio_corr = gr.Radio(
                choices=["Masculin", "Féminin"],
                label="Genre Réel", value=None
            )
            btn_save = gr.Button("💾 Enregistrer", variant="secondary")
        out_status = gr.Textbox(label="Statut", interactive=False)

    # ── Liaisons ────────────────────────────────────────────────
    btn_predict.click(
        fn=predict_single_input,
        inputs=[input_name],
        outputs=[out_genre, out_conf, out_cleaned, out_html, radio_corr]
    )
    btn_save.click(
        fn=save_correction,
        inputs=[input_name, out_genre, radio_corr, out_conf],
        outputs=[out_status]
    )

if __name__ == "__main__":
    print(">>> Lancement de l'interface Gradio...")
    # Accès protégé par mot de passe si le secret APP_PASSWORD est défini
    auth = None
    if os.environ.get("APP_PASSWORD"):
        auth = (os.environ.get("APP_USER", "jury"), os.environ["APP_PASSWORD"])
    # Hébergeur (Render) : écouter sur toutes les interfaces, au port fourni
    launch_kwargs = {}
    if os.environ.get("PORT"):
        launch_kwargs = dict(server_name="0.0.0.0",
                             server_port=int(os.environ["PORT"]))
    demo.launch(share=False, auth=auth, **launch_kwargs)
