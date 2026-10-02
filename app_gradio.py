"""
app_gradio.py - Interface Gradio reconnaissance de genre - INP-HB STIC

Le modèle servi est le paquet « production » du registre de modèles (registre.ipynb) :
un dossier contenant modele.json et les fichiers du modèle, chargé par adaptateurs.py.
Changer de modèle ne demande donc aucune modification de ce fichier.

  - Interprétabilité par masquage des mots, commune à tous les modèles
  - Mode « incertain » : sous le seuil de confiance, la prédiction est à vérifier
"""

import os, datetime
import pandas as pd
import gradio as gr
from adaptateurs import charger, clean_text, FICHE

# ═══════════════════════════════
# CHARGEMENT
# ═══════════════════════════════
PAQUET_DIR       = os.environ.get("PAQUET_DIR", "production")
CORRECTIONS_FILE = "corrections_dataset.csv"

# En ligne, le paquet n'est pas dans le dépôt : il est téléchargé depuis le dossier
# production/ du dépôt modèle PRIVÉ (variables HF_MODEL_REPO et HF_TOKEN).
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")

print(">>> Chargement du modèle de production...")

if HF_MODEL_REPO and not os.path.exists(os.path.join(PAQUET_DIR, FICHE)):
    from huggingface_hub import snapshot_download
    snapshot_download(HF_MODEL_REPO, allow_patterns=["production/*"], local_dir=".",
                      token=os.environ.get("HF_TOKEN"))
    PAQUET_DIR = "production"
    print(f">>> Paquet téléchargé depuis {HF_MODEL_REPO}/production")

try:
    modele = charger(PAQUET_DIR)
except FileNotFoundError as e:
    print(f">>> ERREUR : {e}\nPromouvez d'abord un modèle avec registre.ipynb.")
    raise SystemExit(1)

SEUIL_INCERTAIN = float(os.environ.get("SEUIL_INCERTAIN",
                                       modele.fiche.get("seuil_incertain", 0.70)))
version = modele.fiche.get("version_registre")
ENGINE = modele.nom + (f" (version {version} du registre)" if version else "")
print(f">>> Modèle chargé : {ENGINE} | seuil de confiance {SEUIL_INCERTAIN:.0%}")


def predict_texts(texts: list):
    """P(Masculin) pour une liste de textes nettoyés (inférence batch)."""
    return modele.predict_proba(texts)


def predict_proba(text: str) -> float:
    """Retourne P(Masculin) pour un texte nettoyé."""
    return float(predict_texts([text])[0])


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

    probs = predict_texts(masked_texts)

    return [(w, float(abs(original_prob - probs[i])))
            for i, w in enumerate(words)]


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


# ═══════════════════════════════
# PRÉDICTION PRINCIPALE
# ═══════════════════════════════
def predict_single_input(full_name_input: str):
    """
    Pipeline complet :
    saisie → nettoyage → prédiction → importance → HTML

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

    # ── Mode « incertain » : sous le seuil, la prédiction est à vérifier ──
    affichage = (f"Incertain, à vérifier (plutôt {genre_pred})"
                 if confidence < SEUIL_INCERTAIN else genre_pred)

    # ── Importance par mot ───────────────────────────────────────
    w_scores = word_importance(cleaned, prob_m)

    # ── Rendu HTML ───────────────────────────────────────────────
    html_word = render_word_html(w_scores, genre_pred)

    top_mot = max(w_scores, key=lambda x: x[1])[0] if w_scores else "-"
    cleaned_info = (
        f" **Nom traité :** `{cleaned}`  \n"
        f" **Mot le + influent :** `{top_mot}`  \n"
        f" **Modèle :** {ENGINE}"
    )

    html_full = html_word

    return affichage, conf_str, cleaned_info, html_full, None


# ═══════════════════════════════
# CORRECTION
# ═══════════════════════════════
def save_correction(nom_complet, prediction_modele,
                    correction_utilisateur, confiance):
    if not correction_utilisateur:
        return "Sélectionnez le genre correct avant d'enregistrer."
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
        f"Correction enregistrée : "
        f"'{nom_complet}' marqué comme '{correction_utilisateur}'."
    )


# ═══════════════════════════════
# INTERFACE GRADIO
# ═══════════════════════════════
with gr.Blocks(theme=gr.themes.Soft(),
               title="Reconnaissance Genre") as demo:

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.Markdown("# Reconnaissance de Genre - Noms Locaux Ivoiriens")
        # Déconnexion : route /logout de Gradio, utile seulement si l'accès est protégé
        if os.environ.get("APP_PASSWORD"):
            with gr.Column(scale=1, min_width=140):
                gr.Button("Déconnexion", link="/logout", variant="secondary", size="sm")
    gr.Markdown("Saisissez le **nom complet**")

    with gr.Row():
        with gr.Column(scale=1):
            input_name = gr.Textbox(
                label="Nom complet",
                placeholder="Ex : ASSIE EMMANUEL  ou  EMMANUEL ASSIE",
                lines=1
            )
            btn_predict = gr.Button("Prédire", variant="primary")

        with gr.Column(scale=1):
            out_genre   = gr.Textbox(label="Genre Prédit")
            out_conf    = gr.Textbox(label="Confiance")
            out_cleaned = gr.Markdown(label="Nom traité / mot clé")

    gr.Markdown("### Analyse d'importance")
    out_html = gr.HTML(label="Interprétabilité")

    with gr.Accordion("Corriger une mauvaise prédiction", open=False):
        gr.Markdown(
            "Si le genre prédit est incorrect, sélectionnez le genre réel "
            "puis enregistrez. Ces données serviront au fine-tuning futur."
        )
        with gr.Row():
            radio_corr = gr.Radio(
                choices=["Masculin", "Féminin"],
                label="Genre Réel", value=None
            )
            btn_save = gr.Button("Enregistrer", variant="secondary")
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
    # File d'attente : au plus 2 requêtes traitées en même temps (mémoire limitée en ligne)
    demo.queue(default_concurrency_limit=2, max_size=32)
    demo.launch(share=False, auth=auth, **launch_kwargs)
