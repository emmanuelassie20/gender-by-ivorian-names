"""
adaptateurs.py - Interface commune aux modèles de reconnaissance du genre.

Un modèle est distribué sous forme de « paquet » : un dossier contenant modele.json
(type, fichiers, fiche descriptive) et les fichiers du modèle. charger(dossier) renvoie
un adaptateur qui expose la même interface quel que soit le modèle :

  predict_proba(textes)            -> P(Masculin) pour des noms déjà nettoyés
  nb_caracteres(texte)             -> nombre de caractères réellement lus par le modèle
  proba_caracteres_masques(texte)  -> P(Masculin) en masquant chaque caractère tour à tour
                                      (interprétabilité par caractère)

Types pris en charge :
  lr       régression logistique (pipeline scikit-learn, joblib)
  v4_onnx  CNN + BiLSTM + MHA sur la chaîne complète (ONNX)
  v5_onnx  modèle par mot (ONNX)

L'app n'a besoin ni de TensorFlow ni de MLflow pour utiliser un paquet.
"""

import json, os, re, unicodedata
import numpy as np

FICHE = "modele.json"


def clean_text(text: str) -> str:
    """Normalise un nom propre : minuscules, ASCII, espaces (commun à tous les modèles)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r"['\-]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Adaptateur:
    """Base commune : fiche du paquet et masquage par défaut (caractère retiré)."""

    def __init__(self, dossier: str, fiche: dict):
        self.dossier, self.fiche = dossier, fiche

    @property
    def nom(self) -> str:
        return self.fiche.get('nom', self.fiche['type'])

    def chemin(self, cle: str) -> str:
        return os.path.join(self.dossier, self.fiche['fichiers'][cle])

    def predict_proba(self, textes) -> np.ndarray:
        raise NotImplementedError

    def nb_caracteres(self, texte: str) -> int:
        return len(texte)

    def proba_caracteres_masques(self, texte: str) -> np.ndarray:
        """Par défaut, le caractère i est retiré du texte."""
        return self.predict_proba([texte[:i] + texte[i + 1:]
                                   for i in range(self.nb_caracteres(texte))])


class AdaptateurLR(Adaptateur):
    def __init__(self, dossier, fiche):
        super().__init__(dossier, fiche)
        import joblib
        self.pipeline = joblib.load(self.chemin('pipeline'))

    def predict_proba(self, textes):
        return self.pipeline.predict_proba(list(textes))[:, 1]


class _AdaptateurOnnx(Adaptateur):
    def __init__(self, dossier, fiche):
        super().__init__(dossier, fiche)
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.chemin('onnx'),
                                            providers=['CPUExecutionProvider'])
        with open(self.chemin('vocab'), encoding='utf-8') as f:
            self._init_vocab(json.load(f))

    def _init_vocab(self, vocab: dict):
        raise NotImplementedError

    def _run(self, feeds: dict) -> np.ndarray:
        return self.session.run(None, feeds)[0].ravel()


class AdaptateurV4Onnx(_AdaptateurOnnx):
    """Chaîne complète : séquence de caractères, complétée à max_len (troncature au début)."""

    def _init_vocab(self, vocab):
        self.c2i, self.max_len = vocab['char_to_index'], int(vocab['max_len'])
        self.entree = self.session.get_inputs()[0].name

    def _seq(self, texte):
        return [self.c2i.get(c, self.c2i['<UNK>']) for c in texte]

    def _pad(self, seqs):
        X = np.zeros((len(seqs), self.max_len), dtype=np.int32)
        for i, seq in enumerate(seqs):
            seq = seq[-self.max_len:]
            X[i, :len(seq)] = seq
        return X

    def predict_proba(self, textes):
        return self._run({self.entree: self._pad([self._seq(t) for t in textes])})

    def nb_caracteres(self, texte):
        return min(len(texte), self.max_len)

    def proba_caracteres_masques(self, texte):
        """Le caractère i est remplacé par le jeton PAD (position conservée)."""
        base, seqs = self._seq(texte), []
        for i in range(self.nb_caracteres(texte)):
            masque = base.copy()
            masque[i] = 0
            seqs.append(masque)
        return self._run({self.entree: self._pad(seqs)})


class AdaptateurV5Onnx(_AdaptateurOnnx):
    """Par mot : max_tokens mots × max_chars caractères, plus l'identifiant de chaque mot."""

    def _init_vocab(self, vocab):
        self.c2i, self.w2i = vocab['char_to_index'], vocab['word_to_index']
        self.max_tokens, self.max_chars = int(vocab['max_tokens']), int(vocab['max_chars'])

    def predict_proba(self, textes):
        textes = list(textes)
        C = np.zeros((len(textes), self.max_tokens, self.max_chars), dtype=np.int32)
        W = np.zeros((len(textes), self.max_tokens), dtype=np.int32)
        for i, t in enumerate(textes):
            for j, w in enumerate(t.split()[:self.max_tokens]):
                W[i, j] = self.w2i.get(w, 1)
                seq = [self.c2i.get(c, 1) for c in w[:self.max_chars]]
                C[i, j, :len(seq)] = seq
        return self._run({'chars': C, 'words': W})


TYPES = {'lr': AdaptateurLR, 'v4_onnx': AdaptateurV4Onnx, 'v5_onnx': AdaptateurV5Onnx}


def charger(dossier: str) -> Adaptateur:
    """Charge le paquet d'un modèle à partir de son dossier."""
    with open(os.path.join(dossier, FICHE), encoding='utf-8') as f:
        fiche = json.load(f)
    if fiche['type'] not in TYPES:
        raise ValueError(f"type de modèle inconnu : {fiche['type']}")
    return TYPES[fiche['type']](dossier, fiche)
