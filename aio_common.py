"""aio_common.py - shared core for the AI Olympics kit.
Schema everywhere: sample_id, image_path, text, label
Set AIO_ROOT env var if the package moves."""
import os, re, time, random
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ROOT = (r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
                r"\AI_Olympics_2026_Student_Release_v1")
ROOT = Path(os.environ.get("AIO_ROOT", DEFAULT_ROOT))
SEED = 42
TEAM = os.environ.get("AIO_TEAM", "Ded_Sec")


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_image_path(rel):
    rel = str(rel)
    for c in (ROOT / rel, Path(rel)):
        if c.exists():
            return c
    raise FileNotFoundError(f"Image not found: {rel} (ROOT={ROOT})")


def load_pair(train_csv, val_csv):
    tr = pd.read_csv(ROOT / train_csv if not Path(train_csv).exists() else train_csv)
    va = pd.read_csv(ROOT / val_csv if not Path(val_csv).exists() else val_csv)
    for df, n in ((tr, "train"), (va, "val")):
        missing = {"sample_id", "image_path", "text", "label"} - set(df.columns)
        if missing:
            raise SystemExit(f"{n} csv missing columns {missing}; has {list(df.columns)}")
    return tr, va


def safe_text(v):
    return "" if pd.isna(v) else str(v)


def norm_text(v):
    s = safe_text(v).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "precision": round(precision_score(y_true, y_pred, average="macro",
                                               zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="macro",
                                         zero_division=0), 4)}


def tfidf_baseline():
    """The exact baseline from the official Game 1 starter."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), min_df=2,
                                  max_features=40000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=SEED)),
    ])


def eval_baseline(tr, va):
    m = tfidf_baseline()
    m.fit(tr["text"].map(safe_text), tr["label"])
    pred = m.predict(va["text"].map(safe_text))
    return metrics(va["label"], pred), m


class Timer:
    def __enter__(self):
        self.t = time.time(); return self
    def __exit__(self, *a):
        self.sec = round(time.time() - self.t, 2)


def out_dir(name):
    d = Path(name); d.mkdir(parents=True, exist_ok=True); return d
