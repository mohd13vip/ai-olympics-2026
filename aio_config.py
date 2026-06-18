"""Shared config for the AI Olympics toolkit.

Official competition schema (cheat sheet / all game CSVs):
    sample_id, image_path, text, label        label in {real, fake}

Edit DEFAULT_ROOT if you move the folder, or set the AIO_ROOT environment
variable to override without editing.
"""
import os
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = (r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
                r"\AI_Olympics_2026_Student_Release_v1")
ROOT = Path(os.environ.get("AIO_ROOT", DEFAULT_ROOT))
IMAGES_DIR = ROOT / "data" / "images"
TEAM = os.environ.get("AIO_TEAM", "Ded_Sec")
SEED = 42

# filename prefix -> split patterns (first match wins; fallbacks cover naming drift)
SPLITS = {
    "train": ["train__*", "train_*", "train*"],
    "val": ["validation__*", "validation_*", "validation*", "val_*"],
    "test": ["test1_*", "test_*", "test*"],
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def seed_all(seed=SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------- image paths
def list_split(split: str):
    """All image paths in data/images belonging to a split, by filename pattern."""
    if not IMAGES_DIR.exists():
        return []
    for pat in SPLITS[split]:
        hits = sorted(p for p in IMAGES_DIR.glob(pat)
                      if p.suffix.lower() in IMG_EXTS)
        if hits:
            return hits
    return []


def split_of(name: str):
    for s, pats in SPLITS.items():
        if any(fnmatch(name, pat) for pat in pats):
            return s
    return None


@lru_cache(maxsize=1)
def _basename_index():
    """basename -> full path for every image anywhere under ROOT (built once).
    Lets us resolve any image_path stored in a CSV, including files that live in
    games/game2_data_reconnaissance/modified_images or game1 injected_images."""
    idx = {}
    if ROOT.exists():
        for p in ROOT.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                idx.setdefault(p.name, p)
    return idx


def resolve_path(rel):
    """Resolve an image_path value from a CSV to an existing file.
    Tries the literal path, then relative to ROOT and common bases, then a
    basename lookup across the whole release."""
    rel = str(rel).strip().replace("\\", "/")
    p = Path(rel)
    candidates = [p, ROOT / rel, ROOT / "data" / rel, IMAGES_DIR / rel,
                  IMAGES_DIR / p.name]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    hit = _basename_index().get(p.name)
    if hit:
        return hit
    raise FileNotFoundError(f"Image not found: {rel} (ROOT={ROOT})")


# --------------------------------------------------------------- CSV columns
ID_COL_GUESSES = ["sample_id", "id", "uid", "sample", "row_id"]
IMG_COL_GUESSES = ["image_path", "image", "image_name", "img_path", "filename",
                   "file_name", "file", "img", "image_id", "image_file", "path"]
TEXT_COL_GUESSES = ["text", "caption", "title", "description", "content", "tweet"]
LABEL_COL_GUESSES = ["label", "target", "class", "y", "is_fake", "fake",
                     "real_or_fake", "category", "ground_truth"]


def _pick(df, guesses, substr=None):
    low = {c.lower(): c for c in df.columns}
    for g in guesses:
        if g in low:
            return low[g]
    if substr:
        for c in df.columns:
            if any(s in c.lower() for s in substr):
                return c
    return None


def autodetect_cols(df: pd.DataFrame, img_col=None, label_col=None):
    """(img_col, label_col) - now recognises the official 'image_path' column."""
    if img_col is None:
        img_col = _pick(df, IMG_COL_GUESSES, substr=("image", "img", "path"))
    if label_col is None:
        label_col = _pick(df, LABEL_COL_GUESSES, substr=("label", "class", "target"))
    return img_col, label_col


def detect_schema(df: pd.DataFrame, img_col=None, text_col=None,
                  label_col=None, id_col=None):
    """Detect all four schema columns. Any of them may come back None."""
    img_col, label_col = autodetect_cols(df, img_col, label_col)
    if text_col is None:
        text_col = _pick(df, TEXT_COL_GUESSES, substr=("text", "caption"))
    if id_col is None:
        id_col = _pick(df, ID_COL_GUESSES)
    return {"id": id_col, "image": img_col, "text": text_col, "label": label_col}


def load_split_csv(csv_path, **overrides):
    """Read a competition CSV and return (df, schema_dict).
    df keeps the original columns; schema maps logical names to real columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists() and (ROOT / csv_path).exists():
        csv_path = ROOT / csv_path
    df = pd.read_csv(csv_path)
    schema = detect_schema(df, **overrides)
    if schema["image"] is None and schema["text"] is None:
        raise SystemExit(f"{csv_path}: could not find an image or text column. "
                         f"Columns: {list(df.columns)}")
    return df, schema


def load_label_map(csv_path, img_col=None, label_col=None):
    """Returns ({basename: label}, sorted_class_list, img_col, label_col)."""
    df = pd.read_csv(csv_path)
    img_col, label_col = autodetect_cols(df, img_col, label_col)
    if not img_col or not label_col:
        raise SystemExit(f"Could not auto-detect columns in {csv_path}. "
                         f"Found {list(df.columns)} - pass --img-col / --label-col.")
    mapping = {Path(str(r[img_col])).name: str(r[label_col]) for _, r in df.iterrows()}
    classes = sorted(set(mapping.values()))
    return mapping, classes, img_col, label_col


# --------------------------------------------------------------- CSV guessing
def find_game_csvs(keyword=None):
    out = []
    for base in (ROOT / "games", ROOT / "data", ROOT):
        if base.exists():
            depth = base.rglob("*.csv") if base.name == "games" else base.glob("*.csv")
            out += [p for p in depth]
    out = sorted(set(out))
    if keyword:
        out = [p for p in out if keyword.lower() in p.name.lower()]
    return out


def _first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def guess_train_csv():
    """Prefer the team's processed file (Games 4+), then game CSVs."""
    return _first_existing(
        ROOT / "processed_train.csv",
        Path("processed_train.csv"),
        *find_game_csvs("processed_train"),
        *find_game_csvs("corrected_train"),
        *[p for p in find_game_csvs("train")],
        *find_game_csvs("label"),
    )


def guess_val_csv():
    return _first_existing(
        ROOT / "processed_validation.csv",
        Path("processed_validation.csv"),
        *find_game_csvs("processed_validation"),
        *find_game_csvs("corrected_validation"),
        *[p for p in find_game_csvs("validation")],
        *find_game_csvs("valid"),
    )
