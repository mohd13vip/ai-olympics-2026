#!/usr/bin/env python3
"""Strip Final_System score caches down to reference + public-test keys.

Benchmark and evaluation runs append their scores to the cache CSVs. Rows
keyed by material outside the shipped reference data do not belong in the
submission: they bloat the folder and a hidden-test file that happened to
reuse one of those names would silently inherit a stale score instead of
being scored fresh.

Usage:  python clean_score_cache.py <Final_System dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def normalize_image_path(value: object) -> str:
    return f"data/images/{Path(str(value)).name}"


def model_text(frame: pd.DataFrame) -> pd.Series:
    raw = frame["text"]
    missing = raw.isna() | raw.fillna("").astype(str).str.strip().eq("")
    text = raw.fillna("").astype(str)
    text[missing] = "MISSING_TEXT"
    return text


def main() -> None:
    system_root = Path(sys.argv[1]).resolve()
    reference = system_root / "reference_data"
    cache = system_root / "score_cache"
    public_test = Path(
        r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
        r"\AI_Olympics_2026_Game8_Release_v1\data\game8_public_test.csv"
    )
    frames = [
        pd.read_csv(reference / "game8_train.csv"),
        pd.read_csv(reference / "game8_validation.csv"),
        pd.read_csv(public_test),
    ]
    allowed_images: set[str] = set()
    allowed_texts: set[str] = set()
    allowed_pairs: set[tuple[str, str]] = set()
    for frame in frames:
        images = frame["image_path"].map(normalize_image_path)
        texts = model_text(frame)
        allowed_images.update(images)
        allowed_texts.update(texts)
        allowed_pairs.update(zip(images, texts))

    image_cache = pd.read_csv(cache / "image_scores.csv")
    kept = image_cache[
        image_cache["normalized_image_path"].isin(allowed_images)
    ]
    print(f"image_scores: {len(image_cache)} -> {len(kept)}")
    missing = allowed_images - set(kept["normalized_image_path"])
    if missing:
        raise SystemExit(f"image cache lost coverage: {sorted(missing)[:5]}")
    kept.to_csv(cache / "image_scores.csv", index=False)

    text_cache = pd.read_csv(cache / "text_scores.csv")
    text_cache["model_text"] = text_cache["model_text"].astype(str)
    kept = text_cache[text_cache["model_text"].isin(allowed_texts)]
    print(f"text_scores: {len(text_cache)} -> {len(kept)}")
    missing_texts = allowed_texts - set(kept["model_text"])
    if missing_texts:
        raise SystemExit(
            f"text cache lost coverage: {len(missing_texts)} texts"
        )
    kept.to_csv(cache / "text_scores.csv", index=False)

    clip_cache = pd.read_csv(cache / "clip_scores.csv")
    clip_cache["model_text"] = clip_cache["model_text"].astype(str)
    pair_mask = [
        (image, text) in allowed_pairs
        for image, text in zip(
            clip_cache["normalized_image_path"], clip_cache["model_text"]
        )
    ]
    kept = clip_cache[pair_mask]
    print(f"clip_scores: {len(clip_cache)} -> {len(kept)}")
    missing_pairs = allowed_pairs - set(
        zip(kept["normalized_image_path"], kept["model_text"])
    )
    if missing_pairs:
        raise SystemExit(
            f"clip cache lost coverage: {len(missing_pairs)} pairs"
        )
    kept.to_csv(cache / "clip_scores.csv", index=False)
    print("coverage check passed: all reference+public keys retained")


if __name__ == "__main__":
    main()
