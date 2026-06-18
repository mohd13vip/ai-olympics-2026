#!/usr/bin/env python3
"""Build local hidden-domain evaluation suites for the Game 8 Final System.

The Phase 2 boss test ships material the canonical pair manifest has never
seen, which silences the relation features. These suites reproduce that
regime locally by copying validation images under fresh file names (fresh
normalized keys -> relation_known_both = 0) so candidate system changes can
be compared on more than the official validation split:

  sim_suite       500 untouched validation pairs, new image names, original
                  labels. Hidden-regime sanity check.
  mismatch_suite  500 REAL validation rows; half keep their own caption
                  (label real), half get a caption from a different real row
                  (label fake). Only semantic image-text agreement can solve
                  the fake half.
  stress_suite    500 validation pairs, original labels, images degraded
                  (gaussian blur / JPEG q30 / down-up resample) to mimic the
                  lower-quality imagery of a stress test.

Usage:  python make_eval_suites.py
Outputs CSV + image folder per suite under _v2_staging\\eval_suites.
Deterministic: seed 2026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

SEED = 2026
SUITE_ROWS = 500
PROJECT = Path(__file__).resolve().parent.parent
VALIDATION_CSV = (
    PROJECT
    / "_v2_staging"
    / "Final_System"
    / "reference_data"
    / "game8_validation.csv"
)
IMAGES_ROOT = (
    PROJECT / "AI_Olympics_2026_Student_Release_v1" / "data" / "images"
)
OUT_ROOT = PROJECT / "_v2_staging" / "eval_suites"


def source_image(relative: str) -> Path:
    path = IMAGES_ROOT / Path(relative).name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def derangement(count: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation of range(count) with no fixed points."""
    while True:
        permutation = rng.permutation(count)
        if not np.any(permutation == np.arange(count)):
            return permutation


def degrade(image: Image.Image, mode: int, rng: np.random.Generator) -> Image.Image:
    image = image.convert("RGB")
    if mode == 0:
        return image.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(1.2, 2.0)))
        )
    if mode == 1:
        return image  # saved at JPEG quality 30 by the caller
    small = image.resize(
        (max(1, image.width // 2), max(1, image.height // 2)), Image.BILINEAR
    )
    return small.resize((image.width, image.height), Image.BILINEAR)


def write_suite(name: str, frame: pd.DataFrame) -> None:
    out_csv = OUT_ROOT / f"{name}.csv"
    frame.to_csv(out_csv, index=False)
    counts = frame["label"].value_counts().to_dict()
    print(f"{name}: {len(frame)} rows {counts} -> {out_csv}")


def build_sim(validation: pd.DataFrame, rng: np.random.Generator) -> None:
    images_dir = OUT_ROOT / "sim_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        validation.groupby("label")
        .apply(
            lambda grp: grp.sample(SUITE_ROWS // 2, random_state=SEED),
            include_groups=False,
        )
        .reset_index()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    records = []
    for index, row in rows.iterrows():
        new_name = f"sim_{index:04d}.jpg"
        source = source_image(row["image_path"])
        (images_dir / new_name).write_bytes(source.read_bytes())
        records.append(
            {
                "sample_id": f"sim__{index:04d}",
                "image_path": new_name,
                "text": row["text"],
                "label": row["label"],
            }
        )
    write_suite("sim_suite", pd.DataFrame(records))


def build_mismatch(validation: pd.DataFrame, rng: np.random.Generator) -> None:
    images_dir = OUT_ROOT / "mismatch_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    real = (
        validation[validation["label"] == "real"]
        .sample(SUITE_ROWS, random_state=SEED)
        .reset_index(drop=True)
    )
    half = SUITE_ROWS // 2
    swapped = real.iloc[half:].reset_index(drop=True)
    shuffled_texts = swapped["text"].to_numpy()[
        derangement(len(swapped), rng)
    ]
    records = []
    for index, row in real.iterrows():
        new_name = f"mm_{index:04d}.jpg"
        source = source_image(row["image_path"])
        (images_dir / new_name).write_bytes(source.read_bytes())
        if index < half:
            text, label = row["text"], "real"
        else:
            text, label = shuffled_texts[index - half], "fake"
        records.append(
            {
                "sample_id": f"mm__{index:04d}",
                "image_path": new_name,
                "text": text,
                "label": label,
            }
        )
    write_suite("mismatch_suite", pd.DataFrame(records))


def build_stress(validation: pd.DataFrame, rng: np.random.Generator) -> None:
    images_dir = OUT_ROOT / "stress_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        validation.groupby("label")
        .apply(
            lambda grp: grp.sample(SUITE_ROWS // 2, random_state=SEED + 1),
            include_groups=False,
        )
        .reset_index()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    records = []
    for index, row in rows.iterrows():
        new_name = f"st_{index:04d}.jpg"
        mode = index % 3
        with Image.open(source_image(row["image_path"])) as image:
            damaged = degrade(image, mode, rng)
            quality = 30 if mode == 1 else 85
            damaged.save(images_dir / new_name, "JPEG", quality=quality)
        records.append(
            {
                "sample_id": f"st__{index:04d}",
                "image_path": new_name,
                "text": row["text"],
                "label": row["label"],
            }
        )
    write_suite("stress_suite", pd.DataFrame(records))


def main() -> None:
    rng = np.random.default_rng(SEED)
    validation = pd.read_csv(VALIDATION_CSV)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_sim(validation, rng)
    build_mismatch(validation, rng)
    build_stress(validation, rng)


if __name__ == "__main__":
    main()
