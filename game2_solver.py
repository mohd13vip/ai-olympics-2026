#!/usr/bin/env python3
"""Generate the required AI Olympics 2026 Game 2 reconnaissance artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = ["sample_id", "image_path", "text", "label"]


def safe_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def build_text_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    text = output["text"].map(safe_text)
    length = text.str.len()
    letters = text.str.count(r"[A-Za-z]")
    output["safe_text"] = text
    output["text_length_chars"] = length
    output["text_length_words"] = text.str.split().str.len()
    output["is_empty_text"] = text.str.strip().eq("")
    output["symbol_count"] = text.str.count(r"[^A-Za-z0-9\s]")
    output["symbol_ratio"] = output["symbol_count"] / length.clip(lower=1)
    output["uppercase_count"] = text.str.count(r"[A-Z]")
    output["uppercase_ratio"] = output["uppercase_count"] / letters.clip(lower=1)
    output["digit_count"] = text.str.count(r"\d")
    output["digit_ratio"] = output["digit_count"] / length.clip(lower=1)
    output["url_count"] = text.str.count(r"https?://|www\.")
    output["repeated_punctuation"] = text.str.contains(
        r"(?:!{3,}|\?{3,}|\.{3,}|,{3,})", regex=True, na=False
    )
    return output


def resolve_image(root: Path, relative_path: object) -> Path:
    value = Path(str(relative_path))
    for candidate in (value, root / value):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {relative_path}")


def image_statistics(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
        exif_present = bool(image.getexif())
        gray = np.asarray(
            image.convert("L").resize((256, 256)), dtype=np.float32
        )
        rgb = np.asarray(
            image.convert("RGB").resize((128, 128)), dtype=np.float32
        )

    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    yy, xx = np.mgrid[0:256, 0:256]
    center = ((yy - 128) ** 2 + (xx - 128) ** 2) <= 32**2
    total_frequency = float(spectrum.sum()) + 1e-9
    red_green = rgb[..., 0] - rgb[..., 1]
    yellow_blue = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(
        np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
        + 0.3
        * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    )
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / max(height, 1),
        "file_size_kb": path.stat().st_size / 1024,
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "laplacian_variance": float(laplacian.var()),
        "high_frequency_ratio": float(
            spectrum[~center].sum() / total_frequency
        ),
        "colorfulness": colorfulness,
        "exif_present": exif_present,
    }


def build_image_statistics(
    frame: pd.DataFrame,
    root: Path,
    workers: int,
    cache: dict[Path, dict[str, object]],
) -> pd.DataFrame:
    paths = frame["image_path"].map(lambda value: resolve_image(root, value))
    missing_paths = list(dict.fromkeys(path for path in paths if path not in cache))
    if missing_paths:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(image_statistics, missing_paths))
        cache.update(dict(zip(missing_paths, results)))

    records = []
    for (_, row), path in zip(frame.iterrows(), paths):
        record = {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "label": row["label"],
            "image_source": (
                "modified"
                if "modified_images" in str(row["image_path"])
                else "shared"
            ),
        }
        record.update(cache[path])
        records.append(record)
    return pd.DataFrame(records)


def standardized_mean_difference(
    train: pd.Series, validation: pd.Series
) -> float:
    train = pd.to_numeric(train, errors="coerce").dropna()
    validation = pd.to_numeric(validation, errors="coerce").dropna()
    pooled = np.sqrt((train.var(ddof=1) + validation.var(ddof=1)) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((validation.mean() - train.mean()) / pooled)


def make_risk_table(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_text: pd.DataFrame,
    validation_text: pd.DataFrame,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
) -> pd.DataFrame:
    risks: list[dict[str, object]] = []

    def add(
        modality: str,
        pattern: str,
        evidence: str,
        split: str,
        affected: int,
        severity: str,
        action: str,
    ) -> None:
        risks.append(
            {
                "risk_id": f"R{len(risks) + 1:02d}",
                "modality": modality,
                "discovered_pattern": pattern,
                "evidence": evidence,
                "affected_split": split,
                "affected_samples": int(affected),
                "severity": severity,
                "recommended_next_action": action,
            }
        )

    train_counts = train["label"].value_counts()
    validation_counts = validation["label"].value_counts()
    train_gap = int(train_counts.max() - train_counts.min())
    validation_gap = int(validation_counts.max() - validation_counts.min())
    if train_gap:
        add(
            "label",
            "Training class imbalance",
            f"Train counts={train_counts.to_dict()}; validation counts={validation_counts.to_dict()}",
            "train",
            train_gap,
            "high",
            "Use stratification and test class-weighted loss; do not rebalance validation.",
        )
    if validation_gap:
        add(
            "label",
            "Validation class imbalance",
            f"Validation counts={validation_counts.to_dict()}",
            "validation",
            validation_gap,
            "medium",
            "Use macro-F1 and per-class metrics.",
        )

    for split, stats in (
        ("train", train_text),
        ("validation", validation_text),
    ):
        empty = int(stats["is_empty_text"].sum())
        if empty:
            by_label = (
                stats.loc[stats["is_empty_text"], "label"]
                .value_counts()
                .to_dict()
            )
            add(
                "text",
                "Missing or empty text",
                f"{empty} empty texts; label distribution={by_label}",
                split,
                empty,
                "high",
                "Preserve rows, add a missing-text indicator, and use an explicit placeholder token.",
            )

        extreme_symbols = int((stats["symbol_ratio"] > 0.20).sum())
        if extreme_symbols:
            add(
                "text",
                "High symbol ratio",
                f"{extreme_symbols} rows have symbol_ratio > 0.20",
                split,
                extreme_symbols,
                "medium",
                "Test conservative punctuation normalization while retaining an original-text copy.",
            )

        uppercase = int(
            (
                (stats["uppercase_ratio"] > 0.80)
                & (stats["text_length_chars"] >= 20)
            ).sum()
        )
        if uppercase:
            add(
                "text",
                "Mostly uppercase text",
                f"{uppercase} rows have uppercase_ratio > 0.80 and at least 20 characters",
                split,
                uppercase,
                "medium",
                "Test lowercase normalization and record whether capitalization carries label signal.",
            )

    modified_train = int((train_images["image_source"] == "modified").sum())
    modified_validation = int(
        (validation_images["image_source"] == "modified").sum()
    )
    if modified_train or modified_validation:
        modified = pd.concat(
            [
                train_images.loc[train_images["image_source"] == "modified"],
                validation_images.loc[
                    validation_images["image_source"] == "modified"
                ],
            ]
        )
        shared = pd.concat(
            [
                train_images.loc[train_images["image_source"] == "shared"],
                validation_images.loc[
                    validation_images["image_source"] == "shared"
                ],
            ]
        )
        evidence = (
            f"modified={len(modified)} (train={modified_train}, validation={modified_validation}); "
            f"median sharpness modified={modified['laplacian_variance'].median():.2f}, "
            f"shared={shared['laplacian_variance'].median():.2f}; "
            f"median size KB modified={modified['file_size_kb'].median():.1f}, "
            f"shared={shared['file_size_kb'].median():.1f}"
        )
        add(
            "image",
            "Altered images mixed with original-quality images",
            evidence,
            "both",
            len(modified),
            "high",
            "Audit alteration types and test quality-robust preprocessing without overwriting originals.",
        )

    all_images = pd.concat(
        [train_images, validation_images], ignore_index=True
    )
    low_resolution = int(
        ((all_images["width"] < 128) | (all_images["height"] < 128)).sum()
    )
    if low_resolution:
        add(
            "image",
            "Low-resolution images",
            f"{low_resolution} images have width or height below 128 pixels",
            "both",
            low_resolution,
            "high",
            "Use aspect-preserving resize with padding; avoid aggressive upscaling assumptions.",
        )

    extreme_aspect = int(
        (
            (all_images["aspect_ratio"] < 0.5)
            | (all_images["aspect_ratio"] > 2.0)
        ).sum()
    )
    if extreme_aspect:
        add(
            "image",
            "Extreme aspect ratios",
            f"{extreme_aspect} images fall outside the 0.5-2.0 aspect-ratio range",
            "both",
            extreme_aspect,
            "medium",
            "Use padding or center-crop experiments and compare information loss.",
        )

    low_sharpness_cutoff = float(
        all_images["laplacian_variance"].quantile(0.05)
    )
    low_sharpness = int(
        (all_images["laplacian_variance"] <= low_sharpness_cutoff).sum()
    )
    add(
        "image",
        "Very low sharpness tail",
        f"{low_sharpness} images are at or below the 5th-percentile sharpness cutoff ({low_sharpness_cutoff:.2f})",
        "both",
        low_sharpness,
        "medium",
        "Test mild sharpening only as a controlled experiment; retain a quality flag.",
    )

    extreme_brightness = int(
        (
            (all_images["brightness_mean"] < 30)
            | (all_images["brightness_mean"] > 225)
        ).sum()
    )
    if extreme_brightness:
        add(
            "image",
            "Extreme brightness",
            f"{extreme_brightness} images have mean brightness below 30 or above 225",
            "both",
            extreme_brightness,
            "medium",
            "Test bounded contrast normalization and compare against unchanged images.",
        )

    shift_sources = {
        "text_length_words": (train_text, validation_text, "text"),
        "symbol_ratio": (train_text, validation_text, "text"),
        "brightness_mean": (train_images, validation_images, "image"),
        "contrast_std": (train_images, validation_images, "image"),
        "laplacian_variance": (train_images, validation_images, "image"),
        "high_frequency_ratio": (train_images, validation_images, "image"),
    }
    for metric, (left, right, modality) in shift_sources.items():
        effect = standardized_mean_difference(left[metric], right[metric])
        if abs(effect) >= 0.20:
            add(
                modality,
                f"Train-validation shift in {metric}",
                f"Standardized mean difference={effect:.3f}; train mean={left[metric].mean():.3f}; validation mean={right[metric].mean():.3f}",
                "both",
                len(left) + len(right),
                "high" if abs(effect) >= 0.50 else "medium",
                "Use identical preprocessing and report split-specific distributions after each transformation.",
            )

    return pd.DataFrame(risks)


def build_notebook(
    path: Path,
    root: Path,
    team_name: str,
    risks: pd.DataFrame,
) -> None:
    def markdown(text: str) -> dict[str, object]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().splitlines()],
        }

    def code(text: str) -> dict[str, object]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().splitlines()],
        }

    high_risks = risks.loc[risks["severity"] == "high", "discovered_pattern"]
    conclusion = (
        "The dataset is not ready for modeling without controlled "
        "preprocessing experiments. The highest-priority risks are: "
        + "; ".join(high_risks.astype(str).tolist())
        + ". Images and texts require separate quality indicators, and train "
        "and validation must receive identical transformations. Game 3 should "
        "test conservative changes one at a time while preserving originals."
    )
    cells = [
        markdown(
            f"""# Game 2 - The Data Reconnaissance Mission

**Team:** {team_name}

This notebook loads the full measured statistics, compares Train and
Validation, visualizes major distributions, and prioritizes risks without
changing the source data."""
        ),
        code(
            f"""from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("AIO_ROOT", r"{root}"))
OUTPUT_DIR = Path.cwd()

train_text = pd.read_csv(OUTPUT_DIR / "game2_train_text_statistics.csv")
validation_text = pd.read_csv(OUTPUT_DIR / "game2_validation_text_statistics.csv")
train_images = pd.read_csv(OUTPUT_DIR / "game2_train_image_statistics.csv")
validation_images = pd.read_csv(OUTPUT_DIR / "game2_validation_image_statistics.csv")
risks = pd.read_csv(OUTPUT_DIR / "game2_risk_priority_table.csv")

print("Train rows:", len(train_text))
print("Validation rows:", len(validation_text))
display(risks)"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train_text["label"].value_counts().plot.bar(ax=axes[0], title="Train labels")
validation_text["label"].value_counts().plot.bar(
    ax=axes[1], title="Validation labels"
)
plt.tight_layout()
plt.show()

display(
    pd.DataFrame({
        "train": train_text[[
            "text_length_chars", "text_length_words", "symbol_ratio"
        ]].mean(),
        "validation": validation_text[[
            "text_length_chars", "text_length_words", "symbol_ratio"
        ]].mean(),
    })
)
print("Empty train texts:", int(train_text["is_empty_text"].sum()))
print("Empty validation texts:", int(validation_text["is_empty_text"].sum()))"""
        ),
        code(
            """metrics = [
    "brightness_mean",
    "contrast_std",
    "laplacian_variance",
    "high_frequency_ratio",
]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, metric in zip(axes.flat, metrics):
    train_images[metric].plot.hist(
        bins=50, alpha=0.5, density=True, ax=ax, label="train"
    )
    validation_images[metric].plot.hist(
        bins=50, alpha=0.5, density=True, ax=ax, label="validation"
    )
    ax.set_title(metric)
    ax.legend()
plt.tight_layout()
plt.show()

display(
    pd.concat([train_images, validation_images])
    .groupby("image_source")[metrics + ["width", "height", "file_size_kb"]]
    .median()
)
display(risks.sort_values(["severity", "risk_id"]))"""
        ),
        markdown("## Final Conclusion\n\n" + conclusion),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game2_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument(
        "--image-workers", type=int, default=min(12, os.cpu_count() or 4)
    )
    args = parser.parse_args()

    started = time.time()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    game_dir = root / "games" / "game2_data_reconnaissance"

    train = pd.read_csv(game_dir / "game2_train_eda.csv")
    validation = pd.read_csv(game_dir / "game2_validation_eda.csv")
    for name, frame in (("train", train), ("validation", validation)):
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise SystemExit(f"{name} missing columns: {sorted(missing)}")

    print("Building text statistics...")
    train_text = build_text_statistics(train)
    validation_text = build_text_statistics(validation)

    print(
        f"Building image statistics for {len(train) + len(validation):,} rows..."
    )
    cache: dict[Path, dict[str, object]] = {}
    train_images = build_image_statistics(
        train, root, args.image_workers, cache
    )
    validation_images = build_image_statistics(
        validation, root, args.image_workers, cache
    )

    print("Building risk priority table...")
    risks = make_risk_table(
        train,
        validation,
        train_text,
        validation_text,
        train_images,
        validation_images,
    )

    train_text.to_csv(
        output / "game2_train_text_statistics.csv", index=False
    )
    validation_text.to_csv(
        output / "game2_validation_text_statistics.csv", index=False
    )
    train_images.to_csv(
        output / "game2_train_image_statistics.csv", index=False
    )
    validation_images.to_csv(
        output / "game2_validation_image_statistics.csv", index=False
    )
    risks.to_csv(output / "game2_risk_priority_table.csv", index=False)

    audit = {
        "team_name": args.team_name,
        "rows": {"train": len(train), "validation": len(validation)},
        "labels": {
            "train": train["label"].value_counts().to_dict(),
            "validation": validation["label"].value_counts().to_dict(),
        },
        "empty_texts": {
            "train": int(train_text["is_empty_text"].sum()),
            "validation": int(validation_text["is_empty_text"].sum()),
        },
        "modified_images": {
            "train": int((train_images["image_source"] == "modified").sum()),
            "validation": int(
                (validation_images["image_source"] == "modified").sum()
            ),
        },
        "risk_count": len(risks),
        "risk_severity_counts": risks["severity"].value_counts().to_dict(),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output / "game2_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    build_notebook(
        output / "Game_2_Data_Reconnaissance_Mission_Completed.ipynb",
        root,
        args.team_name,
        risks,
    )
    print(risks.to_string(index=False))
    print(f"Artifacts written to {output}")


if __name__ == "__main__":
    main()
