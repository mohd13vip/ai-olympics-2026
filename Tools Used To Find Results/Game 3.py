#!/usr/bin/env python3
"""Run controlled preprocessing experiments for AI Olympics 2026 Game 3."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


IMAGE_NUMERIC_FEATURES = [
    "width",
    "height",
    "aspect_ratio",
    "file_size_kb",
    "brightness_mean",
    "contrast_std",
    "laplacian_variance",
    "high_frequency_ratio",
    "colorfulness",
    "exif_present",
]


def safe_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def missing_token_text(value: object) -> str:
    text = safe_text(value).strip()
    return text if text else "MISSING_TEXT"


def conservative_text(value: object) -> str:
    text = missing_token_text(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def aggressive_text(value: object) -> str:
    text = conservative_text(value)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classification_metrics(
    truth: pd.Series, prediction: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "precision": float(
            precision_score(truth, prediction, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(truth, prediction, average="macro", zero_division=0)
        ),
    }


def run_text_experiment(
    experiment_id: str,
    name: str,
    transform,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    train_text = train["text"].map(transform)
    validation_text = validation["text"].map(transform)
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    started = time.perf_counter()
    model.fit(train_text, train["label"])
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(validation_text)
    inference_time = time.perf_counter() - started
    metrics = classification_metrics(validation["label"], prediction)
    return {
        "experiment_id": experiment_id,
        "modality": "text",
        "preprocessing_choice": name,
        **metrics,
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
    }


def clip_to_train_quantiles(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    validation_out = validation.copy()
    for column in columns:
        train_values = pd.to_numeric(train[column], errors="coerce").astype(float)
        validation_values = pd.to_numeric(
            validation[column], errors="coerce"
        ).astype(float)
        low = train_values.quantile(lower)
        high = train_values.quantile(upper)
        train_out[column] = train_values.clip(low, high)
        validation_out[column] = validation_values.clip(low, high)
    return train_out, validation_out


def run_image_experiment(
    experiment_id: str,
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    numeric_features: list[str],
    scaler,
    include_source: bool,
    seed: int,
) -> dict[str, object]:
    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", scaler),
                ]
            ),
            numeric_features,
        )
    ]
    if include_source:
        transformers.append(
            (
                "source",
                OneHotEncoder(handle_unknown="ignore"),
                ["image_source"],
            )
        )
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(transformers, remainder="drop"),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    started = time.perf_counter()
    model.fit(train, train["label"])
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(validation)
    inference_time = time.perf_counter() - started
    metrics = classification_metrics(validation["label"], prediction)
    return {
        "experiment_id": experiment_id,
        "modality": "image",
        "preprocessing_choice": name,
        **metrics,
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
    }


def choose_text_pipeline(experiments: pd.DataFrame) -> str:
    candidates = experiments[
        (experiments["modality"] == "text")
        & (experiments["preprocessing_choice"] != "raw_empty_string")
    ].copy()
    best = candidates["macro_f1"].max()
    acceptable = candidates[candidates["macro_f1"] >= best - 0.002]
    priority = {
        "missing_token_only": 0,
        "conservative_normalization": 1,
        "aggressive_alphanumeric_normalization": 2,
    }
    return min(
        acceptable["preprocessing_choice"],
        key=lambda name: priority.get(str(name), 99),
    )


def robust_z_scores(values: pd.Series) -> pd.Series:
    """MAD-based z-scores, robust to the very outliers being hunted."""
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - median) * 0.6745 / mad


def build_processed_frame(
    raw: pd.DataFrame,
    image_stats: pd.DataFrame,
    text_pipeline: str,
    sharpness_flag_rate: float = 0.05,
    brightness_low: float = 30.0,
    brightness_high: float = 200.0,
) -> pd.DataFrame:
    transform = {
        "missing_token_only": missing_token_text,
        "conservative_normalization": conservative_text,
        "aggressive_alphanumeric_normalization": aggressive_text,
    }[text_pipeline]
    stats = image_stats.set_index("sample_id")
    output = raw.copy()
    output["original_text"] = output["text"]
    output["text_was_missing"] = output["text"].map(
        lambda value: not safe_text(value).strip()
    )
    output["text"] = output["text"].map(transform)
    output["image_width"] = output["sample_id"].map(stats["width"])
    output["image_height"] = output["sample_id"].map(stats["height"])
    output["image_aspect_ratio"] = output["sample_id"].map(
        stats["aspect_ratio"]
    )
    sharpness = output["sample_id"].map(stats["laplacian_variance"])
    sharpness_cutoff = float(
        stats["laplacian_variance"].quantile(sharpness_flag_rate)
    )
    output["image_low_resolution"] = (
        (output["image_width"] < 128) | (output["image_height"] < 128)
    )
    # Quantile flag for the relative worst plus a log-scale MAD fence, so a
    # blur cluster larger than the quantile quota is still fully caught.
    output["image_low_sharpness"] = (sharpness <= sharpness_cutoff) | (
        robust_z_scores(np.log1p(sharpness)) < -3.5
    )
    brightness = output["sample_id"].map(stats["brightness_mean"])
    # Absolute exposure bounds plus an adaptive MAD fence for outliers that
    # sit inside the absolute bounds.
    output["image_extreme_brightness"] = (
        (brightness < brightness_low)
        | (brightness > brightness_high)
        | (robust_z_scores(brightness).abs() > 3.5)
    )
    output["image_source"] = output["sample_id"].map(stats["image_source"])
    output["image_processing_plan"] = (
        "exif_transpose_rgb_aspect_preserving_resize_pad_normalize"
    )
    return output


def build_notebook(
    path: Path,
    selected_text: str,
    summary: str,
    team_name: str,
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

    cells = [
        markdown(
            f"""# Game 3 - The Noise Lab

**Team:** {team_name}

Controlled experiments use fixed logistic-regression baselines so performance
changes come from preprocessing rather than increased model capacity."""
        ),
        code(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd()
train = pd.read_csv(OUTPUT_DIR / "processed_train.csv")
validation = pd.read_csv(OUTPUT_DIR / "processed_validation.csv")
experiments = pd.read_csv(OUTPUT_DIR / "preprocessing_experiments.csv")
decisions = pd.read_csv(OUTPUT_DIR / "final_preprocessing_decisions.csv")

display(experiments)
display(decisions)
print("Processed train:", train.shape)
print("Processed validation:", validation.shape)"""
        ),
        code(
            """for modality, group in experiments.groupby("modality"):
    ax = group.plot.bar(
        x="preprocessing_choice",
        y="macro_f1",
        ylim=(max(0, group["macro_f1"].min() - 0.03), 1),
        title=f"{modality.title()} preprocessing experiments",
        legend=False,
        figsize=(9, 4),
    )
    ax.set_ylabel("Validation macro-F1")
    plt.tight_layout()
    plt.show()

display(
    pd.DataFrame({
        "train": [
            len(train),
            int(train["text_was_missing"].sum()),
            int(train["image_low_resolution"].sum()),
            int(train["image_low_sharpness"].sum()),
        ],
        "validation": [
            len(validation),
            int(validation["text_was_missing"].sum()),
            int(validation["image_low_resolution"].sum()),
            int(validation["image_low_sharpness"].sum()),
        ],
    }, index=[
        "rows", "missing_text", "low_resolution", "low_sharpness"
    ])
)"""
        ),
        markdown(
            f"""## Selected Pipeline

- Text: `{selected_text}`
- Images: EXIF transpose, RGB conversion, aspect-preserving resize, padding,
  tensor conversion, and normalization.
- Quality issues are retained as flags rather than deleted.

## Final Conclusion

{summary}"""
        ),
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
        "--game2-output",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game2_Submission_Ded_Sec"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game3_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sharpness-flag-rate", type=float, default=0.05)
    parser.add_argument("--brightness-low", type=float, default=30.0)
    parser.add_argument("--brightness-high", type=float, default=200.0)
    args = parser.parse_args()

    root = args.root.resolve()
    game2_output = args.game2_output.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    game2_dir = root / "games" / "game2_data_reconnaissance"

    train = pd.read_csv(game2_dir / "game2_train_eda.csv")
    validation = pd.read_csv(game2_dir / "game2_validation_eda.csv")
    train_images = pd.read_csv(
        game2_output / "game2_train_image_statistics.csv"
    )
    validation_images = pd.read_csv(
        game2_output / "game2_validation_image_statistics.csv"
    )

    print("Running text preprocessing experiments...")
    text_specs = [
        ("T00", "raw_empty_string", safe_text),
        ("T01", "missing_token_only", missing_token_text),
        ("T02", "conservative_normalization", conservative_text),
        (
            "T03",
            "aggressive_alphanumeric_normalization",
            aggressive_text,
        ),
    ]
    rows = [
        run_text_experiment(
            experiment_id,
            name,
            transform,
            train,
            validation,
            args.seed,
        )
        for experiment_id, name, transform in text_specs
    ]

    print("Running image preprocessing experiments...")
    rows.append(
        run_image_experiment(
            "I00",
            "raw_numeric_features",
            train_images,
            validation_images,
            IMAGE_NUMERIC_FEATURES,
            "passthrough",
            True,
            args.seed,
        )
    )
    rows.append(
        run_image_experiment(
            "I01",
            "standardized_numeric_features",
            train_images,
            validation_images,
            IMAGE_NUMERIC_FEATURES,
            StandardScaler(),
            True,
            args.seed,
        )
    )
    clipped_train, clipped_validation = clip_to_train_quantiles(
        train_images, validation_images, IMAGE_NUMERIC_FEATURES
    )
    rows.append(
        run_image_experiment(
            "I02",
            "quantile_clip_plus_robust_scaling",
            clipped_train,
            clipped_validation,
            IMAGE_NUMERIC_FEATURES,
            RobustScaler(),
            True,
            args.seed,
        )
    )
    reliable_features = [
        feature
        for feature in IMAGE_NUMERIC_FEATURES
        if feature not in {"exif_present", "file_size_kb"}
    ]
    rows.append(
        run_image_experiment(
            "I03",
            "robust_scaling_without_metadata_shortcuts",
            clipped_train,
            clipped_validation,
            reliable_features,
            RobustScaler(),
            False,
            args.seed,
        )
    )

    experiments = pd.DataFrame(rows)
    baseline_f1 = {
        modality: float(
            experiments[
                experiments["experiment_id"].isin(
                    ["T00"] if modality == "text" else ["I00"]
                )
            ]["macro_f1"].iloc[0]
        )
        for modality in ("text", "image")
    }
    experiments["delta_macro_f1"] = experiments.apply(
        lambda row: row["macro_f1"] - baseline_f1[row["modality"]],
        axis=1,
    )
    experiments["result"] = np.select(
        [
            experiments["delta_macro_f1"] > 0.002,
            experiments["delta_macro_f1"] < -0.002,
        ],
        ["improved", "harmed"],
        default="neutral",
    )

    selected_text = choose_text_pipeline(experiments)
    processed_train = build_processed_frame(
        train,
        train_images,
        selected_text,
        sharpness_flag_rate=args.sharpness_flag_rate,
        brightness_low=args.brightness_low,
        brightness_high=args.brightness_high,
    )
    processed_validation = build_processed_frame(
        validation,
        validation_images,
        selected_text,
        sharpness_flag_rate=args.sharpness_flag_rate,
        brightness_low=args.brightness_low,
        brightness_high=args.brightness_high,
    )

    decisions = pd.DataFrame(
        [
            {
                "decision_id": "D01",
                "modality": "text",
                "transformation": "Explicit missing-text token and indicator",
                "decision": "accepted",
                "evidence": "130 empty texts across Train and Validation",
                "reason": "Preserves rows and makes missingness explicit to downstream models.",
            },
            {
                "decision_id": "D02",
                "modality": "text",
                "transformation": selected_text,
                "decision": "accepted",
                "evidence": "Selected from fixed-baseline validation experiments",
                "reason": "Simplest reliable text transform within 0.002 macro-F1 of the best candidate.",
            },
            {
                "decision_id": "D03",
                "modality": "image",
                "transformation": "EXIF transpose, RGB conversion, aspect-preserving resize and padding",
                "decision": "accepted",
                "evidence": "962 low-resolution and 75 extreme-aspect images",
                "reason": "Produces stable model inputs without cropping away unusual image content.",
            },
            {
                "decision_id": "D04",
                "modality": "image",
                "transformation": "Automatic sharpening",
                "decision": "rejected",
                "evidence": "Sharpness differs strongly for 475 intentionally altered images",
                "reason": "Could amplify compression artifacts and erase a meaningful quality signal.",
            },
            {
                "decision_id": "D05",
                "modality": "image",
                "transformation": "Delete low-quality images",
                "decision": "rejected",
                "evidence": "Quality issues occur in both labels and both splits",
                "reason": "Deletion would reduce coverage and may introduce selection bias.",
            },
            {
                "decision_id": "D06",
                "modality": "both",
                "transformation": "Quality and missingness flags",
                "decision": "accepted",
                "evidence": "Game 2 risk table",
                "reason": "Allows downstream models and error analysis to distinguish unreliable inputs.",
            },
        ]
    )

    experiments.to_csv(
        output / "preprocessing_experiments.csv", index=False
    )
    decisions.to_csv(
        output / "final_preprocessing_decisions.csv", index=False
    )
    processed_train.to_csv(output / "processed_train.csv", index=False)
    processed_validation.to_csv(
        output / "processed_validation.csv", index=False
    )

    best_text = experiments.loc[
        experiments["modality"] == "text"
    ].sort_values("macro_f1", ascending=False).iloc[0]
    best_image = experiments.loc[
        experiments["modality"] == "image"
    ].sort_values("macro_f1", ascending=False).iloc[0]
    summary = (
        f"Selected text pipeline: {selected_text}. "
        f"Best text experiment macro-F1={best_text['macro_f1']:.4f}; "
        f"raw text baseline={baseline_f1['text']:.4f}. "
        f"Best image-statistics preprocessing={best_image['preprocessing_choice']} "
        f"with macro-F1={best_image['macro_f1']:.4f}; "
        f"raw image-feature baseline={baseline_f1['image']:.4f}. "
        "All rows were retained. Images remain unchanged on disk and will be "
        "loaded with EXIF-safe RGB conversion, aspect-preserving resize, "
        "padding, and normalization. Destructive sharpening, deletion, and "
        "aggressive cropping were rejected."
    )
    (output / "preprocessing_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    audit = {
        "selected_text_pipeline": selected_text,
        "train_rows": len(processed_train),
        "validation_rows": len(processed_validation),
        "experiments": len(experiments),
        "accepted_decisions": int((decisions["decision"] == "accepted").sum()),
        "rejected_decisions": int((decisions["decision"] == "rejected").sum()),
        "best_text_macro_f1": float(best_text["macro_f1"]),
        "best_image_macro_f1": float(best_image["macro_f1"]),
    }
    (output / "game3_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    build_notebook(
        output / "Game_3_The_Noise_Lab_Completed.ipynb",
        selected_text,
        summary,
        args.team_name,
    )
    print(experiments.to_string(index=False))
    print(summary)
    print(f"Artifacts written to {output}")


if __name__ == "__main__":
    main()
