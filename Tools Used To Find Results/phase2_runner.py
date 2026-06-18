#!/usr/bin/env python3
"""Run the Ded_Sec Game 8 multimodal system on a new (hidden) test CSV.

Phase 2 usage on results day:

    python phase2_runner.py --test-csv "path\\to\\hidden_test.csv"

If the organizers ship new images in a separate folder, point at it too:

    python phase2_runner.py --test-csv hidden.csv --extra-images "path\\to\\new_images"

The hidden CSV must contain: sample_id, image_path, text. The runner writes
phase2_predictions.csv with the required contract columns:

    sample_id,predicted_label,confidence,primary_evidence,reviewer_flag

Image and text model scores are cached in _phase2_cache so repeat runs only
compute inference for samples not seen before.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import game8_solver
from game8_solver import (
    DEFAULT_MAIN_ROOT,
    FEATURE_COLUMNS,
    PROJECT,
    attach_features,
    binary_metrics,
    canonical_manifest,
    cached_scores,
    evidence_fields,
    model_probability,
    prepare_frame,
    relation_blind_copy,
    tune_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Real/Fake for a new test CSV with the Game 8 system."
    )
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "phase2_predictions.csv"
    )
    parser.add_argument(
        "--extra-images",
        type=Path,
        default=None,
        help="Optional folder with new images shipped alongside the hidden test.",
    )
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument(
        "--game8-source",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Game8_Release_v1"
        ),
    )
    parser.add_argument(
        "--game1-output",
        type=Path,
        default=DEFAULT_MAIN_ROOT / "submissions" / "Game1_Submission_Ded_Sec",
    )
    parser.add_argument(
        "--game3-output",
        type=Path,
        default=DEFAULT_MAIN_ROOT / "submissions" / "Game3_Submission_Ded_Sec",
    )
    parser.add_argument(
        "--game6-output",
        type=Path,
        default=DEFAULT_MAIN_ROOT / "submissions" / "Game6_Submission_Ded_Sec",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=DEFAULT_MAIN_ROOT / "data" / "public_test.csv",
    )
    parser.add_argument(
        "--clip-model",
        type=Path,
        default=DEFAULT_MAIN_ROOT
        / "submissions"
        / "Game8_Submission_Ded_Sec"
        / "Final_System"
        / "model_files"
        / "clip_model",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT / "_phase2_cache"
    )
    parser.add_argument("--image-batch", type=int, default=128)
    parser.add_argument("--text-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def install_extra_image_resolver(extra_images: Path) -> None:
    """Let game8_solver find images that live outside the main release."""
    original = game8_solver.resolve_image

    def resolver(main_root: Path, value: object) -> Path:
        try:
            return original(main_root, value)
        except FileNotFoundError:
            candidate = extra_images / Path(str(value)).name
            if candidate.exists():
                return candidate.resolve()
            raise

    game8_solver.resolve_image = resolver


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args.extra_images is not None:
        install_extra_image_resolver(args.extra_images.resolve())

    test_raw = pd.read_csv(args.test_csv)
    required_inputs = {"sample_id", "image_path", "text"}
    missing_columns = required_inputs - set(test_raw.columns)
    if missing_columns:
        raise SystemExit(
            f"{args.test_csv} is missing required columns: {sorted(missing_columns)}"
        )
    print(f"hidden test: {len(test_raw)} rows from {args.test_csv}")

    data = args.game8_source / "data"
    train = prepare_frame(pd.read_csv(data / "game8_train.csv"))
    validation = prepare_frame(pd.read_csv(data / "game8_validation.csv"))
    test = prepare_frame(test_raw)
    combined = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
        sort=False,
    )

    image_scores, text_scores, clip_scores = cached_scores(
        combined,
        args.main_root.resolve(),
        args.game6_output.resolve(),
        args.clip_model.resolve(),
        args.cache_dir.resolve(),
        device,
        args.image_batch,
        args.text_batch,
    )
    canonical_pairs, known_images, known_texts = canonical_manifest(
        args.game1_output.resolve(),
        args.game3_output.resolve(),
        args.public_manifest.resolve(),
    )
    featured = attach_features(
        combined,
        image_scores,
        text_scores,
        clip_scores,
        canonical_pairs,
        known_images,
        known_texts,
    )
    train_f = featured.loc[featured["split"] == "train"].reset_index(drop=True)
    validation_f = featured.loc[
        featured["split"] == "validation"
    ].reset_index(drop=True)
    test_f = featured.loc[featured["split"] == "test"].reset_index(drop=True)
    y_train = (train_f["label"] == "fake").astype(int)

    image_weight = tune_weight(train_f)
    weighted_validation = (
        image_weight * validation_f["image_p_fake"].to_numpy()
        + (1 - image_weight) * validation_f["text_p_fake"].to_numpy()
    )
    relation_gate_validation = np.where(
        validation_f["relation_mismatch"].to_numpy().astype(bool),
        0.995,
        weighted_validation,
    )
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=args.seed,
        ),
    )
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    blind_rng = np.random.default_rng(args.seed)
    train_fit = pd.concat(
        [
            train_f,
            relation_blind_copy(train_f),
            relation_blind_copy(train_f, blind_rng),
        ],
        ignore_index=True,
    )
    y_train_fit = (train_fit["label"] == "fake").astype(int)
    logistic.fit(train_fit[FEATURE_COLUMNS], y_train_fit)
    boosted.fit(train_fit[FEATURE_COLUMNS], y_train_fit)

    logistic_validation = model_probability(logistic, validation_f)
    boosted_validation = model_probability(boosted, validation_f)
    validation_probabilities = {
        "weighted_probability_fusion": weighted_validation,
        "canonical_relation_gate": relation_gate_validation,
        "relation_logistic_stack": logistic_validation,
        "relation_gradient_boosting": boosted_validation,
        "relation_stack_average": (logistic_validation + boosted_validation) / 2,
    }
    scores = {
        system: binary_metrics(validation_f["label"], values)["macro_f1"]
        for system, values in validation_probabilities.items()
    }
    selected_system = max(scores, key=scores.get)
    print("validation macro-F1 by system:")
    for system, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        marker = " <- selected" if system == selected_system else ""
        print(f"  {system}: {value:.4f}{marker}")

    labeled = pd.concat([train_f, validation_f], ignore_index=True)
    labeled = pd.concat(
        [
            labeled,
            relation_blind_copy(labeled),
            relation_blind_copy(labeled, blind_rng),
        ],
        ignore_index=True,
    )
    y_labeled = (labeled["label"] == "fake").astype(int)
    weighted_test = (
        image_weight * test_f["image_p_fake"].to_numpy()
        + (1 - image_weight) * test_f["text_p_fake"].to_numpy()
    )
    if selected_system == "weighted_probability_fusion":
        selected_test = weighted_test
    elif selected_system == "canonical_relation_gate":
        selected_test = np.where(
            test_f["relation_mismatch"].to_numpy().astype(bool),
            0.995,
            weighted_test,
        )
    elif selected_system == "relation_logistic_stack":
        final_model = clone(logistic).fit(labeled[FEATURE_COLUMNS], y_labeled)
        selected_test = model_probability(final_model, test_f)
    elif selected_system == "relation_stack_average":
        logistic_final = clone(logistic).fit(labeled[FEATURE_COLUMNS], y_labeled)
        boosted_final = clone(boosted).fit(labeled[FEATURE_COLUMNS], y_labeled)
        selected_test = (
            model_probability(logistic_final, test_f)
            + model_probability(boosted_final, test_f)
        ) / 2
    else:
        final_model = clone(boosted).fit(labeled[FEATURE_COLUMNS], y_labeled)
        selected_test = model_probability(final_model, test_f)

    evidence, reviewer = evidence_fields(test_f, selected_test)
    predicted_fake = selected_test >= 0.5
    output = pd.DataFrame(
        {
            "sample_id": test_f["sample_id"],
            "predicted_label": np.where(predicted_fake, "fake", "real"),
            "confidence": np.where(
                predicted_fake, selected_test, 1 - selected_test
            ).round(6),
            "primary_evidence": evidence,
            "reviewer_flag": reviewer,
        }
    )
    output.to_csv(args.output, index=False)
    print(f"\nwrote {len(output)} predictions to {args.output}")
    print(output["predicted_label"].value_counts().to_string())
    print(f"reviewer flags: {int(output['reviewer_flag'].sum())}")

    if "label" in test_raw.columns and test_raw["label"].notna().all():
        metrics = binary_metrics(test_f["label"], selected_test)
        print(f"\nlabeled test metrics: {metrics}")


if __name__ == "__main__":
    main()
