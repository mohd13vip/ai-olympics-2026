#!/usr/bin/env python3
"""Audit Game 8 validation accuracy under stricter relation manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from game8_solver import (
    DEFAULT_MAIN_ROOT,
    FEATURE_COLUMNS,
    PROJECT,
    attach_features,
    binary_metrics,
    clip_similarities,
    image_probabilities,
    normalize_image_path,
    normalize_text,
    prepare_frame,
    relation_blind_copy,
    text_probabilities,
    tune_weight,
)


# Optional local staging copies; fall back to the release packages.
SUBMISSIONS = (
    PROJECT / "work"
    if (PROJECT / "work").exists()
    else DEFAULT_MAIN_ROOT / "submissions"
)
GAME1 = SUBMISSIONS / "Game1_Submission_Ded_Sec"
GAME3 = SUBMISSIONS / "Game3_Submission_Ded_Sec"
GAME6 = SUBMISSIONS / "Game6_Submission_Ded_Sec"
GAME8 = (
    PROJECT / "source" / "Game8_Release" / "data"
    if (PROJECT / "source" / "Game8_Release" / "data").exists()
    else Path(
        r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
        r"\AI_Olympics_2026_Game8_Release_v1"
    ) / "data"
)
PUBLIC_TEST = (
    PROJECT / "source" / "public_test.csv"
    if (PROJECT / "source" / "public_test.csv").exists()
    else DEFAULT_MAIN_ROOT / "data" / "public_test.csv"
)
# Kept outside work/ so the cache never masquerades as a staging copy.
CACHE = PROJECT / "_audit_cache"


def relation_manifest(
    sources: list[pd.DataFrame],
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    manifest = pd.concat(sources, ignore_index=True)
    images = manifest["image_path"].map(normalize_image_path)
    texts = manifest["text"].map(normalize_text)
    return set(zip(images, texts)), set(images), set(texts)


def probability(model, frame: pd.DataFrame) -> np.ndarray:
    values = model.predict_proba(frame[FEATURE_COLUMNS])
    return values[:, list(model.classes_).index(1)]


CLIP_MODEL = (
    DEFAULT_MAIN_ROOT
    / "submissions"
    / "Game8_Submission_Ded_Sec"
    / "Final_System"
    / "model_files"
    / "clip_model"
)


def cached_base_scores(
    combined: pd.DataFrame, device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    image_path = CACHE / "image_scores.csv"
    text_path = CACHE / "text_scores.csv"
    clip_path = CACHE / "clip_scores.csv"
    if image_path.exists() and text_path.exists():
        image_scores = pd.read_csv(image_path)
        text_scores = pd.read_csv(text_path)
    else:
        image_scores = image_probabilities(
            combined, DEFAULT_MAIN_ROOT, GAME6, device, 128
        )
        text_scores = text_probabilities(combined, GAME6, device, 256)
        image_scores.to_csv(image_path, index=False)
        text_scores.to_csv(text_path, index=False)
    if clip_path.exists():
        clip_scores = pd.read_csv(clip_path)
        clip_scores["model_text"] = clip_scores["model_text"].astype(str)
    else:
        pairs = combined[
            ["normalized_image_path", "image_path", "model_text"]
        ].drop_duplicates(["normalized_image_path", "model_text"])
        clip_scores = clip_similarities(
            pairs, DEFAULT_MAIN_ROOT, CLIP_MODEL, device, 128
        )
        clip_scores.to_csv(clip_path, index=False)
    return image_scores, text_scores, clip_scores


def confidence_interval(
    correct: np.ndarray, samples: int = 10000, seed: int = 2026
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(
        correct.astype(float), size=(samples, len(correct)), replace=True
    ).mean(axis=1)
    return tuple(np.quantile(bootstrap, [0.025, 0.975]))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = prepare_frame(pd.read_csv(GAME8 / "game8_train.csv"))
    validation = prepare_frame(pd.read_csv(GAME8 / "game8_validation.csv"))
    public = prepare_frame(pd.read_csv(GAME8 / "game8_public_test.csv"))
    combined = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            public.assign(split="public"),
        ],
        ignore_index=True,
        sort=False,
    )
    image_scores, text_scores, clip_scores = cached_base_scores(
        combined, device
    )

    game1_train = pd.read_csv(GAME1 / "game1_corrected_train.csv")
    game1_validation = pd.read_csv(
        GAME1 / "game1_corrected_validation.csv"
    )
    game3_train = pd.read_csv(GAME3 / "processed_train.csv")
    game3_validation = pd.read_csv(GAME3 / "processed_validation.csv")
    public_manifest = pd.read_csv(PUBLIC_TEST)
    variants = {
        "game8_train_pairs_only": [train],
        "prior_training_pairs_only": [game1_train, game3_train],
        "all_available_manifests": [
            game1_train,
            game1_validation,
            game3_train,
            game3_validation,
            public_manifest,
        ],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    audit_rows: list[dict] = []
    detailed: dict[str, dict[str, object]] = {}
    for variant, sources in variants.items():
        pairs, known_images, known_texts = relation_manifest(sources)
        featured = attach_features(
            combined,
            image_scores,
            text_scores,
            clip_scores,
            pairs,
            known_images,
            known_texts,
        )
        train_f = featured.loc[featured["split"] == "train"].reset_index(
            drop=True
        )
        validation_f = featured.loc[
            featured["split"] == "validation"
        ].reset_index(drop=True)
        y_train = (train_f["label"] == "fake").astype(int)
        weight = tune_weight(train_f)
        weighted = (
            weight * validation_f["image_p_fake"].to_numpy()
            + (1 - weight) * validation_f["text_p_fake"].to_numpy()
        )
        gated = np.where(
            validation_f["relation_mismatch"].astype(bool), 0.995, weighted
        )
        models = {
            "relation_logistic_stack": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=2026,
                ),
            ),
            "relation_gradient_boosting": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=2026,
            ),
        }
        candidate_probabilities = {
            "image_only": validation_f["image_p_fake"].to_numpy(),
            "text_only": validation_f["text_p_fake"].to_numpy(),
            "weighted_probability_fusion": weighted,
            "canonical_relation_gate": gated,
        }
        # Fit exactly like the shipped fusion: original rows plus two
        # relation-blind copies (one clean, one CLIP-jittered).
        blind_rng = np.random.default_rng(2026)
        train_fit = pd.concat(
            [
                train_f,
                relation_blind_copy(train_f),
                relation_blind_copy(train_f, blind_rng),
            ],
            ignore_index=True,
        )
        y_train_fit = (train_fit["label"] == "fake").astype(int)
        cv_scores = {}
        for name, model in models.items():
            scores = cross_val_score(
                model,
                train_f[FEATURE_COLUMNS],
                y_train,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
            )
            cv_scores[name] = float(scores.mean())
            fitted = clone(model).fit(
                train_fit[FEATURE_COLUMNS], y_train_fit
            )
            candidate_probabilities[name] = probability(
                fitted, validation_f
            )
        candidate_probabilities["relation_stack_average"] = (
            candidate_probabilities["relation_logistic_stack"]
            + candidate_probabilities["relation_gradient_boosting"]
        ) / 2
        selected_by_cv = max(cv_scores, key=cv_scores.get)
        for system, values in candidate_probabilities.items():
            metrics = binary_metrics(validation_f["label"], values)
            audit_rows.append(
                {
                    "manifest_variant": variant,
                    "system": system,
                    **metrics,
                    "train_cv_macro_f1": cv_scores.get(system, np.nan),
                    "selected_by_train_cv": system == selected_by_cv,
                    "validation_pair_match_rate": validation_f[
                        "relation_pair_match"
                    ].mean(),
                    "validation_known_mismatch_rate": validation_f[
                        "relation_mismatch"
                    ].mean(),
                }
            )
        selected_values = candidate_probabilities[selected_by_cv]
        truth = validation_f["label"].to_numpy()
        predicted = np.where(selected_values >= 0.5, "fake", "real")
        correct = predicted == truth
        detailed[variant] = {
            "selected_by_train_cv": selected_by_cv,
            "confusion_matrix": confusion_matrix(
                truth, predicted, labels=["fake", "real"]
            ).tolist(),
            "correct": int(correct.sum()),
            "total": len(correct),
            "accuracy_ci_95": confidence_interval(correct),
        }

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        SUBMISSIONS / "Game8_Submission_Ded_Sec"
        / "game8_accuracy_audit.csv",
        index=False,
    )
    print(
        audit.sort_values(
            ["manifest_variant", "macro_f1"],
            ascending=[True, False],
        ).to_string(index=False)
    )
    print("\nSelected systems and confidence intervals:")
    for variant, values in detailed.items():
        print(variant, values)


if __name__ == "__main__":
    main()
