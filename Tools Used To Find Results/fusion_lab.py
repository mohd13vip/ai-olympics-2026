#!/usr/bin/env python3
"""Cross-validated lab for Game 8 fusion-layer experiments (v2 system).

Loads the staging Final_System as a module, rebuilds the 20-feature frame
for train+validation entirely from the score caches (no model inference),
then ranks candidate fusion configurations with 5-fold stratified CV.

Every candidate is scored in two regimes per fold:
  normal  fold-test rows with all features intact (validation-like)
  blind   fold-test rows with relation features zeroed (hidden-domain proxy,
          matching what a boss test of unseen material looks like)

Ranking metric: mean of the two regime macro-F1 means. Suites + validation
inside inference.py remain the final acceptance gate; this lab only ranks.

Usage:  python fusion_lab.py
Deterministic: seed 2026.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
STAGING = PROJECT / "_v2_staging" / "Final_System"
sys.path.insert(0, str(STAGING))

import inference  # noqa: E402  (staging Final_System module)

SEED = 2026
INTERACTION_COLUMNS = ["clip_x_agreement", "clip_x_gap"]


def build_features() -> pd.DataFrame:
    device = torch.device("cpu")
    train = inference.prepare_frame(
        pd.read_csv(STAGING / "reference_data" / "game8_train.csv")
    )
    validation = inference.prepare_frame(
        pd.read_csv(STAGING / "reference_data" / "game8_validation.csv")
    )
    combined = pd.concat(
        [train.assign(split="train"), validation.assign(split="validation")],
        ignore_index=True,
        sort=False,
    )
    image_scores, text_scores, clip_scores = inference.cached_scores(
        combined, device, 64, 64
    )
    canonical_pairs, known_images, known_texts = inference.canonical_manifest()
    featured = inference.attach_features(
        combined,
        image_scores,
        text_scores,
        clip_scores,
        canonical_pairs,
        known_images,
        known_texts,
    )
    featured["clip_x_agreement"] = (
        featured["clip_similarity"] * featured["model_agreement"]
    )
    featured["clip_x_gap"] = (
        featured["clip_similarity"] * featured["probability_gap"]
    )
    return featured


def blind_copy(
    frame: pd.DataFrame,
    rng: np.random.Generator | None,
    jitter: float,
    feature_columns: list[str],
) -> pd.DataFrame:
    copy = frame.copy()
    copy[inference.RELATION_FEATURE_COLUMNS] = 0.0
    if rng is not None and jitter > 0:
        copy["clip_similarity"] = copy["clip_similarity"] - rng.uniform(
            0.0, jitter, len(copy)
        )
    for column in INTERACTION_COLUMNS:
        if column in feature_columns:
            copy["clip_x_agreement"] = (
                copy["clip_similarity"] * copy["model_agreement"]
            )
            copy["clip_x_gap"] = (
                copy["clip_similarity"] * copy["probability_gap"]
            )
            break
    return copy


def macro_f1(y_true: np.ndarray, probability: np.ndarray) -> float:
    frame = pd.DataFrame(
        {"label": np.where(y_true == 1, "fake", "real")}
    )
    return inference.binary_metrics(frame["label"], probability)["macro_f1"]


CANDIDATES: dict[str, dict] = {
    "baseline_v2": {},
    "feat_clip_interactions": {"features": "interactions"},
    "blind_weight_0.5": {"blind_weight": 0.5},
    "jitter_0.00": {"jitter": 0.0},
    "jitter_0.10": {"jitter": 0.10},
    "hgb_lr03_it400": {"learning_rate": 0.03, "max_iter": 400},
    "hgb_it300": {"max_iter": 300},
    "hgb_leaf31": {"max_leaf_nodes": 31},
    "hgb_l2_2.0": {"l2_regularization": 2.0},
    "ens_boost+stack": {"ensemble": True},
    "feat_inter+ens": {"features": "interactions", "ensemble": True},
}


def run_candidate(name: str, config: dict, featured: pd.DataFrame) -> dict:
    feature_columns = list(inference.FEATURE_COLUMNS)
    if config.get("features") == "interactions":
        feature_columns += INTERACTION_COLUMNS
    jitter = config.get("jitter", 0.06)
    blind_weight = config.get("blind_weight", 1.0)
    boosted_template = HistGradientBoostingClassifier(
        learning_rate=config.get("learning_rate", 0.05),
        max_iter=config.get("max_iter", 220),
        max_leaf_nodes=config.get("max_leaf_nodes", 15),
        l2_regularization=config.get("l2_regularization", 1.0),
        random_state=SEED,
    )
    logistic_template = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED
        ),
    )

    y_all = (featured["label"] == "fake").to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    normal_scores: list[float] = []
    blind_scores: list[float] = []
    for fold_index, (train_idx, test_idx) in enumerate(
        folds.split(featured, y_all)
    ):
        rng = np.random.default_rng(SEED + fold_index)
        fold_train = featured.iloc[train_idx].reset_index(drop=True)
        fold_test = featured.iloc[test_idx].reset_index(drop=True)
        train_fit = pd.concat(
            [
                fold_train,
                blind_copy(fold_train, rng, jitter, feature_columns),
            ],
            ignore_index=True,
        )
        y_fit = (train_fit["label"] == "fake").to_numpy(dtype=int)
        weights = np.concatenate(
            [np.ones(len(fold_train)), np.full(len(fold_train), blind_weight)]
        )
        boosted = clone(boosted_template)
        boosted.fit(train_fit[feature_columns], y_fit, sample_weight=weights)
        if config.get("ensemble"):
            logistic = clone(logistic_template)
            logistic.fit(
                train_fit[feature_columns],
                y_fit,
                logisticregression__sample_weight=weights,
            )

        def probability(frame: pd.DataFrame) -> np.ndarray:
            p_boost = boosted.predict_proba(frame[feature_columns])[:, 1]
            if not config.get("ensemble"):
                return p_boost
            p_log = logistic.predict_proba(frame[feature_columns])[:, 1]
            return (p_boost + p_log) / 2

        y_test = (fold_test["label"] == "fake").to_numpy(dtype=int)
        normal_scores.append(macro_f1(y_test, probability(fold_test)))
        fold_blind = blind_copy(fold_test, None, 0.0, feature_columns)
        blind_scores.append(macro_f1(y_test, probability(fold_blind)))

    normal = float(np.mean(normal_scores))
    blind = float(np.mean(blind_scores))
    return {
        "candidate": name,
        "normal_f1": round(normal, 4),
        "blind_f1": round(blind, 4),
        "mean": round((normal + blind) / 2, 4),
    }


def main() -> None:
    featured = build_features()
    print(f"feature rows: {len(featured)}\n")
    results = [
        run_candidate(name, config, featured)
        for name, config in CANDIDATES.items()
    ]
    table = pd.DataFrame(results).sort_values("mean", ascending=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
