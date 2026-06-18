#!/usr/bin/env python3
"""Controlled experiments to improve the Game 8 fusion layer.

Uses the cached base-expert scores from _phase2_cache, so every candidate is
evaluated on identical inputs. Model-class/hyperparameter choices are made by
5-fold CV on the Game 8 training pairs; the validation set is only used for
the final comparison table (same protocol as the submitted solver).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from game8_solver import (
    DEFAULT_MAIN_ROOT,
    FEATURE_COLUMNS,
    PROJECT,
    attach_features,
    binary_metrics,
    canonical_manifest,
    prepare_frame,
    tune_weight,
)

GAME8_DATA = (
    DEFAULT_MAIN_ROOT.parent / "AI_Olympics_2026_Game8_Release_v1" / "data"
)
SUBMISSIONS = DEFAULT_MAIN_ROOT / "submissions"
CACHE = PROJECT / "_phase2_cache"
SEED = 2026


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prepare_frame(pd.read_csv(GAME8_DATA / "game8_train.csv"))
    validation = prepare_frame(
        pd.read_csv(GAME8_DATA / "game8_validation.csv")
    )
    combined = pd.concat(
        [train.assign(split="train"), validation.assign(split="validation")],
        ignore_index=True,
        sort=False,
    )
    image_scores = pd.read_csv(CACHE / "image_scores.csv")
    text_scores = pd.read_csv(CACHE / "text_scores.csv")
    text_scores["model_text"] = text_scores["model_text"].astype(str)
    pairs, known_images, known_texts = canonical_manifest(
        SUBMISSIONS / "Game1_Submission_Ded_Sec",
        SUBMISSIONS / "Game3_Submission_Ded_Sec",
        DEFAULT_MAIN_ROOT / "data" / "public_test.csv",
    )
    featured = attach_features(
        combined, image_scores, text_scores, pairs, known_images, known_texts
    )
    train_f = featured.loc[featured["split"] == "train"].reset_index(drop=True)
    validation_f = featured.loc[
        featured["split"] == "validation"
    ].reset_index(drop=True)
    return train_f, validation_f


def probability(model, frame: pd.DataFrame) -> np.ndarray:
    values = model.predict_proba(frame[FEATURE_COLUMNS])
    return values[:, list(model.classes_).index(1)]


def main() -> None:
    train_f, validation_f = load_features()
    y_train = (train_f["label"] == "fake").astype(int)
    truth = validation_f["label"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    image_weight = tune_weight(train_f)
    weighted = (
        image_weight * validation_f["image_p_fake"].to_numpy()
        + (1 - image_weight) * validation_f["text_p_fake"].to_numpy()
    )
    mismatch = validation_f["relation_mismatch"].to_numpy().astype(bool)

    # --- candidate model classes, hyperparameters chosen by train CV only
    logistic_grid = {
        f"logistic_C{c}": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, max_iter=3000, class_weight="balanced", random_state=SEED
            ),
        )
        for c in (0.25, 0.5, 1.0, 2.0, 4.0)
    }
    boosted_grid = {
        f"hgb_lr{lr}_it{it}_lv{lv}": HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=it,
            max_leaf_nodes=lv,
            l2_regularization=1.0,
            random_state=SEED,
        )
        for lr in (0.03, 0.05, 0.08)
        for it in (220, 400)
        for lv in (15, 31)
    }

    def best_by_cv(grid: dict) -> tuple[str, object, float]:
        results = {}
        for name, model in grid.items():
            scores = cross_val_score(
                model, train_f[FEATURE_COLUMNS], y_train,
                scoring="f1_macro", cv=cv, n_jobs=-1,
            )
            results[name] = float(scores.mean())
        winner = max(results, key=results.get)
        return winner, grid[winner], results[winner]

    logistic_name, logistic_model, logistic_cv = best_by_cv(logistic_grid)
    boosted_name, boosted_model, boosted_cv = best_by_cv(boosted_grid)
    print(f"train-CV picks: {logistic_name} ({logistic_cv:.4f}), "
          f"{boosted_name} ({boosted_cv:.4f})")

    # reference models exactly as the submitted solver configures them
    submitted_logistic = clone(logistic_grid["logistic_C1.0"])
    submitted_boosted = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=220, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=SEED,
    )

    fitted = {}
    for name, model in (
        ("submitted_logistic", submitted_logistic),
        ("submitted_boosted", submitted_boosted),
        ("cv_logistic", clone(logistic_model)),
        ("cv_boosted", clone(boosted_model)),
    ):
        fitted[name] = clone(model).fit(train_f[FEATURE_COLUMNS], y_train)

    p = {name: probability(model, validation_f) for name, model in fitted.items()}

    candidates: dict[str, np.ndarray] = {
        "image_only": validation_f["image_p_fake"].to_numpy(),
        "text_only": validation_f["text_p_fake"].to_numpy(),
        "weighted_probability_fusion": weighted,
        "canonical_relation_gate": np.where(mismatch, 0.995, weighted),
        "relation_logistic_stack (submitted)": p["submitted_logistic"],
        "relation_gradient_boosting (submitted)": p["submitted_boosted"],
        "cv_tuned_logistic_stack": p["cv_logistic"],
        "cv_tuned_gradient_boosting": p["cv_boosted"],
        "stack_average (log+hgb)": (p["submitted_logistic"] + p["submitted_boosted"]) / 2,
        "cv_stack_average": (p["cv_logistic"] + p["cv_boosted"]) / 2,
        "gated_logistic_stack": np.where(mismatch, 0.995, p["submitted_logistic"]),
        "gated_stack_average": np.where(
            mismatch, 0.995, (p["submitted_logistic"] + p["submitted_boosted"]) / 2
        ),
        "gated_cv_stack_average": np.where(
            mismatch, 0.995, (p["cv_logistic"] + p["cv_boosted"]) / 2
        ),
    }

    rows = []
    for name, values in candidates.items():
        metrics = binary_metrics(truth, values)
        rows.append({"system": name, **metrics})
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print()
    print(table.to_string(index=False))
    table.to_csv(PROJECT / "fusion_experiments_results.csv", index=False)


if __name__ == "__main__":
    main()
