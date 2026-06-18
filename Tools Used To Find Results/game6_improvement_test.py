#!/usr/bin/env python3
"""Standalone test of new Game 6 image experiments before integrating them.

Trains candidates with Game 6's exact recipe (same seed, optimizer, schedule,
augmentation), then measures both the Game 6 metric (processed-validation
macro-F1) and the metric that matters most: the Game 8 fusion validation
macro-F1 with the candidate swapped in as image expert.

Does not touch any submission folder.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms as T

from game4_solver import ExifTranspose, LABELS, ResizePad, ScratchImageCNN
from game6_solver import train_image_experiment
from game8_solver import (
    DEFAULT_MAIN_ROOT,
    FEATURE_COLUMNS,
    PROJECT,
    ImageInferenceDataset,
    attach_features,
    binary_metrics,
    canonical_manifest,
    prepare_frame,
    resolve_image,
)

SUBMISSIONS = DEFAULT_MAIN_ROOT / "submissions"
GAME3 = SUBMISSIONS / "Game3_Submission_Ded_Sec"
GAME8_DATA = (
    DEFAULT_MAIN_ROOT.parent / "AI_Olympics_2026_Game8_Release_v1" / "data"
)
CACHE = PROJECT / "_phase2_cache"
OUT = PROJECT / "_game6_improvement"
SEED = 42

# (name, image_size, epochs) — one isolated change per candidate, except the
# combined run that tests whether the two changes stack.
CANDIDATES = [
    ("I04_size160_ep12", 160, 12),
    ("I05_size192_ep6", 192, 6),
    ("I06_size192_ep12", 192, 12),
]


def game8_fusion_score(checkpoint: Path, size: int, device) -> dict[str, float]:
    train = prepare_frame(pd.read_csv(GAME8_DATA / "game8_train.csv"))
    validation = prepare_frame(
        pd.read_csv(GAME8_DATA / "game8_validation.csv")
    )
    combined = pd.concat(
        [train.assign(split="train"), validation.assign(split="validation")],
        ignore_index=True,
        sort=False,
    )
    unique = combined[["normalized_image_path"]].drop_duplicates().reset_index(
        drop=True
    )
    transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    paths = [
        resolve_image(DEFAULT_MAIN_ROOT, p)
        for p in unique["normalized_image_path"]
    ]
    loader = DataLoader(
        ImageInferenceDataset(paths, transform),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = ScratchImageCNN().to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    chunks = []
    with torch.inference_mode():
        for inputs in loader:
            logits = model(inputs.to(device, non_blocking=True))
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(chunks)
    image_scores = unique.copy()
    image_scores["image_p_fake"] = probs[:, LABELS.index("fake")]
    image_scores["image_p_real"] = probs[:, LABELS.index("real")]

    text_scores = pd.read_csv(CACHE / "text_scores.csv")
    text_scores["model_text"] = text_scores["model_text"].astype(str)
    pairs, known_images, known_texts = canonical_manifest(
        SUBMISSIONS / "Game1_Submission_Ded_Sec",
        GAME3,
        DEFAULT_MAIN_ROOT / "data" / "public_test.csv",
    )
    featured = attach_features(
        combined, image_scores, text_scores, pairs, known_images, known_texts
    )
    tr = featured.loc[featured["split"] == "train"].reset_index(drop=True)
    va = featured.loc[featured["split"] == "validation"].reset_index(drop=True)
    y = (tr["label"] == "fake").astype(int)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=2026
        ),
    ).fit(tr[FEATURE_COLUMNS], y)
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=220, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=2026,
    ).fit(tr[FEATURE_COLUMNS], y)
    pl = logistic.predict_proba(va[FEATURE_COLUMNS])[:, 1]
    pb = boosted.predict_proba(va[FEATURE_COLUMNS])[:, 1]
    return {
        "g8_image_only": binary_metrics(va["label"], va["image_p_fake"].to_numpy())["macro_f1"],
        "g8_logistic_stack": binary_metrics(va["label"], pl)["macro_f1"],
        "g8_stack_average": binary_metrics(va["label"], (pl + pb) / 2)["macro_f1"],
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(GAME3 / "processed_train.csv")
    validation = pd.read_csv(GAME3 / "processed_validation.csv")

    rows = []
    baseline = {"name": "I03_size160_ep6 (submitted)", "macro_f1": 0.8493}
    print(f"baseline {baseline['name']}: game6 macro_f1={baseline['macro_f1']}")
    for name, size, epochs in CANDIDATES:
        checkpoint = OUT / f"{name}.pt"
        print(f"\n=== training {name} (size={size}, epochs={epochs})")
        started = time.perf_counter()
        metrics, training_time, history = train_image_experiment(
            train,
            validation,
            DEFAULT_MAIN_ROOT,
            checkpoint,
            device,
            SEED,
            size,
            64,
            2,
            epochs,
            0.0,
        )
        history.to_csv(OUT / f"{name}_history.csv", index=False)
        print(f"{name}: game6 val macro_f1={metrics['macro_f1']:.4f} "
              f"(train {training_time:.0f}s)")
        fusion = game8_fusion_score(checkpoint, size, device)
        print(f"{name}: game8 fusion -> {fusion}")
        rows.append(
            {
                "experiment": name,
                "image_size": size,
                "epochs": epochs,
                "game6_macro_f1": metrics["macro_f1"],
                "game6_accuracy": metrics["accuracy"],
                "training_time_sec": training_time,
                **fusion,
                "total_sec": time.perf_counter() - started,
            }
        )
    table = pd.DataFrame(rows).sort_values("g8_stack_average", ascending=False)
    table.to_csv(PROJECT / "game6_improvement_results.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
