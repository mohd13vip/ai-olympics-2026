#!/usr/bin/env python3
"""Test whether stronger base experts improve the Game 8 fusion.

Image candidates: ensembles of the three submitted checkpoints
(Game 4 scratch @128, Game 6 I02 label-smoothing @128, Game 6 I03 best @160).
Text candidates: alternative heads on the same frozen MiniLM embeddings.
Every variant is scored by the metric that matters: full-fusion validation
macro-F1 (logistic stack, boosted stack, and their average).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchvision import transforms as T

from game4_solver import ExifTranspose, LABELS, ResizePad, ScratchImageCNN
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

GAME8_DATA = (
    DEFAULT_MAIN_ROOT.parent / "AI_Olympics_2026_Game8_Release_v1" / "data"
)
SUBMISSIONS = DEFAULT_MAIN_ROOT / "submissions"
GAME4 = SUBMISSIONS / "Game4_Submission_Ded_Sec"
GAME6 = SUBMISSIONS / "Game6_Submission_Ded_Sec"
CACHE = PROJECT / "_phase2_cache"
EXP_CACHE = PROJECT / "_expert_cache"
SEED = 2026

CHECKPOINTS = {
    "g4_scratch": (GAME4 / "scratch_image_model" / "model.pt", 128),
    "i02_smooth": (GAME6 / "image_label_smoothing.pt", 128),
    # i03_best @160 == the cached scores already in _phase2_cache
}


def checkpoint_scores(
    name: str,
    unique_paths: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    EXP_CACHE.mkdir(exist_ok=True)
    cache_file = EXP_CACHE / f"{name}_scores.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)
    weights, size = CHECKPOINTS[name]
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
        for p in unique_paths["normalized_image_path"]
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
        torch.load(weights, map_location=device, weights_only=True)
    )
    model.eval()
    chunks = []
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs in loader:
            logits = model(inputs.to(device, non_blocking=True))
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(chunks)
    result = unique_paths.copy()
    result["image_p_fake"] = probs[:, LABELS.index("fake")]
    result["image_p_real"] = probs[:, LABELS.index("real")]
    result.to_csv(cache_file, index=False)
    print(f"  {name}: {len(result)} images in {time.perf_counter() - started:.1f}s")
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = prepare_frame(pd.read_csv(GAME8_DATA / "game8_train.csv"))
    validation = prepare_frame(
        pd.read_csv(GAME8_DATA / "game8_validation.csv")
    )
    combined = pd.concat(
        [train.assign(split="train"), validation.assign(split="validation")],
        ignore_index=True,
        sort=False,
    )
    unique_paths = (
        combined[["normalized_image_path"]].drop_duplicates().reset_index(drop=True)
    )

    base_image = pd.read_csv(CACHE / "image_scores.csv")  # I03 best @160
    base_text = pd.read_csv(CACHE / "text_scores.csv")
    base_text["model_text"] = base_text["model_text"].astype(str)

    print("computing checkpoint scores (cached after first run)...")
    extra = {
        name: checkpoint_scores(name, unique_paths, device)
        for name in CHECKPOINTS
    }

    def blend_images(names: list[str]) -> pd.DataFrame:
        frames = [
            base_image[["normalized_image_path", "image_p_fake", "image_p_real"]]
        ]
        frames += [
            extra[n][["normalized_image_path", "image_p_fake", "image_p_real"]]
            for n in names
        ]
        merged = frames[0].rename(
            columns={"image_p_fake": "f0", "image_p_real": "r0"}
        )
        for k, frame in enumerate(frames[1:], 1):
            merged = merged.merge(
                frame.rename(
                    columns={"image_p_fake": f"f{k}", "image_p_real": f"r{k}"}
                ),
                on="normalized_image_path",
            )
        n = len(frames)
        merged["image_p_fake"] = sum(merged[f"f{k}"] for k in range(n)) / n
        merged["image_p_real"] = sum(merged[f"r{k}"] for k in range(n)) / n
        return merged[["normalized_image_path", "image_p_fake", "image_p_real"]]

    image_variants = {
        "i03_alone (submitted)": base_image,
        "i03+g4": blend_images(["g4_scratch"]),
        "i03+i02": blend_images(["i02_smooth"]),
        "i03+g4+i02": blend_images(["g4_scratch", "i02_smooth"]),
    }

    # ---------------- text heads on the same frozen MiniLM embeddings
    print("encoding texts with the submitted MiniLM encoder...")
    model_root = GAME6 / "best_optimized_text_model"
    bundle = joblib.load(model_root / "classifier_bundle.joblib")
    encoder = SentenceTransformer(str(model_root / "encoder"), device=str(device))
    unique_text = (
        combined[["model_text", "text_was_missing"]]
        .drop_duplicates("model_text")
        .reset_index(drop=True)
    )
    embeddings = encoder.encode(
        unique_text["model_text"].tolist(),
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    length = unique_text["model_text"].str.len().to_numpy(float)
    missing = unique_text["text_was_missing"].astype(float).to_numpy()
    quality = np.column_stack([length, missing])
    spread = quality.std(0)
    spread[spread == 0] = 1.0  # missing-flag has zero variance in train+val
    quality = (quality - quality.mean(0)) / spread
    features = np.column_stack([embeddings, quality])

    text_index = combined[["model_text", "label", "split"]].merge(
        unique_text.reset_index(names="row"), on="model_text", how="left"
    )
    train_rows = text_index.loc[text_index["split"] == "train", "row"].to_numpy()
    train_labels = (
        combined.loc[combined["split"] == "train", "label"] == "fake"
    ).astype(int).to_numpy()

    heads = {
        "logreg_T04 (submitted)": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED
        ),
        "hgb_embeddings": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED,
        ),
        "calibrated_linear_svc": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=SEED), cv=3
        ),
        "logreg_C4": LogisticRegression(
            C=4.0, max_iter=3000, class_weight="balanced", random_state=SEED
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    text_variants = {}
    x_train = features[train_rows]
    for name, head in heads.items():
        cv_f1 = cross_val_score(
            clone(head), x_train, train_labels,
            scoring="f1_macro", cv=cv, n_jobs=-1,
        ).mean()
        fitted = clone(head).fit(x_train, train_labels)
        probs = fitted.predict_proba(features)[:, list(fitted.classes_).index(1)]
        scores = unique_text[["model_text"]].copy()
        scores["text_p_fake"] = probs
        scores["text_p_real"] = 1 - probs
        text_variants[name] = (scores, float(cv_f1))
        print(f"  text head {name}: train-CV macro-F1 {cv_f1:.4f}")

    # ---------------- score every expert combination through the full fusion
    pairs, known_images, known_texts = canonical_manifest(
        SUBMISSIONS / "Game1_Submission_Ded_Sec",
        SUBMISSIONS / "Game3_Submission_Ded_Sec",
        DEFAULT_MAIN_ROOT / "data" / "public_test.csv",
    )

    def fusion_scores(image_scores, text_scores) -> dict[str, float]:
        featured = attach_features(
            combined, image_scores, text_scores,
            pairs, known_images, known_texts,
        )
        tr = featured.loc[featured["split"] == "train"].reset_index(drop=True)
        va = featured.loc[featured["split"] == "validation"].reset_index(drop=True)
        y = (tr["label"] == "fake").astype(int)
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED
            ),
        ).fit(tr[FEATURE_COLUMNS], y)
        boosted = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=220, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=SEED,
        ).fit(tr[FEATURE_COLUMNS], y)
        pl = logistic.predict_proba(va[FEATURE_COLUMNS])[:, 1]
        pb = boosted.predict_proba(va[FEATURE_COLUMNS])[:, 1]
        out = {}
        out["image_only"] = binary_metrics(va["label"], va["image_p_fake"].to_numpy())["macro_f1"]
        out["text_only"] = binary_metrics(va["label"], va["text_p_fake"].to_numpy())["macro_f1"]
        out["logistic_stack"] = binary_metrics(va["label"], pl)["macro_f1"]
        out["boosted_stack"] = binary_metrics(va["label"], pb)["macro_f1"]
        out["stack_average"] = binary_metrics(va["label"], (pl + pb) / 2)["macro_f1"]
        return out

    rows = []
    for image_name, image_scores in image_variants.items():
        for text_name, (text_scores, _) in text_variants.items():
            result = fusion_scores(image_scores, text_scores)
            rows.append(
                {"image_expert": image_name, "text_expert": text_name, **result}
            )
            print(
                f"{image_name:24s} x {text_name:28s} "
                f"img={result['image_only']:.4f} txt={result['text_only']:.4f} "
                f"log={result['logistic_stack']:.4f} avg={result['stack_average']:.4f}"
            )
    table = pd.DataFrame(rows).sort_values("stack_average", ascending=False)
    table.to_csv(PROJECT / "expert_experiments_results.csv", index=False)
    print("\nTop combinations by stack_average:")
    print(table.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
