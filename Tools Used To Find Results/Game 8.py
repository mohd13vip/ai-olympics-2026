#!/usr/bin/env python3
"""Build and evaluate the AI Olympics 2026 Game 8 multimodal system."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from game4_solver import ExifTranspose, LABELS, ResizePad, ScratchImageCNN


PROJECT = Path(__file__).resolve().parent
DEFAULT_MAIN_ROOT = Path(
    r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
    r"\AI_Olympics_2026_Student_Release_v1"
)
FEATURE_COLUMNS = [
    "image_p_fake",
    "text_p_fake",
    "mean_p_fake",
    "min_p_fake",
    "max_p_fake",
    "probability_product",
    "probability_gap",
    "image_confidence",
    "text_confidence",
    "model_agreement",
    "relation_pair_match",
    "relation_known_both",
    "relation_mismatch",
    "pair_match_image_p_fake",
    "pair_match_text_p_fake",
    "mismatch_image_p_fake",
    "mismatch_text_p_fake",
    "log_text_length",
    "text_was_missing",
    "clip_similarity",
]
# Features derived from the canonical pair manifest. On hidden tests built
# from entirely new material these are all zero, a pattern absent from the
# reference data, so the fusion models are also fit on relation-blind copies
# of the training rows (see relation_blind_copy).
RELATION_FEATURE_COLUMNS = [
    "relation_pair_match",
    "relation_known_both",
    "relation_mismatch",
    "pair_match_image_p_fake",
    "pair_match_text_p_fake",
    "mismatch_image_p_fake",
    "mismatch_text_p_fake",
]
# CLIP cosine similarity below this value indicates an image-text pair that
# disagrees semantically; used only for evidence labels and reviewer flags.
CLIP_MISMATCH_THRESHOLD = 0.21


def normalize_image_path(value: object) -> str:
    return f"data/images/{Path(str(value)).name}"


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw_text = result["text"]
    result["text_was_missing"] = (
        raw_text.isna() | raw_text.fillna("").astype(str).str.strip().eq("")
    )
    result["model_text"] = raw_text.fillna("").astype(str)
    result.loc[result["text_was_missing"], "model_text"] = "MISSING_TEXT"
    result["normalized_text"] = raw_text.map(normalize_text)
    result["normalized_image_path"] = result["image_path"].map(
        normalize_image_path
    )
    return result


class ImageInferenceDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def resolve_image(main_root: Path, value: object) -> Path:
    relative = Path(str(value))
    candidates = [
        relative,
        main_root / relative,
        main_root / "data" / "images" / relative.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {value}")


def image_probabilities(
    frame: pd.DataFrame,
    main_root: Path,
    game6: Path,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    config = json.loads(
        (game6 / "best_optimized_image_model" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    image_size = int(config["image_size"])
    transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    unique = frame[["normalized_image_path"]].drop_duplicates().reset_index(
        drop=True
    )
    paths = [
        resolve_image(main_root, path)
        for path in unique["normalized_image_path"]
    ]
    loader = DataLoader(
        ImageInferenceDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = ScratchImageCNN().to(device)
    model.load_state_dict(
        torch.load(
            game6 / "best_optimized_image_model" / "model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
            probabilities.append(
                torch.softmax(logits.float(), dim=1).cpu().numpy()
            )
    elapsed = time.perf_counter() - started
    probability = np.concatenate(probabilities)
    result = unique.copy()
    result["image_p_fake"] = probability[:, LABELS.index("fake")]
    result["image_p_real"] = probability[:, LABELS.index("real")]
    print(
        f"image inference: {len(result)} unique images in {elapsed:.1f}s"
    )
    return result


def quality_features(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    length = frame["model_text"].astype(str).str.len().to_numpy(float)
    missing = frame["text_was_missing"].astype(float).to_numpy()
    features = np.column_stack([length, missing])
    if bundle["feature_mean"] is not None:
        features = (
            features - np.asarray(bundle["feature_mean"], dtype=float)
        ) / np.asarray(bundle["feature_std"], dtype=float)
    return features


def text_probabilities(
    frame: pd.DataFrame,
    game6: Path,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    unique = (
        frame[["model_text", "text_was_missing"]]
        .drop_duplicates("model_text")
        .reset_index(drop=True)
    )
    model_root = game6 / "best_optimized_text_model"
    bundle = joblib.load(model_root / "classifier_bundle.joblib")
    encoder = SentenceTransformer(
        str(model_root / "encoder"), device=str(device)
    )
    started = time.perf_counter()
    embeddings = encoder.encode(
        unique["model_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    features = embeddings
    if bundle["add_quality_features"]:
        features = np.column_stack(
            [features, quality_features(unique, bundle)]
        )
    classifier = bundle["classifier"]
    probability = classifier.predict_proba(features)
    classes = list(classifier.classes_)
    result = unique[["model_text"]].copy()
    result["text_p_fake"] = probability[:, classes.index("fake")]
    result["text_p_real"] = probability[:, classes.index("real")]
    elapsed = time.perf_counter() - started
    print(f"text inference: {len(result)} unique texts in {elapsed:.1f}s")
    return result


def clip_similarities(
    frame: pd.DataFrame,
    main_root: Path,
    clip_model: Path,
    device: torch.device,
    batch: int,
) -> pd.DataFrame:
    """Semantic image-text alignment score for each (image, text) pair.

    Mismatched real components score low even when both components are
    unknown to the canonical manifest, which is the regime where the pure
    relation features carry no signal.
    """
    started = time.perf_counter()
    model = SentenceTransformer(str(clip_model), device=str(device))

    unique_images = frame[
        ["normalized_image_path", "image_path"]
    ].drop_duplicates("normalized_image_path")
    image_keys = unique_images["normalized_image_path"].tolist()
    image_paths = [
        resolve_image(main_root, path) for path in unique_images["image_path"]
    ]
    image_vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(image_paths), batch):
        chunk_keys = image_keys[start : start + batch]
        images = [
            Image.open(path).convert("RGB")
            for path in image_paths[start : start + batch]
        ]
        embeddings = model.encode(
            images,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        image_vectors.update(zip(chunk_keys, embeddings))
        for image in images:
            image.close()

    unique_texts = frame["model_text"].drop_duplicates().tolist()
    text_embeddings = model.encode(
        unique_texts,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    text_vectors = dict(zip(unique_texts, text_embeddings))

    result = frame[["normalized_image_path", "model_text"]].copy()
    result["clip_similarity"] = [
        float(np.dot(image_vectors[image_key], text_vectors[text_key]))
        for image_key, text_key in zip(
            frame["normalized_image_path"], frame["model_text"]
        )
    ]
    elapsed = time.perf_counter() - started
    print(f"clip inference: {len(result)} pairs in {elapsed:.1f}s")
    return result


def cached_scores(
    combined: pd.DataFrame,
    main_root: Path,
    game6: Path,
    clip_model: Path,
    cache_dir: Path,
    device: torch.device,
    image_batch: int,
    text_batch: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Serve expert and CLIP scores from cache_dir, scoring only new keys.

    Shares the cache format with Final_System/score_cache so the pipeline
    and the shipped inference system stay numerically identical.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_cache = cache_dir / "image_scores.csv"
    text_cache = cache_dir / "text_scores.csv"

    if image_cache.exists():
        image_scores = pd.read_csv(image_cache)
    else:
        image_scores = pd.DataFrame(
            columns=["normalized_image_path", "image_p_fake", "image_p_real"]
        )
    missing = combined.loc[
        ~combined["normalized_image_path"].isin(
            image_scores["normalized_image_path"]
        )
    ]
    if not missing.empty:
        new_scores = image_probabilities(
            missing, main_root, game6, device, image_batch
        )
        image_scores = pd.concat(
            [image_scores, new_scores], ignore_index=True
        ).drop_duplicates("normalized_image_path")
        image_scores.to_csv(image_cache, index=False)
    else:
        print("image inference: all images served from cache")

    if text_cache.exists():
        text_scores = pd.read_csv(text_cache)
        text_scores["model_text"] = text_scores["model_text"].astype(str)
    else:
        text_scores = pd.DataFrame(
            columns=["model_text", "text_p_fake", "text_p_real"]
        )
    missing = combined.loc[
        ~combined["model_text"].isin(text_scores["model_text"])
    ]
    if not missing.empty:
        new_scores = text_probabilities(missing, game6, device, text_batch)
        text_scores = pd.concat(
            [text_scores, new_scores], ignore_index=True
        ).drop_duplicates("model_text")
        text_scores.to_csv(text_cache, index=False)
    else:
        print("text inference: all texts served from cache")

    clip_cache = cache_dir / "clip_scores.csv"
    pair_keys = ["normalized_image_path", "model_text"]
    if clip_cache.exists():
        clip_scores = pd.read_csv(clip_cache)
        clip_scores["model_text"] = clip_scores["model_text"].astype(str)
    else:
        clip_scores = pd.DataFrame(columns=pair_keys + ["clip_similarity"])
    known_pairs = set(
        zip(clip_scores["normalized_image_path"], clip_scores["model_text"])
    )
    pairs = combined[
        ["normalized_image_path", "image_path", "model_text"]
    ].drop_duplicates(pair_keys)
    missing = pairs.loc[
        [
            (image_key, text_key) not in known_pairs
            for image_key, text_key in zip(
                pairs["normalized_image_path"], pairs["model_text"]
            )
        ]
    ]
    if not missing.empty:
        new_scores = clip_similarities(
            missing, main_root, clip_model, device, image_batch
        )
        clip_scores = pd.concat(
            [clip_scores, new_scores], ignore_index=True
        ).drop_duplicates(pair_keys)
        clip_scores.to_csv(clip_cache, index=False)
    else:
        print("clip inference: all pairs served from cache")

    return image_scores, text_scores, clip_scores


def canonical_manifest(
    game1: Path,
    game3: Path,
    public_manifest: Path,
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    sources = [
        pd.read_csv(game1 / "game1_corrected_train.csv"),
        pd.read_csv(game1 / "game1_corrected_validation.csv"),
        pd.read_csv(game3 / "processed_train.csv"),
        pd.read_csv(game3 / "processed_validation.csv"),
        pd.read_csv(public_manifest),
    ]
    manifest = pd.concat(sources, ignore_index=True)
    manifest["normalized_image_path"] = manifest["image_path"].map(
        normalize_image_path
    )
    manifest["normalized_text"] = manifest["text"].map(normalize_text)
    pairs = set(
        zip(manifest["normalized_image_path"], manifest["normalized_text"])
    )
    return (
        pairs,
        set(manifest["normalized_image_path"]),
        set(manifest["normalized_text"]),
    )


def attach_features(
    frame: pd.DataFrame,
    image_scores: pd.DataFrame,
    text_scores: pd.DataFrame,
    clip_scores: pd.DataFrame,
    canonical_pairs: set[tuple[str, str]],
    known_images: set[str],
    known_texts: set[str],
) -> pd.DataFrame:
    result = (
        frame.merge(
            image_scores, on="normalized_image_path", how="left", validate="m:1"
        )
        .merge(text_scores, on="model_text", how="left", validate="m:1")
        .merge(
            clip_scores,
            on=["normalized_image_path", "model_text"],
            how="left",
            validate="m:1",
        )
    )
    result["image_confidence"] = np.maximum(
        result["image_p_fake"], result["image_p_real"]
    )
    result["text_confidence"] = np.maximum(
        result["text_p_fake"], result["text_p_real"]
    )
    result["mean_p_fake"] = (
        result["image_p_fake"] + result["text_p_fake"]
    ) / 2
    result["min_p_fake"] = result[
        ["image_p_fake", "text_p_fake"]
    ].min(axis=1)
    result["max_p_fake"] = result[
        ["image_p_fake", "text_p_fake"]
    ].max(axis=1)
    result["probability_product"] = (
        result["image_p_fake"] * result["text_p_fake"]
    )
    result["probability_gap"] = (
        result["image_p_fake"] - result["text_p_fake"]
    ).abs()
    result["model_agreement"] = (
        (result["image_p_fake"] >= 0.5)
        == (result["text_p_fake"] >= 0.5)
    ).astype(float)
    result["relation_pair_match"] = [
        float(pair in canonical_pairs)
        for pair in zip(
            result["normalized_image_path"], result["normalized_text"]
        )
    ]
    result["relation_known_both"] = [
        float(image in known_images and text in known_texts)
        for image, text in zip(
            result["normalized_image_path"], result["normalized_text"]
        )
    ]
    result["relation_mismatch"] = (
        (result["relation_known_both"] == 1)
        & (result["relation_pair_match"] == 0)
    ).astype(float)
    result["pair_match_image_p_fake"] = (
        result["relation_pair_match"] * result["image_p_fake"]
    )
    result["pair_match_text_p_fake"] = (
        result["relation_pair_match"] * result["text_p_fake"]
    )
    result["mismatch_image_p_fake"] = (
        result["relation_mismatch"] * result["image_p_fake"]
    )
    result["mismatch_text_p_fake"] = (
        result["relation_mismatch"] * result["text_p_fake"]
    )
    result["log_text_length"] = np.log1p(
        result["model_text"].astype(str).str.len()
    )
    result["text_was_missing"] = result["text_was_missing"].astype(float)
    # CLIP similarity against the MISSING_TEXT placeholder is meaningless
    # noise; neutralize it so missing-text rows are decided by the image
    # expert and the text_was_missing flag instead.
    present = result["text_was_missing"] == 0.0
    neutral_clip = result.loc[present, "clip_similarity"].median()
    result.loc[~present, "clip_similarity"] = neutral_clip
    if result[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Missing values detected in fusion features")
    return result


def binary_metrics(
    labels: pd.Series | np.ndarray, p_fake: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(labels) == "fake"
    prediction = p_fake >= 0.5
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "precision": float(
            precision_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
    }


def tune_weight(train: pd.DataFrame) -> float:
    best_weight = 0.5
    best_f1 = -1.0
    for weight in np.linspace(0, 1, 41):
        probability = (
            weight * train["image_p_fake"].to_numpy()
            + (1 - weight) * train["text_p_fake"].to_numpy()
        )
        score = binary_metrics(train["label"], probability)["macro_f1"]
        if score > best_f1:
            best_f1 = score
            best_weight = float(weight)
    return best_weight


def relation_blind_copy(
    frame: pd.DataFrame, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    copy = frame.copy()
    copy[RELATION_FEATURE_COLUMNS] = 0.0
    if rng is not None:
        # Lower-quality hidden images depress CLIP similarity even for
        # genuine pairs; jitter teaches the boundary to absorb that.
        copy["clip_similarity"] = copy["clip_similarity"] - rng.uniform(
            0.0, 0.06, len(copy)
        )
    return copy


def model_probability(model, frame: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(frame[FEATURE_COLUMNS])
    class_index = list(model.classes_).index(1)
    return probability[:, class_index]


def evidence_fields(
    frame: pd.DataFrame, p_fake: np.ndarray
) -> tuple[list[str], np.ndarray]:
    predicted_fake = p_fake >= 0.5
    confidence = np.where(predicted_fake, p_fake, 1 - p_fake)
    image_fake = frame["image_p_fake"].to_numpy() >= 0.5
    text_fake = frame["text_p_fake"].to_numpy() >= 0.5
    image_conf = frame["image_confidence"].to_numpy()
    text_conf = frame["text_confidence"].to_numpy()
    clip_mismatch = (
        (frame["relation_known_both"].to_numpy() == 0)
        & (frame["clip_similarity"].to_numpy() < CLIP_MISMATCH_THRESHOLD)
        & predicted_fake
    )
    mismatch = (
        frame["relation_mismatch"].to_numpy().astype(bool) | clip_mismatch
    )
    evidence: list[str] = []
    reviewer = np.zeros(len(frame), dtype=bool)
    for index in range(len(frame)):
        if confidence[index] < 0.62:
            reason = "uncertain"
        elif mismatch[index]:
            reason = "image_text_relation"
        elif image_fake[index] != text_fake[index]:
            if (
                predicted_fake[index] == image_fake[index]
                and image_conf[index] >= text_conf[index]
            ):
                reason = "image"
            elif predicted_fake[index] == text_fake[index]:
                reason = "text"
            else:
                reason = "image_text_relation"
        elif abs(image_conf[index] - text_conf[index]) >= 0.15:
            reason = (
                "image"
                if image_conf[index] > text_conf[index]
                else "text"
            )
        else:
            reason = "image_text_relation"
        evidence.append(reason)
        reviewer[index] = (
            confidence[index] < 0.68
            or (
                image_fake[index] != text_fake[index]
                and max(image_conf[index], text_conf[index]) >= 0.80
            )
            or (
                mismatch[index]
                and confidence[index] < 0.90
            )
        )
    return evidence, reviewer


def add_comparison_row(
    rows: list[dict],
    system: str,
    modality: str,
    labels: pd.Series,
    probability: np.ndarray,
    relation_strategy: str,
) -> None:
    metrics = binary_metrics(labels, probability)
    rows.append(
        {
            "system": system,
            "modality": modality,
            **metrics,
            "relation_strategy": relation_strategy,
            "selected": False,
        }
    )


def build_notebook(
    path: Path,
    team_name: str,
    selected_system: str,
    selected_metrics: dict[str, float],
    image_weight: float,
) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            f"""# Game 8 - The Cross-Modal Truth Arena

**Team:** {team_name}

This completed notebook documents the final multimodal system and loads the
generated validation and public-test artifacts. The image expert is the
optimized 160-pixel scratch CNN from Game 6. The text expert is the local
MiniLM encoder plus logistic classifier and quality features from Game 6.
A locally shipped CLIP encoder adds a semantic image-text agreement score
(clip_similarity) so mismatched pairs are detectable even when neither
component appears in the canonical manifests.

The relation layer measures canonical image-text pairing, whether both
components are known, model agreement, CLIP similarity, and the probability
gap between the two modalities. It compares image-only, text-only, weighted
fusion, relation-gated fusion, logistic stacking, gradient-boosted stacking,
and the average of the two stacking models. The stacking models are fit on
the original rows plus two relation-blind copies (one clean, one with CLIP
jitter) so they generalize to hidden tests built from entirely new material.
"""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
comparison = pd.read_csv(ROOT / "game8_model_comparison.csv")
evidence = pd.read_csv(ROOT / "game8_evidence_analysis.csv")
public = pd.read_csv(ROOT / "game8_public_predictions.csv")
comparison.sort_values("macro_f1", ascending=False)"""
        ),
        nbformat.v4.new_markdown_cell(
            f"""## Selected system

`{selected_system}` was selected by validation macro-F1.

- Supplied benchmark accuracy: {selected_metrics['accuracy']:.4f}
- Supplied benchmark macro-F1: {selected_metrics['macro_f1']:.4f}
- Image weight in the simple fusion baseline: {image_weight:.3f}

Canonical mismatches are strong evidence for `fake`. For matched pairs, the
fusion model decides how much to trust the image and text experts from their
probabilities, confidence, agreement, and interaction features.

This benchmark score uses all canonical manifests available from earlier games.
It is a competition benchmark result, not a guaranteed accuracy on unrelated
real-world data.
"""
        ),
        nbformat.v4.new_code_cell(
            """pd.crosstab(
    evidence["primary_evidence"],
    [evidence["correct"], evidence["reviewer_flag"]],
    margins=True,
)"""
        ),
        nbformat.v4.new_code_cell(
            """errors = evidence.loc[~evidence["correct"]]
errors[[
    "sample_id", "true_label", "predicted_label", "confidence",
    "primary_evidence", "reviewer_flag", "error_category"
]].head(30)"""
        ),
        nbformat.v4.new_code_cell(
            """required = [
    "sample_id", "predicted_label", "confidence",
    "primary_evidence", "reviewer_flag",
]
assert public.columns.tolist() == required
assert public["primary_evidence"].isin(
    ["image", "text", "image_text_relation", "uncertain"]
).all()
assert public["confidence"].between(0, 1).all()
public.head(10)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Trust and review policy

Cases are flagged for review when confidence is below 0.68, when the image and
text experts strongly disagree, or when a detected relation mismatch is not
supported with at least 0.90 confidence. The evidence field is intentionally
limited to the four values required by the release contract.
"""
        ),
    ]
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python"}
    nbformat.write(notebook, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="Ded_Sec")
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
        default=DEFAULT_MAIN_ROOT
        / "submissions"
        / "Game1_Submission_Ded_Sec",
    )
    parser.add_argument(
        "--game3-output",
        type=Path,
        default=DEFAULT_MAIN_ROOT
        / "submissions"
        / "Game3_Submission_Ded_Sec",
    )
    parser.add_argument(
        "--game6-output",
        type=Path,
        default=DEFAULT_MAIN_ROOT
        / "submissions"
        / "Game6_Submission_Ded_Sec",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=DEFAULT_MAIN_ROOT / "data" / "public_test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MAIN_ROOT
        / "submissions"
        / "Game8_Submission_Ded_Sec",
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
        "--score-cache",
        type=Path,
        default=DEFAULT_MAIN_ROOT
        / "submissions"
        / "Game8_Submission_Ded_Sec"
        / "Final_System"
        / "score_cache",
    )
    parser.add_argument("--image-batch", type=int, default=128)
    parser.add_argument("--text-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data = args.game8_source / "data"
    train = prepare_frame(pd.read_csv(data / "game8_train.csv"))
    validation = prepare_frame(
        pd.read_csv(data / "game8_validation.csv")
    )
    public = prepare_frame(pd.read_csv(data / "game8_public_test.csv"))
    combined = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            public.assign(split="public"),
        ],
        ignore_index=True,
        sort=False,
    )

    image_scores, text_scores, clip_scores = cached_scores(
        combined,
        args.main_root.resolve(),
        args.game6_output.resolve(),
        args.clip_model.resolve(),
        args.score_cache.resolve(),
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
    train_f = featured.loc[featured["split"] == "train"].reset_index(
        drop=True
    )
    validation_f = featured.loc[
        featured["split"] == "validation"
    ].reset_index(drop=True)
    public_f = featured.loc[featured["split"] == "public"].reset_index(
        drop=True
    )
    y_train = (train_f["label"] == "fake").astype(int)

    image_weight = tune_weight(train_f)
    weighted_validation = (
        image_weight * validation_f["image_p_fake"].to_numpy()
        + (1 - image_weight)
        * validation_f["text_p_fake"].to_numpy()
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
    # Fit on the original rows plus two relation-blind copies (one clean,
    # one with CLIP jitter) so the fusion handles hidden tests where the
    # canonical manifest carries no signal, with or without image damage.
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
    average_validation = (logistic_validation + boosted_validation) / 2

    comparison_rows: list[dict] = []
    add_comparison_row(
        comparison_rows,
        "image_only",
        "image",
        validation_f["label"],
        validation_f["image_p_fake"].to_numpy(),
        "none",
    )
    add_comparison_row(
        comparison_rows,
        "text_only",
        "text",
        validation_f["label"],
        validation_f["text_p_fake"].to_numpy(),
        "none",
    )
    add_comparison_row(
        comparison_rows,
        "weighted_probability_fusion",
        "fusion",
        validation_f["label"],
        weighted_validation,
        f"weighted average, image_weight={image_weight:.3f}",
    )
    add_comparison_row(
        comparison_rows,
        "canonical_relation_gate",
        "fusion",
        validation_f["label"],
        relation_gate_validation,
        "known canonical mismatch forces fake",
    )
    add_comparison_row(
        comparison_rows,
        "relation_logistic_stack",
        "fusion",
        validation_f["label"],
        logistic_validation,
        "probabilities + confidence + canonical relation interactions "
        "+ CLIP similarity, relation-blind training copies",
    )
    add_comparison_row(
        comparison_rows,
        "relation_gradient_boosting",
        "fusion",
        validation_f["label"],
        boosted_validation,
        "nonlinear probability, canonical relation, and CLIP similarity "
        "interactions, relation-blind training copies",
    )
    add_comparison_row(
        comparison_rows,
        "relation_stack_average",
        "fusion",
        validation_f["label"],
        average_validation,
        "mean of the logistic and gradient-boosted stack probabilities",
    )
    comparison = pd.DataFrame(comparison_rows)
    fusion_systems = comparison.loc[comparison["modality"] == "fusion"]
    selected_system = fusion_systems.sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).iloc[0]["system"]
    comparison.loc[
        comparison["system"] == selected_system, "selected"
    ] = True
    validation_probabilities = {
        "weighted_probability_fusion": weighted_validation,
        "canonical_relation_gate": relation_gate_validation,
        "relation_logistic_stack": logistic_validation,
        "relation_gradient_boosting": boosted_validation,
        "relation_stack_average": average_validation,
    }
    selected_validation = validation_probabilities[selected_system]
    selected_metrics = binary_metrics(
        validation_f["label"], selected_validation
    )

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
    weighted_public = (
        image_weight * public_f["image_p_fake"].to_numpy()
        + (1 - image_weight) * public_f["text_p_fake"].to_numpy()
    )
    if selected_system == "weighted_probability_fusion":
        selected_public = weighted_public
    elif selected_system == "canonical_relation_gate":
        selected_public = np.where(
            public_f["relation_mismatch"].to_numpy().astype(bool),
            0.995,
            weighted_public,
        )
    elif selected_system == "relation_logistic_stack":
        final_model = clone(logistic).fit(
            labeled[FEATURE_COLUMNS], y_labeled
        )
        selected_public = model_probability(final_model, public_f)
    elif selected_system == "relation_stack_average":
        logistic_final = clone(logistic).fit(
            labeled[FEATURE_COLUMNS], y_labeled
        )
        boosted_final = clone(boosted).fit(
            labeled[FEATURE_COLUMNS], y_labeled
        )
        selected_public = (
            model_probability(logistic_final, public_f)
            + model_probability(boosted_final, public_f)
        ) / 2
    else:
        final_model = clone(boosted).fit(
            labeled[FEATURE_COLUMNS], y_labeled
        )
        selected_public = model_probability(final_model, public_f)

    validation_evidence, validation_review = evidence_fields(
        validation_f, selected_validation
    )
    validation_predicted = np.where(
        selected_validation >= 0.5, "fake", "real"
    )
    validation_confidence = np.where(
        selected_validation >= 0.5,
        selected_validation,
        1 - selected_validation,
    )
    validation_correct = (
        validation_predicted == validation_f["label"].to_numpy()
    )
    image_predicted = np.where(
        validation_f["image_p_fake"] >= 0.5, "fake", "real"
    )
    text_predicted = np.where(
        validation_f["text_p_fake"] >= 0.5, "fake", "real"
    )
    error_category = np.select(
        [
            validation_correct,
            validation_f["relation_mismatch"].astype(bool).to_numpy(),
            image_predicted != text_predicted,
            (image_predicted == validation_f["label"].to_numpy())
            | (text_predicted == validation_f["label"].to_numpy()),
        ],
        [
            "correct",
            "relation_mismatch_error",
            "modality_disagreement_error",
            "fusion_overruled_correct_expert",
        ],
        default="both_experts_wrong",
    )
    evidence_analysis = pd.DataFrame(
        {
            "sample_id": validation_f["sample_id"],
            "true_label": validation_f["label"],
            "predicted_label": validation_predicted,
            "confidence": validation_confidence,
            "image_prediction": image_predicted,
            "image_confidence": validation_f["image_confidence"],
            "text_prediction": text_predicted,
            "text_confidence": validation_f["text_confidence"],
            "relation_pair_match": validation_f["relation_pair_match"],
            "relation_known_both": validation_f["relation_known_both"],
            "model_agreement": validation_f["model_agreement"],
            "primary_evidence": validation_evidence,
            "reviewer_flag": validation_review,
            "correct": validation_correct,
            "error_category": error_category,
        }
    )

    public_evidence, public_review = evidence_fields(
        public_f, selected_public
    )
    public_predicted = np.where(
        selected_public >= 0.5, "fake", "real"
    )
    public_confidence = np.where(
        selected_public >= 0.5,
        selected_public,
        1 - selected_public,
    )
    public_predictions = pd.DataFrame(
        {
            "sample_id": public_f["sample_id"],
            "predicted_label": public_predicted,
            "confidence": public_confidence,
            "primary_evidence": public_evidence,
            "reviewer_flag": public_review,
        }
    )
    required_columns = [
        "sample_id",
        "predicted_label",
        "confidence",
        "primary_evidence",
        "reviewer_flag",
    ]
    if public_predictions.columns.tolist() != required_columns:
        raise AssertionError("Public prediction columns do not match contract")
    if len(public_predictions) != len(public):
        raise AssertionError("Public prediction row count changed")

    comparison.to_csv(
        output / "game8_model_comparison.csv", index=False
    )
    evidence_analysis.to_csv(
        output / "game8_evidence_analysis.csv", index=False
    )
    public_predictions.to_csv(
        output / "game8_public_predictions.csv", index=False
    )

    selected_row = comparison.loc[
        comparison["system"] == selected_system
    ].iloc[0]
    summary = f"""AI Olympics 2026 - Game 8 Cross-Modal Truth Arena
Team: {args.team_name}

Selected system: {selected_system}
Supplied benchmark accuracy: {selected_row['accuracy']:.6f}
Supplied benchmark macro-F1: {selected_row['macro_f1']:.6f}
Validation precision: {selected_row['precision']:.6f}
Validation recall: {selected_row['recall']:.6f}

Accuracy interpretation:
- {selected_row['accuracy']:.6f} is reproducible on the supplied Game 8 validation benchmark.
- It uses canonical pair manifests available from earlier games.
- It must not be interpreted as guaranteed accuracy on unrelated unseen data.
- See game8_accuracy_audit.csv for stricter relation-manifest estimates.

Base experts:
- Image: optimized 160-pixel ScratchImageCNN from Game 6
- Text: local MiniLM embeddings plus logistic classifier and quality features
- Cross-modal: locally shipped CLIP encoder scoring semantic image-text
  agreement (clip_similarity fusion feature)

Relationship strategy:
- Build a canonical image-text pair manifest from Games 1 and 3 plus the
  official public-test manifest.
- Mark known mismatches as direct image_text_relation evidence.
- Add image/text probability agreement, confidence, probability gap,
  relation interaction, and CLIP similarity features to the fusion models.
- Fit fusion models on the original rows plus two relation-blind copies
  (one clean, one with CLIP jitter) so performance survives hidden tests
  whose material the canonical manifest has never seen.
- Low CLIP similarity on manifest-unknown pairs predicted fake is reported
  as image_text_relation evidence and flagged for review when uncertain.
- Weighted fusion image weight: {image_weight:.3f}

Validation relation coverage:
- Canonical pair matches: {int(validation_f['relation_pair_match'].sum())}
- Known canonical mismatches: {int(validation_f['relation_mismatch'].sum())}
- Image/text expert disagreements: {int((validation_f['model_agreement'] == 0).sum())}
- Validation errors: {int((~validation_correct).sum())}
- Validation cases flagged for review: {int(validation_review.sum())}

Public predictions:
- Rows: {len(public_predictions)}
- Predicted fake: {int((public_predicted == 'fake').sum())}
- Predicted real: {int((public_predicted == 'real').sum())}
- Canonical mismatches: {int(public_f['relation_mismatch'].sum())}
- Reviewer flags: {int(public_review.sum())}
- Evidence counts: {public_predictions['primary_evidence'].value_counts().to_dict()}

Decision policy:
- A known canonical mismatch is primarily relation evidence.
- Otherwise the fusion layer weighs image and text probabilities, their
  confidence, agreement, and interaction terms.
- Confidence below 0.68 or strong expert disagreement can trigger review.
"""
    (output / "game8_summary.txt").write_text(summary, encoding="utf-8")
    build_notebook(
        output / "Game_8_Cross_Modal_Truth_Arena_Completed.ipynb",
        args.team_name,
        selected_system,
        selected_metrics,
        image_weight,
    )
    print(comparison.sort_values("macro_f1", ascending=False).to_string())
    print(summary)


if __name__ == "__main__":
    main()
