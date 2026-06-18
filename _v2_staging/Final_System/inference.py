#!/usr/bin/env python3
r"""Ded_Sec Game 8 Final System - standalone Phase 2 (Boss-Test) inference.

Run from this folder:

    python inference.py --test-csv path\to\hidden_test.csv

If the organizers ship new images in a separate folder, point at it too:

    python inference.py --test-csv hidden.csv --images-dir path\to\new_images

Input CSV contract : sample_id,image_path,text
Output CSV contract: sample_id,predicted_label,confidence,primary_evidence,reviewer_flag

Everything the system needs ships inside this folder:

    model_files/image_model   optimized 160px ScratchImageCNN (Game 6)
    model_files/text_model    local MiniLM encoder + classifier bundle (Game 6)
    model_files/clip_model    local CLIP encoder scoring semantic image-text
                              agreement (clip_similarity fusion feature)
    reference_data/           Game 8 train/validation + canonical pair manifests
    score_cache/              precomputed expert and CLIP scores; only unseen
                              images, texts, and pairs are scored at run time

No network access is required: every encoder is loaded from its local folder,
never downloaded.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
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


SYSTEM_ROOT = Path(__file__).resolve().parent
IMAGE_MODEL_DIR = SYSTEM_ROOT / "model_files" / "image_model"
TEXT_MODEL_DIR = SYSTEM_ROOT / "model_files" / "text_model"
CLIP_MODEL_DIR = SYSTEM_ROOT / "model_files" / "clip_model"
REFERENCE_DIR = SYSTEM_ROOT / "reference_data"
CANONICAL_DIR = REFERENCE_DIR / "canonical"
CACHE_DIR = SYSTEM_ROOT / "score_cache"

LABELS = ["fake", "real"]
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

# Folders searched (in order) when an image referenced by the test CSV is not
# already in the score cache. Filled in main() from --images-dir and the test
# CSV location.
EXTRA_IMAGE_DIRS: list[Path] = []


class ExifTranspose:
    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image)


class ResizePad:
    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.size / max(width, 1), self.size / max(height, 1))
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        resized = image.resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - target_width) // 2
        top = (self.size - target_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


class ScratchImageCNN(nn.Module):
    def __init__(self, dropout: float = 0.30):
        super().__init__()

        def block(input_channels: int, output_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, len(LABELS)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


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


def resolve_image(value: object) -> Path:
    relative = Path(str(value))
    candidates = [relative]
    for base in EXTRA_IMAGE_DIRS:
        candidates.extend(
            [
                base / relative,
                base / relative.name,
                base / "data" / "images" / relative.name,
                base / "images" / relative.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(base) for base in EXTRA_IMAGE_DIRS)
    raise FileNotFoundError(
        f"Image not found: {value} (searched relative paths and: {searched}). "
        "Pass the image folder with --images-dir."
    )


def image_probabilities(
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    from torchvision import transforms as T

    config = json.loads(
        (IMAGE_MODEL_DIR / "config.json").read_text(encoding="utf-8")
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
    unique = frame[["normalized_image_path", "image_path"]].drop_duplicates(
        "normalized_image_path"
    ).reset_index(drop=True)
    paths = [resolve_image(path) for path in unique["image_path"]]
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
            IMAGE_MODEL_DIR / "model.pt",
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
    result = unique[["normalized_image_path"]].copy()
    result["image_p_fake"] = probability[:, LABELS.index("fake")]
    result["image_p_real"] = probability[:, LABELS.index("real")]
    print(f"image inference: {len(result)} unique images in {elapsed:.1f}s")
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
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    unique = (
        frame[["model_text", "text_was_missing"]]
        .drop_duplicates("model_text")
        .reset_index(drop=True)
    )
    bundle = joblib.load(TEXT_MODEL_DIR / "classifier_bundle.joblib")
    encoder = SentenceTransformer(
        str(TEXT_MODEL_DIR / "encoder"), device=str(device)
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
    frame: pd.DataFrame, device: torch.device, batch: int
) -> pd.DataFrame:
    """Semantic image-text alignment score for each (image, text) pair.

    Mismatched real components score low even when both components are
    unknown to the canonical manifest, which is the regime where the pure
    relation features carry no signal.
    """
    started = time.perf_counter()
    model = SentenceTransformer(str(CLIP_MODEL_DIR), device=str(device))

    unique_images = frame[
        ["normalized_image_path", "image_path"]
    ].drop_duplicates("normalized_image_path")
    image_keys = unique_images["normalized_image_path"].tolist()
    image_paths = [resolve_image(path) for path in unique_images["image_path"]]
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


def canonical_manifest() -> tuple[set[tuple[str, str]], set[str], set[str]]:
    sources = [
        pd.read_csv(CANONICAL_DIR / "game1_corrected_train.csv"),
        pd.read_csv(CANONICAL_DIR / "game1_corrected_validation.csv"),
        pd.read_csv(CANONICAL_DIR / "processed_train.csv"),
        pd.read_csv(CANONICAL_DIR / "processed_validation.csv"),
        pd.read_csv(CANONICAL_DIR / "public_test.csv"),
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
    result["mean_p_fake"] = (result["image_p_fake"] + result["text_p_fake"]) / 2
    result["min_p_fake"] = result[["image_p_fake", "text_p_fake"]].min(axis=1)
    result["max_p_fake"] = result[["image_p_fake", "text_p_fake"]].max(axis=1)
    result["probability_product"] = (
        result["image_p_fake"] * result["text_p_fake"]
    )
    result["probability_gap"] = (
        result["image_p_fake"] - result["text_p_fake"]
    ).abs()
    result["model_agreement"] = (
        (result["image_p_fake"] >= 0.5) == (result["text_p_fake"] >= 0.5)
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
            precision_score(truth, prediction, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(truth, prediction, average="macro", zero_division=0)
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
                "image" if image_conf[index] > text_conf[index] else "text"
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
            or (mismatch[index] and confidence[index] < 0.90)
        )
    return evidence, reviewer


def cached_scores(
    combined: pd.DataFrame,
    device: torch.device,
    image_batch: int,
    text_batch: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image_cache = CACHE_DIR / "image_scores.csv"
    text_cache = CACHE_DIR / "text_scores.csv"

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
        new_scores = image_probabilities(missing, device, image_batch)
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
        new_scores = text_probabilities(missing, device, text_batch)
        text_scores = pd.concat(
            [text_scores, new_scores], ignore_index=True
        ).drop_duplicates("model_text")
        text_scores.to_csv(text_cache, index=False)
    else:
        print("text inference: all texts served from cache")

    clip_cache = CACHE_DIR / "clip_scores.csv"
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
        new_scores = clip_similarities(missing, device, image_batch)
        clip_scores = pd.concat(
            [clip_scores, new_scores], ignore_index=True
        ).drop_duplicates(pair_keys)
        clip_scores.to_csv(clip_cache, index=False)
    else:
        print("clip inference: all pairs served from cache")

    return image_scores, text_scores, clip_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Real/Fake for a new test CSV with the Ded_Sec "
        "Game 8 multimodal system."
    )
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("phase2_predictions.csv")
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional folder with new images shipped alongside the test CSV.",
    )
    parser.add_argument("--image-batch", type=int, default=128)
    parser.add_argument("--text-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is present."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}")

    if args.images_dir is not None:
        EXTRA_IMAGE_DIRS.append(args.images_dir.resolve())
    EXTRA_IMAGE_DIRS.append(args.test_csv.resolve().parent)
    EXTRA_IMAGE_DIRS.append(SYSTEM_ROOT)

    test_raw = pd.read_csv(args.test_csv)
    required_inputs = {"sample_id", "image_path", "text"}
    missing_columns = required_inputs - set(test_raw.columns)
    if missing_columns:
        raise SystemExit(
            f"{args.test_csv} is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    print(f"hidden test: {len(test_raw)} rows from {args.test_csv}")

    train = prepare_frame(pd.read_csv(REFERENCE_DIR / "game8_train.csv"))
    validation = prepare_frame(
        pd.read_csv(REFERENCE_DIR / "game8_validation.csv")
    )
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
        combined, device, args.image_batch, args.text_batch
    )
    canonical_pairs, known_images, known_texts = canonical_manifest()
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
    validation_probabilities = {
        "weighted_probability_fusion": weighted_validation,
        "canonical_relation_gate": relation_gate_validation,
        "relation_logistic_stack": logistic_validation,
        "relation_gradient_boosting": boosted_validation,
        "relation_stack_average": (logistic_validation + boosted_validation)
        / 2,
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
