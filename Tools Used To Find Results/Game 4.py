#!/usr/bin/env python3
"""Train scratch image and text models for AI Olympics 2026 Game 4."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


LABELS = ["fake", "real"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image(root: Path, relative_path: object) -> Path:
    value = Path(str(relative_path))
    for candidate in (value, root / value):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {relative_path}")


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


class ImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        transform,
    ):
        self.frame = frame.reset_index(drop=True)
        self.paths = [
            resolve_image(root, path) for path in self.frame["image_path"]
        ]
        self.labels = [
            LABEL_TO_INDEX[str(label)] for label in self.frame["label"]
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image)
        return (
            tensor,
            self.labels[index],
            str(self.frame.at[index, "sample_id"]),
        )


TOKEN_PATTERN = re.compile(r"[a-z0-9_']+|[^\w\s]", re.IGNORECASE)


def tokenize(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value).lower()
    return TOKEN_PATTERN.findall(text)


def build_vocabulary(
    texts: pd.Series, min_frequency: int, max_vocabulary: int
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.most_common():
        if count < min_frequency or len(vocabulary) >= max_vocabulary:
            break
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(
    value: object, vocabulary: dict[str, int], max_length: int
) -> list[int]:
    unknown = vocabulary["<UNK>"]
    encoded = [vocabulary.get(token, unknown) for token in tokenize(value)]
    encoded = encoded[:max_length]
    return encoded + [0] * (max_length - len(encoded))


class TextDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_length: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.encoded = torch.tensor(
            [
                encode_text(text, vocabulary, max_length)
                for text in self.frame["text"]
            ],
            dtype=torch.long,
        )
        self.labels = torch.tensor(
            [LABEL_TO_INDEX[str(label)] for label in self.frame["label"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        return (
            self.encoded[index],
            self.labels[index],
            str(self.frame.at[index, "sample_id"]),
        )


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


class ScratchTextCNN(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        channels: int = 128,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dim, padding_idx=0
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, channels, kernel_size=kernel)
                for kernel in (3, 4, 5)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, len(LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs).transpose(1, 2)
        pooled = [
            torch.amax(torch.relu(convolution(embedded)), dim=2)
            for convolution in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def metrics_from_arrays(
    truth: np.ndarray, prediction: np.ndarray
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


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect: bool = False,
):
    model.eval()
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    started = time.perf_counter()
    for inputs, labels, ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probabilities.append(
            torch.softmax(logits.float(), dim=1).cpu().numpy()
        )
        truths.append(labels.numpy())
        sample_ids.extend(list(ids))
    elapsed = time.perf_counter() - started
    truth = np.concatenate(truths)
    probability = np.concatenate(probabilities)
    prediction = probability.argmax(axis=1)
    metrics = metrics_from_arrays(truth, prediction)
    metrics["inference_time_sec"] = elapsed
    metrics["inference_ms_per_sample"] = 1000 * elapsed / max(len(truth), 1)
    if collect:
        return metrics, sample_ids, truth, prediction, probability
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    checkpoint_path: Path,
    modality: str,
    patience: int,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, object]] = []
    best_f1 = -math.inf
    bad_epochs = 0
    total_training_time = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()
        epoch_time = time.perf_counter() - started
        total_training_time += epoch_time
        validation_metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_accuracy": validation_metrics["accuracy"],
            "val_macro_f1": validation_metrics["macro_f1"],
            "val_precision": validation_metrics["precision"],
            "val_recall": validation_metrics["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        print(
            f"{modality} epoch {epoch}/{epochs} "
            f"loss={row['train_loss']:.4f} "
            f"f1={row['val_macro_f1']:.4f} "
            f"acc={row['val_accuracy']:.4f} "
            f"sec={epoch_time:.1f}"
        )
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"{modality}: early stopping after epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    final_metrics = evaluate(model, validation_loader, device)
    return pd.DataFrame(history), final_metrics, total_training_time


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = np.bincount(
        [LABEL_TO_INDEX[str(label)] for label in frame["label"]],
        minlength=len(LABELS),
    )
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def prediction_frame(
    modality: str,
    source: pd.DataFrame,
    ids: list[str],
    truth: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    lookup = source.set_index("sample_id")
    confidence = probability.max(axis=1)
    rows = []
    for index, sample_id in enumerate(ids):
        true_label = LABELS[int(truth[index])]
        predicted_label = LABELS[int(prediction[index])]
        correct = true_label == predicted_label
        if not correct and confidence[index] >= 0.80:
            case_type = "high_confidence_error"
        elif not correct:
            case_type = "error"
        elif confidence[index] < 0.60:
            case_type = "uncertain_correct"
        else:
            case_type = "correct"
        row = lookup.loc[sample_id]
        issue_parts = []
        for column in (
            "text_was_missing",
            "image_low_resolution",
            "image_low_sharpness",
            "image_extreme_brightness",
        ):
            if column in row.index and bool(row[column]):
                issue_parts.append(column)
        rows.append(
            {
                "sample_id": sample_id,
                "modality": modality,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": float(confidence[index]),
                "case_type": case_type,
                "observed_input_flags": "|".join(issue_parts) or "none",
                "text": row.get("text", ""),
                "image_path": row.get("image_path", ""),
            }
        )
    frame = pd.DataFrame(rows)
    selected = pd.concat(
        [
            frame.loc[frame["case_type"] == "high_confidence_error"]
            .sort_values("confidence", ascending=False)
            .head(50),
            frame.loc[frame["case_type"] == "error"]
            .sort_values("confidence", ascending=False)
            .head(50),
            frame.loc[frame["case_type"] == "uncertain_correct"]
            .sort_values("confidence")
            .head(30),
        ],
        ignore_index=True,
    )
    return selected


def build_notebook(path: Path, team_name: str, summary: str) -> None:
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
            f"""# Game 4 - The Zero-to-Hero Sprint

**Team:** {team_name}

Both models were initialized randomly. No pretrained weights, encoders, or
external embeddings were used."""
        ),
        code(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd()
results = pd.read_csv(OUTPUT_DIR / "scratch_model_results.csv")
image_history = pd.read_csv(OUTPUT_DIR / "image_training_history.csv")
text_history = pd.read_csv(OUTPUT_DIR / "text_training_history.csv")
errors = pd.read_csv(OUTPUT_DIR / "scratch_error_analysis.csv")

display(results)
display(errors.head(20))"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for history, label in [
    (image_history, "image CNN"),
    (text_history, "TextCNN"),
]:
    axes[0].plot(
        history["epoch"], history["train_loss"], marker="o", label=label
    )
    axes[1].plot(
        history["epoch"], history["val_macro_f1"], marker="o", label=label
    )
axes[0].set_title("Training loss")
axes[1].set_title("Validation macro-F1")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.show()

display(errors.groupby(["modality", "case_type"]).size().to_frame("cases"))"""
        ),
        markdown("## Final Conclusion\n\n" + summary),
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
        "--game3-output",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game3_Submission_Ded_Sec"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game4_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--image-batch", type=int, default=64)
    parser.add_argument("--image-epochs", type=int, default=8)
    parser.add_argument("--text-batch", type=int, default=256)
    parser.add_argument("--text-epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    root = args.root.resolve()
    game3 = args.game3_output.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_model_dir = output / "scratch_image_model"
    text_model_dir = output / "scratch_text_model"
    image_model_dir.mkdir(exist_ok=True)
    text_model_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train = pd.read_csv(game3 / "processed_train.csv")
    validation = pd.read_csv(game3 / "processed_validation.csv")
    weights = class_weights(train)

    train_transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(args.image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    validation_transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(args.image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image_train_dataset = ImageDataset(train, root, train_transform)
    image_validation_dataset = ImageDataset(
        validation, root, validation_transform
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    image_train_loader = DataLoader(
        image_train_dataset,
        batch_size=args.image_batch,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    image_validation_loader = DataLoader(
        image_validation_dataset,
        batch_size=args.image_batch * 2,
        shuffle=False,
        **loader_options,
    )
    image_model = ScratchImageCNN()
    image_checkpoint = image_model_dir / "model.pt"
    image_history, image_metrics, image_training_time = train_model(
        image_model,
        image_train_loader,
        image_validation_loader,
        device,
        args.image_epochs,
        3e-4,
        1e-4,
        weights,
        image_checkpoint,
        "image",
        args.patience,
    )
    image_model.load_state_dict(
        torch.load(image_checkpoint, map_location=device, weights_only=True)
    )
    (
        image_metrics,
        image_ids,
        image_truth,
        image_prediction,
        image_probability,
    ) = evaluate(image_model.to(device), image_validation_loader, device, True)
    pd.DataFrame(
        confusion_matrix(
            image_truth, image_prediction, labels=range(len(LABELS))
        ),
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(image_model_dir / "confusion_matrix.csv")
    (image_model_dir / "config.json").write_text(
        json.dumps(
            {
                "architecture": "ScratchImageCNN",
                "image_size": args.image_size,
                "labels": LABELS,
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
                "resize": "aspect_preserving_pad",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    vocabulary = build_vocabulary(train["text"], 2, 30000)
    max_length = 96
    text_train_dataset = TextDataset(train, vocabulary, max_length)
    text_validation_dataset = TextDataset(
        validation, vocabulary, max_length
    )
    text_train_loader = DataLoader(
        text_train_dataset,
        batch_size=args.text_batch,
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    text_validation_loader = DataLoader(
        text_validation_dataset,
        batch_size=args.text_batch * 2,
        shuffle=False,
        **loader_options,
    )
    text_model = ScratchTextCNN(len(vocabulary))
    text_checkpoint = text_model_dir / "model.pt"
    text_history, text_metrics, text_training_time = train_model(
        text_model,
        text_train_loader,
        text_validation_loader,
        device,
        args.text_epochs,
        1e-3,
        1e-4,
        weights,
        text_checkpoint,
        "text",
        args.patience,
    )
    text_model.load_state_dict(
        torch.load(text_checkpoint, map_location=device, weights_only=True)
    )
    (
        text_metrics,
        text_ids,
        text_truth,
        text_prediction,
        text_probability,
    ) = evaluate(text_model.to(device), text_validation_loader, device, True)
    pd.DataFrame(
        confusion_matrix(
            text_truth, text_prediction, labels=range(len(LABELS))
        ),
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(text_model_dir / "confusion_matrix.csv")
    torch.save(
        {
            "state_dict": text_model.state_dict(),
            "vocabulary": vocabulary,
            "max_length": max_length,
            "labels": LABELS,
            "architecture": {
                "embedding_dim": 128,
                "channels": 128,
                "kernels": [3, 4, 5],
            },
        },
        text_model_dir / "model_with_vocabulary.pt",
    )

    image_history.to_csv(
        output / "image_training_history.csv", index=False
    )
    text_history.to_csv(output / "text_training_history.csv", index=False)
    image_parameters = sum(
        parameter.numel() for parameter in image_model.parameters()
    )
    text_parameters = sum(
        parameter.numel() for parameter in text_model.parameters()
    )
    results = pd.DataFrame(
        [
            {
                "model_id": "scratch_image_cnn",
                "modality": "image",
                "architecture": "4-block CNN, random initialization",
                "accuracy": image_metrics["accuracy"],
                "macro_f1": image_metrics["macro_f1"],
                "precision": image_metrics["precision"],
                "recall": image_metrics["recall"],
                "training_time": image_training_time,
                "inference_time": image_metrics["inference_time_sec"],
                "notes": (
                    f"{image_parameters:,} parameters; no pretrained weights; "
                    "aspect-preserving resize and padding"
                ),
            },
            {
                "model_id": "scratch_text_cnn",
                "modality": "text",
                "architecture": "learned embedding + 3/4/5-kernel TextCNN",
                "accuracy": text_metrics["accuracy"],
                "macro_f1": text_metrics["macro_f1"],
                "precision": text_metrics["precision"],
                "recall": text_metrics["recall"],
                "training_time": text_training_time,
                "inference_time": text_metrics["inference_time_sec"],
                "notes": (
                    f"{text_parameters:,} parameters; vocabulary={len(vocabulary):,}; "
                    "random embeddings"
                ),
            },
        ]
    )
    results.to_csv(output / "scratch_model_results.csv", index=False)

    error_analysis = pd.concat(
        [
            prediction_frame(
                "image",
                validation,
                image_ids,
                image_truth,
                image_prediction,
                image_probability,
            ),
            prediction_frame(
                "text",
                validation,
                text_ids,
                text_truth,
                text_prediction,
                text_probability,
            ),
        ],
        ignore_index=True,
    )
    error_analysis.to_csv(
        output / "scratch_error_analysis.csv", index=False
    )

    best = results.sort_values("macro_f1", ascending=False).iloc[0]
    summary = (
        f"The scratch image CNN reached accuracy={image_metrics['accuracy']:.4f} "
        f"and macro-F1={image_metrics['macro_f1']:.4f}. The scratch TextCNN "
        f"reached accuracy={text_metrics['accuracy']:.4f} and macro-F1="
        f"{text_metrics['macro_f1']:.4f}. The stronger scratch modality was "
        f"{best['modality']}. Image limitations include low-resolution inputs, "
        "quality variation, and learning visual semantics from only 9,000 "
        "examples. Text limitations include a task-specific vocabulary, short "
        "sequence truncation, and no pretrained language knowledge. Both "
        "models are honest random-initialization baselines for Game 5."
    )
    (output / "scratch_models_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    build_notebook(
        output / "Game_4_Zero_to_Hero_Sprint_Completed.ipynb",
        args.team_name,
        summary,
    )
    print(results.to_string(index=False))
    print(summary)
    print(f"Artifacts written to {output}")


if __name__ == "__main__":
    main()
