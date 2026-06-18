#!/usr/bin/env python3
"""Run controlled optimization experiments for AI Olympics 2026 Game 6."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms as T

from game4_solver import (
    ExifTranspose,
    ImageDataset,
    LABELS,
    ResizePad,
    ScratchImageCNN,
    class_weights,
    evaluate,
    seed_everything,
)


def metric_dict(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
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


def image_transforms(size: int, train: bool):
    steps = [ExifTranspose(), ResizePad(size)]
    if train:
        steps.extend(
            [
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.10
                ),
            ]
        )
    steps.extend(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return T.Compose(steps)


def make_image_loaders(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    root: Path,
    size: int,
    batch: int,
    workers: int,
    device: torch.device,
):
    options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        ImageDataset(train, root, image_transforms(size, True)),
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        **options,
    )
    validation_loader = DataLoader(
        ImageDataset(validation, root, image_transforms(size, False)),
        batch_size=batch * 2,
        shuffle=False,
        **options,
    )
    return train_loader, validation_loader


def train_image_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    root: Path,
    output_checkpoint: Path,
    device: torch.device,
    seed: int,
    size: int,
    batch: int,
    workers: int,
    epochs: int,
    label_smoothing: float,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    seed_everything(seed)
    train_loader, validation_loader = make_image_loaders(
        train, validation, root, size, batch, workers, device
    )
    model = ScratchImageCNN().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train).to(device),
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_f1 = -1.0
    training_time = 0.0
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
                enabled=device.type == "cuda",
            ):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()
        elapsed = time.perf_counter() - started
        training_time += elapsed
        metrics = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "epoch_time_sec": elapsed,
            }
        )
        print(
            f"image size={size} smoothing={label_smoothing} "
            f"epoch={epoch}/{epochs} f1={metrics['macro_f1']:.4f} "
            f"loss={history[-1]['train_loss']:.4f} sec={elapsed:.1f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), output_checkpoint)
    model.load_state_dict(
        torch.load(output_checkpoint, map_location=device, weights_only=True)
    )
    return (
        evaluate(model, validation_loader, device),
        training_time,
        pd.DataFrame(history),
    )


@torch.inference_mode()
def evaluate_tta(
    model: nn.Module,
    original_loader: DataLoader,
    flipped_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    original_probabilities = []
    flipped_probabilities = []
    truths = []
    started = time.perf_counter()
    for inputs, labels, _ in original_loader:
        logits = model(inputs.to(device, non_blocking=True))
        original_probabilities.append(
            torch.softmax(logits.float(), dim=1).cpu().numpy()
        )
        truths.append(labels.numpy())
    for inputs, _, _ in flipped_loader:
        logits = model(inputs.to(device, non_blocking=True))
        flipped_probabilities.append(
            torch.softmax(logits.float(), dim=1).cpu().numpy()
        )
    elapsed = time.perf_counter() - started
    truth = np.concatenate(truths)
    probability = (
        np.concatenate(original_probabilities)
        + np.concatenate(flipped_probabilities)
    ) / 2
    prediction = probability.argmax(axis=1)
    metrics = metric_dict(truth, prediction)
    metrics["inference_time_sec"] = elapsed
    return metrics


def text_quality_features(frame: pd.DataFrame) -> np.ndarray:
    length = frame["text"].fillna("").astype(str).str.len().to_numpy(float)
    missing = frame["text_was_missing"].astype(float).to_numpy()
    return np.column_stack([length, missing])


def run_text_experiment(
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    c_value: float,
    use_class_weights: bool,
    add_quality_features: bool,
    seed: int,
) -> tuple[dict[str, float], float, float, dict[str, object]]:
    train_features = train_embeddings
    validation_features = validation_embeddings
    metadata = {
        "add_quality_features": add_quality_features,
        "feature_mean": None,
        "feature_std": None,
    }
    if add_quality_features:
        train_quality = text_quality_features(train)
        validation_quality = text_quality_features(validation)
        mean = train_quality.mean(axis=0)
        std = train_quality.std(axis=0)
        std[std == 0] = 1
        train_quality = (train_quality - mean) / std
        validation_quality = (validation_quality - mean) / std
        train_features = np.column_stack(
            [train_embeddings, train_quality]
        )
        validation_features = np.column_stack(
            [validation_embeddings, validation_quality]
        )
        metadata["feature_mean"] = mean.tolist()
        metadata["feature_std"] = std.tolist()
    classifier = LogisticRegression(
        max_iter=3000,
        C=c_value,
        class_weight="balanced" if use_class_weights else None,
        random_state=seed,
    )
    started = time.perf_counter()
    classifier.fit(train_features, train["label"])
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = classifier.predict(validation_features)
    inference_time = time.perf_counter() - started
    metrics = metric_dict(validation["label"].to_numpy(), prediction)
    metadata["classifier"] = classifier
    return metrics, training_time, inference_time, metadata


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
            f"""# Game 6 - The Optimization Decathlon

**Team:** {team_name}

Each experiment changes one major factor. Failed experiments remain in the
history and are explicitly rejected."""
        ),
        code(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd()
experiments = pd.read_csv(OUTPUT_DIR / "optimization_experiments.csv")
best = pd.read_csv(OUTPUT_DIR / "best_models_summary.csv")

display(experiments)
display(best)"""
        ),
        code(
            """for modality, group in experiments.groupby("modality"):
    group.plot.bar(
        x="experiment_id",
        y="macro_f1",
        figsize=(9, 4),
        title=f"{modality.title()} controlled experiments",
        legend=False,
    )
    plt.ylabel("Validation macro-F1")
    plt.tight_layout()
    plt.show()

display(
    experiments.groupby(["modality", "decision"])
    .size()
    .to_frame("experiments")
)"""
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
        "--game4-output",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game4_Submission_Ded_Sec"
        ),
    )
    parser.add_argument(
        "--game5-output",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game5_Submission_Ded_Sec"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game6_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-batch", type=int, default=64)
    parser.add_argument("--image-epochs", type=int, default=6)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    root = args.root.resolve()
    game3 = args.game3_output.resolve()
    game4 = args.game4_output.resolve()
    game5 = args.game5_output.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_output = output / "best_optimized_image_model"
    text_output = output / "best_optimized_text_model"
    image_output.mkdir(exist_ok=True)
    text_output.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train = pd.read_csv(game3 / "processed_train.csv")
    validation = pd.read_csv(game3 / "processed_validation.csv")
    scratch_results = pd.read_csv(game4 / "scratch_model_results.csv")
    transfer_results = pd.read_csv(game5 / "transfer_model_results.csv")
    image_baseline = scratch_results.loc[
        scratch_results["modality"] == "image"
    ].iloc[0]
    text_baseline = transfer_results.loc[
        transfer_results["model_id"]
        == "minilm_frozen_embeddings_logreg"
    ].iloc[0]
    experiments: list[dict[str, object]] = []
    image_candidates: list[tuple[str, Path, dict[str, float], float, str]] = []

    baseline_checkpoint = (
        game4 / "scratch_image_model" / "model.pt"
    )
    baseline_model = ScratchImageCNN().to(device)
    baseline_model.load_state_dict(
        torch.load(
            baseline_checkpoint, map_location=device, weights_only=True
        )
    )
    _, baseline_validation_loader = make_image_loaders(
        train,
        validation,
        root,
        128,
        args.image_batch,
        args.workers,
        device,
    )
    baseline_metrics = evaluate(
        baseline_model, baseline_validation_loader, device
    )
    experiments.append(
        {
            "experiment_id": "I00",
            "modality": "image",
            "base_model": "scratch_image_cnn",
            "hypothesis": "The Game 4 checkpoint is the strongest honest image baseline.",
            "change_applied": "none",
            **{
                key: baseline_metrics[key]
                for key in ("accuracy", "macro_f1", "precision", "recall")
            },
            "training_time": float(image_baseline["training_time"]),
            "inference_time": baseline_metrics["inference_time_sec"],
            "decision": "accepted",
            "reason": "Reference baseline.",
        }
    )
    image_candidates.append(
        (
            "I00",
            baseline_checkpoint,
            baseline_metrics,
            float(image_baseline["training_time"]),
            "Game 4 scratch checkpoint",
        )
    )

    flip_transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(128),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    flip_loader = DataLoader(
        ImageDataset(validation, root, flip_transform),
        batch_size=args.image_batch * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    tta_metrics = evaluate_tta(
        baseline_model,
        baseline_validation_loader,
        flip_loader,
        device,
    )
    experiments.append(
        {
            "experiment_id": "I01",
            "modality": "image",
            "base_model": "scratch_image_cnn",
            "hypothesis": "Horizontal-flip TTA will reduce orientation-sensitive errors.",
            "change_applied": "average original and horizontal-flip probabilities",
            **{
                key: tta_metrics[key]
                for key in ("accuracy", "macro_f1", "precision", "recall")
            },
            "training_time": 0.0,
            "inference_time": tta_metrics["inference_time_sec"],
            "decision": (
                "accepted"
                if tta_metrics["macro_f1"]
                > baseline_metrics["macro_f1"] + 0.001
                else "rejected"
            ),
            "reason": (
                "Improved macro-F1 enough to justify doubled inference."
                if tta_metrics["macro_f1"]
                > baseline_metrics["macro_f1"] + 0.001
                else "No meaningful gain for doubled inference cost."
            ),
        }
    )
    image_candidates.append(
        (
            "I01",
            baseline_checkpoint,
            tta_metrics,
            0.0,
            "horizontal-flip test-time augmentation",
        )
    )

    smoothing_checkpoint = output / "image_label_smoothing.pt"
    smoothing_metrics, smoothing_time, smoothing_history = (
        train_image_experiment(
            train,
            validation,
            root,
            smoothing_checkpoint,
            device,
            args.seed,
            128,
            args.image_batch,
            args.workers,
            args.image_epochs,
            0.05,
        )
    )
    smoothing_history.assign(experiment_id="I02").to_csv(
        output / "image_label_smoothing_history.csv", index=False
    )
    experiments.append(
        {
            "experiment_id": "I02",
            "modality": "image",
            "base_model": "scratch_image_cnn",
            "hypothesis": "Mild label smoothing will reduce overconfidence and improve generalization.",
            "change_applied": "CrossEntropy label_smoothing=0.05",
            **{
                key: smoothing_metrics[key]
                for key in ("accuracy", "macro_f1", "precision", "recall")
            },
            "training_time": smoothing_time,
            "inference_time": smoothing_metrics["inference_time_sec"],
            "decision": (
                "accepted"
                if smoothing_metrics["macro_f1"]
                > baseline_metrics["macro_f1"] + 0.001
                else "rejected"
            ),
            "reason": (
                "Improved validation macro-F1."
                if smoothing_metrics["macro_f1"]
                > baseline_metrics["macro_f1"] + 0.001
                else "Did not beat the baseline by the acceptance margin."
            ),
        }
    )
    image_candidates.append(
        (
            "I02",
            smoothing_checkpoint,
            smoothing_metrics,
            smoothing_time,
            "label smoothing 0.05",
        )
    )

    resolution_checkpoint = output / "image_size_160.pt"
    resolution_metrics, resolution_time, resolution_history = (
        train_image_experiment(
            train,
            validation,
            root,
            resolution_checkpoint,
            device,
            args.seed,
            160,
            args.image_batch,
            args.workers,
            args.image_epochs,
            0.0,
        )
    )
    resolution_history.assign(experiment_id="I03").to_csv(
        output / "image_size_160_history.csv", index=False
    )
    experiments.append(
        {
            "experiment_id": "I03",
            "modality": "image",
            "base_model": "scratch_image_cnn",
            "hypothesis": "Increasing input size from 128 to 160 will preserve useful visual detail.",
            "change_applied": "image_size=160",
            **{
                key: resolution_metrics[key]
                for key in ("accuracy", "macro_f1", "precision", "recall")
            },
            "training_time": resolution_time,
            "inference_time": resolution_metrics["inference_time_sec"],
            "decision": (
                "accepted"
                if resolution_metrics["macro_f1"]
                > baseline_metrics["macro_f1"] + 0.001
                else "rejected"
            ),
            "reason": (
                "Improved validation macro-F1."
                if resolution_metrics["macro_f1"]
                > baseline_metrics["macro_f1"] + 0.001
                else "Extra resolution did not justify its compute cost."
            ),
        }
    )
    image_candidates.append(
        (
            "I03",
            resolution_checkpoint,
            resolution_metrics,
            resolution_time,
            "160-pixel input",
        )
    )

    image_sizes = {"I00": 128, "I01": 128, "I02": 128, "I03": 160}
    extended_specs = [
        (
            "I04",
            160,
            args.image_epochs * 2,
            "image_size_160_epochs_12",
            "The 160-pixel model was still improving when its schedule ended, "
            "so doubling the schedule should let it converge.",
            "image_size=160, epochs=12",
            "160-pixel input with a doubled training schedule",
        ),
        (
            "I05",
            192,
            args.image_epochs,
            "image_size_192",
            "Increasing input size from 160 to 192 may preserve more detail.",
            "image_size=192",
            "192-pixel input",
        ),
        (
            "I06",
            192,
            args.image_epochs * 2,
            "image_size_192_epochs_12",
            "A larger 192-pixel input may need the longer schedule to converge.",
            "image_size=192, epochs=12",
            "192-pixel input with a doubled training schedule",
        ),
    ]
    for (
        experiment_id,
        size,
        epochs,
        artifact_stem,
        hypothesis,
        change,
        label,
    ) in extended_specs:
        checkpoint = output / f"{artifact_stem}.pt"
        metrics, training_time, history = train_image_experiment(
            train,
            validation,
            root,
            checkpoint,
            device,
            args.seed,
            size,
            args.image_batch,
            args.workers,
            epochs,
            0.0,
        )
        history.assign(experiment_id=experiment_id).to_csv(
            output / f"{artifact_stem}_history.csv", index=False
        )
        improved = (
            metrics["macro_f1"] > baseline_metrics["macro_f1"] + 0.001
        )
        experiments.append(
            {
                "experiment_id": experiment_id,
                "modality": "image",
                "base_model": "scratch_image_cnn",
                "hypothesis": hypothesis,
                "change_applied": change,
                **{
                    key: metrics[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "precision",
                        "recall",
                    )
                },
                "training_time": training_time,
                "inference_time": metrics["inference_time_sec"],
                "decision": "accepted" if improved else "rejected",
                "reason": (
                    "Improved validation macro-F1."
                    if improved
                    else "Did not beat the baseline by the acceptance margin."
                ),
            }
        )
        image_candidates.append(
            (experiment_id, checkpoint, metrics, training_time, label)
        )
        image_sizes[experiment_id] = size

    print("Encoding text with the local MiniLM model...")
    encoder_path = game5 / "transfer_text_model" / "encoder"
    encoder = SentenceTransformer(str(encoder_path), device=str(device))
    started = time.perf_counter()
    train_embeddings = encoder.encode(
        train["text"].fillna("").astype(str).tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    train_embedding_time = time.perf_counter() - started
    started = time.perf_counter()
    validation_embeddings = encoder.encode(
        validation["text"].fillna("").astype(str).tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    validation_embedding_time = time.perf_counter() - started

    text_specs = [
        (
            "T00",
            1.0,
            True,
            False,
            "Reference MiniLM logistic classifier.",
            "none",
        ),
        (
            "T01",
            0.25,
            True,
            False,
            "Stronger regularization may improve generalization.",
            "logistic C=0.25",
        ),
        (
            "T02",
            4.0,
            True,
            False,
            "Weaker regularization may fit the task boundary better.",
            "logistic C=4.0",
        ),
        (
            "T03",
            1.0,
            False,
            False,
            "Removing class weights may help because validation is balanced.",
            "class_weight=None",
        ),
        (
            "T04",
            1.0,
            True,
            True,
            "Text length and missingness may add useful reliability context.",
            "append standardized text length and missing-text flag",
        ),
    ]
    text_candidates = []
    for experiment_id, c_value, weighted, quality, hypothesis, change in text_specs:
        metrics, training_time, inference_time, bundle = run_text_experiment(
            train_embeddings,
            validation_embeddings,
            train,
            validation,
            c_value,
            weighted,
            quality,
            args.seed,
        )
        decision = (
            "accepted"
            if experiment_id == "T00"
            or metrics["macro_f1"] > float(text_baseline["macro_f1"]) + 0.001
            else "rejected"
        )
        experiments.append(
            {
                "experiment_id": experiment_id,
                "modality": "text",
                "base_model": "minilm_frozen_embeddings_logreg",
                "hypothesis": hypothesis,
                "change_applied": change,
                **metrics,
                "training_time": training_time + train_embedding_time,
                "inference_time": inference_time + validation_embedding_time,
                "decision": decision,
                "reason": (
                    "Reference baseline."
                    if experiment_id == "T00"
                    else (
                        "Improved validation macro-F1."
                        if decision == "accepted"
                        else "Did not beat the baseline by the acceptance margin."
                    )
                ),
            }
        )
        text_candidates.append(
            (
                experiment_id,
                metrics,
                training_time + train_embedding_time,
                bundle,
            )
        )

    experiment_frame = pd.DataFrame(experiments)
    experiment_frame.to_csv(
        output / "optimization_experiments.csv", index=False
    )

    selected_image = max(
        image_candidates, key=lambda candidate: candidate[2]["macro_f1"]
    )
    shutil.copy2(selected_image[1], image_output / "model.pt")
    selected_image_size = image_sizes[selected_image[0]]
    (image_output / "config.json").write_text(
        json.dumps(
            {
                "architecture": "ScratchImageCNN",
                "selected_experiment": selected_image[0],
                "image_size": selected_image_size,
                "test_time_augmentation": (
                    selected_image[0] == "I01"
                ),
                "labels": LABELS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_text = max(
        text_candidates, key=lambda candidate: candidate[1]["macro_f1"]
    )
    shutil.copytree(
        encoder_path, text_output / "encoder", dirs_exist_ok=True
    )
    joblib.dump(
        {
            "classifier": selected_text[3]["classifier"],
            "add_quality_features": selected_text[3][
                "add_quality_features"
            ],
            "feature_mean": selected_text[3]["feature_mean"],
            "feature_std": selected_text[3]["feature_std"],
            "selected_experiment": selected_text[0],
        },
        text_output / "classifier_bundle.joblib",
    )

    best_summary = pd.DataFrame(
        [
            {
                "modality": "image",
                "selected_model": f"ScratchImageCNN/{selected_image[0]}",
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "final_macro_f1": selected_image[2]["macro_f1"],
                "macro_f1_gain": selected_image[2]["macro_f1"]
                - baseline_metrics["macro_f1"],
                "training_time": selected_image[3],
                "inference_time": selected_image[2][
                    "inference_time_sec"
                ],
                "selection_reason": selected_image[4],
            },
            {
                "modality": "text",
                "selected_model": f"MiniLMLogReg/{selected_text[0]}",
                "baseline_macro_f1": float(text_baseline["macro_f1"]),
                "final_macro_f1": selected_text[1]["macro_f1"],
                "macro_f1_gain": selected_text[1]["macro_f1"]
                - float(text_baseline["macro_f1"]),
                "training_time": selected_text[2],
                "inference_time": experiment_frame.loc[
                    experiment_frame["experiment_id"] == selected_text[0],
                    "inference_time",
                ].iloc[0],
                "selection_reason": (
                    "Highest measured validation macro-F1 among controlled text experiments."
                ),
            },
        ]
    )
    best_summary.to_csv(
        output / "best_models_summary.csv", index=False
    )
    accepted = experiment_frame.loc[
        experiment_frame["decision"] == "accepted", "experiment_id"
    ].tolist()
    rejected = experiment_frame.loc[
        experiment_frame["decision"] == "rejected", "experiment_id"
    ].tolist()
    summary = (
        f"Selected image experiment {selected_image[0]} with macro-F1="
        f"{selected_image[2]['macro_f1']:.4f} versus baseline "
        f"{baseline_metrics['macro_f1']:.4f}. Selected text experiment "
        f"{selected_text[0]} with macro-F1={selected_text[1]['macro_f1']:.4f} "
        f"versus baseline {float(text_baseline['macro_f1']):.4f}. Accepted "
        f"experiments: {accepted}. Rejected experiments: {rejected}. "
        "Every rejected result is retained because optimization decisions were "
        "based on measured macro-F1 and cost rather than selective reporting."
    )
    (output / "optimization_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    build_notebook(
        output / "Game_6_Optimization_Decathlon_Completed.ipynb",
        args.team_name,
        summary,
    )
    print(experiment_frame.to_string(index=False))
    print(best_summary.to_string(index=False))
    print(summary)
    print(f"Artifacts written to {output}")


if __name__ == "__main__":
    main()
