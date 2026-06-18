#!/usr/bin/env python3
"""Train efficient transfer-learning models for AI Olympics 2026 Game 5."""

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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision import transforms as T

from game4_solver import (
    ExifTranspose,
    ImageDataset,
    LABELS,
    ResizePad,
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


def train_image_phase(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
    epochs: int,
    learning_rate: float,
    checkpoint: Path,
    phase_name: str,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_f1 = -1.0
    total_time = 0.0
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
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()
        elapsed = time.perf_counter() - started
        total_time += elapsed
        metrics = evaluate(model, validation_loader, device)
        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": elapsed,
        }
        history.append(row)
        print(
            f"{phase_name} epoch {epoch}/{epochs} "
            f"loss={row['train_loss']:.4f} "
            f"f1={row['val_macro_f1']:.4f} "
            f"acc={row['val_accuracy']:.4f} "
            f"sec={elapsed:.1f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return pd.DataFrame(history), evaluate(model, validation_loader, device), total_time


def directory_size_mb(path: Path) -> float:
    return sum(
        file.stat().st_size for file in path.rglob("*") if file.is_file()
    ) / 1e6


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
            f"""# Game 5 - The Transfer Relay

**Team:** {team_name}

The image experiment compares frozen and partially fine-tuned pretrained
features. The text experiment keeps the MiniLM encoder frozen and trains only
the task classifier."""
        ),
        code(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd()
results = pd.read_csv(OUTPUT_DIR / "transfer_model_results.csv")
comparison = pd.read_csv(OUTPUT_DIR / "scratch_vs_transfer_comparison.csv")
efficiency = pd.read_csv(OUTPUT_DIR / "efficiency_comparison.csv")
history = pd.read_csv(OUTPUT_DIR / "image_transfer_history.csv")

display(results)
display(comparison)
display(efficiency)"""
        ),
        code(
            """for phase, group in history.groupby("phase"):
    plt.plot(
        range(1, len(group) + 1),
        group["val_macro_f1"],
        marker="o",
        label=phase,
    )
plt.xlabel("Epoch within phase")
plt.ylabel("Validation macro-F1")
plt.title("MobileNetV3 transfer phases")
plt.legend()
plt.tight_layout()
plt.show()

comparison.plot.bar(
    x="modality",
    y=["scratch_macro_f1", "transfer_macro_f1"],
    figsize=(8, 4),
    title="Scratch vs transfer",
)
plt.tight_layout()
plt.show()"""
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
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game5_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--image-batch", type=int, default=64)
    parser.add_argument("--frozen-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    root = args.root.resolve()
    game3 = args.game3_output.resolve()
    game4 = args.game4_output.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "transfer_image_model"
    text_dir = output / "transfer_text_model"
    image_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train = pd.read_csv(game3 / "processed_train.csv")
    validation = pd.read_csv(game3 / "processed_validation.csv")
    weights = class_weights(train)

    image_train_transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(args.image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    image_validation_transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(args.image_size),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        ImageDataset(train, root, image_train_transform),
        batch_size=args.image_batch,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        ImageDataset(validation, root, image_validation_transform),
        batch_size=args.image_batch * 2,
        shuffle=False,
        **loader_options,
    )

    pretrained_weights = MobileNet_V3_Small_Weights.DEFAULT
    image_model = models.mobilenet_v3_small(weights=pretrained_weights)
    image_model.classifier[3] = nn.Linear(
        image_model.classifier[3].in_features, len(LABELS)
    )
    image_model = image_model.to(device)
    for parameter in image_model.features.parameters():
        parameter.requires_grad = False
    frozen_checkpoint = image_dir / "frozen_head.pt"
    frozen_history, frozen_metrics, frozen_time = train_image_phase(
        image_model,
        train_loader,
        validation_loader,
        device,
        weights,
        args.frozen_epochs,
        1e-3,
        frozen_checkpoint,
        "frozen_backbone",
    )
    frozen_trainable = sum(
        parameter.numel()
        for parameter in image_model.parameters()
        if parameter.requires_grad
    )

    for block in image_model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    finetuned_checkpoint = image_dir / "partially_finetuned.pt"
    finetune_history, finetuned_metrics, finetune_time = train_image_phase(
        image_model,
        train_loader,
        validation_loader,
        device,
        weights,
        args.finetune_epochs,
        8e-5,
        finetuned_checkpoint,
        "last_three_blocks_trainable",
    )
    finetuned_trainable = sum(
        parameter.numel()
        for parameter in image_model.parameters()
        if parameter.requires_grad
    )
    image_history = pd.concat(
        [frozen_history, finetune_history], ignore_index=True
    )
    image_history.to_csv(
        output / "image_transfer_history.csv", index=False
    )
    if finetuned_metrics["macro_f1"] >= frozen_metrics["macro_f1"]:
        selected_image_checkpoint = finetuned_checkpoint
        selected_image_metrics = finetuned_metrics
        selected_image_phase = "partially_finetuned"
        selected_image_time = frozen_time + finetune_time
    else:
        selected_image_checkpoint = frozen_checkpoint
        selected_image_metrics = frozen_metrics
        selected_image_phase = "frozen_backbone"
        selected_image_time = frozen_time
    shutil.copy2(selected_image_checkpoint, image_dir / "model.pt")
    image_model.load_state_dict(
        torch.load(
            selected_image_checkpoint,
            map_location=device,
            weights_only=True,
        )
    )
    image_eval, _, image_truth, image_prediction, image_probability = evaluate(
        image_model, validation_loader, device, collect=True
    )
    pd.DataFrame(
        confusion_matrix(
            image_truth, image_prediction, labels=range(len(LABELS))
        ),
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(image_dir / "confusion_matrix.csv")
    total_image_parameters = sum(
        parameter.numel() for parameter in image_model.parameters()
    )
    (image_dir / "config.json").write_text(
        json.dumps(
            {
                "architecture": "mobilenet_v3_small",
                "pretrained_source": "ImageNet-1K V1",
                "selected_phase": selected_image_phase,
                "image_size": args.image_size,
                "labels": LABELS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Loading pretrained MiniLM sentence encoder...")
    text_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=str(device),
    )
    train_texts = train["text"].fillna("").astype(str).tolist()
    validation_texts = validation["text"].fillna("").astype(str).tolist()
    started = time.perf_counter()
    train_embeddings = text_encoder.encode(
        train_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encoding_train_time = time.perf_counter() - started
    text_classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        C=1.0,
    )
    started = time.perf_counter()
    text_classifier.fit(train_embeddings, train["label"])
    classifier_training_time = time.perf_counter() - started
    started = time.perf_counter()
    validation_embeddings = text_encoder.encode(
        validation_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    text_probability = text_classifier.predict_proba(validation_embeddings)
    text_prediction = text_classifier.classes_[
        text_probability.argmax(axis=1)
    ]
    text_inference_time = time.perf_counter() - started
    text_metrics = metric_dict(
        validation["label"].to_numpy(), text_prediction
    )
    text_encoder.save_pretrained(str(text_dir / "encoder"))
    joblib.dump(text_classifier, text_dir / "classifier.joblib")
    np.save(text_dir / "validation_embeddings.npy", validation_embeddings)
    pd.DataFrame(
        confusion_matrix(
            validation["label"],
            text_prediction,
            labels=LABELS,
        ),
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(text_dir / "confusion_matrix.csv")
    text_train_time = encoding_train_time + classifier_training_time

    image_size_mb = (image_dir / "model.pt").stat().st_size / 1e6
    text_size_mb = directory_size_mb(text_dir / "encoder") + (
        text_dir / "classifier.joblib"
    ).stat().st_size / 1e6
    transfer_results = pd.DataFrame(
        [
            {
                "model_id": "mobilenet_v3_small_frozen",
                "modality": "image",
                "pretrained_source": "ImageNet-1K V1",
                "frozen_components": "all feature blocks",
                "trainable_components": f"classifier ({frozen_trainable:,} params)",
                "accuracy": frozen_metrics["accuracy"],
                "macro_f1": frozen_metrics["macro_f1"],
                "precision": frozen_metrics["precision"],
                "recall": frozen_metrics["recall"],
                "training_time": frozen_time,
                "inference_time": frozen_metrics["inference_time_sec"],
                "model_size": image_size_mb,
            },
            {
                "model_id": "mobilenet_v3_small_partial_finetune",
                "modality": "image",
                "pretrained_source": "ImageNet-1K V1",
                "frozen_components": "early feature blocks",
                "trainable_components": f"classifier + last 3 blocks ({finetuned_trainable:,} params)",
                "accuracy": finetuned_metrics["accuracy"],
                "macro_f1": finetuned_metrics["macro_f1"],
                "precision": finetuned_metrics["precision"],
                "recall": finetuned_metrics["recall"],
                "training_time": frozen_time + finetune_time,
                "inference_time": finetuned_metrics["inference_time_sec"],
                "model_size": image_size_mb,
            },
            {
                "model_id": "minilm_frozen_embeddings_logreg",
                "modality": "text",
                "pretrained_source": "sentence-transformers/all-MiniLM-L6-v2",
                "frozen_components": "all MiniLM encoder layers",
                "trainable_components": "balanced logistic classifier",
                "accuracy": text_metrics["accuracy"],
                "macro_f1": text_metrics["macro_f1"],
                "precision": text_metrics["precision"],
                "recall": text_metrics["recall"],
                "training_time": text_train_time,
                "inference_time": text_inference_time,
                "model_size": text_size_mb,
            },
        ]
    )
    transfer_results.to_csv(
        output / "transfer_model_results.csv", index=False
    )

    scratch = pd.read_csv(game4 / "scratch_model_results.csv")
    selected_transfer = {
        "image": transfer_results.loc[
            transfer_results["model_id"]
            == (
                "mobilenet_v3_small_partial_finetune"
                if selected_image_phase == "partially_finetuned"
                else "mobilenet_v3_small_frozen"
            )
        ].iloc[0],
        "text": transfer_results.loc[
            transfer_results["model_id"]
            == "minilm_frozen_embeddings_logreg"
        ].iloc[0],
    }
    comparison_rows = []
    for modality in ("image", "text"):
        scratch_row = scratch.loc[scratch["modality"] == modality].iloc[0]
        transfer_row = selected_transfer[modality]
        gain = transfer_row["macro_f1"] - scratch_row["macro_f1"]
        cost_difference = (
            transfer_row["training_time"] - scratch_row["training_time"]
        )
        comparison_rows.append(
            {
                "modality": modality,
                "scratch_model": scratch_row["model_id"],
                "transfer_model": transfer_row["model_id"],
                "scratch_macro_f1": scratch_row["macro_f1"],
                "transfer_macro_f1": transfer_row["macro_f1"],
                "macro_f1_gain": gain,
                "cost_difference": cost_difference,
                "final_observation": (
                    "Transfer justified by measurable macro-F1 gain."
                    if gain > 0.005
                    else "Transfer gain is small; prefer the cheaper model unless later fusion benefits."
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output / "scratch_vs_transfer_comparison.csv", index=False
    )

    efficiency = pd.concat(
        [
            scratch.assign(
                pretrained_source="none",
                model_size=np.nan,
                parameter_count=np.nan,
            )[
                [
                    "model_id",
                    "modality",
                    "pretrained_source",
                    "accuracy",
                    "macro_f1",
                    "training_time",
                    "inference_time",
                    "model_size",
                    "parameter_count",
                ]
            ],
            pd.DataFrame(
                [
                    {
                        "model_id": selected_transfer["image"]["model_id"],
                        "modality": "image",
                        "pretrained_source": "ImageNet-1K V1",
                        "accuracy": image_eval["accuracy"],
                        "macro_f1": image_eval["macro_f1"],
                        "training_time": selected_image_time,
                        "inference_time": image_eval["inference_time_sec"],
                        "model_size": image_size_mb,
                        "parameter_count": total_image_parameters,
                    },
                    {
                        "model_id": selected_transfer["text"]["model_id"],
                        "modality": "text",
                        "pretrained_source": "all-MiniLM-L6-v2",
                        "accuracy": text_metrics["accuracy"],
                        "macro_f1": text_metrics["macro_f1"],
                        "training_time": text_train_time,
                        "inference_time": text_inference_time,
                        "model_size": text_size_mb,
                        "parameter_count": "encoder frozen + 385x2 classifier",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    efficiency.to_csv(output / "efficiency_comparison.csv", index=False)

    best_image = selected_transfer["image"]
    best_text = selected_transfer["text"]
    summary = (
        f"Best image transfer model: {best_image['model_id']} with macro-F1="
        f"{best_image['macro_f1']:.4f}; scratch image macro-F1="
        f"{comparison.loc[comparison['modality'] == 'image', 'scratch_macro_f1'].iloc[0]:.4f}. "
        f"Best text transfer model: {best_text['model_id']} with macro-F1="
        f"{best_text['macro_f1']:.4f}; scratch text macro-F1="
        f"{comparison.loc[comparison['modality'] == 'text', 'scratch_macro_f1'].iloc[0]:.4f}. "
        "MobileNetV3 was selected for a strong accuracy-to-cost ratio on 4 GB "
        "VRAM. MiniLM was kept frozen to obtain semantic representations with "
        "low task-training cost. Final selections balance validation macro-F1, "
        "inference time, model size, and training cost."
    )
    (output / "transfer_learning_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    build_notebook(
        output / "Game_5_The_Transfer_Relay_Completed.ipynb",
        args.team_name,
        summary,
    )
    print(transfer_results.to_string(index=False))
    print(comparison.to_string(index=False))
    print(summary)
    print(f"Artifacts written to {output}")


if __name__ == "__main__":
    main()
