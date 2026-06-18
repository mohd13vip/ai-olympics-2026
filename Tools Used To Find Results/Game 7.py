#!/usr/bin/env python3
"""Generate explainability galleries and trust analysis for Game 7."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torchvision import transforms as T

from game4_solver import (
    ExifTranspose,
    ImageDataset,
    LABELS,
    ResizePad,
    ScratchImageCNN,
    evaluate,
    resolve_image,
    seed_everything,
)


def quality_features(
    frame: pd.DataFrame,
    mean: list[float] | None,
    std: list[float] | None,
) -> np.ndarray:
    length = frame["text"].fillna("").astype(str).str.len().to_numpy(float)
    missing = frame["text_was_missing"].astype(float).to_numpy()
    features = np.column_stack([length, missing])
    if mean is not None and std is not None:
        features = (features - np.asarray(mean)) / np.asarray(std)
    return features


def text_model_features(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    bundle: dict[str, object],
) -> np.ndarray:
    if not bundle["add_quality_features"]:
        return embeddings
    return np.column_stack(
        [
            embeddings,
            quality_features(
                frame,
                bundle["feature_mean"],
                bundle["feature_std"],
            ),
        ]
    )


def prediction_cases(
    frame: pd.DataFrame,
    predictions: list[str] | np.ndarray,
    confidences: np.ndarray,
) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_label"] = predictions
    result["confidence"] = confidences
    result["correct"] = result["label"] == result["predicted_label"]
    result["case_type"] = np.select(
        [
            (~result["correct"]) & (result["confidence"] >= 0.80),
            ~result["correct"],
            result["confidence"] < 0.60,
        ],
        ["high_confidence_error", "incorrect", "uncertain"],
        default="correct",
    )
    return result


def select_cases(
    frame: pd.DataFrame,
    limit_per_type: int = 4,
    target_total: int = 16,
) -> pd.DataFrame:
    selected = []
    high_errors = frame.loc[
        frame["case_type"] == "high_confidence_error"
    ].sort_values("confidence", ascending=False)
    selected.append(high_errors.head(limit_per_type))
    high_ids = set(high_errors.head(limit_per_type)["sample_id"])
    selected.append(
        frame.loc[
            (frame["case_type"] == "incorrect")
            & (~frame["sample_id"].isin(high_ids))
        ]
        .sort_values("confidence", ascending=False)
        .head(limit_per_type)
    )
    selected.append(
        frame.loc[frame["case_type"] == "uncertain"]
        .sort_values("confidence")
        .head(limit_per_type)
    )
    for label in LABELS:
        selected.append(
            frame.loc[
                (frame["case_type"] == "correct")
                & (frame["label"] == label)
            ]
            .sort_values("confidence", ascending=False)
            .head(2)
        )
    combined = (
        pd.concat(selected, ignore_index=True)
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )
    if len(combined) < target_total:
        # On easy data there are too few error/uncertain cases to fill the
        # 16-case submission quota; backfill with the remaining cases,
        # preferring leftover errors and then the least-confident correct
        # predictions, so the gallery count stays at the validator contract.
        priority = {
            "high_confidence_error": 0,
            "incorrect": 1,
            "uncertain": 2,
            "correct": 3,
        }
        backfill = frame.loc[
            ~frame["sample_id"].isin(set(combined["sample_id"]))
        ].copy()
        backfill["backfill_rank"] = backfill["case_type"].map(priority)
        backfill = (
            backfill.sort_values(["backfill_rank", "confidence", "sample_id"])
            .head(target_total - len(combined))
            .drop(columns="backfill_rank")
        )
        combined = pd.concat([combined, backfill], ignore_index=True)
    return combined.reset_index(drop=True)


class GradCAM:
    def __init__(self, model: ScratchImageCNN):
        self.activations = None
        self.gradients = None
        target = model.features[-1][3]
        self.forward_handle = target.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        model: ScratchImageCNN,
        tensor: torch.Tensor,
        class_index: int,
    ) -> np.ndarray:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, class_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1))
        heatmap = F.interpolate(
            heatmap.unsqueeze(1),
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        return heatmap.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def heatmap_geometry(heatmap: np.ndarray) -> dict[str, float]:
    height, width = heatmap.shape
    top, bottom = height // 4, 3 * height // 4
    left, right = width // 4, 3 * width // 4
    center_mask = np.zeros_like(heatmap, dtype=bool)
    center_mask[top:bottom, left:right] = True
    border_mask = ~center_mask
    return {
        "center_mean": float(heatmap[center_mask].mean()),
        "border_mean": float(heatmap[border_mask].mean()),
        "hotspot_fraction": float((heatmap >= 0.60).mean()),
    }


def save_gradcam(
    display_image: Image.Image,
    heatmap: np.ndarray,
    path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(display_image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(display_image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    figure.suptitle(title, fontsize=9)
    plt.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def token_occlusion(
    text: str,
    row: pd.Series,
    encoder: SentenceTransformer,
    bundle: dict[str, object],
    predicted_label: str,
    batch_size: int = 64,
) -> tuple[list[str], np.ndarray, float]:
    tokens = text.split()
    if not tokens:
        tokens = ["MISSING_TEXT"]
    tokens = tokens[:60]
    classifier = bundle["classifier"]
    base_frame = pd.DataFrame([row])
    base_embedding = encoder.encode(
        [text],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    base_features = text_model_features(
        base_embedding, base_frame, bundle
    )
    class_index = list(classifier.classes_).index(predicted_label)
    base_probability = float(
        classifier.predict_proba(base_features)[0, class_index]
    )

    variants = [
        " ".join(tokens[:index] + tokens[index + 1 :])
        for index in range(len(tokens))
    ]
    variant_embeddings = encoder.encode(
        variants,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    variant_rows = pd.DataFrame([row.to_dict()] * len(variants))
    variant_rows["text"] = variants
    variant_features = text_model_features(
        variant_embeddings, variant_rows, bundle
    )
    variant_probability = classifier.predict_proba(variant_features)[
        :, class_index
    ]
    importance = base_probability - variant_probability
    return tokens, importance, base_probability


def save_text_html(
    path: Path,
    tokens: list[str],
    importance: np.ndarray,
    title: str,
    details: str,
) -> None:
    max_importance = max(float(np.max(np.abs(importance))), 1e-8)
    spans = []
    for token, score in zip(tokens, importance):
        intensity = min(abs(float(score)) / max_importance, 1.0)
        color = (
            f"rgba(255,80,80,{0.15 + 0.65 * intensity:.3f})"
            if score >= 0
            else f"rgba(80,120,255,{0.15 + 0.65 * intensity:.3f})"
        )
        spans.append(
            f'<span title="importance={score:.5f}" '
            f'style="background:{color};padding:2px 3px;margin:1px;'
            f'display:inline-block;border-radius:3px">'
            f"{html.escape(token)}</span>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:30px auto;line-height:1.6}}
.legend{{font-size:14px;color:#444}} .tokens{{font-size:18px}}</style></head>
<body><h1>{html.escape(title)}</h1><p>{html.escape(details)}</p>
<p class="legend">Red supports the predicted class; blue opposes it. Hover for score.</p>
<div class="tokens">{' '.join(spans)}</div></body></html>"""
    path.write_text(document, encoding="utf-8")


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
            f"""# Game 7 - The Black-Box Torch

**Team:** {team_name}

Image explanations use Grad-CAM. Text explanations use token occlusion. The
galleries are interpreted through structured case and trust tables."""
        ),
        code(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd()
cases = pd.read_csv(OUTPUT_DIR / "explainability_case_analysis.csv")
trust = pd.read_csv(OUTPUT_DIR / "model_trust_assessment.csv")

display(cases)
display(trust)
print("Image gallery files:", len(list((OUTPUT_DIR / "image_explainability_gallery").glob("*.png"))))
print("Text gallery files:", len(list((OUTPUT_DIR / "text_explainability_gallery").glob("*.html"))))"""
        ),
        code(
            """cases.groupby(["modality", "case_type"]).size().unstack(fill_value=0).plot.bar(
    figsize=(9, 4), title="Explained case coverage"
)
plt.ylabel("Cases")
plt.tight_layout()
plt.show()

display(
    trust.groupby(["modality", "risk_level"])
    .size()
    .to_frame("patterns")
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
        "--game6-output",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game6_Submission_Ded_Sec"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game7_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    root = args.root.resolve()
    game3 = args.game3_output.resolve()
    game6 = args.game6_output.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_gallery = output / "image_explainability_gallery"
    text_gallery = output / "text_explainability_gallery"
    image_gallery.mkdir(exist_ok=True)
    text_gallery.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = pd.read_csv(game3 / "processed_validation.csv")

    image_config = json.loads(
        (
            game6 / "best_optimized_image_model" / "config.json"
        ).read_text(encoding="utf-8")
    )
    image_size = int(image_config["image_size"])
    image_transform = T.Compose(
        [
            ExifTranspose(),
            ResizePad(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image_dataset = ImageDataset(validation, root, image_transform)
    image_loader = DataLoader(
        image_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    image_model = ScratchImageCNN().to(device)
    image_model.load_state_dict(
        torch.load(
            game6 / "best_optimized_image_model" / "model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    (
        _image_metrics,
        image_ids,
        image_truth,
        image_prediction_indices,
        image_probabilities,
    ) = evaluate(image_model, image_loader, device, collect=True)
    image_predictions = [
        LABELS[index] for index in image_prediction_indices
    ]
    image_cases = prediction_cases(
        validation.set_index("sample_id").loc[image_ids].reset_index(),
        image_predictions,
        image_probabilities.max(axis=1),
    )

    text_bundle = joblib.load(
        game6
        / "best_optimized_text_model"
        / "classifier_bundle.joblib"
    )
    text_encoder = SentenceTransformer(
        str(game6 / "best_optimized_text_model" / "encoder"),
        device=str(device),
    )
    validation_embeddings = text_encoder.encode(
        validation["text"].fillna("").astype(str).tolist(),
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    text_features = text_model_features(
        validation_embeddings, validation, text_bundle
    )
    text_classifier = text_bundle["classifier"]
    text_probabilities = text_classifier.predict_proba(text_features)
    text_predictions = text_classifier.classes_[
        text_probabilities.argmax(axis=1)
    ]
    text_cases = prediction_cases(
        validation,
        text_predictions,
        text_probabilities.max(axis=1),
    )

    selected_images = select_cases(image_cases)
    selected_texts = select_cases(text_cases)
    case_rows = []

    gradcam = GradCAM(image_model)
    display_transform = T.Compose([ExifTranspose(), ResizePad(image_size)])
    normalized_transform = image_transform
    for index, row in selected_images.iterrows():
        source = resolve_image(root, row["image_path"])
        with Image.open(source) as image:
            display_image = display_transform(image).convert("RGB")
            tensor = normalized_transform(image).unsqueeze(0).to(device)
        predicted_index = LABELS.index(row["predicted_label"])
        heatmap = gradcam.generate(image_model, tensor, predicted_index)
        geometry = heatmap_geometry(heatmap)
        flags = [
            name
            for name in (
                "image_low_resolution",
                "image_low_sharpness",
                "image_extreme_brightness",
            )
            if bool(row.get(name, False))
        ]
        focus = (
            "border-heavy"
            if geometry["border_mean"] > geometry["center_mean"] * 1.15
            else "center/content-heavy"
        )
        observed = (
            f"{focus} activation; center_mean={geometry['center_mean']:.3f}; "
            f"border_mean={geometry['border_mean']:.3f}; "
            f"hotspot_fraction={geometry['hotspot_fraction']:.3f}; "
            f"quality_flags={flags or ['none']}"
        )
        interpretation = (
            "High-confidence error with concentrated evidence; visual shortcuts or dataset bias should reduce image weight."
            if row["case_type"] == "high_confidence_error"
            else (
                "Uncertain activation pattern; fusion should seek corroborating text evidence."
                if row["case_type"] == "uncertain"
                else (
                    "Activation is compatible with the correct decision, but geometry alone cannot prove semantic relevance."
                    if row["correct"]
                    else "Incorrect prediction; the highlighted region should be reviewed for irrelevant texture or quality cues."
                )
            )
        )
        case_id = f"IMG_{index + 1:03d}"
        save_gradcam(
            display_image,
            heatmap,
            image_gallery / f"{case_id}_{row['sample_id']}.png",
            (
                f"{row['sample_id']} true={row['label']} "
                f"pred={row['predicted_label']} conf={row['confidence']:.3f}"
            ),
        )
        case_rows.append(
            {
                "case_id": case_id,
                "sample_id": row["sample_id"],
                "modality": "image",
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "case_type": row["case_type"],
                "observed_evidence": observed,
                "interpretation": interpretation,
            }
        )
    gradcam.close()

    for index, row in selected_texts.iterrows():
        tokens, importance, base_probability = token_occlusion(
            str(row["text"]),
            row,
            text_encoder,
            text_bundle,
            str(row["predicted_label"]),
        )
        top_support = np.argsort(importance)[::-1][:5]
        top_oppose = np.argsort(importance)[:3]
        supporting = [
            f"{tokens[position]}:{importance[position]:.4f}"
            for position in top_support
            if importance[position] > 0
        ]
        opposing = [
            f"{tokens[position]}:{importance[position]:.4f}"
            for position in top_oppose
            if importance[position] < 0
        ]
        observed = (
            f"supporting_tokens={supporting or ['none']}; "
            f"opposing_tokens={opposing or ['none']}; "
            f"base_predicted_probability={base_probability:.3f}; "
            f"missing_text={bool(row.get('text_was_missing', False))}"
        )
        interpretation = (
            "High-confidence error suggests topic or lexical shortcut reliance; text weight should be capped when image evidence disagrees."
            if row["case_type"] == "high_confidence_error"
            else (
                "Diffuse token influence supports uncertainty-aware review or stronger image corroboration."
                if row["case_type"] == "uncertain"
                else (
                    "Dominant tokens support the correct class, but topic-specific wording may not generalize."
                    if row["correct"]
                    else "Influential tokens led to an incorrect class and should be treated as potentially superficial evidence."
                )
            )
        )
        case_id = f"TXT_{index + 1:03d}"
        save_text_html(
            text_gallery / f"{case_id}_{row['sample_id']}.html",
            tokens,
            importance,
            (
                f"{row['sample_id']} true={row['label']} "
                f"pred={row['predicted_label']}"
            ),
            observed,
        )
        case_rows.append(
            {
                "case_id": case_id,
                "sample_id": row["sample_id"],
                "modality": "text",
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "case_type": row["case_type"],
                "observed_evidence": observed,
                "interpretation": interpretation,
            }
        )

    case_analysis = pd.DataFrame(case_rows)
    case_analysis.to_csv(
        output / "explainability_case_analysis.csv", index=False
    )

    image_high_error = int(
        (image_cases["case_type"] == "high_confidence_error").sum()
    )
    text_high_error = int(
        (text_cases["case_type"] == "high_confidence_error").sum()
    )
    image_uncertain = int(
        (image_cases["case_type"] == "uncertain").sum()
    )
    text_uncertain = int((text_cases["case_type"] == "uncertain").sum())
    disagreement = image_cases[
        ["sample_id", "predicted_label", "correct"]
    ].merge(
        text_cases[["sample_id", "predicted_label", "correct"]],
        on="sample_id",
        suffixes=("_image", "_text"),
    )
    disagreement = disagreement[
        disagreement["predicted_label_image"]
        != disagreement["predicted_label_text"]
    ]
    image_wins = int(
        (
            disagreement["correct_image"]
            & ~disagreement["correct_text"]
        ).sum()
    )
    text_wins = int(
        (
            disagreement["correct_text"]
            & ~disagreement["correct_image"]
        ).sum()
    )
    border_cases = int(
        case_analysis.loc[
            case_analysis["modality"] == "image",
            "observed_evidence",
        ]
        .str.contains("border-heavy")
        .sum()
    )
    trust = pd.DataFrame(
        [
            {
                "modality": "image",
                "observed_pattern": "High-confidence errors",
                "evidence_count": image_high_error,
                "risk_level": "high" if image_high_error else "low",
                "impact_on_trust": "Image confidence alone is not sufficient.",
                "recommended_action_for_fusion": "Cap image dominance on disagreement and route very confident conflicts for review.",
            },
            {
                "modality": "image",
                "observed_pattern": "Uncertain predictions",
                "evidence_count": image_uncertain,
                "risk_level": "medium",
                "impact_on_trust": "Weak visual evidence needs corroboration.",
                "recommended_action_for_fusion": "Increase text/relation weight when image confidence is low.",
            },
            {
                "modality": "image",
                "observed_pattern": "Border-heavy Grad-CAM focus in gallery",
                "evidence_count": border_cases,
                "risk_level": "medium" if border_cases else "low",
                "impact_on_trust": "May indicate quality, padding, or framing shortcuts.",
                "recommended_action_for_fusion": "Downweight image evidence when quality flags and border focus coincide.",
            },
            {
                "modality": "text",
                "observed_pattern": "High-confidence errors",
                "evidence_count": text_high_error,
                "risk_level": "high" if text_high_error else "low",
                "impact_on_trust": "Lexical confidence can reflect topic shortcuts.",
                "recommended_action_for_fusion": "Require image or relation agreement before trusting highly confident text-only decisions.",
            },
            {
                "modality": "text",
                "observed_pattern": "Uncertain predictions",
                "evidence_count": text_uncertain,
                "risk_level": "medium",
                "impact_on_trust": "Diffuse token evidence weakens text reliability.",
                "recommended_action_for_fusion": "Increase image/relation weight or flag for review.",
            },
            {
                "modality": "fusion",
                "observed_pattern": "Image-text prediction disagreement",
                "evidence_count": len(disagreement),
                "risk_level": "high",
                "impact_on_trust": f"Image alone correct={image_wins}; text alone correct={text_wins} among disagreements.",
                "recommended_action_for_fusion": "Learn reliability-aware fusion and expose disagreement as primary evidence.",
            },
        ]
    )
    trust.to_csv(output / "model_trust_assessment.csv", index=False)
    summary = (
        f"Explained {len(case_analysis)} selected cases. Across the full "
        f"validation set, image high-confidence errors={image_high_error}, "
        f"text high-confidence errors={text_high_error}, image uncertain="
        f"{image_uncertain}, and text uncertain={text_uncertain}. The models "
        f"disagreed on {len(disagreement)} samples; image was uniquely correct "
        f"on {image_wins} and text was uniquely correct on {text_wins}. Game 8 "
        "should therefore use confidence- and quality-aware fusion, explicitly "
        "model disagreement, and flag strong cross-modal conflicts for review."
    )
    (output / "explainability_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    build_notebook(
        output / "Game_7_Black_Box_Torch_Completed.ipynb",
        args.team_name,
        summary,
    )
    print(case_analysis.to_string(index=False))
    print(trust.to_string(index=False))
    print(summary)
    print(f"Artifacts written to {output}")


if __name__ == "__main__":
    main()
