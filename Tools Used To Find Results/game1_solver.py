#!/usr/bin/env python3
"""Solve AI Olympics 2026 Game 1 without modifying the provided package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


REQUIRED_COLUMNS = ["sample_id", "image_path", "text", "label"]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_baseline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_baseline(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, seed: int
) -> tuple[dict[str, object], Pipeline]:
    model = build_baseline(seed)
    model.fit(train_df["text"].fillna(""), train_df["label"])
    predictions = model.predict(validation_df["text"].fillna(""))
    metrics = {
        "accuracy": float(accuracy_score(validation_df["label"], predictions)),
        "macro_f1": float(
            f1_score(validation_df["label"], predictions, average="macro")
        ),
        "precision": float(
            precision_score(
                validation_df["label"],
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                validation_df["label"],
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            validation_df["label"], predictions, digits=4
        ),
        "confusion_matrix": confusion_matrix(
            validation_df["label"], predictions
        ).tolist(),
    }
    return metrics, model


def resolve_image(root: Path, relative_path: object) -> Path:
    value = Path(str(relative_path))
    candidates = [value, root / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {relative_path}")


def image_fingerprints(path: Path) -> tuple[str, int]:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            md5.update(chunk)
    with Image.open(path) as image:
        phash = int(str(imagehash.phash(image.convert("RGB"))), 16)
    return md5.hexdigest(), phash


def band_keys(value: int, bands: int) -> list[tuple[int, int]]:
    base, remainder = divmod(64, bands)
    keys = []
    shift = 0
    for band in range(bands):
        width = base + (1 if band < remainder else 0)
        keys.append((band, (value >> shift) & ((1 << width) - 1)))
        shift += width
    return keys


def near_hash_pairs(
    hashes: list[int], max_distance: int
) -> list[tuple[int, int, int]]:
    bands = max_distance + 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    pairs: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for right, value in enumerate(hashes):
        candidates: set[int] = set()
        for key in band_keys(value, bands):
            candidates.update(buckets.get(key, ()))
        for left in candidates:
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            distance = (hashes[left] ^ value).bit_count()
            if distance <= max_distance:
                pairs.append((left, right, distance))
        for key in band_keys(value, bands):
            buckets[key].append(right)
    return pairs


def add_evidence(
    rows: list[dict[str, object]],
    seen: set[tuple[str, str, str]],
    combined: pd.DataFrame,
    left: int,
    right: int,
    evidence_type: str,
    score: float | int,
) -> None:
    left_split = combined.at[left, "original_split"]
    right_split = combined.at[right, "original_split"]
    if left_split == right_split:
        return
    train_index, validation_index = (
        (left, right) if left_split == "train" else (right, left)
    )
    train_id = str(combined.at[train_index, "sample_id"])
    validation_id = str(combined.at[validation_index, "sample_id"])
    key = (validation_id, train_id, evidence_type)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "validation_sample_id": validation_id,
            "related_train_sample_id": train_id,
            "evidence_type": evidence_type,
            "similarity_score_or_distance": round(float(score), 6),
            "label": combined.at[validation_index, "label"],
        }
    )


def union_groups(
    values: pd.Series, union_find: UnionFind
) -> int:
    grouped: dict[object, list[int]] = defaultdict(list)
    for index, value in values.items():
        if value not in ("", None) and not pd.isna(value):
            grouped[value].append(index)
    edges = 0
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        anchor = indexes[0]
        for index in indexes[1:]:
            union_find.union(anchor, index)
            edges += 1
    return edges


def build_completed_notebook(
    output_path: Path,
    root: Path,
    initial_metrics: dict[str, object],
    corrected_metrics: dict[str, object],
    suspicious_cases: int,
    crossed_edges: int,
    team_name: str,
) -> None:
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

    conclusion = (
        f"The initial validation score was not trustworthy. The audit found "
        f"{suspicious_cases:,} train-validation evidence rows caused by "
        "normalized text overlap and exact or perceptually similar images. "
        "Related samples were connected into groups and assigned to one fold, "
        f"leaving {crossed_edges} known relation edges across the corrected "
        "boundary. The unchanged TF-IDF plus logistic-regression baseline "
        f"changed from accuracy {initial_metrics['accuracy']:.4f} and macro-F1 "
        f"{initial_metrics['macro_f1']:.4f} to accuracy "
        f"{corrected_metrics['accuracy']:.4f} and macro-F1 "
        f"{corrected_metrics['macro_f1']:.4f}. This demonstrates why leakage "
        "auditing must precede stronger model training."
    )

    cells = [
        markdown(
            f"""# Game 1 - The Mirror Maze

**Team:** {team_name}

This completed notebook verifies the generated evidence and reproduces the
provided text baseline before and after the corrected group-aware split."""
        ),
        code(
            f"""from pathlib import Path
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(os.environ.get("AIO_ROOT", r"{root}"))
GAME1_DIR = ROOT / "games" / "game1_mirror_maze"
OUTPUT_DIR = Path.cwd()

naive_train = pd.read_csv(GAME1_DIR / "game1_train_naive.csv")
naive_validation = pd.read_csv(GAME1_DIR / "game1_validation_naive.csv")
corrected_train = pd.read_csv(OUTPUT_DIR / "game1_corrected_train.csv")
corrected_validation = pd.read_csv(OUTPUT_DIR / "game1_corrected_validation.csv")
evidence = pd.read_csv(OUTPUT_DIR / "game1_evidence_table.csv")

print("Naive:", len(naive_train), len(naive_validation))
print("Corrected:", len(corrected_train), len(corrected_validation))
print("Evidence rows:", len(evidence))"""
        ),
        code(
            """def baseline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

def evaluate(train_df, validation_df):
    model = baseline()
    model.fit(train_df["text"].fillna(""), train_df["label"])
    predictions = model.predict(validation_df["text"].fillna(""))
    return {
        "accuracy": accuracy_score(validation_df["label"], predictions),
        "macro_f1": f1_score(
            validation_df["label"], predictions, average="macro"
        ),
        "report": classification_report(
            validation_df["label"], predictions, digits=4
        ),
        "confusion_matrix": confusion_matrix(
            validation_df["label"], predictions
        ),
    }"""
        ),
        code(
            """initial = evaluate(naive_train, naive_validation)
corrected = evaluate(corrected_train, corrected_validation)

comparison = pd.DataFrame([
    {
        "evaluation_setup": "Initial validation",
        "accuracy": initial["accuracy"],
        "macro_f1": initial["macro_f1"],
    },
    {
        "evaluation_setup": "Corrected validation",
        "accuracy": corrected["accuracy"],
        "macro_f1": corrected["macro_f1"],
    },
])
display(comparison)
print("\\nInitial classification report:\\n", initial["report"])
print("Initial confusion matrix:\\n", initial["confusion_matrix"])
print("\\nCorrected classification report:\\n", corrected["report"])
print("Corrected confusion matrix:\\n", corrected["confusion_matrix"])"""
        ),
        code(
            """display(
    evidence.groupby("evidence_type")
    .size()
    .rename("cases")
    .sort_values(ascending=False)
    .to_frame()
)
display(evidence.head(20))"""
        ),
        markdown("## Final Conclusion\n\n" + conclusion),
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
    output_path.write_text(
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
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
            r"\AI_Olympics_2026_Student_Release_v1\submissions"
            r"\Game1_Submission_Ded_Sec"
        ),
    )
    parser.add_argument("--team-name", default="Ded_Sec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-text-threshold", type=float, default=0.94)
    parser.add_argument("--near-image-distance", type=int, default=6)
    parser.add_argument("--image-workers", type=int, default=min(12, os.cpu_count() or 4))
    args = parser.parse_args()

    started = time.time()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    game_dir = root / "games" / "game1_mirror_maze"

    train_df = pd.read_csv(game_dir / "game1_train_naive.csv")
    validation_df = pd.read_csv(game_dir / "game1_validation_naive.csv")
    for name, frame in (("train", train_df), ("validation", validation_df)):
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise SystemExit(f"{name} is missing columns: {sorted(missing)}")

    train_df = train_df[REQUIRED_COLUMNS].copy()
    validation_df = validation_df[REQUIRED_COLUMNS].copy()
    train_df["original_split"] = "train"
    validation_df["original_split"] = "validation"
    combined = pd.concat([train_df, validation_df], ignore_index=True)
    combined["normalized_text"] = combined["text"].map(normalize_text)
    combined["resolved_image"] = combined["image_path"].map(
        lambda path: resolve_image(root, path)
    )

    print("Running unchanged baseline on the naive split...")
    initial_metrics, _ = evaluate_baseline(train_df, validation_df, args.seed)
    print(
        f"Initial accuracy={initial_metrics['accuracy']:.4f} "
        f"macro_f1={initial_metrics['macro_f1']:.4f}"
    )

    union_find = UnionFind(len(combined))
    evidence_rows: list[dict[str, object]] = []
    evidence_seen: set[tuple[str, str, str]] = set()
    relation_edges: list[tuple[int, int, str, float]] = []

    print("Finding exact normalized-text relationships...")
    text_groups: dict[str, list[int]] = defaultdict(list)
    for index, text in combined["normalized_text"].items():
        if text:
            text_groups[text].append(index)
    for indexes in text_groups.values():
        if len(indexes) < 2:
            continue
        anchor = indexes[0]
        for index in indexes[1:]:
            union_find.union(anchor, index)
            relation_edges.append((anchor, index, "exact_text", 1.0))
        train_indexes = [
            index
            for index in indexes
            if combined.at[index, "original_split"] == "train"
        ]
        validation_indexes = [
            index
            for index in indexes
            if combined.at[index, "original_split"] == "validation"
        ]
        for validation_index in validation_indexes:
            for train_index in train_indexes[:3]:
                add_evidence(
                    evidence_rows,
                    evidence_seen,
                    combined,
                    train_index,
                    validation_index,
                    "exact_normalized_text",
                    1.0,
                )

    print("Finding near-text relationships...")
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    text_matrix = text_vectorizer.fit_transform(combined["normalized_text"])
    text_neighbors = NearestNeighbors(
        n_neighbors=4, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(text_matrix)
    distances, neighbors = text_neighbors.kneighbors(text_matrix)
    for left in range(len(combined)):
        for distance, right in zip(distances[left, 1:], neighbors[left, 1:]):
            similarity = 1.0 - float(distance)
            right = int(right)
            if similarity < args.near_text_threshold or left >= right:
                continue
            union_find.union(left, right)
            relation_edges.append((left, right, "near_text", similarity))
            add_evidence(
                evidence_rows,
                evidence_seen,
                combined,
                left,
                right,
                "near_text_cosine",
                similarity,
            )

    unique_paths = list(dict.fromkeys(combined["resolved_image"].tolist()))
    print(
        f"Hashing {len(unique_paths):,} unique images with "
        f"{args.image_workers} workers..."
    )
    with ThreadPoolExecutor(max_workers=args.image_workers) as pool:
        fingerprints = list(pool.map(image_fingerprints, unique_paths))
    path_to_fingerprint = dict(zip(unique_paths, fingerprints))
    combined["image_md5"] = combined["resolved_image"].map(
        lambda path: path_to_fingerprint[path][0]
    )
    combined["image_phash"] = combined["resolved_image"].map(
        lambda path: path_to_fingerprint[path][1]
    )

    print("Finding exact-image relationships...")
    md5_groups: dict[str, list[int]] = defaultdict(list)
    for index, value in combined["image_md5"].items():
        md5_groups[value].append(index)
    for indexes in md5_groups.values():
        if len(indexes) < 2:
            continue
        anchor = indexes[0]
        for index in indexes[1:]:
            union_find.union(anchor, index)
            relation_edges.append((anchor, index, "exact_image", 0.0))
            add_evidence(
                evidence_rows,
                evidence_seen,
                combined,
                anchor,
                index,
                "exact_image_md5",
                0,
            )

    print("Finding perceptually similar image relationships...")
    phashes = combined["image_phash"].astype(object).tolist()
    for left, right, distance in near_hash_pairs(
        phashes, args.near_image_distance
    ):
        if combined.at[left, "image_md5"] == combined.at[right, "image_md5"]:
            continue
        union_find.union(left, right)
        relation_edges.append((left, right, "near_image", float(distance)))
        add_evidence(
            evidence_rows,
            evidence_seen,
            combined,
            left,
            right,
            "perceptual_image_hash_distance",
            distance,
        )

    combined["group_id"] = [
        union_find.find(index) for index in range(len(combined))
    ]
    group_count = combined["group_id"].nunique()
    related_group_count = int(
        (combined.groupby("group_id").size() > 1).sum()
    )
    print(
        f"Built {group_count:,} relation groups; "
        f"{related_group_count:,} contain multiple samples."
    )

    print("Creating a stratified group-aware corrected split...")
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=args.seed
    )
    split_train_indexes, split_validation_indexes = next(
        splitter.split(
            np.zeros(len(combined)),
            combined["label"],
            groups=combined["group_id"],
        )
    )
    corrected_train = combined.iloc[split_train_indexes].copy()
    corrected_validation = combined.iloc[split_validation_indexes].copy()
    corrected_fold = np.full(len(combined), "train", dtype=object)
    corrected_fold[split_validation_indexes] = "validation"

    crossed_edges = sum(
        1
        for left, right, _, _ in relation_edges
        if corrected_fold[left] != corrected_fold[right]
    )
    if crossed_edges:
        raise RuntimeError(
            f"Corrected split still crosses {crossed_edges} relation edges"
        )

    print("Running unchanged baseline on the corrected split...")
    corrected_metrics, _ = evaluate_baseline(
        corrected_train, corrected_validation, args.seed
    )
    print(
        f"Corrected accuracy={corrected_metrics['accuracy']:.4f} "
        f"macro_f1={corrected_metrics['macro_f1']:.4f}"
    )

    evidence_df = pd.DataFrame(
        evidence_rows,
        columns=[
            "validation_sample_id",
            "related_train_sample_id",
            "evidence_type",
            "similarity_score_or_distance",
            "label",
        ],
    ).sort_values(
        ["validation_sample_id", "related_train_sample_id", "evidence_type"]
    )
    output_columns = REQUIRED_COLUMNS
    corrected_train[output_columns].to_csv(
        output_dir / "game1_corrected_train.csv", index=False
    )
    corrected_validation[output_columns].to_csv(
        output_dir / "game1_corrected_validation.csv", index=False
    )
    evidence_df.to_csv(
        output_dir / "game1_evidence_table.csv", index=False
    )

    evidence_counts = (
        evidence_df["evidence_type"].value_counts().sort_index().to_dict()
        if len(evidence_df)
        else {}
    )
    summary = f"""Team Name: {args.team_name}
Initial Accuracy: {initial_metrics['accuracy']:.4f}
Initial Macro F1: {initial_metrics['macro_f1']:.4f}
Corrected Accuracy: {corrected_metrics['accuracy']:.4f}
Corrected Macro F1: {corrected_metrics['macro_f1']:.4f}
Number of suspicious cases found: {len(evidence_df)}
Main evidence: {json.dumps(evidence_counts, sort_keys=True)}
Correction strategy: Combined both naive splits, linked exact/near text and exact/perceptually-similar image samples, then used a stratified group-aware split so every related component stays in one fold.
Final conclusion: The initial validation score was not trustworthy because related content crossed the naive train-validation boundary. The corrected score is the more credible estimate of generalization.
"""
    (output_dir / "game1_summary.txt").write_text(
        summary, encoding="utf-8"
    )

    audit = {
        "team_name": args.team_name,
        "root": str(root),
        "naive_rows": {
            "train": len(train_df),
            "validation": len(validation_df),
        },
        "corrected_rows": {
            "train": len(corrected_train),
            "validation": len(corrected_validation),
        },
        "corrected_label_counts": {
            "train": corrected_train["label"].value_counts().to_dict(),
            "validation": corrected_validation["label"].value_counts().to_dict(),
        },
        "initial_metrics": initial_metrics,
        "corrected_metrics": corrected_metrics,
        "evidence_counts": evidence_counts,
        "evidence_rows": len(evidence_df),
        "relation_edges": len(relation_edges),
        "relation_groups": group_count,
        "related_groups": related_group_count,
        "crossed_edges_after_correction": crossed_edges,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "game1_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    build_completed_notebook(
        output_dir / "Game_1_The_Mirror_Maze_Completed.ipynb",
        root,
        initial_metrics,
        corrected_metrics,
        len(evidence_df),
        crossed_edges,
        args.team_name,
    )
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
