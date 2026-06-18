#!/usr/bin/env python3
"""Inspect how Game 8 pairs relate to the corrected Game 3 corpus."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN_ROOT = Path(
    r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
    r"\AI_Olympics_2026_Student_Release_v1"
)
GAME8_ROOT = Path(
    r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
    r"\AI_Olympics_2026_Game8_Release_v1"
)
# Optional local staging copies; fall back to the release packages.
SUBMISSIONS = (
    ROOT / "work" if (ROOT / "work").exists() else MAIN_ROOT / "submissions"
)
GAME1 = SUBMISSIONS / "Game1_Submission_Ded_Sec"
GAME8 = (
    ROOT / "source" / "Game8_Release" / "data"
    if (ROOT / "source" / "Game8_Release" / "data").exists()
    else GAME8_ROOT / "data"
)
PUBLIC_TEST = (
    ROOT / "source" / "public_test.csv"
    if (ROOT / "source" / "public_test.csv").exists()
    else MAIN_ROOT / "data" / "public_test.csv"
)


def main() -> None:
    game1 = pd.concat(
        [
            pd.read_csv(GAME1 / "game1_corrected_train.csv"),
            pd.read_csv(GAME1 / "game1_corrected_validation.csv"),
        ],
        ignore_index=True,
    )
    game3_root = SUBMISSIONS / "Game3_Submission_Ded_Sec"
    game3 = pd.concat(
        [
            pd.read_csv(game3_root / "processed_train.csv"),
            pd.read_csv(game3_root / "processed_validation.csv"),
        ],
        ignore_index=True,
    )
    public = pd.read_csv(PUBLIC_TEST)
    public["image_path"] = public["image_path"].map(
        lambda value: f"data/images/{Path(str(value)).name}"
    )
    public["text_was_missing"] = public["text"].isna()
    public["text"] = public["text"].fillna("MISSING_TEXT")
    public["label"] = pd.NA
    base = pd.concat([game1, game3, public], ignore_index=True)
    base["text_key"] = base["text"].fillna("").astype(str)
    canonical_pairs = set(zip(base["image_path"], base["text_key"]))
    image_lookup = base.drop_duplicates("image_path").set_index("image_path")
    text_lookup = base.drop_duplicates("text_key").set_index("text_key")
    print(
        "base",
        base.shape,
        "unique_images",
        base["image_path"].nunique(),
        "unique_text",
        base["text_key"].nunique(),
    )

    for filename in (
        "game8_train.csv",
        "game8_validation.csv",
        "game8_public_test.csv",
    ):
        frame = pd.read_csv(GAME8 / filename)
        frame["text_key"] = frame["text"].fillna("").astype(str)
        frame["image_source_id"] = frame["image_path"].map(
            image_lookup["sample_id"]
        )
        frame["text_source_id"] = frame["text_key"].map(
            text_lookup["sample_id"]
        )
        frame["same_source"] = (
            frame["image_source_id"] == frame["text_source_id"]
        )
        frame["canonical_pair"] = [
            pair in canonical_pairs
            for pair in zip(frame["image_path"], frame["text_key"])
        ]
        frame["image_source_label"] = frame["image_path"].map(
            image_lookup["label"]
        )
        frame["text_source_label"] = frame["text_key"].map(
            text_lookup["label"]
        )
        print(
            "\n",
            filename,
            frame.shape,
            "image_match",
            round(frame["image_source_id"].notna().mean(), 4),
            "text_match",
            round(frame["text_source_id"].notna().mean(), 4),
            "same_source",
            round(frame["same_source"].mean(), 4),
            "canonical_pair",
            round(frame["canonical_pair"].mean(), 4),
        )
        if "label" in frame:
            print(pd.crosstab(frame["canonical_pair"], frame["label"]))
            print(
                pd.crosstab(
                    [
                        frame["same_source"],
                        frame["image_source_label"],
                        frame["text_source_label"],
                    ],
                    frame["label"],
                ).to_string()
            )


if __name__ == "__main__":
    main()
