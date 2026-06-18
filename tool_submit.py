#!/usr/bin/env python3
"""TOOL 8 - SUBMISSION BUILDER
Reads the official submission_template.csv, runs your checkpoint on the
test1_* images, and writes a correctly formatted submission.

Usage:
  python tool_submit.py --ckpt runs/<run>/best.pt
  python tool_submit.py --dry-random          # valid-format random baseline (no torch needed)
Always open the output and eyeball 5 rows before uploading.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from aio_config import ROOT, list_split, autodetect_cols


def template_df():
    tpl = ROOT / "submission_template.csv"
    if not tpl.exists():
        raise SystemExit(f"Template not found: {tpl}")
    return pd.read_csv(tpl), tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dry-random", action="store_true",
                    help="random predictions, just to validate the format")
    a = ap.parse_args()

    df, tpl_path = template_df()
    img_col, label_col = autodetect_cols(df)
    if img_col is None:
        img_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else "label"
    print(f"Template: {tpl_path.name}  columns={list(df.columns)}  "
          f"-> using img_col='{img_col}', label_col='{label_col}'")

    test_paths = list_split("test")
    print(f"Test images: {len(test_paths)}")

    if a.dry_random:
        classes = ["fake", "real"]
        preds = {p.name: random.choice(classes) for p in test_paths}
        proba = {p.name: round(random.random(), 4) for p in test_paths}
        suffix = "random"
    else:
        if not a.ckpt:
            raise SystemExit("Pass --ckpt runs/<run>/best.pt  (or --dry-random).")
        import torch
        from tool_dataset import AIODataset, build_transforms
        from tool_train import build_model
        from torch.utils.data import DataLoader

        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        classes, size = ck["classes"], ck.get("size", 224)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(ck["model_name"], len(classes), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()

        ds = AIODataset("test", transform=build_transforms(False, size))
        loader = DataLoader(ds, batch_size=a.batch, num_workers=4)
        preds, proba = {}, {}
        with torch.no_grad():
            for x, _, names in loader:
                p = torch.softmax(model(x.to(device)).float(), 1).cpu()
                for n, row in zip(names, p):
                    k = int(row.argmax())
                    preds[n] = classes[k]
                    proba[n] = round(float(row[1] if len(classes) == 2 else row[k]), 5)
        suffix = Path(a.ckpt).parent.name

    # fill the template: by existing ids if present, else from test files
    if df[img_col].notna().any() and len(df):
        df[label_col] = [preds.get(Path(str(v)).name, "") for v in df[img_col]]
        missing = int((df[label_col] == "").sum())
        if missing:
            print(f"[!] {missing} template rows had no matching test image - check naming!")
    else:
        df = pd.DataFrame({img_col: [p.name for p in test_paths],
                           label_col: [preds[p.name] for p in test_paths]})
    for c in df.columns:
        if c.lower() in ("prob", "proba", "probability", "score", "confidence"):
            df[c] = [proba.get(Path(str(v)).name, "") for v in df[img_col]]

    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    out = a.out or sub_dir / f"submission_{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")
    print(df.head(5).to_string())
    print("\nNow check submissions/README_SUBMISSION_RULES.txt for the required "
          "FILENAME convention before uploading.")


if __name__ == "__main__":
    main()
