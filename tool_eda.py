#!/usr/bin/env python3
"""TOOL 2 - DATA RECONNAISSANCE (Game 2)
Per-image statistics + charts, with three signals that often separate
AI-generated from real photos: EXIF presence, high-frequency FFT energy,
and Laplacian sharpness.

Usage:
  python tool_eda.py                       # samples 800 images per split
  python tool_eda.py --sample 0            # all images (slower)
  python tool_eda.py --train-csv x.csv --val-csv y.csv
Outputs: eda_out/eda_stats.csv + PNG charts + printed insights.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from aio_config import SPLITS, list_split, load_label_map, guess_train_csv, find_game_csvs


def stats_for(path: Path):
    try:
        with Image.open(path) as im:
            w, h = im.size
            exif = 1 if (im.getexif() and len(im.getexif()) > 0) else 0
            g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)
            rgb = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.float32)
    except Exception as e:
        print(f"  [warn] {path.name}: {e}")
        return None
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    f = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    yy, xx = np.mgrid[0:256, 0:256]
    center = ((yy - 128) ** 2 + (xx - 128) ** 2) <= 32 ** 2
    total = f.sum() + 1e-9
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return dict(file=path.name, width=w, height=h, aspect=round(w / h, 3),
                kb=round(path.stat().st_size / 1024, 1),
                brightness=float(g.mean()), contrast=float(g.std()),
                sharpness=float(lap.var()),
                hf_ratio=float(f[~center].sum() / total),
                colorfulness=colorfulness, exif=exif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800, help="images per split (0 = all)")
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--val-csv", type=Path, default=None)
    ap.add_argument("--img-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--out", type=Path, default=Path("eda_out"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    random.seed(42)

    labels = {}
    for csv in filter(None, [a.train_csv or guess_train_csv(),
                             a.val_csv or next(iter(find_game_csvs("valid")), None)]):
        if Path(csv).exists():
            try:
                m, classes, ic, lc = load_label_map(csv, a.img_col, a.label_col)
                labels.update(m)
                print(f"Labels from {Path(csv).name}: classes={classes}")
            except SystemExit as e:
                print(e)

    rows = []
    for s in SPLITS:
        paths = list_split(s)
        if a.sample and len(paths) > a.sample:
            paths = random.sample(paths, a.sample)
        print(f"Analyzing {len(paths)} images [{s}] ...")
        for i, p in enumerate(paths, 1):
            r = stats_for(p)
            if r:
                r["split"] = s
                r["label"] = labels.get(p.name, "?")
                rows.append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out / "eda_stats.csv", index=False)
    print(f"\nWrote {a.out/'eda_stats.csv'}  ({len(df)} rows)")

    # ---------- charts ----------
    def bar_counts(series, title, fname):
        plt.figure(figsize=(6, 4))
        series.value_counts().plot.bar()
        plt.title(title); plt.tight_layout()
        plt.savefig(a.out / fname, dpi=120); plt.close()

    bar_counts(df["split"], "Images per split (sampled)", "split_counts.png")
    has_labels = (df["label"] != "?").any()
    if has_labels:
        bar_counts(df.loc[df.label != "?", "label"],
                   "Class balance (labeled images)", "class_balance.png")

    plt.figure(figsize=(6, 4))
    for s, g in df.groupby("split"):
        plt.scatter(g["width"], g["height"], s=6, alpha=0.4, label=s)
    plt.xlabel("width"); plt.ylabel("height"); plt.legend()
    plt.title("Resolutions"); plt.tight_layout()
    plt.savefig(a.out / "resolutions.png", dpi=120); plt.close()

    metrics = ["brightness", "contrast", "sharpness", "hf_ratio", "colorfulness", "kb"]
    key = "label" if has_labels else "split"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axes.flat, metrics):
        for k, g in df[df[key] != "?"].groupby(key):
            ax.hist(g[m], bins=40, alpha=0.55, label=str(k), density=True)
        ax.set_title(m); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(a.out / f"distributions_by_{key}.png", dpi=120); plt.close()

    # ---------- printed insights ----------
    print("\n" + "=" * 66)
    print("RECON INSIGHTS")
    print("=" * 66)
    if has_labels:
        lab = df[df.label != "?"]
        summary = lab.groupby("label")[metrics + ["exif"]].mean().round(3)
        print(summary.to_string())
        ex = summary["exif"]
        if len(ex) == 2 and abs(ex.iloc[0] - ex.iloc[1]) > 0.2:
            print(f"\n [signal] EXIF presence differs strongly by class "
                  f"({ex.to_dict()}) - real photos keep camera EXIF, AI images rarely do.")
        hf = summary["hf_ratio"]
        if len(hf) == 2 and abs(hf.iloc[0] - hf.iloc[1]) / (hf.mean() + 1e-9) > 0.05:
            print(" [signal] High-frequency FFT energy differs by class - a known "
                  "AI-generation artifact. A model can learn this; so can a baseline.")
    else:
        print(df.groupby("split")[metrics + ["exif"]].mean().round(3).to_string())
        print("\n No labels matched - pass --train-csv to unlock per-class insights.")
    print(f"\nCharts saved in {a.out}/")


if __name__ == "__main__":
    main()
