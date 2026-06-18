#!/usr/bin/env python3
"""
GAME 1 - THE MIRROR MAZE: data leakage detector
================================================
Finds the leaks that inflate your scores before they embarrass you:

  1. Exact duplicate images WITHIN train  (+ label conflicts)
  2. Exact duplicate images ACROSS train/test   <- classic leakage
  3. Near-duplicate images ACROSS train/test    <- sneaky leakage (perceptual hash)
  4. Near-duplicate images WITHIN train
  5. Duplicate/overlapping TEXT across train/test
  6. Same text, different labels within train   <- label noise

Requires:  pip install pillow imagehash pandas

Usage (point at what you have, it skips the rest):
  python check_leakage.py --train-dir data/train_images --test-dir data/test_images
  python check_leakage.py --train-dir data/train_images --test-dir data/test_images \
      --train-csv data/train.csv --test-csv data/test.csv \
      --img-col image --text-col text --label-col label

Outputs CSV reports into ./leakage_report/ and prints a summary verdict.
"""

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    from PIL import Image
    import imagehash
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun:  pip install pillow imagehash pandas")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
HASH_BITS = 64  # phash default 8x8


# ----------------------------------------------------------------- helpers
def collect_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_int(path: Path):
    try:
        with Image.open(path) as im:
            return int(str(imagehash.phash(im.convert("RGB"))), 16)
    except Exception as exc:
        print(f"  [warn] unreadable image {path}: {exc}")
        return None


def hash_folder(paths, tag):
    md5s, phashes = {}, {}
    n = len(paths)
    print(f"Hashing {n} images in {tag} ...")
    for i, p in enumerate(paths, 1):
        md5s[p] = md5_of(p)
        ph = phash_int(p)
        if ph is not None:
            phashes[p] = ph
        if i % 500 == 0 or i == n:
            print(f"  {i}/{n}")
    return md5s, phashes


def band_keys(h: int, bands: int):
    """Split a 64-bit hash into `bands` segments (pigeonhole for Hamming search)."""
    base, rem = HASH_BITS // bands, HASH_BITS % bands
    keys, shift = [], 0
    for b in range(bands):
        width = base + (1 if b < rem else 0)
        keys.append((b, (h >> shift) & ((1 << width) - 1)))
        shift += width
    return keys


def near_pairs(hashes_a: dict, hashes_b: dict, max_dist: int, same_set=False):
    """All (a, b, hamming) pairs with distance <= max_dist. Banded LSH: if two
    64-bit hashes differ in <= d bits, at least one of d+1 bands is identical."""
    bands = max_dist + 1
    buckets = defaultdict(list)
    for p, h in hashes_a.items():
        for key in band_keys(h, bands):
            buckets[key].append((p, h))
    out, seen_pairs = [], set()
    for q, h in hashes_b.items():
        checked = set()
        for key in band_keys(h, bands):
            for p, h2 in buckets.get(key, ()):
                if p in checked or (same_set and p >= q):
                    continue
                checked.add(p)
                d = bin(h ^ h2).count("1")
                if d <= max_dist and (p, q) not in seen_pairs:
                    seen_pairs.add((p, q))
                    out.append((p, q, d))
    return sorted(out, key=lambda t: t[2])


def norm_text(s) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_csv(path, img_col, text_col, label_col, tag):
    df = pd.read_csv(path)
    for col, name in [(img_col, "img-col"), (text_col, "text-col"), (label_col, "label-col")]:
        if col and col not in df.columns:
            print(f"  [warn] {tag}: column '{col}' not found ({name}); have {list(df.columns)}")
    return df


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Game 1: train/test leakage detector")
    ap.add_argument("--train-dir", type=Path, help="folder of training images")
    ap.add_argument("--test-dir", type=Path, help="folder of test images")
    ap.add_argument("--train-csv", type=Path, help="CSV with train metadata")
    ap.add_argument("--test-csv", type=Path, help="CSV with test metadata")
    ap.add_argument("--img-col", default="image", help="CSV column with image filename")
    ap.add_argument("--text-col", default="text", help="CSV column with the text")
    ap.add_argument("--label-col", default="label", help="CSV column with the label")
    ap.add_argument("--near-dist", type=int, default=6,
                    help="max Hamming distance (of 64) to call images near-duplicates")
    ap.add_argument("--report-dir", type=Path, default=Path("leakage_report"))
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    findings = []

    def save(df: pd.DataFrame, name: str, message: str):
        if len(df):
            out = args.report_dir / name
            df.to_csv(out, index=False)
            findings.append(f"[!] {message}: {len(df)}  ->  {out}")
        else:
            findings.append(f"[ok] {message}: 0")

    # ---------- image checks ----------
    train_md5 = train_ph = test_md5 = test_ph = None
    if args.train_dir and args.train_dir.exists():
        train_paths = collect_images(args.train_dir)
        train_md5, train_ph = hash_folder(train_paths, "TRAIN")
    if args.test_dir and args.test_dir.exists():
        test_paths = collect_images(args.test_dir)
        test_md5, test_ph = hash_folder(test_paths, "TEST")

    # label lookup by filename (for conflict detection)
    name2label = {}
    train_df = None
    if args.train_csv and args.train_csv.exists():
        train_df = load_csv(args.train_csv, args.img_col, args.text_col, args.label_col, "train csv")
        if args.img_col in train_df.columns and args.label_col in train_df.columns:
            for _, r in train_df.iterrows():
                name2label[Path(str(r[args.img_col])).name] = r[args.label_col]

    if train_md5:
        groups = defaultdict(list)
        for p, h in train_md5.items():
            groups[h].append(p)
        dup_rows, conflict_rows = [], []
        for h, ps in groups.items():
            if len(ps) > 1:
                labels = {name2label.get(p.name, "?") for p in ps}
                dup_rows.append({"files": " | ".join(str(p) for p in ps),
                                 "labels": " | ".join(map(str, labels))})
                if len({l for l in labels if l != "?"}) > 1:
                    conflict_rows.append(dup_rows[-1])
        save(pd.DataFrame(dup_rows), "exact_dups_within_train.csv",
             "Exact duplicate image groups WITHIN train")
        save(pd.DataFrame(conflict_rows), "label_conflicts_images.csv",
             "Duplicate train images with CONFLICTING labels")

    if train_md5 and test_md5:
        md5_to_train = defaultdict(list)
        for p, h in train_md5.items():
            md5_to_train[h].append(p)
        exact_rows = [{"train_file": str(tp), "test_file": str(q)}
                      for q, h in test_md5.items() for tp in md5_to_train.get(h, ())]
        save(pd.DataFrame(exact_rows), "image_leaks_exact.csv",
             "EXACT image leaks train->test")

        pairs = near_pairs(train_ph, test_ph, args.near_dist)
        exact_set = {(r["train_file"], r["test_file"]) for r in exact_rows}
        near_rows = [{"train_file": str(a), "test_file": str(b), "hamming": d}
                     for a, b, d in pairs if (str(a), str(b)) not in exact_set]
        save(pd.DataFrame(near_rows), "image_leaks_near.csv",
             f"NEAR-duplicate image leaks train->test (hamming<={args.near_dist})")

    if train_ph:
        pairs = near_pairs(train_ph, train_ph, args.near_dist, same_set=True)
        rows = [{"file_a": str(a), "file_b": str(b), "hamming": d}
                for a, b, d in pairs if train_md5.get(a) != train_md5.get(b)]
        save(pd.DataFrame(rows), "near_dups_within_train.csv",
             f"Near-duplicate pairs WITHIN train (hamming<={args.near_dist})")

    # ---------- text checks ----------
    test_df = None
    if args.test_csv and args.test_csv.exists():
        test_df = load_csv(args.test_csv, args.img_col, args.text_col, args.label_col, "test csv")

    if train_df is not None and args.text_col in train_df.columns:
        tnorm = train_df[args.text_col].map(norm_text)
        if args.label_col in train_df.columns:
            tmp = pd.DataFrame({"text": tnorm, "label": train_df[args.label_col]})
            conf = (tmp.groupby("text")["label"].nunique()
                       .pipe(lambda s: s[s > 1]).index)
            rows = tmp[tmp["text"].isin(conf)].drop_duplicates()
            save(rows, "text_label_conflicts.csv",
                 "Same text with CONFLICTING labels within train")
        if test_df is not None and args.text_col in test_df.columns:
            snorm = set(tnorm) - {""}
            mask = test_df[args.text_col].map(norm_text).isin(snorm)
            rows = test_df.loc[mask, [c for c in [args.img_col, args.text_col]
                                      if c in test_df.columns]]
            save(rows, "text_leaks.csv", "Test texts that also appear in train")

    # ---------- verdict ----------
    print("\n" + "=" * 64)
    print("MIRROR MAZE REPORT")
    print("=" * 64)
    for line in findings:
        print(" " + line)
    leaks = sum(1 for f in findings if f.startswith("[!]"))
    print("-" * 64)
    if leaks:
        print(" VERDICT: leakage found. Fix protocol:")
        print("   1. DROP every train item listed in image_leaks_* / text_leaks")
        print("      (clean TRAIN, never touch TEST).")
        print("   2. Resolve label conflicts: keep one copy with the correct label.")
        print("   3. Re-create your validation split AFTER cleaning, stratified,")
        print("      and keep near-duplicates inside the same fold.")
        print("   4. Re-run this script until it comes back clean.")
    else:
        print(" VERDICT: no leakage detected. Your baseline scores are honest.")
    print("=" * 64)


if __name__ == "__main__":
    main()
