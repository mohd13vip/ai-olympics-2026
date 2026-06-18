#!/usr/bin/env python3
"""TOOL 1 - MIRROR MAZE LEAKAGE DETECTOR (Game 1)
Adapted to the AI Olympics layout: one data/images folder, splits by filename prefix.

Checks:
  1. Exact duplicates (MD5) within each split, with label conflicts
  2. Exact + near duplicates (perceptual hash) ACROSS train/val/test  <- leakage
  3. games/game1_mirror_maze/injected_images compared against ALL splits

Usage:
  python tool_leakage.py                      # auto-detects train labels CSV
  python tool_leakage.py --train-csv path.csv --near-dist 6
Requires: pip install pillow imagehash pandas
"""
import argparse
import hashlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from PIL import Image
import imagehash

from aio_config import ROOT, SPLITS, list_split, load_label_map, guess_train_csv

HASH_BITS = 64


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_int(path):
    try:
        with Image.open(path) as im:
            return int(str(imagehash.phash(im.convert("RGB"))), 16)
    except Exception as e:
        print(f"  [warn] unreadable {path.name}: {e}")
        return None


def hash_many(paths, tag):
    md5s, phs = {}, {}
    n = len(paths)
    print(f"Hashing {n} images [{tag}] ...")
    for i, p in enumerate(paths, 1):
        md5s[p] = md5_of(p)
        ph = phash_int(p)
        if ph is not None:
            phs[p] = ph
        if i % 1000 == 0 or i == n:
            print(f"  {i}/{n}")
    return md5s, phs


def band_keys(h, bands):
    base, rem = HASH_BITS // bands, HASH_BITS % bands
    out, shift = [], 0
    for b in range(bands):
        width = base + (1 if b < rem else 0)
        out.append((b, (h >> shift) & ((1 << width) - 1)))
        shift += width
    return out


def near_pairs(ha, hb, max_dist, same_set=False):
    """(a, b, hamming) with distance <= max_dist via banded LSH (pigeonhole)."""
    bands = max_dist + 1
    buckets = defaultdict(list)
    for p, h in ha.items():
        for k in band_keys(h, bands):
            buckets[k].append((p, h))
    out, seen = [], set()
    for q, h in hb.items():
        checked = set()
        for k in band_keys(h, bands):
            for p, h2 in buckets.get(k, ()):
                if p in checked or (same_set and str(p) >= str(q)):
                    continue
                checked.add(p)
                d = bin(h ^ h2).count("1")
                if d <= max_dist and (p, q) not in seen:
                    seen.add((p, q))
                    out.append((p, q, d))
    return sorted(out, key=lambda t: t[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="CSV with train labels (auto-detected if omitted)")
    ap.add_argument("--img-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--near-dist", type=int, default=6)
    ap.add_argument("--report-dir", type=Path, default=Path("leakage_report"))
    a = ap.parse_args()
    a.report_dir.mkdir(exist_ok=True)
    findings = []

    def save(rows, name, msg):
        df = pd.DataFrame(rows)
        if len(df):
            out = a.report_dir / name
            df.to_csv(out, index=False)
            findings.append(f"[!] {msg}: {len(df)}  ->  {out}")
        else:
            findings.append(f"[ok] {msg}: 0")

    # ---- labels (optional, for conflict detection)
    name2label = {}
    csv = a.train_csv or guess_train_csv()
    if csv and Path(csv).exists():
        try:
            name2label, classes, ic, lc = load_label_map(csv, a.img_col, a.label_col)
            print(f"Labels: {csv}  (img-col='{ic}', label-col='{lc}', classes={classes})")
        except SystemExit as e:
            print(e)

    # ---- hash every split
    md5s, phs = {}, {}
    for s in SPLITS:
        paths = list_split(s)
        if paths:
            md5s[s], phs[s] = hash_many(paths, s)

    # ---- within-split exact dups + label conflicts
    for s, mm in md5s.items():
        groups = defaultdict(list)
        for p, h in mm.items():
            groups[h].append(p)
        dup, conf = [], []
        for ps in groups.values():
            if len(ps) > 1:
                labels = {name2label.get(p.name, "?") for p in ps}
                row = {"files": " | ".join(p.name for p in ps),
                       "labels": " | ".join(sorted(map(str, labels)))}
                dup.append(row)
                if len({l for l in labels if l != "?"}) > 1:
                    conf.append(row)
        save(dup, f"exact_dups_within_{s}.csv", f"Exact duplicates WITHIN {s}")
        if s == "train":
            save(conf, "label_conflicts_train.csv",
                 "Duplicate train images with CONFLICTING labels")

    # ---- cross-split exact + near (the actual leakage)
    for s1, s2 in combinations([s for s in SPLITS if s in md5s], 2):
        inv = defaultdict(list)
        for p, h in md5s[s1].items():
            inv[h].append(p)
        exact = [{f"{s1}_file": p.name, f"{s2}_file": q.name}
                 for q, h in md5s[s2].items() for p in inv.get(h, ())]
        save(exact, f"leak_exact_{s1}_{s2}.csv", f"EXACT leaks {s1}<->{s2}")
        exact_set = {(r[f"{s1}_file"], r[f"{s2}_file"]) for r in exact}
        near = [{f"{s1}_file": p.name, f"{s2}_file": q.name, "hamming": d}
                for p, q, d in near_pairs(phs[s1], phs[s2], a.near_dist)
                if (p.name, q.name) not in exact_set]
        save(near, f"leak_near_{s1}_{s2}.csv",
             f"NEAR-dup leaks {s1}<->{s2} (hamming<={a.near_dist})")

    # ---- injected_images vs all splits (Game 1's planted trap)
    inj_dir = ROOT / "games" / "game1_mirror_maze" / "injected_images"
    if inj_dir.exists():
        inj_paths = sorted(p for p in inj_dir.rglob("*")
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
        if inj_paths:
            imd5, iph = hash_many(inj_paths, "injected")
            for s in md5s:
                inv = defaultdict(list)
                for p, h in md5s[s].items():
                    inv[h].append(p)
                hits = [{"injected": p.name, "found_in": q.name, "match": "exact"}
                        for p, h in imd5.items() for q in inv.get(h, ())]
                hits += [{"injected": p.name, "found_in": q.name,
                          "match": f"near(h={d})"}
                         for p, q, d in near_pairs(iph, phs[s], a.near_dist)
                         if not any(x["injected"] == p.name and x["found_in"] == q.name
                                    for x in hits)]
                save(hits, f"injected_hits_in_{s}.csv",
                     f"Injected images found in {s}")
    else:
        findings.append(f"[--] injected_images dir not found at {inj_dir}")

    print("\n" + "=" * 66)
    print("MIRROR MAZE REPORT")
    print("=" * 66)
    for f in findings:
        print(" " + f)
    n_bad = sum(1 for f in findings if f.startswith("[!]"))
    print("-" * 66)
    if n_bad:
        print(" VERDICT: leakage found. Drop flagged TRAIN items (never touch")
        print(" val/test), resolve conflicts, document everything in the notebook,")
        print(" then re-run until clean.")
    else:
        print(" VERDICT: clean. Your validation scores can be trusted.")
    print("=" * 66)


if __name__ == "__main__":
    main()
