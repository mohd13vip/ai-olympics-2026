#!/usr/bin/env python3
"""TOOL 0 - PROJECT RECON
Scans the entire competition release and writes project_report.txt:
folder tree, image counts per split, every CSV's schema + sample rows,
every notebook's instructions (markdown cells), READMEs, submission template,
and the .docx handbook text. Run this FIRST and share the report with your AI assistant.

Usage:  python tool_inspect.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from aio_config import ROOT, IMAGES_DIR, SPLITS, list_split

OUT = Path("project_report.txt")
MAX_MD_CHARS = 2000      # per markdown cell
MAX_DOCX_CHARS = 12000   # per docx
lines = []


def w(s=""):
    lines.append(str(s))


def section(title):
    w("\n" + "=" * 70)
    w(title)
    w("=" * 70)


def tree(root: Path, depth=0, max_depth=3):
    if depth > max_depth:
        return
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return
    files = [p for p in entries if p.is_file()]
    by_ext = Counter(p.suffix.lower() for p in files)
    pad = "  " * depth
    if depth:
        summary = ", ".join(f"{n}x {e or 'noext'}" for e, n in by_ext.most_common())
        w(f"{pad}{root.name}/  ({summary or 'empty'})")
    for f in files:
        if len(files) <= 12:
            w(f"{pad}  {f.name}")
    for d in (p for p in entries if p.is_dir()):
        tree(d, depth + 1, max_depth)


def main():
    if not ROOT.exists():
        sys.exit(f"ROOT not found: {ROOT}\nSet AIO_ROOT env var or edit aio_config.py")

    section(f"PROJECT TREE  ({ROOT})")
    tree(ROOT)

    section("IMAGE COUNTS PER SPLIT")
    def split_patterns(value):
        return [value] if isinstance(value, str) else list(value)

    for s in SPLITS:
        paths = list_split(s)
        w(f"{s:>6} ({', '.join(split_patterns(SPLITS[s]))}): {len(paths)} images"
          + (f"   e.g. {paths[0].name}" if paths else ""))
    if IMAGES_DIR.exists():
        all_patterns = [pat for value in SPLITS.values()
                        for pat in split_patterns(value)]
        others = [p for p in IMAGES_DIR.iterdir() if p.is_file()
                  and not any(p.match(pat) for pat in all_patterns)]
        if others:
            w(f"  [!] {len(others)} files match NO split pattern, e.g. {others[0].name}")

    section("EVERY CSV: SCHEMA + SAMPLE")
    for csv in sorted(ROOT.rglob("*.csv")):
        w(f"\n--- {csv.relative_to(ROOT)} ---")
        try:
            df = pd.read_csv(csv)
            w(f"shape={df.shape}  columns={list(df.columns)}")
            with pd.option_context("display.max_colwidth", 60, "display.width", 120):
                w(df.head(3).to_string())
            for c in df.columns:
                u = df[c].nunique(dropna=True)
                if u <= 12:
                    w(f"  value_counts[{c}]: {df[c].value_counts(dropna=False).to_dict()}")
        except Exception as e:
            w(f"  [error reading: {e}]")

    section("NOTEBOOK INSTRUCTIONS (markdown cells)")
    for nb in sorted(ROOT.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in str(nb):
            continue
        w(f"\n############ {nb.relative_to(ROOT)} ############")
        try:
            cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
            n_code = sum(1 for c in cells if c.get("cell_type") == "code")
            w(f"[{len(cells)} cells, {n_code} code]")
            for c in cells:
                if c.get("cell_type") == "markdown":
                    txt = "".join(c.get("source", []))[:MAX_MD_CHARS]
                    w(txt)
                    w("- - -")
        except Exception as e:
            w(f"  [error reading: {e}]")

    section("TEXT FILES (READMEs, rules)")
    for txt in sorted(ROOT.rglob("*.txt")):
        w(f"\n--- {txt.relative_to(ROOT)} ---")
        try:
            w(txt.read_text(encoding="utf-8", errors="replace")[:6000])
        except Exception as e:
            w(f"  [error: {e}]")

    section("DOCX HANDBOOK / CHEAT SHEET")
    try:
        from docx import Document
        for dx in sorted(ROOT.rglob("*.docx")):
            if dx.name.startswith("~$"):
                continue
            w(f"\n--- {dx.relative_to(ROOT)} ---")
            try:
                text = "\n".join(p.text for p in Document(str(dx)).paragraphs if p.text.strip())
                w(text[:MAX_DOCX_CHARS])
                if len(text) > MAX_DOCX_CHARS:
                    w(f"...[truncated, {len(text)} chars total]")
            except Exception as e:
                w(f"  [error: {e}]")
    except ImportError:
        w("python-docx not installed - run:  pip install python-docx  and re-run to include the handbook.")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT.resolve()}  ({OUT.stat().st_size/1024:.0f} KB)")
    print("Open it, skim it, and paste it to your AI assistant for fully tailored help.")


if __name__ == "__main__":
    main()
