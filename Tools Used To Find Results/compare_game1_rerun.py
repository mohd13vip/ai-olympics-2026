#!/usr/bin/env python3
"""Compare a fresh Game 1 re-run against the submitted Game 1 package."""

import json
from pathlib import Path

import pandas as pd

SUBMITTED = Path(
    r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
    r"\AI_Olympics_2026_Student_Release_v1\submissions\Game1_Submission_Ded_Sec"
)
RERUN = Path(r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc\_game1_retest")

CSV_FILES = [
    "game1_corrected_train.csv",
    "game1_corrected_validation.csv",
    "game1_evidence_table.csv",
]
# Environment-dependent audit fields that are allowed to differ.
IGNORED_AUDIT_KEYS = {"root", "runtime_seconds", "elapsed_seconds"}


def compare_csv(name: str) -> str:
    old = pd.read_csv(SUBMITTED / name)
    new = pd.read_csv(RERUN / name)
    if old.shape != new.shape:
        return f"DIFFERENT shape: submitted {old.shape} vs re-run {new.shape}"
    if list(old.columns) != list(new.columns):
        return f"DIFFERENT columns: {list(old.columns)} vs {list(new.columns)}"
    sort_cols = list(old.columns)
    old_sorted = old.sort_values(sort_cols).reset_index(drop=True)
    new_sorted = new.sort_values(sort_cols).reset_index(drop=True)
    if old_sorted.equals(new_sorted):
        order = "identical" if old.reset_index(drop=True).equals(
            new.reset_index(drop=True)
        ) else "identical content, different row order"
        return f"MATCH ({len(old)} rows, {order})"
    diff_mask = (old_sorted != new_sorted) & ~(
        old_sorted.isna() & new_sorted.isna()
    )
    cells = int(diff_mask.to_numpy().sum())
    return f"DIFFERENT: {cells} differing cells of {old_sorted.size}"


def main() -> None:
    print(f"submitted: {SUBMITTED}")
    print(f"re-run:    {RERUN}\n")
    for name in CSV_FILES:
        print(f"{name}: {compare_csv(name)}")

    old_summary = (SUBMITTED / "game1_summary.txt").read_text(encoding="utf-8")
    new_summary = (RERUN / "game1_summary.txt").read_text(encoding="utf-8")
    print(
        "game1_summary.txt:",
        "MATCH" if old_summary == new_summary else "DIFFERENT",
    )
    if old_summary != new_summary:
        for old_line, new_line in zip(
            old_summary.splitlines(), new_summary.splitlines()
        ):
            if old_line != new_line:
                print(f"  submitted: {old_line}")
                print(f"  re-run:    {new_line}")

    old_audit = json.loads(
        (SUBMITTED / "game1_audit.json").read_text(encoding="utf-8")
    )
    new_audit = json.loads(
        (RERUN / "game1_audit.json").read_text(encoding="utf-8")
    )
    keys = sorted(
        (set(old_audit) | set(new_audit)) - IGNORED_AUDIT_KEYS
    )
    mismatched = [
        key for key in keys if old_audit.get(key) != new_audit.get(key)
    ]
    print(
        "game1_audit.json:",
        "MATCH (all compared fields)"
        if not mismatched
        else f"DIFFERENT fields: {mismatched}",
    )
    for key in mismatched:
        print(f"  {key}:")
        print(f"    submitted: {old_audit.get(key)}")
        print(f"    re-run:    {new_audit.get(key)}")

    print("\nHeadline metrics (submitted vs re-run):")
    for metric in ("accuracy", "macro_f1"):
        for stage in ("initial_metrics", "corrected_metrics"):
            old_value = old_audit[stage][metric]
            new_value = new_audit[stage][metric]
            flag = "OK" if abs(old_value - new_value) < 1e-9 else "DRIFT"
            print(
                f"  {stage}.{metric}: {old_value:.6f} vs {new_value:.6f} [{flag}]"
            )


if __name__ == "__main__":
    main()
