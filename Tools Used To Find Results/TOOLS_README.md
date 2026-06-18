---
date: 2026-06-11
tags:
  - ai-olympics
  - python-tools
  - reproducibility
---

# Exact Tools Used to Find the Results

This folder contains the complete Python source used to generate and verify the
AI Olympics results.

## Main Tools

- `Game 1.py` through `Game 8.py`: complete, directly runnable game solutions.
- `game1_solver.py` through `game8_solver.py`: original solver module names.

The `Game N.py` and matching `gameN_solver.py` files contain the same full
implementation. Both names are included for clarity and compatibility because
later games import functions and model classes from earlier solver modules.

## Verification Tools

- `game8_probe.py`: inspects how Game 8 image-text pairs were constructed.
- `game8_accuracy_audit.py`: measures the reported Game 8 result under multiple
  canonical-manifest assumptions.
- `validate_submissions.py`: checks all eight submission folders, notebooks,
  required files, galleries, and the Game 8 output contract.

## Phase 2 Tool (16 June 2026)

- `phase2_runner.py`: runs the final Game 8 system on a new test CSV and
  writes the contract output. Dress-rehearsal verified: reproduces the
  submitted public-test predictions with 100% agreement.

```powershell
python phase2_runner.py --test-csv "path\to\hidden_test.csv"
```

## Running a Game

Open PowerShell in the tools folder and run:

```powershell
python "Game 1.py"
```

Replace `1` with the required game number. The scripts contain default paths
for the release packages under:

`C:\Users\mohd1\OneDrive\Desktop\ai olpyc`

Use `python "Game 1.py" --help` to see optional path and training arguments.

