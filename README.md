# AI Olympics 2026 — The Truth Challenge

**Team: Ded_Sec**

**Contributors:** [mohd13vip](https://github.com/mohd13vip) and [Thrictical](https://github.com/Thrictical)

An end-to-end multimodal Real/Fake content detection system built across the
eight connected games of Phase 1, ready for the Phase 2 Final Boss Test on
16 June 2026. This README is the complete manual: results, how to run every
game, how to use every tool, and how to verify everything.

---

## 1. Results at a Glance

| Game | Challenge | Key Result |
|------|-----------|------------|
| 1 — The Mirror Maze | Expose the untrustworthy validation split | Found **1,555 leaked cases** (exact/near-duplicate text and perceptually similar images crossing the split). Honest score after group-aware correction: 0.791 → **0.762 accuracy** |
| 2 — Data Reconnaissance | Measure data-quality risks before modeling | Full text/image statistics + prioritized risk table for both splits |
| 3 — The Noise Lab | Evidence-based preprocessing | 8 controlled experiments; destructive transforms **rejected** with evidence; all rows retained |
| 4 — Zero-to-Hero Sprint | Scratch models, no pretrained weights | Scratch CNN **0.8422 macro-F1** (image), TextCNN 0.7108 (text) |
| 5 — The Transfer Relay | Pretrained vs scratch, cost-aware | MobileNetV3 0.8377 (scratch CNN kept — it won), frozen MiniLM + LogReg **0.7646** for text |
| 6 — Optimization Decathlon | Hypothesis-driven tuning | Image **0.8682** (I04: 160px, doubled schedule), text **0.7674** (T04); 7 image + 5 text experiments, rejected ones retained in the log |
| 7 — Black-Box Torch | Explainability and trust | 32 explained cases, 16-image + 16-text galleries; 745 cross-modal disagreements found → motivated the Game 8 design |
| 8 — Cross-Modal Truth Arena | Relation-aware multimodal fusion | `relation_gradient_boosting`: **0.9300 accuracy / 0.9300 macro-F1** on the supplied benchmark (95% CI 0.9140–0.9348) |

**Honesty note (Game 8):** 0.9300 uses all canonical manifests from earlier
games. Our own stricter audit (`game8_accuracy_audit.csv`) shows 0.748–0.806
when relation knowledge is restricted — reported transparently in
`game8_summary.txt`.

**Reproducibility — both ends of the pipeline verified:**
- Game 1 re-run from scratch: all 5 artifacts byte-identical, metrics equal
  to 6 decimal places (`compare_game1_rerun.py`).
- Game 8 re-run on the public test: 100% agreement with the submitted
  predictions (`phase2_runner.py` dress rehearsal).

---

## 2. Setup

Python 3.13 with an NVIDIA GPU (CUDA) recommended; everything also runs on CPU.

```powershell
pip install torch torchvision scikit-learn pandas numpy pillow `
    sentence-transformers joblib nbformat imagehash opencv-python `
    matplotlib tqdm optuna
```

All scripts default to this folder layout. If the package moves, either pass
`--root` / path arguments, or set the environment variable once:

```powershell
$env:AIO_ROOT = "D:\new\location\AI_Olympics_2026_Student_Release_v1"
```

---

## 3. Quick Start — Verify Everything (3 commands)

```powershell
cd "Tools Used To Find Results"

# 1. Validate all 8 submission packages (files, notebooks, galleries, contract)
python validate_submissions.py

# 2. Prove Game 1 reproduces exactly (re-runs the full solver, then diffs)
python "..\Game 1.py" --output-dir "..\_game1_retest"
python compare_game1_rerun.py

# 3. Prove the final system reproduces the submitted predictions
python phase2_runner.py --test-csv "..\AI_Olympics_2026_Game8_Release_v1\data\game8_public_test.csv" --output dress_rehearsal.csv
```

Expected: `PASS` on all 8 games; `MATCH` on every Game 1 artifact; the
phase2 run selects `relation_gradient_boosting` at validation macro-F1 0.9300.

---

## 4. Running the Games

Each game is one self-contained script. Defaults point at the release
folders, so the plain command reproduces our submission. Outputs land in the
matching `submissions\GameN_Submission_Ded_Sec` folder (override with
`--output-dir` to keep the submitted artifacts untouched).

```powershell
python "Game 1.py"     # leakage hunt + corrected split        (~3 min)
python "Game 2.py"     # EDA statistics + risk table           (~5 min)
python "Game 3.py"     # preprocessing experiments             (~10 min)
python "Game 4.py"     # scratch CNN + TextCNN training        (GPU, ~30 min)
python "Game 5.py"     # MobileNetV3 + MiniLM transfer         (GPU, ~30 min)
python "Game 6.py"     # optimization experiments              (GPU, ~1 h)
python "Game 7.py"     # Grad-CAM galleries + trust analysis   (GPU, ~15 min)
python "Game 8.py"     # final multimodal system + predictions (GPU, ~5 min)
```

Every script supports `--help` for all options (`--root`, `--seed`,
`--team-name`, thresholds, batch sizes). Order matters: 3 needs 1–2,
4–7 need 3, 8 needs 1, 3 and 6. Seeds are fixed (42 for Games 1–7,
2026 for Game 8) — results are deterministic.

Hardening updates (June 2026):
- Game 3 quality flags are now adaptive: brightness combines absolute
  bounds (`--brightness-low` 30 / `--brightness-high` 200, was a hardcoded
  225) with a MAD outlier fence, and the sharpness flag
  (`--sharpness-flag-rate`, default 0.05) adds a log-scale MAD fence so a
  blur cluster larger than the quota is still fully caught.
- Game 7 backfills its case selection to exactly 16 + 16 gallery entries
  when error/uncertain cases are scarce (easy data), so
  `validate_submissions.py` can never fail on gallery counts. Selection is
  unchanged whenever the quota fills naturally — re-runs on the
  competition data still reproduce the submitted galleries.

---

## 5. Phase 2 — Final Boss Test (16 June 2026)

When the organizers provide the hidden test CSV
(`sample_id, image_path, text`):

```powershell
cd "Tools Used To Find Results"
python phase2_runner.py --test-csv "path\to\hidden_test.csv"
```

- New images in a separate folder? Add `--extra-images "path\to\new_images"`.
- Output: `phase2_predictions.csv` with
  `sample_id, predicted_label, confidence, primary_evidence, reviewer_flag`.
- If the organizers want the simple template format instead
  (`sample_id, predicted_label`), trim it with:

```powershell
python -c "import pandas as pd; pd.read_csv('phase2_predictions.csv')[['sample_id','predicted_label']].to_csv('phase2_simple.csv', index=False)"
```

- Scores are cached in `_phase2_cache`: the first run inferences the whole
  corpus (~2 min on GPU); later runs only compute the new file (<1 min).

**Day-of checklist:** confirm output format with organizers → run the
command → open the CSV and check the row count matches their file → submit.

---

## 6. Toolkit Reference (`tool_*.py`)

Reusable scripts in the root folder. All auto-detect CSVs/columns from the
release; every flag is optional unless marked.

| Tool | What it does | Example |
|------|--------------|---------|
| `tool_inspect.py` | Scans the whole release: folder tree, image counts per split, every CSV schema + sample rows → `project_report.txt` | `python tool_inspect.py` |
| `tool_eda.py` | Label distributions, text statistics, image brightness/contrast/sharpness plots | `python tool_eda.py --sample 500` |
| `tool_leakage.py` | Train/validation leakage scan: MD5 + perceptual-hash duplicates within and across splits | `python tool_leakage.py --near-dist 6` |
| `check_leakage.py` | Standalone leakage detector for arbitrary folders/CSVs (works outside this release) | `python check_leakage.py --train-csv a.csv --test-csv b.csv` |
| `tool_train.py` | Train an image classifier (scratch CNN or any torchvision/timm-style model) with logging, early stopping, eval artifacts and curves per run | `python tool_train.py --model scratch --epochs 12 --size 160` |
| `tool_tune.py` | Optuna hyperparameter search wrapped around `tool_train` (needs `pip install optuna`) | `python tool_tune.py --model scratch --epochs 8` |
| `tool_explain.py` | Grad-CAM visualizations for a trained checkpoint, correct vs incorrect cases | `python tool_explain.py --ckpt runs\<run>\best.pt --n 8` |
| `tool_submit.py` | Predict the `test1_*` images with a checkpoint → `sample_id,predicted_label` CSV (the `submission_template.csv` format); `--dry-random` validates the format only | `python tool_submit.py --ckpt runs\<run>\best.pt` |
| `tool_dataset.py` / `aio_config.py` / `aio_common.py` | Importable library: dataset/dataloader builders, path resolution, schema detection, seeding, the official TF-IDF baseline | `from aio_config import load_split_csv` |

---

## 7. Verification Tools (`Tools Used To Find Results\`)

| Tool | Purpose |
|------|---------|
| `validate_submissions.py` | Checks all 8 packages: required files, non-empty, notebooks executed, gallery counts (16+16), Game 8 contract columns/values, main vs dedicated package consistency |
| `compare_game1_rerun.py` | Diffs a fresh Game 1 re-run against the submitted package, file by file and metric by metric |
| `game8_probe.py` | Audits how Game 8 pairs relate to the earlier corpus (100% traceability, 80% canonical pairs) |
| `game8_accuracy_audit.py` | Stress-tests the 0.922 result under three relation-knowledge assumptions with bootstrap CIs |
| `phase2_runner.py` | Runs the final system on any new CSV (see section 5) |

---

## 8. Final System Architecture (Game 8)

1. **Image expert** — optimized 160-pixel scratch CNN from Game 6
   (experiment I04: doubled training schedule, macro-F1 0.8682).
2. **Text expert** — local MiniLM sentence embeddings + logistic classifier
   with text-quality features from Game 6.
3. **Relation layer** — canonical image-text pair manifest built from Games 1
   and 3 plus the public manifest; a known image paired with the wrong known
   text is strong fake evidence.
4. **Fusion** — gradient-boosted stack over 19 features (per-modality
   probabilities, confidence, agreement, relation matches/mismatches,
   interactions, text quality). Compared against image-only, text-only,
   weighted fusion, relation gating, a logistic stack, and a stack average;
   selected by validation macro-F1.
5. **Reviewer flag (bonus)** — flags confidence < 0.68, strong cross-modal
   disagreement, and weakly supported relation mismatches.

---

## 9. Folder Map

```
ai olpyc/
├── README.md                            <- this file
├── aio_config.py / aio_common.py        <- shared toolkit config (path auto-detection)
├── tool_*.py                            <- reusable toolkit (section 6)
├── check_leakage.py                     <- standalone split-leakage auditor
├── Game 1.py ... Game 8.py              <- complete, runnable game solutions
├── game1_solver.py ... game8_solver.py  <- same code under importable module names
├── project_report.txt                   <- generated by tool_inspect.py
├── AI_Olympics_2026_Student_Release_v1/
│   ├── data/images/                     <- 13,000 shared images (untouched)
│   ├── data/public_test.csv             <- public test manifest
│   ├── docs/                            <- handbook + cheat sheet (organizer originals)
│   ├── games/                           <- original starter notebooks (untouched)
│   └── submissions/
│       ├── Game1_Submission_Ded_Sec ... Game8_Submission_Ded_Sec
│       └── Game2_Submission_Ded_Sec.zip
├── AI_Olympics_2026_Game8_Release_v1/
│   ├── data/                            <- game8 train / validation / public test
│   └── submissions/Game8_Submission_Ded_Sec
└── Tools Used To Find Results/          <- verification + Phase 2 tools (section 7)
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Image not found` / `FileNotFoundError` | The package moved — set `$env:AIO_ROOT` (section 2) or pass `--root` |
| `pip install optuna - then re-run` | Only `tool_tune.py` needs it: `pip install optuna` |
| CUDA out of memory | Lower `--image-batch` / `--batch` (e.g. 32) |
| Slow on CPU | Everything works, just slower; `phase2_runner.py` cache makes repeat runs fast |
| Validator FAIL after editing a submission | Re-run the matching `Game N.py` so artifacts and notebook agree again |
| New test file has unknown columns | `phase2_runner.py` checks and names exactly what is missing |

---

*Verify first. Optimize second. Submit with confidence.* — Ded_Sec
