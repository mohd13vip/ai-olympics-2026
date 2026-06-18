# AI Olympics 2026 — Game 8 (Cross-Modal Truth Arena)
**Team Ded_Sec · Phase 2 Boss Test summary**

> Merged from three source docs: `game8_v3_summary.txt`,
> `Final_System_v3/README_SYSTEM.txt`, and the Boss-Test `README.md`.
> Covers the task, results, system, the v2→v5 improvement, how to run it, and status.

---

## 📋 Challenge Requirements (official — Game 8)

> **The Cross-Modal Truth Arena — "When Image and Text Disagree, Who Should You Trust?"**
> Each sample contains an image and a text. Sometimes both support the same truth;
> sometimes the content is fake; sometimes the image looks real and the text sounds
> believable, but the *relationship between them* is wrong. **Mission: build a
> multimodal system that decides whether the content is Real or Fake** — not as two
> independent signals, but understanding what the image suggests, what the text
> suggests, whether they support each other, which modality to trust, and when a
> case should be sent for review.

**Your team must:**

1. Build an image-based model or representation.
2. Build a text-based model or representation.
3. Build a fusion system that combines image and text.
4. Include a strategy for measuring / modeling the image–text relationship.
5. Compare at least three systems: **image-only**, **text-only**, **fusion**.
6. Explain how the system decides which signal is more important.
7. Generate predictions for the public test file.
8. Provide confidence scores.
9. Provide an evidence field: `image` · `text` · `image_text_relation` · `uncertain`.
10. Optionally flag uncertain cases for human review.
11. Analyze errors on the validation set.
12. Save all required outputs in the correct format.

> **Important rule:** the goal is *not only* high accuracy — it is a system that can
> **reason across modalities and justify its decision**.

**Provided files:** `game8_train.csv`, `game8_validation.csv`, `game8_public_test.csv`,
`Game_8_Cross_Modal_Truth_Arena.ipynb` (images come from the shared `data/images/` folder; not repeated in the Game 8 zip).

**Required output — `game8_public_predictions.csv`:**
`sample_id, predicted_label, confidence, primary_evidence, reviewer_flag`

**Submission folder — `Game8_Submission_<Team>/`:** completed notebook +
`game8_public_predictions.csv`, `game8_model_comparison.csv`,
`game8_evidence_analysis.csv`, `game8_summary.txt`.

**Scoring (Phase 2):** `phase2_score = 70·macro_F1 + 20·accuracy + 10` — organizers
run our `inference.py` on hidden, out-of-distribution data.

**Submission rules:** use the required folder; don't rename required files; submit
completed notebooks + CSVs; preserve earlier-game outputs; the notebook must run
top-to-bottom.

---

## TL;DR — best result

| System | Boss test (2,000 balanced OOD imgs) | phase2_score |
|---|---|---|
| v2 (original + recalibration) | acc 0.722 / macro-F1 0.722 | **74.98** |
| **v5 (generalization-hardened)** | **acc 0.792 / macro-F1 0.792** | **≈ 81.3** |
| **Improvement** | **+7.0 pts**, wins on all 4 unseen generators | **+6.3** |

`phase2_score = 70·macro_F1 + 20·accuracy + 10` — organizers run **our `inference.py`** on hidden data.

---

## 1. The task

Real / fake **news-content detection**: each sample is an image + caption; a
*fake* is an AI-generated image and/or a fabricated caption. The system outputs a
label, confidence, the *primary evidence* it used, and a reviewer flag — an
**explainable** verdict, robust to brand-new (out-of-distribution) material.

The Phase 2 hidden Boss Test is **out-of-domain by design**: new generators
(DALL-E & SDXL) and new news sources (BBC & CNN style) never seen in training.

---

## 2. Results

### Boss test — v5 vs v2 (2,000 balanced images)

| system | acc / macro-F1 | bbc·dalle | cnn·dalle | bbc·sdxl | cnn·sdxl |
|---|---|---|---|---|---|
| v2 (recalibrated) | 0.7220 | 0.686 | 0.706 | 0.740 | 0.756 |
| **v5 (improved)** | **0.7920** | 0.756 | 0.804 | 0.770 | 0.838 |
| **Δ v5 − v2** | **+0.070** | +0.070 | +0.098 | +0.030 | +0.082 |

Gains on **all four unseen generators**; balanced confusion matrix **792 / 792**.

### Honest validation progression (macro-F1)

Fit on `game8_train`, measured on `game8_validation` (zero image/text/pair overlap):

| regime | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| **manifest-blind (true floor)** | 0.817 | 0.843 | 0.852 | **0.863** (+4.6) |
| prior-training manifest | 0.876 | 0.901 | 0.911 | 0.919 |
| full manifest (benchmark) | 0.935 | 0.967 | 0.976 | **0.984** |
| image expert alone | 0.724 | 0.754 | 0.762 | 0.762 |
| text expert alone | 0.652 | 0.681 | 0.681 | **0.722** |

> The number a run prints ("validation macro-F1 by system") uses the **full
> canonical manifest** and is the *optimistic* end. The realistic expectation on
> entirely new hidden material is the **manifest-blind floor (~0.84)**, not ~0.97.

---

## 3. Data — training, validation & the final test

| dataset | rows | columns | role |
|---|---|---|---|
| `game8_train.csv` | 10,000 | sample_id, image_path, text, label | model training |
| `game8_validation.csv` | 2,500 | (same) | honest yardstick — **zero** image/text/pair overlap with train |
| **`game8_boss_test.csv`** | **2,000** | sample_id, image_path, text | **the final Phase 2 boss test** (balanced, fully OOD) |
| `game8_boss_test_labels.csv` | 2,000 | sample_id, label | boss ground truth |

Canonical relation manifests (`reference_data/canonical/`): `game1_corrected_train`
9,999 / `validation` 2,501 · `processed_train` 9,000 / `validation` 2,500 · `public_test` 500.

**The final test lives in `model test\`.** That folder is where `game8_boss_test.csv`
(2,000 balanced OOD pairs) was scored end-to-end: `v2_boss_predictions.csv` (0.722)
vs `v5_boss_predictions.csv` (0.792), via `score_v5.py` / `score_compare.py` / `make_viz.py`.

> This summary **documents** the data; it does not embed the raw rows. The actual
> CSVs live in `Final_System_v3\reference_data\` (train/val + canonical) and
> `model test\` (the final boss test + labels + predictions).

### Before vs after cleaning

The data went through a deliberate **"clean carefully, preserve what matters"**
pipeline (label correction in Game 1, preprocessing in Game 3). It is
**non-destructive**: no rows deleted, images untouched on disk, `original_text`
kept — cleaning only *adds* quality/missingness flags and a normalization plan.

**Before cleaning** — raw corrected split, 4 columns:

```csv
sample_id,image_path,text,label
train__007432,data/images/train__007432.jpg,"Tourists and locals walked by an expanding pile of trash in the historic neighborhood of Trastevere in Rome earlier this month.",real
```

**After cleaning** — processed split, 14 columns (same row, + metadata):

```csv
sample_id,image_path,text,label,original_text,text_was_missing,image_width,image_height,image_aspect_ratio,image_low_resolution,image_low_sharpness,image_extreme_brightness,image_source,image_processing_plan
train__007432,data/images/train__007432.jpg,"Tourists and locals walked by an expanding pile of trash...",real,"Tourists and locals walked by an expanding pile of trash...",False,600,400,1.5,False,False,False,shared,exif_transpose_rgb_aspect_preserving_resize_pad_normalize
```

**Cleaning decisions (Game 3, evidence-driven):**

| transformation | modality | kept? | why |
|---|---|---|---|
| Explicit missing-text token + indicator | text | ✅ | 130 empty texts → make missingness explicit, preserve rows |
| `missing_token_only` transform | text | ✅ | within 0.002 macro-F1 of best, simplest reliable |
| EXIF transpose + RGB + aspect-preserving resize/pad | image | ✅ | 962 low-res + 75 extreme-aspect → stable inputs, no cropping |
| Quality & missingness flags | both | ✅ | lets models / error-analysis spot unreliable inputs |
| Automatic sharpening | image | ❌ | could amplify compression artifacts / erase a quality signal |
| Delete low-quality images | image | ❌ | would cut coverage + add selection bias |

Net effect on accuracy was **neutral** (raw-text baseline 0.7475 macro-F1; chosen
pipeline within 0.002) — cleaning was about **reliability and preserved signal,
not a score bump**. Quality audit of the processed split: ~760 low-resolution,
~450 low-sharpness, ~182 extreme-brightness, ~380 modified-source images;
~100 missing-text rows — all flagged, none dropped.

---

## 4. The v5 system (what ships in `Final_System_v3`)

A fully self-contained, **offline** multimodal Real/Fake classifier. Everything
needed ships in the folder; no network access, no external files.

- **Image expert** (dominant signal) — three independent, generalizing views:
  - `0.30` · mean of 3 scratch CNNs (Game 6: 160 / 160 / 192 px)
  - `0.35` · **frozen** logistic probe over CLIP + ImageNet ConvNeXt features
    *(backbones frozen → detect synthetic artifacts, cannot memorize pixels)*
  - `0.35` · fine-tuned EfficientNet-B0 *(heavy aug + early-stop → decorrelated view)*
- **Text expert** (the v5 win) — dominated by a fine-tuned MiniLM (BERT 22M)
  trained end-to-end as a fabrication detector, early-stopped on clean validation:
  - `0.70` · fine-tuned MiniLM   ·   `0.20` · char n-gram TF-IDF LogReg   ·   `0.10` · MiniLM+artifacts LogReg
  - Captures generation/alteration **style**, not specific names or topics.
- **Relation layer** —
  - (1) canonical image–text pair manifest (Games 1 & 3 + official public-test manifest); known parts in a non-canonical combo = strong "fake" evidence.
  - (2) local CLIP ViT-B/32 scores semantic image–text alignment → catches mismatches on new material.
- **Fusion** — 20 features (expert probs, confidences, agreement, prob gap/product,
  relation match/mismatch, CLIP alignment, text length/missingness) feed 5 candidate
  fusion systems; the best by validation macro-F1 is auto-selected and refit on
  train+validation. Also fit on **relation-blind** copies so it works when the
  manifest matches nothing. A final **median recalibration** recenters scores to the
  known balanced class prior of the boss test.

---

## 5. What changed v2 → v5, and why it matters

v2 leaned heavily on the canonical pair manifest — i.e. **memorization**. On a
hidden test built from brand-new material the manifest is blind, so most of v2's
0.935 headline evaporated (true floor ~0.817). v5 keeps the manifest as a bonus
but strengthens the two experts that actually **generalize**, raising the
manifest-blind floor to **0.863**.

**Why it generalizes:** two of three image views (generic scratch features + the
**frozen** CLIP/ConvNeXt probe) cannot memorize the hidden test; the fine-tuned
view is regularized and selected on clean validation. Text gains come from
character-level fabrication style, not specific names.

**What did NOT help (kept for the record):**
- Concatenating EfficientNet + ViT into the frozen probe **hurt** (val AUC 0.79→0.75) — weak backbones dilute CLIP. Probability-level ensembling is the right combiner.
- FFT / spectral fingerprint features: +0.001 only → dropped to keep inference simple.
- Fine-tuning EfficientNet-B0 **alone** only tied the frozen probe (0.755) — value is purely as a decorrelated ensemble member. Visual signal saturates near AUC 0.79 for every model tried.

---

## 6. How to run

```bash
# 1. install (Python 3.13 recommended; 3.11+ expected to work)
pip install -r requirements.txt

# 2. run from inside Final_System_v3/
python inference.py --test-csv path\to\hidden_test.csv
# if new images ship in a separate folder:
python inference.py --test-csv hidden.csv --images-dir path\to\new_images
```

Optional flags: `--output` (default `phase2_predictions.csv`), `--cpu`,
`--image-batch`, `--text-batch`, `--seed`. Validated on a **4 GB RTX 3050**;
lower `--image-batch` to 32 on out-of-memory.

### Input / output contract

| | columns |
|---|---|
| **Input CSV** | `sample_id`, `image_path`, `text` |
| **Output CSV** | `sample_id`, `predicted_label` (fake/real), `confidence` (0–1), `primary_evidence` (image / text / image_text_relation / uncertain), `reviewer_flag` |

If the input CSV also has a populated `label` column, the script prints
accuracy / macro-F1 / precision / recall at the end.

### Robustness to damaged inputs
- Truncated images are decoded as far as the data allows.
- Missing/undecodable images never stop the run: rows fall back to text + relation evidence, get neutral image features, and are flagged for review (console WARNING lists each).
- Missing/empty text handled symmetrically (neutralized CLIP similarity, empty TF-IDF string).
- One bad row costs at most that row — never the whole run.

---

## 7. Folder layout (`Final_System_v3/`)

```
inference.py                          entry point (self-contained, relative paths)
requirements.txt                      pinned dependencies
model_files/
  image_model/                        model.pt + g6_160_e12.pt + g6_192_e12.pt + config
  image_probe.joblib                  frozen CLIP+ConvNeXt logistic probe
  finetune/effnet_b0.pt               fine-tuned EfficientNet-B0 image detector
  finetune/minilm_text.pt             fine-tuned MiniLM text detector
  image_expert_config.json            image 3-view recipe + blend weights (v4)
  text_expert_config.json             text 3-view recipe + blend weights (v5)
  convnext/                           offline ImageNet ConvNeXt-Tiny weights
  text_model/encoder/                 local MiniLM  [+ legacy v2 bundle, unused]
  text_models.joblib                  MiniLM+artifacts LR, char-TFIDF, TFIDF LR
  clip_model/                         local CLIP ViT-B/32 (probe + alignment)
reference_data/                       game8_train.csv, game8_validation.csv, canonical/
score_cache/                          precomputed v3 expert + CLIP scores for reference rows
```

---

## 8. Notes for evaluators
- **scikit-learn must be 1.8.0** (pinned) — the joblib bundles were serialized with it.
- **Determinism:** seed fixed at 2026; repeated runs on the same machine produce byte-identical output (verified).
- The score cache makes reference scoring instant; deleting `score_cache/*.csv` is safe but the first run must re-score all reference rows (needs the original release images via `--images-dir`, slower).
- Everything loads from local folders; **no network access required**.

---

## 9. Status & open action

- **Boss submission of record:** original upload scored **74.98** (v2 `Final_System` +
  balanced/median recalibration; baseline before recal was 72.81 at threshold 0.5).
- **v5 (`Final_System_v3`) raises this to ≈ 81.3** on the boss test — confirmed by
  its configs: text = `v5` (`minilm_text.pt`), image = `v4` three-view blend.

> ⚠️ **Open action:** re-zip the `Solutions` deliverable **with `Final_System_v3`**
> and re-upload it, so the judges run the +7.0-point v5 system rather than the
> stale v2. Until then the uploaded package still scores 74.98.

### Source locations
- v5 deliverable: `solution\Game_8\Final_System_v3\inference.py`
- Working runner: `ai olpyc\Tools Used To Find Results\phase2_runner.py`
- Experiments: `solution\Game_8\_v3_experiments\`
- Training / validation data: `solution\Game_8\Final_System_v3\reference_data\` (`game8_train.csv` 10k, `game8_validation.csv` 2.5k, `canonical\`)
- **Final boss test (run + scoring): `model test\`** — `game8_boss_test.csv` (2,000), labels, `v2_/v5_boss_predictions.csv`, `score_v5.py`, `score_compare.py`, `make_viz.py`
- Boss judge package: `Boss_final\` (incl. `phase2_results\Ded_Sec\`)
