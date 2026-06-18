AI Olympics 2026 - Game 8 Final System
Team: Ded_Sec
======================================

WHAT THIS IS
------------
A fully self-contained multimodal Real/Fake classifier for the Phase 2
hidden Boss-Test. Everything needed to run it ships inside this folder;
no network access and no files outside this folder are required.

HOW TO RUN
----------
1. Install dependencies (Python 3.13 recommended, 3.11+ expected to work):

       pip install -r requirements.txt

2. From inside this folder, run:

       python inference.py --test-csv path\to\hidden_test.csv

   If the hidden test ships new images in a separate folder, add:

       python inference.py --test-csv hidden.csv --images-dir path\to\new_images

   Optional flags: --output (default phase2_predictions.csv in the current
   directory), --cpu (force CPU), --image-batch, --text-batch, --seed.

INPUT / OUTPUT CONTRACT
-----------------------
Input CSV columns : sample_id, image_path, text
Output CSV columns: sample_id, predicted_label, confidence,
                    primary_evidence, reviewer_flag
- predicted_label is "fake" or "real".
- confidence is the probability of the predicted label (0..1).
- primary_evidence is one of: image, text, image_text_relation, uncertain.
- reviewer_flag marks low-confidence or conflicting cases for human review.
If the input CSV also contains a fully populated "label" column, the script
additionally prints accuracy / macro-F1 / precision / recall at the end.

EXPECTED CONSOLE OUTPUT
-----------------------
The run prints a validation leaderboard. The selected system should be:

    relation_gradient_boosting: 0.9340 <- selected

(verified reproducible on CPU; repeated runs on the same machine are
byte-identical, and predictions for the public test match the submitted
game8_public_predictions.csv. Hardware-level float differences may shift
the validation score by ~0.001 on other machines without changing the
selected system.)

HOW IT WORKS
------------
- Image expert : optimized 160px ScratchImageCNN from Game 6
                 (model_files/image_model).
- Text expert  : local MiniLM sentence encoder + classifier bundle with text
                 quality features from Game 6 (model_files/text_model).
- Cross-modal expert: local CLIP encoder (model_files/clip_model) scoring
  semantic image-text agreement (clip_similarity). This detects mismatched
  pairs even when neither component appears in any canonical manifest -
  exactly the regime of a hidden test built from new material.
- Relation layer: a canonical image-text pair manifest built from Games 1
  and 3 plus the official public-test manifest (reference_data/canonical).
  Known image and text components appearing in a non-canonical combination
  are strong "fake" relation evidence.
- Fusion: 20 features (expert probabilities, confidences, agreement,
  probability gap/product, relation match/mismatch interactions, CLIP
  similarity, text length/missingness) feed five candidate fusion systems.
  The one with the best validation macro-F1 is selected automatically
  (expected: HistGradientBoostingClassifier stack, refit on
  train+validation before scoring the hidden test).
- Hidden-domain robustness: the stacking models are fit on the original
  rows plus two relation-blind copies (canonical-manifest features zeroed;
  one copy clean, one with CLIP-similarity jitter simulating degraded
  imagery), so performance holds when the manifest carries no signal.
- Low CLIP similarity on manifest-unknown pairs predicted fake is reported
  as image_text_relation evidence and flagged for review when uncertain.

FOLDER LAYOUT
-------------
inference.py               entry point (self-contained, relative paths only)
requirements.txt           pinned dependencies
model_files/image_model/   config.json + model.pt
model_files/text_model/    classifier_bundle.joblib + encoder/ (local MiniLM)
model_files/clip_model/    local CLIP encoder (sentence-transformers format)
reference_data/            game8_train.csv, game8_validation.csv,
                           canonical/ pair manifests (5 CSVs)
score_cache/               precomputed image, text, and CLIP scores for all
                           13,000 reference samples plus the public test;
                           only images/texts/pairs not already in the cache
                           are scored at run time, so a hidden test of N
                           samples costs at most N image + N text + N CLIP
                           inferences (roughly 2-3 minutes on CPU for 500
                           new images, seconds when cached)

NOTES FOR EVALUATORS
--------------------
- The score cache makes reference-data scoring instant; it is safe to
  delete score_cache/*.csv, but then the first run must re-score all
  reference images, which requires the original release images and is much
  slower. Please keep the cache files in place.
- Determinism: seed fixed at 2026; repeated runs on the same machine
  produce identical outputs.
