#!/usr/bin/env python3
"""Validate local and packaged AI Olympics 2026 submissions."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pandas as pd


PROJECT = Path(__file__).resolve().parent
MAIN_SUBMISSIONS = Path(
    r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
    r"\AI_Olympics_2026_Student_Release_v1\submissions"
)
# Optional local staging copy; when absent, validate the packaged
# submissions directly.
WORK = PROJECT / "work"
LOCAL_BASE = WORK if WORK.exists() else MAIN_SUBMISSIONS
GAME8_SUBMISSIONS = Path(
    r"C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
    r"\AI_Olympics_2026_Game8_Release_v1\submissions"
)

REQUIRED = {
    1: {
        "Game_1_The_Mirror_Maze_Completed.ipynb",
        "game1_corrected_train.csv",
        "game1_corrected_validation.csv",
        "game1_evidence_table.csv",
        "game1_audit.json",
        "game1_summary.txt",
    },
    2: {
        "Game_2_Data_Reconnaissance_Mission_Completed.ipynb",
        "game2_train_text_statistics.csv",
        "game2_validation_text_statistics.csv",
        "game2_train_image_statistics.csv",
        "game2_validation_image_statistics.csv",
        "game2_risk_priority_table.csv",
        "game2_audit.json",
    },
    3: {
        "Game_3_The_Noise_Lab_Completed.ipynb",
        "processed_train.csv",
        "processed_validation.csv",
        "preprocessing_experiments.csv",
        "final_preprocessing_decisions.csv",
        "preprocessing_summary.txt",
    },
    4: {
        "Game_4_Zero_to_Hero_Sprint_Completed.ipynb",
        "scratch_model_results.csv",
        "scratch_error_analysis.csv",
        "scratch_models_summary.txt",
    },
    5: {
        "Game_5_The_Transfer_Relay_Completed.ipynb",
        "transfer_model_results.csv",
        "scratch_vs_transfer_comparison.csv",
        "efficiency_comparison.csv",
        "transfer_learning_summary.txt",
    },
    6: {
        "Game_6_Optimization_Decathlon_Completed.ipynb",
        "optimization_experiments.csv",
        "best_models_summary.csv",
        "optimization_summary.txt",
    },
    7: {
        "Game_7_Black_Box_Torch_Completed.ipynb",
        "explainability_case_analysis.csv",
        "model_trust_assessment.csv",
        "explainability_summary.txt",
    },
    8: {
        "Game_8_Cross_Modal_Truth_Arena_Completed.ipynb",
        "game8_public_predictions.csv",
        "game8_model_comparison.csv",
        "game8_evidence_analysis.csv",
        "game8_summary.txt",
    },
}


def inventory(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> None:
    rows = []
    for game, required in REQUIRED.items():
        name = f"Game{game}_Submission_Ded_Sec"
        local = LOCAL_BASE / name
        packaged = MAIN_SUBMISSIONS / name
        missing = sorted(
            filename for filename in required if not (local / filename).is_file()
        )
        empty = sorted(
            str(path.relative_to(local))
            for path in local.rglob("*")
            if path.is_file() and path.stat().st_size == 0
        )
        notebooks = list(local.glob("*.ipynb"))
        readable_notebooks = 0
        executed_notebooks = 0
        for notebook_path in notebooks:
            notebook = nbformat.read(str(notebook_path), as_version=4)
            readable_notebooks += 1
            code_cells = [
                cell for cell in notebook.cells if cell.cell_type == "code"
            ]
            if not code_cells or any(
                cell.execution_count is not None for cell in code_cells
            ):
                executed_notebooks += 1
        local_inventory = inventory(local)
        packaged_inventory = inventory(packaged) if packaged.exists() else {}
        rows.append(
            {
                "game": game,
                "required_missing": len(missing),
                "empty_files": len(empty),
                "notebooks": len(notebooks),
                "readable_notebooks": readable_notebooks,
                "executed_notebooks": executed_notebooks,
                "local_files": len(local_inventory),
                "packaged_files": len(packaged_inventory),
                "package_inventory_matches": (
                    local_inventory == packaged_inventory
                ),
                "status": (
                    "PASS"
                    if not missing
                    and not empty
                    and notebooks
                    and readable_notebooks == len(notebooks)
                    and executed_notebooks == len(notebooks)
                    and local_inventory == packaged_inventory
                    else "FAIL"
                ),
                "details": "; ".join(
                    filter(
                        None,
                        [
                            f"missing={missing}" if missing else "",
                            f"empty={empty}" if empty else "",
                        ],
                    )
                ),
            }
        )

    game7 = LOCAL_BASE / "Game7_Submission_Ded_Sec"
    image_gallery_count = len(
        list((game7 / "image_explainability_gallery").glob("*.png"))
    )
    text_gallery_count = len(
        list((game7 / "text_explainability_gallery").glob("*.html"))
    )
    game8 = LOCAL_BASE / "Game8_Submission_Ded_Sec"
    public = pd.read_csv(game8 / "game8_public_predictions.csv")
    required_columns = [
        "sample_id",
        "predicted_label",
        "confidence",
        "primary_evidence",
        "reviewer_flag",
    ]
    game8_dedicated_matches = (
        inventory(game8)
        == inventory(GAME8_SUBMISSIONS / "Game8_Submission_Ded_Sec")
    )

    audit = pd.DataFrame(rows)
    audit.to_csv(PROJECT / "submission_audit.csv", index=False)
    print(audit.to_string(index=False))
    print(
        "game7_gallery_counts",
        {"image": image_gallery_count, "text": text_gallery_count},
    )
    print(
        "game8_contract",
        {
            "rows": len(public),
            "columns_match": public.columns.tolist() == required_columns,
            "labels_valid": public["predicted_label"].isin(
                ["fake", "real"]
            ).all(),
            "evidence_valid": public["primary_evidence"].isin(
                ["image", "text", "image_text_relation", "uncertain"]
            ).all(),
            "confidence_valid": public["confidence"].between(0, 1).all(),
            "dedicated_package_matches": game8_dedicated_matches,
        },
    )
    if not (audit["status"] == "PASS").all():
        raise SystemExit("One or more submissions failed validation")
    if image_gallery_count != 16 or text_gallery_count != 16:
        raise SystemExit("Game 7 gallery count mismatch")
    if (
        len(public) != 500
        or public.columns.tolist() != required_columns
        or not game8_dedicated_matches
    ):
        raise SystemExit("Game 8 package validation failed")


if __name__ == "__main__":
    main()
