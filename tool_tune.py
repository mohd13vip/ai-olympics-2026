#!/usr/bin/env python3
"""TOOL 6b - OPTIMIZATION DECATHLON (Optuna search)
Autonomous hyperparameter search over lr / weight decay / dropout /
label smoothing using short training runs. Your tuning agent.

Usage:
  pip install optuna
  python tool_tune.py --model efficientnet_b0 --trials 15 --epochs 5
Best params land in tune_results.csv; retrain the winner with full epochs.
"""
import argparse

try:
    import optuna
except ImportError:
    raise SystemExit("pip install optuna  - then re-run.")

from tool_train import get_parser, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--img-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-list", default=None)
    t = ap.parse_args()

    def objective(trial):
        args = get_parser().parse_args([])
        args.model = t.model
        args.epochs = t.epochs
        args.batch = t.batch
        args.csv = t.csv
        args.img_col, args.label_col = t.img_col, t.label_col
        args.drop_list = t.drop_list
        args.patience = 3
        args.lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        args.freeze_epochs = trial.suggest_int("freeze_epochs", 0, 2)
        score, _ = run_training(args)
        return score

    study = optuna.create_study(direction="maximize",
                                study_name=f"aio_{t.model}")
    study.optimize(objective, n_trials=t.trials)
    print("\nBEST:", study.best_value, study.best_params)
    study.trials_dataframe().to_csv("tune_results.csv", index=False)
    print("Full table -> tune_results.csv. Retrain best params with --epochs 15+.")


if __name__ == "__main__":
    main()
