#!/usr/bin/env python3
"""TOOL 4/5/6 - TRAINING ENGINE (images)
Game 4 (scratch), Game 5 (transfer), Game 6 (optimization) in one script.
Every run logs to runs/<name>/ with:
  config.json, log.csv (training history), learning_curve.png,
  confusion_matrix.csv/.png, val_predictions.csv (for error analysis),
  results.json (metrics + cost, consumed by tool_results.py), best.pt

Examples:
  python tool_train.py --model scratch --epochs 15 --no-pretrained   # Game 4
  python tool_train.py --model efficientnet_b0 --freeze-epochs 2     # Game 5
  python tool_train.py --model resnet50 --lr 3e-5 --label-smoothing 0.1
Models: scratch | resnet18 | resnet50 | efficientnet_b0 | efficientnet_b3 | vit_b_16
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

from aio_config import seed_all
from tool_dataset import make_loaders


class ScratchCNN(nn.Module):
    """Game 4: built from scratch, no pretrained weights."""

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, num_classes, dropout=0.3, pretrained=True):
    if name == "scratch":
        return ScratchCNN(num_classes, dropout)
    from torchvision import models
    w = "DEFAULT" if pretrained else None
    if name in ("resnet18", "resnet50"):
        m = getattr(models, name)(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name in ("efficientnet_b0", "efficientnet_b3"):
        m = getattr(models, name)(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "vit_b_16":
        m = models.vit_b_16(weights=w)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    raise SystemExit(f"Unknown model '{name}'")


def head_parameters(model, name):
    if name == "scratch":
        return model.head.parameters()
    if name.startswith("resnet"):
        return model.fc.parameters()
    if name.startswith("efficientnet"):
        return model.classifier.parameters()
    if name == "vit_b_16":
        return model.heads.parameters()
    return model.parameters()


@torch.no_grad()
def evaluate(model, loader, device, num_classes, collect=False):
    model.eval()
    ys, ps, ids = [], [], []
    t0 = time.time()
    for x, y, sid in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        ps.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
        ids += list(sid)
    infer_sec = time.time() - t0
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = p.argmax(1)
    out = {"acc": accuracy_score(y, pred),
           "f1": f1_score(y, pred, average="macro"),
           "precision": precision_score(y, pred, average="macro", zero_division=0),
           "recall": recall_score(y, pred, average="macro", zero_division=0),
           "ms_per_image": 1000 * infer_sec / max(len(y), 1)}
    try:
        out["auc"] = (roc_auc_score(y, p[:, 1]) if num_classes == 2
                      else roc_auc_score(y, p, multi_class="ovr"))
    except ValueError:
        out["auc"] = float("nan")
    if collect:
        return out, (ids, y, pred, p)
    return out


def save_eval_artifacts(run_dir, classes, ids, y, pred, p):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = p.max(1)
    pd.DataFrame({"sample_id": ids,
                  "true_label": [classes[i] for i in y],
                  "predicted_label": [classes[i] for i in pred],
                  "confidence": conf.round(4),
                  "correct": (y == pred)}).to_csv(
        run_dir / "val_predictions.csv", index=False)

    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                 columns=[f"pred_{c}" for c in classes]).to_csv(
        run_dir / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=120); plt.close()


def plot_curves(run_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = pd.read_csv(run_dir / "log.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(log.epoch, log.train_loss, marker="o")
    ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(log.epoch, log.val_acc, marker="o", label="val acc")
    ax[1].plot(log.epoch, log.val_f1, marker="s", label="val macro-F1")
    ax[1].set_title("validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(run_dir / "learning_curve.png", dpi=120)
    plt.close()


def run_training(a):
    seed_all(getattr(a, "seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={device}  bf16={use_bf16}")

    drop = None
    if a.drop_list and Path(a.drop_list).exists():
        drop = [l.strip() for l in Path(a.drop_list).read_text().splitlines() if l.strip()]
        print(f"Dropping {len(drop)} flagged train rows from {a.drop_list}")

    train_loader, val_loader, classes = make_loaders(
        a.batch, a.size, a.workers, a.train_csv or a.csv, a.val_csv,
        a.img_col, a.label_col, drop, augment=not a.no_augment)
    nc = len(classes)
    print(f"classes={classes}  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}")

    model = build_model(a.model, nc, a.dropout, pretrained=not a.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    weight = (train_loader.dataset.class_weights().to(device)
              if a.class_weights else None)
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing, weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    run_dir = Path(a.runs_dir) / f"{time.strftime('%m%d_%H%M%S')}_{a.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))
    log_path = run_dir / "log.csv"
    log_path.write_text("epoch,train_loss,val_acc,val_f1,val_auc,lr,sec\n")

    best_metric, bad_epochs, train_sec = -1.0, 0, 0.0
    for epoch in range(1, a.epochs + 1):
        if a.freeze_epochs:
            freeze = epoch <= a.freeze_epochs and a.model != "scratch"
            for p in model.parameters():
                p.requires_grad = not freeze
            for p in head_parameters(model, a.model):
                p.requires_grad = True

        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        sec = time.time() - t0
        train_sec += sec

        m = evaluate(model, val_loader, device, nc)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:>2}/{a.epochs}  loss={total/seen:.4f}  "
              f"acc={m['acc']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  "
              f"lr={lr_now:.2e}  {sec:.0f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{total/seen:.5f},{m['acc']:.5f},{m['f1']:.5f},"
                    f"{m['auc']:.5f},{lr_now:.6e},{sec:.1f}\n")

        score = m["f1"] if a.select == "f1" or m["auc"] != m["auc"] else m["auc"]
        if score > best_metric:
            best_metric, bad_epochs = score, 0
            torch.save({"model_name": a.model, "state_dict": model.state_dict(),
                        "classes": classes, "size": a.size, "score": score},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f"Early stop (no val improvement for {a.patience} epochs).")
                break

    # ---- final artifacts from the BEST checkpoint
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    m, (ids, y, pred, p) = evaluate(model, val_loader, device, nc, collect=True)
    save_eval_artifacts(run_dir, classes, ids, y, pred, p)
    plot_curves(run_dir)

    results = {
        "model_id": run_dir.name, "modality": "image", "architecture": a.model,
        "pretrained": not a.no_pretrained and a.model != "scratch",
        "accuracy": round(m["acc"], 4), "macro_f1": round(m["f1"], 4),
        "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
        "auc": round(m["auc"], 4) if m["auc"] == m["auc"] else "",
        "training_time_sec": round(train_sec, 1),
        "inference_ms_per_sample": round(m["ms_per_image"], 2),
        "params": n_params,
        "model_size_mb": round((run_dir / "best.pt").stat().st_size / 1e6, 1),
        "checkpoint": str(run_dir / "best.pt"),
        "train_csv": str(a.train_csv or a.csv), "val_csv": str(a.val_csv),
        "config": {k: vars(a)[k] for k in ("lr", "weight_decay", "dropout",
                   "label_smoothing", "freeze_epochs", "batch", "size", "epochs")},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nBest val score {best_metric:.4f}")
    print(f"Artifacts in {run_dir}/ : best.pt, log.csv, learning_curve.png, "
          f"confusion_matrix.*, val_predictions.csv, results.json")
    return best_metric, run_dir


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="freeze backbone for first N epochs (transfer learning)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--class-weights", action="store_true",
                    help="weight the loss by inverse class frequency")
    ap.add_argument("--select", choices=["auc", "f1"], default="auc",
                    help="metric used to keep the best checkpoint")
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--val-csv", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None, help="alias for --train-csv")
    ap.add_argument("--img-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-list", type=Path, default=None,
                    help="txt of leaked train sample_ids/filenames to exclude (Game 1)")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return ap


if __name__ == "__main__":
    run_training(get_parser().parse_args())
