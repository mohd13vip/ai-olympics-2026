#!/usr/bin/env python3
"""TOOL 7 - BLACK-BOX TORCH (Grad-CAM, no extra dependencies)
Visualize WHERE the model looks when it calls an image real or fake.
Works with checkpoints saved by tool_train.py (scratch / resnet / efficientnet).

Usage:
  python tool_explain.py --ckpt runs/<run>/best.pt --n 6
Outputs heatmap overlays to explain_out/.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from tool_dataset import AIODataset, build_transforms
from tool_train import build_model


def target_layer(model, name):
    if name == "scratch":
        return model.features[-1][3]          # last conv in last block
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("efficientnet"):
        return model.features[-1]
    raise SystemExit(f"Grad-CAM target not defined for '{name}' "
                     "(ViT needs attention rollout - ask your assistant).")


class GradCAM:
    def __init__(self, model, layer):
        self.acts, self.grads = None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0]))
        self.model = model

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        idx = logits.argmax(1) if class_idx is None else torch.tensor([class_idx])
        logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(1)[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6, help="images per class")
    ap.add_argument("--split", default="val")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("explain_out"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    random.seed(0)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    name, classes, size = ck["model_name"], ck["classes"], ck.get("size", 224)
    model = build_model(name, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cam = GradCAM(model, target_layer(model, name))

    ds = AIODataset(a.split, csv=a.csv, transform=build_transforms(False, size),
                    classes=classes)
    by_class = {i: [] for i in range(len(classes))}
    for j, y in enumerate(ds.labels):
        by_class[y].append(j)

    for ci, idxs in by_class.items():
        for j in random.sample(idxs, min(a.n, len(idxs))):
            x, y, fname = ds[j]
            heat, probs = cam(x.unsqueeze(0))
            with Image.open(ds.paths[j]) as im:
                im = im.convert("RGB").resize((size, size))
            heat_img = np.array(Image.fromarray(
                np.uint8(heat * 255)).resize((size, size), Image.BILINEAR)) / 255.0

            fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
            ax[0].imshow(im); ax[0].axis("off")
            ax[0].set_title(f"true: {classes[y]}", fontsize=9)
            ax[1].imshow(im); ax[1].imshow(heat_img, cmap="jet", alpha=0.45)
            ax[1].axis("off")
            pred = int(probs.argmax())
            ax[1].set_title(f"pred: {classes[pred]} ({probs[pred]:.2f})", fontsize=9)
            plt.tight_layout()
            safe = Path(fname).stem
            plt.savefig(a.out / f"{classes[y]}_{safe}_cam.png", dpi=110)
            plt.close()
    print(f"Saved Grad-CAM overlays to {a.out}/ - look for whether the model "
          "focuses on faces/textures/backgrounds, and include 3-4 in your notebook.")


if __name__ == "__main__":
    main()
