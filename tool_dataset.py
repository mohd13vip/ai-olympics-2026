"""TOOL 3 - DATASET + TRANSFORMS (Game 3 foundation)
PyTorch Dataset for the AI Olympics. CSV-first: every game runs off a labels
CSV with the official schema (sample_id, image_path, text, label), so the
dataset is built directly from CSV rows and image paths are resolved through
aio_config.resolve_path (covers data/images, game2 modified_images, etc.).

Requires: torch, torchvision
"""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

from aio_config import (list_split, load_split_csv, resolve_path,
                        guess_train_csv, guess_val_csv)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, size: int = 224, augment: bool = True):
    if train and augment:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(size * 1.15)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CSVImageDataset(Dataset):
    """Built from a competition CSV. Returns (image_tensor, label_idx, sample_id).
    label_idx = -1 when the CSV has no label column (e.g. public test)."""

    def __init__(self, csv, transform=None, classes=None,
                 drop=None, img_col=None, label_col=None):
        if isinstance(csv, (str, Path)):
            df, schema = load_split_csv(csv, img_col=img_col, label_col=label_col)
            self.source = str(csv)
        else:  # already a DataFrame
            from aio_config import detect_schema
            df, schema = csv.copy(), detect_schema(csv, img_col=img_col,
                                                   label_col=label_col)
            self.source = "<dataframe>"
        if schema["image"] is None:
            raise SystemExit(f"{self.source}: no image column found.")

        if drop:  # drop by sample_id OR image basename (Game 1 leak list)
            dropset = {str(d).strip() for d in drop}
            before = len(df)
            keep = ~(df[schema["image"]].map(lambda v: Path(str(v)).name).isin(dropset))
            if schema["id"]:
                keep &= ~df[schema["id"]].astype(str).isin(dropset)
            df = df[keep]
            print(f"[dataset] dropped {before - len(df)} flagged rows from {self.source}")

        self.paths, self.ids, raw_labels, skipped = [], [], [], 0
        for _, r in df.iterrows():
            try:
                p = resolve_path(r[schema["image"]])
            except FileNotFoundError:
                skipped += 1
                continue
            self.paths.append(p)
            self.ids.append(str(r[schema["id"]]) if schema["id"] else p.name)
            raw_labels.append(str(r[schema["label"]]) if schema["label"] else None)
        if skipped:
            print(f"[dataset] {skipped} rows in {self.source} had missing image files - skipped.")
        if not self.paths:
            raise SystemExit(f"{self.source}: 0 usable rows. Check image paths / AIO_ROOT.")

        if schema["label"]:
            self.classes = classes or sorted(set(raw_labels))
            idx = {c: i for i, c in enumerate(self.classes)}
            unknown = [l for l in raw_labels if l not in idx]
            if unknown:
                raise SystemExit(f"{self.source}: labels {sorted(set(unknown))} not in "
                                 f"classes {self.classes}.")
            self.labels = [idx[l] for l in raw_labels]
        else:
            self.classes, self.labels = classes, None
        self.transform = transform or build_transforms(False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        y = self.labels[i] if self.labels is not None else -1
        return x, y, self.ids[i]

    def class_weights(self):
        counts = torch.bincount(torch.tensor(self.labels),
                                minlength=len(self.classes)).float()
        return counts.sum() / (len(self.classes) * counts.clamp(min=1))


class AIODataset(CSVImageDataset):
    """Back-compat wrapper: AIODataset('test') still works for the unlabeled
    test1_* images in data/images; train/val now require a CSV."""

    def __init__(self, split, csv=None, img_col=None, label_col=None,
                 transform=None, classes=None, drop_files=None):
        if split == "test" and csv is None:
            paths = list_split("test")
            if not paths:
                raise SystemExit("No test images found in data/images.")
            df = pd.DataFrame({"sample_id": [p.name for p in paths],
                               "image_path": [str(p) for p in paths]})
            super().__init__(df, transform=transform or build_transforms(False),
                             classes=classes, drop=drop_files)
            return
        if csv is None:
            csv = guess_train_csv() if split == "train" else guess_val_csv()
            if csv is None:
                raise SystemExit(f"No labels CSV found for split '{split}' - pass csv=.")
            print(f"[dataset:{split}] using {csv}")
        super().__init__(csv, transform=transform or build_transforms(split == "train"),
                         classes=classes, drop=drop_files,
                         img_col=img_col, label_col=label_col)


def make_loaders(batch=32, size=224, workers=4, train_csv=None, val_csv=None,
                 img_col=None, label_col=None, drop_files=None, augment=True,
                 csv=None):
    """csv= kept for backwards compatibility (treated as train_csv)."""
    train_csv = train_csv or csv or guess_train_csv()
    val_csv = val_csv or guess_val_csv()
    if not train_csv or not val_csv:
        raise SystemExit("Could not find train/val CSVs - pass --train-csv/--val-csv.")
    print(f"[data] train={train_csv}\n[data] val  ={val_csv}")
    tr = CSVImageDataset(train_csv, build_transforms(True, size, augment),
                         drop=drop_files, img_col=img_col, label_col=label_col)
    va = CSVImageDataset(val_csv, build_transforms(False, size),
                         classes=tr.classes, img_col=img_col, label_col=label_col)
    pin = torch.cuda.is_available()
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers,
                       pin_memory=pin, drop_last=len(tr) > batch),
            DataLoader(va, batch_size=batch * 2, shuffle=False,
                       num_workers=workers, pin_memory=pin),
            tr.classes)
