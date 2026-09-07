from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.transforms import build_transforms


def _validate_dataset_structure(data_dir: Path) -> None:
    required = [
        data_dir / "train" / "benign",
        data_dir / "train" / "malignant",
        data_dir / "test" / "benign",
        data_dir / "test" / "malignant",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing:\n"
            f"{msg}\n\nExpected data/input/train|test/benign|malignant."
        )


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    validation_ratio: float,
    seed: int,
    num_workers: int,
):
    """
    Uses the Kaggle train folder to create TRAIN + VALIDATION.
    The original Kaggle test folder remains untouched until final evaluation.
    """
    _validate_dataset_structure(data_dir)

    train_transform, eval_transform = build_transforms(image_size)

    # Read labels once without transforms to create a stratified split.
    raw_train = datasets.ImageFolder(data_dir / "train", transform=None)

    class_to_idx = {k.lower(): v for k, v in raw_train.class_to_idx.items()}
    if "benign" not in class_to_idx or "malignant" not in class_to_idx:
        raise ValueError(
            f"Expected classes benign/malignant, got {raw_train.classes}"
        )

    indices = np.arange(len(raw_train))
    targets = np.asarray(raw_train.targets)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        stratify=targets,
    )

    # Separate ImageFolder instances prevent validation from receiving augmentation.
    train_base = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_base = datasets.ImageFolder(data_dir / "train", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    train_dataset = Subset(train_base, train_idx.tolist())
    val_dataset = Subset(val_base, val_idx.tolist())

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    train_targets = targets[train_idx]
    counts = {
        int(cls_idx): int((train_targets == cls_idx).sum())
        for cls_idx in np.unique(train_targets)
    }

    metadata = {
        "class_names": raw_train.classes,
        "class_to_idx": raw_train.class_to_idx,
        "positive_class_idx": raw_train.class_to_idx["malignant"],
        "train_class_counts": counts,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return loaders, metadata


def compute_pos_weight(metadata: Dict) -> float:
    """
    BCEWithLogitsLoss pos_weight = negatives / positives.
    Malignant is treated as the positive class.
    """
    positive_idx = metadata["positive_class_idx"]
    negative_idx = 1 - positive_idx

    n_pos = metadata["train_class_counts"][positive_idx]
    n_neg = metadata["train_class_counts"][negative_idx]

    if n_pos == 0:
        raise ValueError("Training split contains zero malignant samples.")

    return n_neg / n_pos
