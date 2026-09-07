import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from src.metrics import binary_metrics
from src.model_factory import build_model


@torch.no_grad()
def collect_predictions(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_labels = []
    all_probabilities = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits)

        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.numpy())

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
    )


def evaluate_production_model(
    checkpoint_path: Path,
    test_loader,
    device,
    figures_dir: Path,
    output_json: Path,
) -> Dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        checkpoint["model_name"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    y_true, probabilities = collect_predictions(model, test_loader, device)
    threshold = float(checkpoint["threshold"])
    metrics = binary_metrics(y_true, probabilities, threshold)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(y_true, probabilities, threshold, figures_dir)
    _save_roc_curve(y_true, probabilities, figures_dir)
    _save_pr_curve(y_true, probabilities, figures_dir)

    return metrics


def _save_confusion_matrix(y_true, probabilities, threshold, figures_dir):
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["benign", "malignant"],
    ).plot(ax=ax, values_format="d")
    ax.set_title("Production model - Test confusion matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def _save_roc_curve(y_true, probabilities, figures_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Production model - Test ROC curve")
    fig.tight_layout()
    fig.savefig(figures_dir / "test_roc_curve.png", dpi=160)
    plt.close(fig)


def _save_pr_curve(y_true, probabilities, figures_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Production model - Test Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(figures_dir / "test_pr_curve.png", dpi=160)
    plt.close(fig)
