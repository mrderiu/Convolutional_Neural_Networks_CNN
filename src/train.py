import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.evaluate import collect_predictions
from src.metrics import binary_metrics, find_best_threshold
from src.model_factory import build_model, freeze_backbone, unfreeze_all
from src.utils import count_parameters


def _loss_on_loader(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            n += images.size(0)

    return running_loss / max(n, 1)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n += images.size(0)

    return running_loss / max(n, 1)


def train_experiment(
    model_name: str,
    train_loader,
    val_loader,
    metadata: Dict,
    device,
    experiments_dir: Path,
    pos_weight: float,
    head_epochs: int,
    fine_tune_epochs: int,
    head_lr: float,
    fine_tune_lr: float,
    weight_decay: float,
    scheduler_patience: int,
    early_stopping_patience: int,
    threshold_metric: str,
    pretrained: bool = True,
):
    started = time.perf_counter()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}_{model_name}"
    experiment_dir = experiments_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)

    model = build_model(model_name, pretrained=pretrained).to(device)
    freeze_backbone(model, model_name)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = float("-inf")
    epochs_without_improvement = 0
    global_epoch = 0

    def run_stage(stage_name, epochs, learning_rate):
        nonlocal best_state, best_val_auc
        nonlocal epochs_without_improvement, global_epoch

        if epochs <= 0:
            return

        optimizer = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=scheduler_patience,
        )

        epochs_without_improvement = 0

        for _ in range(epochs):
            global_epoch += 1
            train_loss = _train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = _loss_on_loader(
                model, val_loader, criterion, device
            )

            y_val, p_val = collect_predictions(model, val_loader, device)
            val_metrics = binary_metrics(y_val, p_val, threshold=0.50)
            val_auc = val_metrics["roc_auc"]

            scheduler.step(val_auc)

            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy_0.5": val_metrics["accuracy"],
                "val_recall_malignant_0.5": val_metrics["recall_malignant"],
                "val_specificity_0.5": val_metrics["specificity"],
                "val_f1_0.5": val_metrics["f1"],
                "val_roc_auc": val_auc,
                "val_pr_auc": val_metrics["pr_auc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            print(
                f"[{model_name}] {stage_name} epoch {global_epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"[{model_name}] Early stopping in {stage_name}."
                )
                break

    # Stage 1: train only classification head.
    run_stage("head", head_epochs, head_lr)

    # Stage 2: fine-tune the whole network at a smaller LR.
    if fine_tune_epochs > 0:
        model.load_state_dict(best_state)
        unfreeze_all(model)
        run_stage("fine_tune", fine_tune_epochs, fine_tune_lr)

    model.load_state_dict(best_state)

    # Select threshold ONLY on validation.
    y_val, p_val = collect_predictions(model, val_loader, device)
    threshold, _ = find_best_threshold(
        y_val,
        p_val,
        metric=threshold_metric,
    )
    val_metrics = binary_metrics(y_val, p_val, threshold)

    total_params, trainable_params = count_parameters(model)

    checkpoint_path = experiment_dir / "best.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "image_size": 224,
            "class_to_idx": metadata["class_to_idx"],
            "positive_class": "malignant",
            "validation_metrics": val_metrics,
        },
        checkpoint_path,
    )

    history_path = experiment_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    duration = time.perf_counter() - started

    summary = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "created_at_utc": timestamp,
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision_malignant": val_metrics["precision_malignant"],
        "val_recall_malignant": val_metrics["recall_malignant"],
        "val_specificity": val_metrics["specificity"],
        "val_f1": val_metrics["f1"],
        "threshold": threshold,
        "duration_seconds": round(duration, 2),
        "total_parameters": total_params,
        "trainable_parameters_final_stage": trainable_params,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
    }

    (experiment_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
