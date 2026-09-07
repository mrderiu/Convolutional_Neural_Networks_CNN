from pathlib import Path

import pandas as pd


DISPLAY_COLUMNS = [
    "experiment_id",
    "model_name",
    "val_roc_auc",
    "val_pr_auc",
    "val_recall_malignant",
    "val_specificity",
    "val_f1",
    "val_accuracy",
    "threshold",
    "duration_seconds",
]


def comparison_table(log_path: Path, metric: str = "val_roc_auc"):
    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} does not exist. Train at least one model."
        )

    df = pd.read_csv(log_path)

    if metric not in df.columns:
        raise KeyError(f"{metric} not found in experiments log.")

    columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    return df.sort_values(metric, ascending=False)[columns]
