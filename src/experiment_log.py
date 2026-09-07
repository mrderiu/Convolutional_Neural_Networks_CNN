import csv
from pathlib import Path
from typing import Dict, List


def append_experiment(log_path: Path, row: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()

    fieldnames = list(row.keys())

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_experiments(log_path: Path) -> List[Dict[str, str]]:
    if not log_path.exists():
        return []

    with log_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_experiment(log_path: Path, metric: str) -> Dict[str, str]:
    rows = read_experiments(log_path)
    if not rows:
        raise FileNotFoundError(
            f"No experiments found in {log_path}. Train models first."
        )

    if metric not in rows[0]:
        raise KeyError(
            f"Metric '{metric}' not found. Available: {list(rows[0].keys())}"
        )

    # Only validation metrics are allowed for model promotion.
    if not metric.startswith("val_"):
        raise ValueError(
            "Production model selection must use a validation metric, not test."
        )

    return max(rows, key=lambda row: float(row[metric]))
