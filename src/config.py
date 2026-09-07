from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:

    data_dir: Path = ROOT_DIR / "input" / "data"

    logs_dir: Path = ROOT_DIR / "logs"
    experiments_dir: Path = ROOT_DIR / "models" / "experiments"
    production_dir: Path = ROOT_DIR / "models" / "production"
    figures_dir: Path = ROOT_DIR / "reports" / "figures"

    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    validation_ratio: float = 0.20
    num_workers: int = 2

    head_epochs: int = 8
    fine_tune_epochs: int = 7

    head_lr: float = 1e-3
    fine_tune_lr: float = 1e-5

    weight_decay: float = 1e-4

    scheduler_patience: int = 2
    early_stopping_patience: int = 5

    selection_metric: str = "val_roc_auc"
    threshold_metric: str = "f1"

    model_names: Tuple[str, ...] = (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet152",
        "efficientnet_b0",
        "densenet121",
    )


settings = Settings()