import json
import shutil
from pathlib import Path

from src.experiment_log import best_experiment


def promote_best(
    experiments_log: Path,
    production_dir: Path,
    metric: str,
):
    best = best_experiment(experiments_log, metric)

    source_checkpoint = Path(best["checkpoint_path"])
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    production_dir.mkdir(parents=True, exist_ok=True)

    destination_checkpoint = production_dir / "model.pt"
    shutil.copy2(source_checkpoint, destination_checkpoint)

    metadata = {
        "selection_metric": metric,
        "selected_experiment": best,
        "model_file": str(destination_checkpoint),
    }

    (production_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return best, destination_checkpoint
