import argparse
from pathlib import Path

from src.compare_models import comparison_table
from src.config import settings
from src.data_loader import compute_pos_weight, create_dataloaders
from src.evaluate import evaluate_production_model
from src.experiment_log import append_experiment
from src.model_factory import AVAILABLE_MODELS
from src.promote import promote_best
from src.train import train_experiment
from src.utils import ensure_directories, get_device, set_seed


def load_data():
    return create_dataloaders(
        data_dir=settings.data_dir,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
        validation_ratio=settings.validation_ratio,
        seed=settings.seed,
        num_workers=settings.num_workers,
    )


def cmd_train(args):
    ensure_directories(
        settings.logs_dir,
        settings.experiments_dir,
        settings.production_dir,
        settings.figures_dir,
    )
    set_seed(settings.seed)
    device = get_device()

    print(f"Device: {device}")

    loaders, metadata = load_data()
    pos_weight = compute_pos_weight(metadata)

    print("Dataset metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print(f"  malignant pos_weight: {pos_weight:.4f}")

    models_to_train = (
        list(AVAILABLE_MODELS)
        if args.models == ["all"]
        else args.models
    )

    unknown = [m for m in models_to_train if m not in AVAILABLE_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Available: {AVAILABLE_MODELS}"
        )

    log_path = settings.logs_dir / "experiments.csv"

    for model_name in models_to_train:
        print("\n" + "=" * 72)
        print(f"TRAINING {model_name}")
        print("=" * 72)

        summary = train_experiment(
            model_name=model_name,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            metadata=metadata,
            device=device,
            experiments_dir=settings.experiments_dir,
            pos_weight=pos_weight,
            head_epochs=args.head_epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            head_lr=settings.head_lr,
            fine_tune_lr=settings.fine_tune_lr,
            weight_decay=settings.weight_decay,
            scheduler_patience=settings.scheduler_patience,
            early_stopping_patience=settings.early_stopping_patience,
            threshold_metric=settings.threshold_metric,
            pretrained=not args.no_pretrained,
        )
        append_experiment(log_path, summary)

    print("\nMODEL COMPARISON (validation only)")
    print(comparison_table(log_path, settings.selection_metric).to_string(index=False))

    if args.promote_best:
        best, model_path = promote_best(
            log_path,
            settings.production_dir,
            settings.selection_metric,
        )
        print(
            f"\nPromoted: {best['model_name']} "
            f"({settings.selection_metric}={best[settings.selection_metric]})"
        )

        if args.test_after_promotion:
            metrics = evaluate_production_model(
                checkpoint_path=model_path,
                test_loader=loaders["test"],
                device=device,
                figures_dir=settings.figures_dir,
                output_json=settings.production_dir / "test_metrics.json",
            )
            print("\nFINAL TEST METRICS (production model only)")
            for key, value in metrics.items():
                print(f"  {key}: {value}")


def cmd_compare(args):
    log_path = settings.logs_dir / "experiments.csv"
    df = comparison_table(log_path, args.metric)
    print(df.to_string(index=False))


def cmd_promote(args):
    log_path = settings.logs_dir / "experiments.csv"
    best, destination = promote_best(
        log_path,
        settings.production_dir,
        args.metric,
    )
    print(f"Promoted {best['model_name']} -> {destination}")


def cmd_test_production(args):
    set_seed(settings.seed)
    device = get_device()
    loaders, _ = load_data()

    metrics = evaluate_production_model(
        checkpoint_path=settings.production_dir / "model.pt",
        test_loader=loaders["test"],
        device=device,
        figures_dir=settings.figures_dir,
        output_json=settings.production_dir / "test_metrics.json",
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Skin cancer binary image classification project"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        help=f"Models or 'all'. Available: {AVAILABLE_MODELS}",
    )
    train_parser.add_argument(
        "--head-epochs",
        type=int,
        default=settings.head_epochs,
    )
    train_parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=settings.fine_tune_epochs,
    )
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--promote-best", action="store_true")
    train_parser.add_argument("--test-after-promotion", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--metric",
        default=settings.selection_metric,
    )
    compare_parser.set_defaults(func=cmd_compare)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument(
        "--metric",
        default=settings.selection_metric,
    )
    promote_parser.set_defaults(func=cmd_promote)

    test_parser = subparsers.add_parser("test-production")
    test_parser.set_defaults(func=cmd_test_production)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
