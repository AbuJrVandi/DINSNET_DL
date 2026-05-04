"""Main entry point for DINSNet training and inference."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Mitigate Intel OpenMP runtime crashes on macOS (libiomp5 aborts).
# These defaults are applied only when the shell hasn't set them already.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch

from data.dataset import build_dataloaders
from models.dinsnet import DINSNet
from models.unet import SimpleUNet
from trainer.trainer import Trainer
from utils import (
    calculate_flops,
    copy_config_to_experiment,
    count_parameters,
    create_child_experiment_paths,
    create_experiment_paths,
    export_run_metadata,
    load_config,
    save_model_profile,
    select_device,
    set_random_seed,
    setup_logger,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DINSNet: cross-dataset polyp segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Run mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path for inference or training resume override.",
    )
    parser.add_argument(
        "--inference-split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used in inference mode.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.loader.batch_size from config.")
    parser.add_argument("--lr", type=float, default=None, help="Override training.optimizer.lr from config.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.loader.num_workers from config.")
    parser.add_argument("--output-root", type=str, default=None, help="Override project.output_root from config.")
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default=None,
        help="Override project.experiment_prefix from config.",
    )
    parser.add_argument(
        "--disable-baseline",
        action="store_true",
        help="Disable baseline U-Net comparison for this run.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Run test evaluation after training (disabled by default to avoid leakage).",
    )
    return parser.parse_args()


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply optional CLI overrides onto loaded configuration."""
    overrides: Dict[str, Any] = {}

    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be > 0.")
        config["training"]["epochs"] = int(args.epochs)
        overrides["training.epochs"] = int(args.epochs)

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0.")
        config["data"]["loader"]["batch_size"] = int(args.batch_size)
        overrides["data.loader.batch_size"] = int(args.batch_size)

    if args.lr is not None:
        if args.lr <= 0.0:
            raise ValueError("--lr must be > 0.")
        config["training"]["optimizer"]["lr"] = float(args.lr)
        overrides["training.optimizer.lr"] = float(args.lr)

    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be >= 0.")
        config["data"]["loader"]["num_workers"] = int(args.num_workers)
        overrides["data.loader.num_workers"] = int(args.num_workers)

    if args.output_root:
        config["project"]["output_root"] = str(args.output_root)
        overrides["project.output_root"] = str(args.output_root)

    if args.experiment_prefix:
        config["project"]["experiment_prefix"] = str(args.experiment_prefix)
        overrides["project.experiment_prefix"] = str(args.experiment_prefix)

    if args.disable_baseline:
        config.setdefault("comparison", {})["run_baseline"] = False
        overrides["comparison.run_baseline"] = False

    return overrides


def split_statistics(split_samples: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build split-level counts and per-domain sample statistics."""
    output: Dict[str, Dict[str, Any]] = {}
    for split, samples in split_samples.items():
        domain_counts: Dict[str, int] = {}
        for sample in samples:
            domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1
        output[split] = {
            "num_samples": len(samples),
            "domain_counts": domain_counts,
        }
    return output


def _format_duration(total_seconds: float) -> str:
    seconds = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def _format_device(device: torch.device) -> str:
    if device.type == "cuda":
        device_idx = 0 if device.index is None else int(device.index)
        return f"CUDA ({torch.cuda.get_device_name(device_idx)})"
    return "CPU"


def _format_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    if value >= 1_000:
        return f"{value / 1_000:.2f} K"
    return str(value)


def _format_flops(flops: float | None) -> str:
    if flops is None:
        return "N/A"
    if flops >= 1_000_000_000:
        return f"{flops / 1_000_000_000:.2f} GMac"
    if flops >= 1_000_000:
        return f"{flops / 1_000_000:.2f} MMac"
    return f"{flops:.0f}"


def _relative_path(path: Path) -> str:
    abs_path = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(abs_path.relative_to(cwd))
    except Exception:
        return str(abs_path)


def _best_metrics_from_history(csv_path: Path) -> tuple[int | None, Dict[str, float]]:
    if not csv_path.exists():
        return None, {}
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, {}

    def _safe_float(row: Dict[str, str], key: str) -> float:
        return float(row.get(key, "0.0") or 0.0)

    best_row = max(rows, key=lambda row: _safe_float(row, "val_dice"))
    best_epoch = int(best_row["epoch"]) if best_row.get("epoch") else None
    metrics = {
        "dice": _safe_float(best_row, "val_dice"),
        "iou": _safe_float(best_row, "val_iou"),
        "precision": _safe_float(best_row, "val_precision"),
        "recall": _safe_float(best_row, "val_recall"),
    }
    return best_epoch, metrics


def _format_metric(metrics: Dict[str, float] | None, key: str) -> str:
    if not metrics or key not in metrics:
        return "N/A"
    try:
        return f"{float(metrics[key]):.4f}"
    except Exception:
        return "N/A"


def _serialize_metrics(metrics: Dict[str, float] | None) -> Dict[str, float] | None:
    if metrics is None:
        return None
    serialized: Dict[str, float] = {}
    for key, value in metrics.items():
        serialized[key] = float(value)
    return serialized


def _build_summary_report(
    exp_id: str,
    total_images: int,
    train_images: int,
    val_images: int,
    test_images: int,
    image_h: int,
    image_w: int,
    batch_size: int,
    completed_epochs: int,
    max_epochs: int,
    early_stopped: bool,
    training_seconds: float,
    device: torch.device,
    best_epoch: int | None,
    best_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float] | None,
    total_params: int,
    trainable_params: int,
    flops: float | None,
    checkpoint_path: Path,
    prediction_path: Path,
    figure_path: Path,
    metrics_file: Path,
    summary_text_path: Path,
    summary_json_path: Path,
) -> str:
    separator = "=" * 60
    divider = "-" * 60
    report_lines = [
        separator,
        "                DINSNet Training Completed",
        separator,
        "",
        f"Experiment ID        : {exp_id}",
        f"Total Images         : {total_images}",
        f"Train Images         : {train_images}",
        f"Validation Images    : {val_images}",
        f"Test Images          : {test_images}",
        "",
        f"Image Resolution     : {image_h} x {image_w}",
        f"Batch Size           : {batch_size}",
        f"Epochs Completed     : {completed_epochs} / {max_epochs}",
        f"Early Stopped        : {'YES' if early_stopped else 'NO'}",
        f"Training Time        : {_format_duration(training_seconds)}",
        f"Device               : {_format_device(device)}",
        "",
        divider,
        "BEST VALIDATION PERFORMANCE",
        divider,
        f"Best Epoch           : {best_epoch if best_epoch is not None else 'N/A'}",
        f"Best Dice Score      : {_format_metric(best_metrics, 'dice')}",
        f"Best IoU             : {_format_metric(best_metrics, 'iou')}",
        f"Precision            : {_format_metric(best_metrics, 'precision')}",
        f"Recall               : {_format_metric(best_metrics, 'recall')}",
        "",
        divider,
        "FINAL TEST PERFORMANCE",
        divider,
        f"Test Loss            : {_format_metric(test_metrics, 'loss')}",
        f"Test Dice            : {_format_metric(test_metrics, 'dice')}",
        f"Test IoU             : {_format_metric(test_metrics, 'iou')}",
        f"Test Precision       : {_format_metric(test_metrics, 'precision')}",
        f"Test Recall          : {_format_metric(test_metrics, 'recall')}",
        "",
        divider,
        "MODEL COMPLEXITY",
        divider,
        f"Total Parameters     : {_format_params(total_params)}",
        f"Trainable Params     : {_format_params(trainable_params)}",
        f"FLOPs                : {_format_flops(flops)}",
        "",
    ]

    if baseline_metrics:
        test_dice = float(test_metrics.get("dice", 0.0))
        base_dice = float(baseline_metrics.get("dice", 0.0))
        test_iou = float(test_metrics.get("iou", 0.0))
        base_iou = float(baseline_metrics.get("iou", 0.0))
        dice_gap = test_dice - base_dice
        iou_gap = test_iou - base_iou
        report_lines.extend(
            [
                divider,
                "BASELINE COMPARISON (Simple U-Net)",
                divider,
                f"Baseline Dice        : {_format_metric(baseline_metrics, 'dice')}",
                f"Baseline IoU         : {_format_metric(baseline_metrics, 'iou')}",
                f"Dice Gain            : {dice_gap:+.4f}",
                f"IoU Gain             : {iou_gap:+.4f}",
                "",
            ]
        )

    report_lines.extend(
        [
            divider,
            "OUTPUT LOCATION",
            divider,
            f"Best Checkpoint      : {_relative_path(checkpoint_path)}",
            f"Predictions          : {_relative_path(prediction_path)}",
            f"Figures              : {_relative_path(figure_path)}",
            f"Metrics CSV          : {_relative_path(metrics_file)}",
            f"Summary Text         : {_relative_path(summary_text_path)}",
            f"Summary JSON         : {_relative_path(summary_json_path)}",
            "",
            separator,
            "Training finished successfully.",
            separator,
        ]
    )
    return "\n".join(report_lines)


def _build_inference_report(
    exp_id: str,
    split: str,
    checkpoint_path: Path,
    metrics: Dict[str, float],
    prediction_path: Path,
    report_json_path: Path,
    report_text_path: Path,
) -> str:
    separator = "=" * 60
    divider = "-" * 60
    return "\n".join(
        [
            separator,
            "                DINSNet Inference Completed",
            separator,
            "",
            f"Experiment ID        : {exp_id}",
            f"Split                : {split}",
            f"Checkpoint           : {_relative_path(checkpoint_path)}",
            "",
            divider,
            "INFERENCE METRICS",
            divider,
            f"Loss                 : {_format_metric(metrics, 'loss')}",
            f"Dice                 : {_format_metric(metrics, 'dice')}",
            f"IoU                  : {_format_metric(metrics, 'iou')}",
            f"Precision            : {_format_metric(metrics, 'precision')}",
            f"Recall               : {_format_metric(metrics, 'recall')}",
            "",
            divider,
            "OUTPUT LOCATION",
            divider,
            f"Predictions          : {_relative_path(prediction_path)}",
            f"Summary Text         : {_relative_path(report_text_path)}",
            f"Summary JSON         : {_relative_path(report_json_path)}",
            "",
            separator,
            "Inference finished successfully.",
            separator,
        ]
    )


def run() -> None:
    """Execute the configured training or inference pipeline."""
    warnings.filterwarnings(
        "ignore",
        message="Error fetching version info",
        module="albumentations.check_version",
    )
    args = parse_args()
    config = load_config(args.config)
    cli_overrides = _apply_cli_overrides(config, args)

    # 1) Prepare config + device.
    if args.mode == "inference":
        config["runtime"]["inference_only"] = True
    if args.checkpoint and args.mode == "train":
        config["training"]["resume_checkpoint"] = args.checkpoint
    validate_config(config)

    set_random_seed(
        seed=int(config["project"]["seed"]),
        deterministic=bool(config["project"]["deterministic"]),
    )
    torch.backends.cudnn.benchmark = bool(config["runtime"]["cudnn_benchmark"])
    device = select_device(config["runtime"])

    exp_paths = create_experiment_paths(
        output_root=config["project"]["output_root"],
        prefix=config["project"]["experiment_prefix"],
    )
    logger = setup_logger(
        exp_paths.logs,
        enable_console=bool(config.get("runtime", {}).get("enable_console_logs", False)),
    )
    copied_cfg = copy_config_to_experiment(args.config, exp_paths.root)
    effective_cfg_path = exp_paths.root / "config.effective.json"
    with effective_cfg_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    if cli_overrides:
        with (exp_paths.metrics / "cli_overrides.json").open("w", encoding="utf-8") as handle:
            json.dump(cli_overrides, handle, indent=2)

    logger.info("Experiment directory: %s", exp_paths.root)
    logger.info("Using device: %s", device)
    logger.info("Saved run config to: %s", copied_cfg)
    logger.info("Saved effective config to: %s", effective_cfg_path)
    if cli_overrides:
        logger.info("Applied CLI overrides: %s", cli_overrides)

    # 2) Build data loaders and record split metadata.
    dataloaders, split = build_dataloaders(
        config=config,
        seed=int(config["project"]["seed"]),
        split_output_dir=exp_paths.metrics / "splits",
    )
    split_stats = split_statistics(split)
    with (exp_paths.metrics / "data_split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_stats, handle, indent=2)
    logger.info("Data split stats: %s", split_stats)
    env_metadata_path = export_run_metadata(
        output_dir=exp_paths.metrics,
        config=config,
        config_path=args.config,
        cli_overrides=cli_overrides,
        split_manifest_path=exp_paths.metrics / "splits" / "split_manifest.json",
        dataset_manifest_path=exp_paths.metrics / "dataset_manifest.json",
    )
    logger.info("Saved run environment metadata to: %s", env_metadata_path)

    # 3) Build model + trainer.
    model = DINSNet(model_cfg=config["model"]).to(device)
    image_h, image_w = (int(v) for v in config["data"]["image_size"])
    input_shape = (int(config["model"]["in_channels"]), image_h, image_w)
    trainable_param_count = count_parameters(model)
    total_param_count = sum(param.numel() for param in model.parameters())
    flops = calculate_flops(model=model, input_shape=input_shape, device=device)
    save_model_profile(
        path=exp_paths.metrics / "model_profile.json",
        param_count=trainable_param_count,
        flops=flops,
        input_shape=input_shape,
    )
    logger.info("Trainable parameters: %d", trainable_param_count)
    if flops is not None:
        logger.info("FLOPs (single forward pass): %.2f", flops)
    else:
        logger.warning("FLOPs could not be calculated. Install 'thop' to enable profiling.")

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        exp_paths=exp_paths,
        logger=logger,
    )
    baseline_trainer: Trainer | None = None
    try:
        inference_only = bool(config["runtime"]["inference_only"]) or args.mode == "inference"
        if inference_only:
            checkpoint_path = args.checkpoint or str(config["inference"]["checkpoint"]).strip()
            if not checkpoint_path:
                raise ValueError("Inference mode requires --checkpoint or inference.checkpoint in config.")
            inference_split = args.inference_split if args.mode == "inference" else str(config["inference"]["split"])
            metrics = trainer.inference(
                checkpoint_path=checkpoint_path,
                split=inference_split,
                save_predictions=bool(config["inference"]["save_predictions"]),
            )
            logger.info("Inference complete: %s", metrics)
            inference_summary_json = exp_paths.metrics / "inference_summary.json"
            inference_summary_txt = exp_paths.metrics / "inference_summary.txt"
            inference_payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": exp_paths.root.name,
                "mode": "inference",
                "split": inference_split,
                "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
                "metrics": _serialize_metrics(metrics),
                "paths": {
                    "experiment_root": str(exp_paths.root),
                    "predictions": str(exp_paths.predictions / inference_split),
                    "inference_summary_json": str(inference_summary_json),
                    "inference_summary_text": str(inference_summary_txt),
                },
            }
            with inference_summary_json.open("w", encoding="utf-8") as handle:
                json.dump(inference_payload, handle, indent=2)
            inference_report = _build_inference_report(
                exp_id=exp_paths.root.name,
                split=inference_split,
                checkpoint_path=Path(checkpoint_path),
                metrics=metrics,
                prediction_path=exp_paths.predictions / inference_split,
                report_json_path=inference_summary_json,
                report_text_path=inference_summary_txt,
            )
            with inference_summary_txt.open("w", encoding="utf-8") as handle:
                handle.write(inference_report + "\n")
            print("\n" + inference_report)
            return

        training_start = time.perf_counter()
        best_checkpoint = trainer.fit()
        _ = trainer.evaluate(
            split="val",
            checkpoint_path=best_checkpoint,
            save_predictions=True,
            allow_test=False,
        )
        test_metrics: Dict[str, float] = {}
        if args.evaluate_test:
            logger.warning(
                "Test evaluation enabled via --evaluate-test. "
                "Ensure you are not selecting/tuning models based on test results."
            )
            test_metrics = trainer.evaluate(
                split="test",
                checkpoint_path=best_checkpoint,
                save_predictions=bool(config["inference"]["save_predictions"]),
                allow_test=True,
            )
            logger.info("Final test metrics (best checkpoint): %s", test_metrics)
        else:
            logger.info("Skipping test evaluation. Use --evaluate-test to run on the test split.")
        training_seconds = time.perf_counter() - training_start

        comparison_cfg = config.get("comparison", {})
        baseline_metrics: Dict[str, float] | None = None
        run_baseline = bool(comparison_cfg.get("run_baseline", True)) and not args.disable_baseline
        if run_baseline and not args.evaluate_test:
            logger.warning(
                "Baseline comparison is disabled unless --evaluate-test is provided, "
                "to avoid implicit test-set usage."
            )
            run_baseline = False

        if run_baseline:
            logger.info("Running baseline comparison model (Simple U-Net).")
            baseline_cfg = copy.deepcopy(config)
            baseline_cfg["training"]["resume_checkpoint"] = ""
            set_random_seed(
                seed=int(baseline_cfg["project"]["seed"]),
                deterministic=bool(baseline_cfg["project"]["deterministic"]),
            )

            baseline_paths = create_child_experiment_paths(exp_paths.root, "baseline_unet")
            logger.info("Baseline experiment directory: %s", baseline_paths.root)
            baseline_dataloaders, _ = build_dataloaders(
                config=baseline_cfg,
                seed=int(baseline_cfg["project"]["seed"]),
                split_output_dir=baseline_paths.metrics / "splits",
            )

            baseline_model_cfg = comparison_cfg.get("baseline", {})
            norm_cfg = baseline_model_cfg.get("normalization", {})
            baseline_norm = str(norm_cfg.get("type", baseline_model_cfg.get("norm", "auto"))).lower()
            if baseline_norm in {"groupnorm", "group"}:
                baseline_norm = "group"
            if baseline_norm in {"batchnorm", "batch"}:
                baseline_norm = "batch"
            if baseline_norm == "auto":
                batch_size = int(config["data"]["loader"]["batch_size"])
                baseline_norm = "group" if batch_size < 8 else "batch"
            baseline_norm_groups = int(norm_cfg.get("groups", baseline_model_cfg.get("norm_groups", 8)))
            baseline_model = SimpleUNet(
                in_channels=int(config["model"]["in_channels"]),
                num_classes=int(config["model"]["num_classes"]),
                base_channels=int(baseline_model_cfg.get("base_channels", config["model"]["base_channels"])),
                depth=int(baseline_model_cfg.get("depth", 4)),
                norm_type=baseline_norm,
                norm_groups=baseline_norm_groups,
            ).to(device)
            baseline_param_count = count_parameters(baseline_model)
            baseline_flops = calculate_flops(model=baseline_model, input_shape=input_shape, device=device)
            save_model_profile(
                path=baseline_paths.metrics / "model_profile.json",
                param_count=baseline_param_count,
                flops=baseline_flops,
                input_shape=input_shape,
            )

            baseline_trainer = Trainer(
                model=baseline_model,
                config=baseline_cfg,
                device=device,
                dataloaders=baseline_dataloaders,
                exp_paths=baseline_paths,
                logger=logger,
            )
            baseline_best_checkpoint = baseline_trainer.fit()
            baseline_metrics = baseline_trainer.evaluate(
                split="test",
                checkpoint_path=baseline_best_checkpoint,
                save_predictions=False,
                allow_test=True,
            )
            logger.info("Baseline test metrics: %s", baseline_metrics)

            with (exp_paths.metrics / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dinsnet": test_metrics,
                        "baseline_unet": baseline_metrics,
                    },
                    handle,
                    indent=2,
                )

        try:
            from trainer.visualization import generate_publication_figures
        except ImportError as exc:
            raise RuntimeError(
                "Figure generation requires matplotlib. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        generate_publication_figures(
            config=config,
            exp_paths=exp_paths,
            split_samples=split,
            device=device,
            model_builder=lambda: DINSNet(model_cfg=config["model"]),
            checkpoint_path=Path(best_checkpoint),
            dinsnet_test_metrics=test_metrics,
            baseline_test_metrics=baseline_metrics,
            logger=logger,
        )

        best_epoch = trainer.best_epoch
        best_metrics = dict(trainer.best_metrics)
        if not best_metrics:
            hist_epoch, hist_metrics = _best_metrics_from_history(exp_paths.metrics / "training_history.csv")
            if hist_metrics:
                best_epoch = hist_epoch
                best_metrics = hist_metrics

        split_counts = split_statistics(split)
        train_count = int(split_counts["train"]["num_samples"])
        val_count = int(split_counts["val"]["num_samples"])
        test_count = int(split_counts["test"]["num_samples"])
        total_images = train_count + val_count + test_count

        metrics_csv_path = exp_paths.metrics / "evaluation_metrics.csv"
        if not metrics_csv_path.exists():
            metrics_csv_path = exp_paths.metrics / "training_history.csv"

        summary_text_path = exp_paths.metrics / "run_summary.txt"
        summary_json_path = exp_paths.metrics / "run_summary.json"
        report = _build_summary_report(
            exp_id=exp_paths.root.name,
            total_images=total_images,
            train_images=train_count,
            val_images=val_count,
            test_images=test_count,
            image_h=image_h,
            image_w=image_w,
            batch_size=int(config["data"]["loader"]["batch_size"]),
            completed_epochs=int(trainer.completed_epochs),
            max_epochs=int(config["training"]["epochs"]),
            early_stopped=bool(trainer.early_stopped),
            training_seconds=training_seconds,
            device=device,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            test_metrics=test_metrics,
            baseline_metrics=baseline_metrics,
            total_params=total_param_count,
            trainable_params=trainable_param_count,
            flops=flops,
            checkpoint_path=Path(best_checkpoint),
            prediction_path=exp_paths.predictions,
            figure_path=exp_paths.figures,
            metrics_file=metrics_csv_path,
            summary_text_path=summary_text_path,
            summary_json_path=summary_json_path,
        )
        with summary_text_path.open("w", encoding="utf-8") as handle:
            handle.write(report + "\n")

        summary_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": exp_paths.root.name,
            "mode": "train",
            "dataset": {
                "total_images": total_images,
                "train_images": train_count,
                "val_images": val_count,
                "test_images": test_count,
                "split_stats": split_counts,
            },
            "training": {
                "completed_epochs": int(trainer.completed_epochs),
                "max_epochs": int(config["training"]["epochs"]),
                "early_stopped": bool(trainer.early_stopped),
                "duration_seconds": float(training_seconds),
            },
            "best_validation": {
                "epoch": best_epoch,
                "metrics": _serialize_metrics(best_metrics),
            },
            "test_metrics": _serialize_metrics(test_metrics),
            "baseline_test_metrics": _serialize_metrics(baseline_metrics),
            "metric_settings": {
                "mode": str(config["training"]["metrics"].get("mode", "raw")),
                "threshold": float(config["training"]["metrics"]["threshold"]),
            },
            "model_profile": {
                "total_parameters": int(total_param_count),
                "trainable_parameters": int(trainable_param_count),
                "flops": float(flops) if flops is not None else None,
                "input_shape": [int(config["model"]["in_channels"]), image_h, image_w],
            },
            "cli_overrides": cli_overrides,
            "paths": {
                "experiment_root": str(exp_paths.root),
                "checkpoint": str(Path(best_checkpoint).expanduser().resolve()),
                "predictions": str(exp_paths.predictions),
                "figures": str(exp_paths.figures),
                "metrics_csv": str(metrics_csv_path.expanduser().resolve()),
                "summary_text": str(summary_text_path.expanduser().resolve()),
                "summary_json": str(summary_json_path.expanduser().resolve()),
                "config_effective": str(effective_cfg_path.expanduser().resolve()),
                "environment": str(env_metadata_path.expanduser().resolve()),
                "split_manifest": str((exp_paths.metrics / "splits" / "split_manifest.json").expanduser().resolve()),
                "dataset_manifest": str((exp_paths.metrics / "dataset_manifest.json").expanduser().resolve()),
            },
        }
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)

        print("\n" + report)
    finally:
        if baseline_trainer is not None:
            baseline_trainer.close()
        trainer.close()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        raise RuntimeError(f"DINSNet execution failed: {exc}") from exc
