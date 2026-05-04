"""Publication-quality figure generation for DINSNet experiments."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data.dataset import SampleRecord
from utils import ExperimentPaths, sanitize_model_state_dict


def _configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
        }
    )


def _figure_dirs(root: Path) -> Dict[str, Path]:
    output = {
        "qualitative": root / "qualitative_results",
        "progression": root / "training_progress",
        "curves": root / "metric_curves",
        "comparison": root / "comparison_plots",
    }
    for directory in output.values():
        directory.mkdir(parents=True, exist_ok=True)
    return output


def _sample_stem(sample: SampleRecord) -> str:
    return f"{sample.domain}_{sample.sample_id}"


def _sort_samples(samples: Iterable[SampleRecord]) -> List[SampleRecord]:
    return sorted(samples, key=lambda item: (item.domain, item.sample_id))


def _read_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Unable to read mask: {path}")
    return (mask.astype(np.float32) >= 127.5).astype(np.uint8)


def _to_model_tensor(
    image_rgb: np.ndarray,
    image_size: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    height, width = image_size
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _predict_mask(
    model: nn.Module,
    sample: SampleRecord,
    image_size: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_rgb = _read_rgb_image(sample.image_path)
    input_tensor = _to_model_tensor(
        image_rgb=image_rgb,
        image_size=image_size,
        mean=mean,
        std=std,
        device=device,
    )
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
    prob_map = cv2.resize(probs, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred_mask = (prob_map >= threshold).astype(np.uint8)
    return image_rgb, pred_mask, prob_map


def _overlay_prediction(image_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    overlay = image.copy()
    color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[pred_mask == 1] = (1.0 - alpha) * overlay[pred_mask == 1] + alpha * color
    return overlay


def _error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)

    err = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    err[tp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # green
    err[fp] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red
    err[fn] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # blue
    return err


def _binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred_mask.astype(np.float32).reshape(-1)
    gt = gt_mask.astype(np.float32).reshape(-1)
    tp = float((pred * gt).sum())
    fp = float((pred * (1.0 - gt)).sum())
    fn = float(((1.0 - pred) * gt).sum())
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def _load_model_checkpoint(
    model_builder: Callable[[], nn.Module],
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Optional[int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = sanitize_model_state_dict(checkpoint.get("model_state_dict", checkpoint))
    epoch = checkpoint.get("epoch", None)

    model = model_builder().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    epoch_idx = int(epoch) + 1 if isinstance(epoch, int) else None
    return model, epoch_idx


def _resolve_progression_checkpoints(checkpoint_dir: Path) -> Dict[str, Path]:
    epoch_paths = sorted(checkpoint_dir.glob("epoch_*.pth"))
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    fallback = best_path if best_path.exists() else last_path
    if not fallback.exists():
        raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")

    if not epoch_paths:
        return {
            "Epoch Early": fallback,
            "Epoch Mid": fallback,
            "Epoch Late": fallback,
            "Final": fallback,
        }

    return {
        "Epoch Early": epoch_paths[0],
        "Epoch Mid": epoch_paths[len(epoch_paths) // 2],
        "Epoch Late": epoch_paths[-1],
        "Final": best_path if best_path.exists() else fallback,
    }


def _save_qualitative_figures(
    model: nn.Module,
    samples: Sequence[SampleRecord],
    save_dir: Path,
    image_size: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    for sample in samples:
        image_rgb, pred_mask, _ = _predict_mask(
            model=model,
            sample=sample,
            image_size=image_size,
            mean=mean,
            std=std,
            threshold=threshold,
            device=device,
        )
        gt_mask = _read_binary_mask(sample.mask_path)
        overlay = _overlay_prediction(image_rgb=image_rgb, pred_mask=pred_mask)
        err = _error_map(pred_mask=pred_mask, gt_mask=gt_mask)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")
        axes[4].imshow(err)
        axes[4].set_title("Error Map")
        for ax in axes:
            ax.axis("off")

        fig.suptitle(_sample_stem(sample), y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / f"{_sample_stem(sample)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved qualitative figures to: %s", save_dir)


def _save_progression_figures(
    model_builder: Callable[[], nn.Module],
    samples: Sequence[SampleRecord],
    save_dir: Path,
    checkpoint_dir: Path,
    image_size: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    plan = _resolve_progression_checkpoints(checkpoint_dir=checkpoint_dir)
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    title_with_epoch: Dict[str, str] = {}

    for title, checkpoint_path in plan.items():
        model, epoch_idx = _load_model_checkpoint(
            model_builder=model_builder,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        title_with_epoch[title] = title if epoch_idx is None else f"{title}\n(E{epoch_idx:03d})"
        predictions[title] = {}
        for sample in samples:
            _, pred_mask, _ = _predict_mask(
                model=model,
                sample=sample,
                image_size=image_size,
                mean=mean,
                std=std,
                threshold=threshold,
                device=device,
            )
            predictions[title][_sample_stem(sample)] = pred_mask

    for sample in samples:
        image_rgb = _read_rgb_image(sample.image_path)
        gt_mask = _read_binary_mask(sample.mask_path)
        stem = _sample_stem(sample)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for idx, title in enumerate(["Epoch Early", "Epoch Mid", "Epoch Late", "Final"], start=1):
            pred_mask = predictions[title][stem]
            dice = _binary_dice(pred_mask=pred_mask, gt_mask=gt_mask)
            axes[idx].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
            axes[idx].set_title(f"{title_with_epoch[title]}\nDice={dice:.3f}")
            axes[idx].axis("off")

        fig.suptitle(stem, y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved training progression figures to: %s", save_dir)


def _plot_metric_curves(training_history_csv: Path, save_dir: Path, logger: logging.Logger) -> None:
    if not training_history_csv.exists():
        logger.warning("Metric curve plotting skipped. File missing: %s", training_history_csv)
        return

    rows: List[Dict[str, str]] = []
    with training_history_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        logger.warning("Metric curve plotting skipped. Empty file: %s", training_history_csv)
        return

    epochs = np.array([int(row["epoch"]) for row in rows], dtype=np.int32)
    train_dice = np.array([float(row["train_dice"]) for row in rows], dtype=np.float32)
    val_dice = np.array([float(row["val_dice"]) for row in rows], dtype=np.float32)
    train_iou = np.array([float(row["train_iou"]) for row in rows], dtype=np.float32)
    val_iou = np.array([float(row["val_iou"]) for row in rows], dtype=np.float32)
    train_loss = np.array([float(row["train_loss"]) for row in rows], dtype=np.float32)
    val_loss = np.array([float(row["val_loss"]) for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, train_dice, marker="o", color="#1f77b4", label="Train")
    axes[0].plot(epochs, val_dice, marker="s", color="#d62728", label="Val")
    axes[0].set_title("Dice vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    axes[1].plot(epochs, train_iou, marker="o", color="#1f77b4", label="Train")
    axes[1].plot(epochs, val_iou, marker="s", color="#d62728", label="Val")
    axes[1].set_title("IoU vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(epochs, train_loss, marker="o", color="#1f77b4", label="Train")
    axes[2].plot(epochs, val_loss, marker="s", color="#d62728", label="Val")
    axes[2].set_title("Loss vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_dir / "training_metric_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric curves to: %s", save_dir)


def _plot_comparison(
    dinsnet_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    save_dir: Path,
    logger: logging.Logger,
) -> None:
    dins_iou = float(dinsnet_metrics["iou"])
    base_iou = float(baseline_metrics["iou"])
    dins_dice = float(dinsnet_metrics["dice"])
    base_dice = float(baseline_metrics["dice"])

    eps = 1e-7
    iou_improvement = 100.0 * (dins_iou - base_iou) / max(abs(base_iou), eps)
    dice_improvement = 100.0 * (dins_dice - base_dice) / max(abs(base_dice), eps)

    labels = ["IoU", "Dice"]
    baseline_vals = [base_iou, base_dice]
    dinsnet_vals = [dins_iou, dins_dice]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_base = ax.bar(x - width / 2, baseline_vals, width=width, label="Baseline U-Net", color="#9ea7b8")
    bars_dins = ax.bar(x + width / 2, dinsnet_vals, width=width, label="DINSNet", color="#1f77b4")

    for bars in [bars_base, bars_dins]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    improvement_text = f"IoU Improvement: {iou_improvement:+.2f}%\nDice Improvement: {dice_improvement:+.2f}%"
    ax.text(
        0.02,
        0.98,
        improvement_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#bdbdbd"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs DINSNet Segmentation Performance")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_dir / "baseline_vs_dinsnet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot to: %s", save_dir)


def generate_publication_figures(
    config: Dict[str, Any],
    exp_paths: ExperimentPaths,
    split_samples: Dict[str, List[SampleRecord]],
    device: torch.device,
    model_builder: Callable[[], nn.Module],
    checkpoint_path: Path,
    dinsnet_test_metrics: Dict[str, float],
    baseline_test_metrics: Optional[Dict[str, float]],
    logger: logging.Logger,
) -> None:
    """Generate publication-ready figures after model training."""
    vis_cfg = config.get("visualization", {})
    if not bool(vis_cfg.get("enabled", True)):
        logger.info("Figure generation is disabled via visualization.enabled=false.")
        return

    _configure_plot_style()
    dirs = _figure_dirs(exp_paths.figures)

    qual_split = str(vis_cfg.get("qualitative_split", "val"))
    all_samples = _sort_samples(split_samples.get(qual_split, []))
    if not all_samples:
        logger.warning("No samples found for qualitative split '%s'.", qual_split)
    max_qualitative = int(vis_cfg.get("max_qualitative_samples", 0))
    if max_qualitative > 0:
        all_samples = all_samples[:max_qualitative]

    image_size = (int(config["data"]["image_size"][0]), int(config["data"]["image_size"][1]))
    mean = np.array(config["data"]["preprocessing"]["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(config["data"]["preprocessing"]["std"], dtype=np.float32).reshape(1, 1, 3)
    threshold = float(config["training"]["metrics"]["threshold"])

    if all_samples:
        final_model, _ = _load_model_checkpoint(
            model_builder=model_builder,
            checkpoint_path=Path(checkpoint_path),
            device=device,
        )
        _save_qualitative_figures(
            model=final_model,
            samples=all_samples,
            save_dir=dirs["qualitative"],
            image_size=image_size,
            mean=mean,
            std=std,
            threshold=threshold,
            device=device,
            logger=logger,
        )

        progression_count = max(1, int(vis_cfg.get("progression_samples", 4)))
        progression_samples = all_samples[: min(progression_count, len(all_samples))]
        _save_progression_figures(
            model_builder=model_builder,
            samples=progression_samples,
            save_dir=dirs["progression"],
            checkpoint_dir=exp_paths.checkpoints,
            image_size=image_size,
            mean=mean,
            std=std,
            threshold=threshold,
            device=device,
            logger=logger,
        )

    _plot_metric_curves(
        training_history_csv=exp_paths.metrics / "training_history.csv",
        save_dir=dirs["curves"],
        logger=logger,
    )

    if baseline_test_metrics is not None:
        _plot_comparison(
            dinsnet_metrics=dinsnet_test_metrics,
            baseline_metrics=baseline_test_metrics,
            save_dir=dirs["comparison"],
            logger=logger,
        )
