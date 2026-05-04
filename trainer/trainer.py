"""Training and evaluation engine for DINSNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    ExperimentPaths,
    MetricTracker,
    batch_segmentation_metrics,
    load_checkpoint,
    sanitize_model_state_dict,
    save_checkpoint,
    write_csv_row,
)


class DiceBCELoss(nn.Module):
    """Combined Dice and BCEWithLogits loss for binary segmentation."""

    def __init__(self, dice_weight: float, bce_weight: float, smooth: float = 1e-6) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1.0 - dice.mean()
        total = self.dice_weight * dice_loss + self.bce_weight * bce
        return total, {"dice_loss": float(dice_loss.item()), "bce_loss": float(bce.item())}


class Trainer:
    """End-to-end trainer for DINSNet with checkpointing and inference support."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        dataloaders: Dict[str, DataLoader],
        exp_paths: ExperimentPaths,
        logger: logging.Logger,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.dataloaders = dataloaders
        self.exp_paths = exp_paths
        self.logger = logger

        self.train_cfg = config["training"]
        self.infer_cfg = config["inference"]
        self.runtime_cfg = config.get("runtime", {})
        self.metric_threshold = float(self.train_cfg["metrics"]["threshold"])
        self.metric_mode = str(self.train_cfg["metrics"].get("mode", "raw")).lower()
        if self.metric_mode not in {"raw", "postprocess"}:
            raise ValueError("training.metrics.mode must be one of: raw, postprocess.")
        if self.metric_mode == "postprocess":
            # Postprocess mode evaluates metrics after mask cleanup.
            self.logger.info(
                "Metric mode is set to postprocess. Reported metrics will use post-processed masks."
            )
        self.show_progress_bar = bool(self.runtime_cfg.get("show_progress_bar", False))
        preprocessing_cfg = config.get("data", {}).get("preprocessing", {})
        mean = preprocessing_cfg.get("mean", [0.485, 0.456, 0.406])
        std = preprocessing_cfg.get("std", [0.229, 0.224, 0.225])
        self.image_mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.pred_overlay_alpha = float(config.get("inference", {}).get("overlay_alpha", 0.45))
        self.pred_apply_postprocess = bool(config.get("inference", {}).get("postprocess_predictions", True))
        self.pred_min_component_area = int(config.get("inference", {}).get("postprocess_min_component_area", 64))

        loss_cfg = self.train_cfg["loss"]
        self.criterion = DiceBCELoss(
            dice_weight=float(loss_cfg["dice_weight"]),
            bce_weight=float(loss_cfg["bce_weight"]),
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scheduler_type = self.train_cfg["scheduler"]["type"]
        self.use_amp = bool(self.train_cfg["amp"]) and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_epochs = int(self.train_cfg["epochs"])
        self.grad_clip = float(self.train_cfg["gradient_clip_norm"])
        self.ckpt_interval = int(self.train_cfg["checkpoint_interval"])

        early_stop_cfg = self.train_cfg["early_stopping"]
        self.early_stop_enabled = bool(early_stop_cfg["enabled"])
        self.early_stop_patience = int(early_stop_cfg["patience"])
        self.early_stop_min_delta = float(early_stop_cfg["min_delta"])
        self.early_stop_counter = 0

        self.best_val_dice = float("-inf")
        self.best_epoch: Optional[int] = None
        self.best_metrics: Dict[str, float] = {}
        self.completed_epochs = 0
        self.early_stopped = False
        self.start_epoch = 0
        self.history_csv = self.exp_paths.metrics / "training_history.csv"
        self.eval_csv = self.exp_paths.metrics / "evaluation_metrics.csv"
        self.writer = SummaryWriter(log_dir=str(self.exp_paths.logs / "tensorboard"))

        resume_ckpt = str(self.train_cfg["resume_checkpoint"]).strip()
        if resume_ckpt:
            self.resume_from_checkpoint(resume_ckpt)

    def _build_optimizer(self) -> Adam:
        optim_cfg = self.train_cfg["optimizer"]
        return Adam(
            params=self.model.parameters(),
            lr=float(optim_cfg["lr"]),
            weight_decay=float(optim_cfg["weight_decay"]),
            betas=tuple(float(x) for x in optim_cfg["betas"]),
        )

    def _build_scheduler(self) -> Any:
        sched_cfg = self.train_cfg["scheduler"]
        sched_type = sched_cfg["type"]
        if sched_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_cfg["t_max"]),
                eta_min=float(sched_cfg["min_lr"]),
            )
        if sched_type == "step":
            return StepLR(
                self.optimizer,
                step_size=int(sched_cfg["step_size"]),
                gamma=float(sched_cfg["gamma"]),
            )
        return ReduceLROnPlateau(
            self.optimizer,
            mode=str(sched_cfg["mode"]),
            factor=float(sched_cfg["factor"]),
            patience=int(sched_cfg["patience"]),
            min_lr=float(sched_cfg["min_lr"]),
            verbose=True,
        )

    def _checkpoint_state(self, epoch: int) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": sanitize_model_state_dict(self.model.state_dict()),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": self.config,
        }

    def _save_epoch_checkpoints(self, epoch: int, is_best: bool) -> None:
        last_path = self.exp_paths.checkpoints / "last.pth"
        save_checkpoint(self._checkpoint_state(epoch), last_path)
        if (epoch + 1) % self.ckpt_interval == 0:
            epoch_path = self.exp_paths.checkpoints / f"epoch_{epoch + 1:03d}.pth"
            save_checkpoint(self._checkpoint_state(epoch), epoch_path)
        if is_best:
            best_path = self.exp_paths.checkpoints / "best.pth"
            save_checkpoint(self._checkpoint_state(epoch), best_path)

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training state from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        model_state = sanitize_model_state_dict(checkpoint["model_state_dict"])
        self.model.load_state_dict(model_state, strict=True)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_val_dice = float(checkpoint.get("best_val_dice", float("-inf")))
        self.logger.info(
            "Resumed checkpoint from %s (start_epoch=%d, best_val_dice=%.6f)",
            checkpoint_path,
            self.start_epoch,
            self.best_val_dice,
        )

    def fit(self) -> Path:
        """Run training and validation loops."""
        self.logger.info("Starting training for %d epochs", self.max_epochs)
        last_completed_epoch = self.start_epoch - 1
        self.early_stopped = False
        for epoch in range(self.start_epoch, self.max_epochs):
            train_metrics = self._run_epoch(
                loader=self.dataloaders["train"],
                training=True,
                split="train",
                epoch=epoch,
            )
            val_metrics = self._run_epoch(
                loader=self.dataloaders["val"],
                training=False,
                split="val",
                epoch=epoch,
            )

            if self.scheduler_type == "plateau":
                self.scheduler.step(val_metrics["dice"])
            else:
                self.scheduler.step()

            lr_value = float(self.optimizer.param_groups[0]["lr"])
            summary_row = {
                "epoch": epoch + 1,
                "lr": lr_value,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
            write_csv_row(self.history_csv, summary_row)
            self._log_to_tensorboard(epoch, lr_value, train_metrics, val_metrics)

            improved = val_metrics["dice"] > (self.best_val_dice + self.early_stop_min_delta)
            if improved:
                self.best_val_dice = val_metrics["dice"]
                self.best_epoch = epoch + 1
                self.best_metrics = dict(val_metrics)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self._save_epoch_checkpoints(epoch=epoch, is_best=improved)
            last_completed_epoch = epoch
            self.logger.info(
                "Epoch [%d/%d] lr=%.8f train_loss=%.5f val_loss=%.5f val_dice=%.5f best_val_dice=%.5f",
                epoch + 1,
                self.max_epochs,
                lr_value,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["dice"],
                self.best_val_dice,
            )

            if self.early_stop_enabled and self.early_stop_counter >= self.early_stop_patience:
                # Stop if validation Dice has not improved for N epochs.
                self.early_stopped = True
                self.logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch + 1,
                    self.early_stop_patience,
                )
                break

        self.completed_epochs = max(last_completed_epoch + 1, 0)
        self.writer.flush()
        best_path = self.exp_paths.checkpoints / "best.pth"
        if not best_path.exists():
            best_path = self.exp_paths.checkpoints / "last.pth"
        self.logger.info("Training complete. Best checkpoint: %s", best_path)
        return best_path

    def _log_to_tensorboard(
        self,
        epoch: int,
        learning_rate: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        self.writer.add_scalar("lr", learning_rate, epoch)
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        split: str,
        epoch: int,
        save_predictions: bool = False,
    ) -> Dict[str, float]:
        self.model.train(mode=training)
        tracker = MetricTracker()
        desc = f"{split.upper()} E{epoch + 1:03d}"
        progress = tqdm(loader, desc=desc, leave=False, disable=not self.show_progress_bar)

        if save_predictions:
            pred_dir = self.exp_paths.predictions / split
            pred_dir.mkdir(parents=True, exist_ok=True)
        else:
            pred_dir = None

        for batch_idx, batch in enumerate(progress):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            batch_size = images.size(0)

            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss, loss_parts = self.criterion(logits, masks)

                if training:
                    # Standard AMP-safe optimization step.
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            probs = torch.sigmoid(logits.detach())
            if self.metric_mode == "raw":
                metric_values = batch_segmentation_metrics(
                    probs=probs,
                    targets=masks.detach(),
                    threshold=self.metric_threshold,
                )
            else:
                # Compute metrics after postprocessing the predicted mask.
                probs_cpu = probs.detach().cpu()
                masks_cpu = masks.detach().cpu()
                pred_stack = []
                for idx in range(probs_cpu.size(0)):
                    pred_raw = ((probs_cpu[idx, 0] >= self.metric_threshold).numpy().astype(np.uint8) * 255)
                    pred_clean = self._postprocess_mask(pred_raw, force=True)
                    pred_stack.append((pred_clean > 127).astype(np.float32))
                preds_tensor = torch.from_numpy(np.stack(pred_stack, axis=0)).unsqueeze(1)
                metric_values = batch_segmentation_metrics(
                    probs=preds_tensor,
                    targets=masks_cpu,
                    threshold=0.5,
                )
            aggregate = {
                "loss": float(loss.item()),
                "dice": metric_values["dice"],
                "iou": metric_values["iou"],
                "precision": metric_values["precision"],
                "recall": metric_values["recall"],
                "dice_loss": loss_parts["dice_loss"],
                "bce_loss": loss_parts["bce_loss"],
            }
            tracker.update(aggregate, n=batch_size)
            progress.set_postfix({"loss": f"{aggregate['loss']:.4f}", "dice": f"{aggregate['dice']:.4f}"})

            if pred_dir is not None:
                self._save_batch_predictions(
                    probs=probs,
                    batch=batch,
                    pred_dir=pred_dir,
                    split=split,
                    batch_idx=batch_idx,
                )

        return tracker.averages()

    def _safe_token(self, token: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(token))
        cleaned = cleaned.strip("_")
        return cleaned or "sample"

    def _to_bgr_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        image = (image * self.image_std + self.image_mean).clip(0.0, 1.0)
        image_rgb = (image * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _mask_tensor_to_uint8(mask_tensor: torch.Tensor) -> np.ndarray:
        mask = mask_tensor.detach().cpu().numpy().squeeze(0)
        return ((mask >= 0.5).astype(np.uint8) * 255)

    def _postprocess_mask(self, mask_uint8: np.ndarray, force: bool = False) -> np.ndarray:
        if not force and not self.pred_apply_postprocess:
            return mask_uint8

        binary = (mask_uint8 > 127).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels <= 1:
            return (closed * 255).astype(np.uint8)

        filtered = np.zeros_like(closed, dtype=np.uint8)
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area >= self.pred_min_component_area:
                filtered[labels == label_idx] = 1

        if filtered.sum() == 0:
            filtered = closed
        return (filtered * 255).astype(np.uint8)

    def _overlay_mask(self, image_bgr: np.ndarray, mask_uint8: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
        overlay = image_bgr.copy()
        mask = mask_uint8 > 127
        if mask.any():
            color = np.array(color_bgr, dtype=np.float32)
            blended = (
                (1.0 - self.pred_overlay_alpha) * overlay[mask].astype(np.float32)
                + self.pred_overlay_alpha * color
            )
            overlay[mask] = blended.clip(0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, color_bgr, 2)
        return overlay

    def _comparison_overlay(self, image_bgr: np.ndarray, gt_uint8: np.ndarray, pred_uint8: np.ndarray) -> np.ndarray:
        overlay = image_bgr.copy().astype(np.float32)
        gt = gt_uint8 > 127
        pred = pred_uint8 > 127

        tp = gt & pred
        fp = (~gt) & pred
        fn = gt & (~pred)

        tp_color = np.array([60, 200, 60], dtype=np.float32)    # green
        fp_color = np.array([35, 35, 240], dtype=np.float32)    # red
        fn_color = np.array([0, 190, 255], dtype=np.float32)    # amber

        alpha = 0.5
        for mask, color in [(tp, tp_color), (fp, fp_color), (fn, fn_color)]:
            if mask.any():
                overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

        output = overlay.clip(0, 255).astype(np.uint8)
        gt_contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if gt_contours:
            cv2.drawContours(output, gt_contours, -1, (0, 255, 255), 2)  # yellow
        if pred_contours:
            cv2.drawContours(output, pred_contours, -1, (255, 180, 0), 2)  # cyan
        return output

    @staticmethod
    def _tile_with_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        header = 34
        tile = np.full((height + header, width, 3), 248, dtype=np.uint8)
        tile[header:, :, :] = image_bgr
        cv2.putText(
            tile,
            title,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        return tile

    def _build_panel(
        self,
        image_bgr: np.ndarray,
        gt_uint8: np.ndarray,
        prob_overlay: np.ndarray,
        pred_uint8: np.ndarray,
        compare_overlay: np.ndarray,
    ) -> np.ndarray:
        gt_bgr = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR)

        tiles = [
            self._tile_with_title(image_bgr, "Input Image"),
            self._tile_with_title(gt_bgr, "Ground Truth"),
            self._tile_with_title(prob_overlay, "Probability Heatmap"),
            self._tile_with_title(pred_bgr, "Prediction"),
            self._tile_with_title(compare_overlay, "Overlay (TP/FP/FN)"),
        ]
        return np.hstack(tiles)

    def _save_batch_predictions(
        self,
        probs: torch.Tensor,
        batch: Dict[str, Any],
        pred_dir: Path,
        split: str,
        batch_idx: int,
    ) -> None:
        probs_np = probs.squeeze(1).detach().cpu().numpy()
        images = batch["image"]
        masks = batch["mask"]

        sample_ids = batch["sample_id"]
        domains = batch["domain"]
        raw_dir = pred_dir / "raw"
        visual_dir = pred_dir / "visual"
        panel_dir = pred_dir / "panels"
        for directory in [raw_dir, visual_dir, panel_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        for idx in range(len(sample_ids)):
            safe_domain = self._safe_token(domains[idx])
            safe_sample = self._safe_token(sample_ids[idx])
            stem = f"{split}_{batch_idx:04d}_{idx:02d}_{safe_domain}_{safe_sample}"

            image_bgr = self._to_bgr_image(images[idx])
            gt_uint8 = self._mask_tensor_to_uint8(masks[idx])
            prob_img = (probs_np[idx] * 255.0).clip(0, 255).astype(np.uint8)
            pred_raw = ((probs_np[idx] >= self.metric_threshold).astype(np.uint8) * 255)
            pred_clean = self._postprocess_mask(pred_raw)

            prob_heatmap = cv2.applyColorMap(prob_img, cv2.COLORMAP_TURBO)
            prob_overlay = cv2.addWeighted(image_bgr, 0.4, prob_heatmap, 0.6, 0.0)
            pred_overlay = self._overlay_mask(image_bgr, pred_clean, (0, 120, 255))
            compare_overlay = self._comparison_overlay(image_bgr, gt_uint8, pred_clean)
            panel = self._build_panel(
                image_bgr=image_bgr,
                gt_uint8=gt_uint8,
                prob_overlay=prob_overlay,
                pred_uint8=pred_clean,
                compare_overlay=compare_overlay,
            )

            cv2.imwrite(str(raw_dir / f"{stem}_image.png"), image_bgr)
            cv2.imwrite(str(raw_dir / f"{stem}_gt.png"), gt_uint8)
            cv2.imwrite(str(raw_dir / f"{stem}_prob.png"), prob_img)
            cv2.imwrite(str(raw_dir / f"{stem}_pred_raw.png"), pred_raw)
            cv2.imwrite(str(raw_dir / f"{stem}_pred_clean.png"), pred_clean)

            cv2.imwrite(str(visual_dir / f"{stem}_heatmap.png"), prob_heatmap)
            cv2.imwrite(str(visual_dir / f"{stem}_prob_overlay.png"), prob_overlay)
            cv2.imwrite(str(visual_dir / f"{stem}_pred_overlay.png"), pred_overlay)
            cv2.imwrite(str(visual_dir / f"{stem}_compare_overlay.png"), compare_overlay)

            cv2.imwrite(str(panel_dir / f"{stem}_panel.png"), panel)

    def evaluate(
        self,
        split: str = "test",
        checkpoint_path: Optional[str | Path] = None,
        save_predictions: bool = True,
        allow_test: bool = False,
    ) -> Dict[str, float]:
        """Evaluate model on a given split and optionally save predictions."""
        if split not in self.dataloaders:
            raise ValueError(f"Unknown split: {split}")
        if split == "test" and not allow_test:
            raise RuntimeError(
                "Test evaluation is disabled by default. "
                "Pass allow_test=True to evaluate on the test split."
            )
        if split == "test" and allow_test:
            self.logger.warning(
                "Test evaluation requested. Ensure model selection is finalized and "
                "do not tune hyperparameters based on test metrics."
            )
        self.logger.info("Evaluation metric mode: %s", self.metric_mode)

        if checkpoint_path:
            checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
            model_state = sanitize_model_state_dict(checkpoint["model_state_dict"])
            self.model.load_state_dict(model_state, strict=True)
            self.logger.info("Loaded checkpoint for evaluation: %s", checkpoint_path)

        metrics = self._run_epoch(
            loader=self.dataloaders[split],
            training=False,
            split=split,
            epoch=0,
            save_predictions=save_predictions,
        )
        row = {"split": split, "metric_mode": self.metric_mode, **metrics}
        write_csv_row(self.eval_csv, row)
        self.logger.info(
            "Evaluation (%s): loss=%.5f dice=%.5f iou=%.5f precision=%.5f recall=%.5f",
            split,
            metrics["loss"],
            metrics["dice"],
            metrics["iou"],
            metrics["precision"],
            metrics["recall"],
        )
        return metrics

    def inference(self, checkpoint_path: str | Path, split: str, save_predictions: bool) -> Dict[str, float]:
        """Inference-only wrapper with metric calculation and prediction export."""
        self.logger.info("Running inference on split='%s' using checkpoint=%s", split, checkpoint_path)
        return self.evaluate(
            split=split,
            checkpoint_path=checkpoint_path,
            save_predictions=save_predictions,
            allow_test=True,
        )

    def close(self) -> None:
        """Close logging resources."""
        self.writer.flush()
        self.writer.close()
