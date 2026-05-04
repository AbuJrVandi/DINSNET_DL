"""Utility functions for DINSNet training and inference pipelines."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import logging
import platform
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from importlib import metadata as importlib_metadata
import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ExperimentPaths:
    """Container for per-run experiment output paths."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    predictions: Path
    figures: Path


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping.")
    return config


def _require_keys(container: Dict[str, Any], keys: Iterable[str], prefix: str) -> None:
    missing = [key for key in keys if key not in container]
    if missing:
        raise ConfigError(f"Missing required keys in '{prefix}': {missing}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required config sections and value ranges."""
    _require_keys(config, ["project", "data", "model", "training", "runtime", "inference"], "root")

    project = config["project"]
    _require_keys(project, ["output_root", "experiment_prefix", "seed", "deterministic"], "project")

    data = config["data"]
    _require_keys(
        data,
        [
            "root_dir",
            "image_dir_name",
            "mask_dir_name",
            "image_extensions",
            "mask_extensions",
            "image_size",
            "split",
            "loader",
            "augmentation",
            "preprocessing",
            "domain_generalization",
        ],
        "data",
    )

    split = data["split"]
    _require_keys(split, ["train", "val", "test", "stratify_by_domain"], "data.split")
    split_sum = float(split["train"]) + float(split["val"]) + float(split["test"])
    if not np.isclose(split_sum, 1.0, atol=1e-6):
        raise ConfigError(f"data.split ratios must sum to 1.0, got {split_sum:.6f}")
    for key in ["train", "val", "test"]:
        if float(split[key]) <= 0.0:
            raise ConfigError(f"data.split.{key} must be > 0.")
    if "allow_image_level_split" in split and not isinstance(split["allow_image_level_split"], bool):
        raise ConfigError("data.split.allow_image_level_split must be a boolean when provided.")

    image_size = data["image_size"]
    if not (isinstance(image_size, list) and len(image_size) == 2 and all(int(v) > 0 for v in image_size)):
        raise ConfigError("data.image_size must be [height, width] with positive integers.")

    preprocessing = data["preprocessing"]
    _require_keys(preprocessing, ["mean", "std"], "data.preprocessing")
    if len(preprocessing["mean"]) != 3 or len(preprocessing["std"]) != 3:
        raise ConfigError("data.preprocessing.mean/std must have three values each.")
    resize_mode = str(preprocessing.get("resize_mode", "pad")).lower()
    if resize_mode not in {"stretch", "pad", "crop"}:
        raise ConfigError("data.preprocessing.resize_mode must be one of: stretch, pad, crop.")
    pad_value = preprocessing.get("pad_value", 0.0)
    try:
        pad_value_float = float(pad_value)
    except Exception as exc:
        raise ConfigError("data.preprocessing.pad_value must be numeric.") from exc
    if pad_value_float < 0.0:
        raise ConfigError("data.preprocessing.pad_value must be >= 0.")

    domain_cfg = data["domain_generalization"]
    _require_keys(domain_cfg, ["enabled", "mode", "held_out_domains"], "data.domain_generalization")
    if domain_cfg["mode"] not in {"random", "leave_one_domain_out"}:
        raise ConfigError("data.domain_generalization.mode must be 'random' or 'leave_one_domain_out'.")

    split_files = data.get("split_files")
    if split_files is not None:
        if not isinstance(split_files, dict):
            raise ConfigError("data.split_files must be a mapping with keys: train/val/test.")
        allowed = {"train", "val", "test"}
        unknown = set(split_files.keys()) - allowed
        if unknown:
            raise ConfigError(f"data.split_files contains unknown keys: {sorted(unknown)}")

    grouping = data.get("grouping")
    if grouping is not None:
        if not isinstance(grouping, dict):
            raise ConfigError("data.grouping must be a mapping.")
        if "enabled" not in grouping:
            raise ConfigError("data.grouping.enabled is required when data.grouping is present.")
        if grouping.get("enabled"):
            mode = str(grouping.get("mode", "none")).lower()
            if mode not in {"none", "regex", "path_parent", "filename_prefix"}:
                raise ConfigError("data.grouping.mode must be one of: none, regex, path_parent, filename_prefix.")
            if mode == "none":
                raise ConfigError("data.grouping.enabled=true requires a non-'none' grouping mode.")
            if mode == "regex" and not grouping.get("regex"):
                raise ConfigError("data.grouping.regex is required when mode is 'regex'.")

    loader = data["loader"]
    _require_keys(loader, ["batch_size", "num_workers", "pin_memory", "persistent_workers"], "data.loader")
    if int(loader["batch_size"]) <= 0:
        raise ConfigError("data.loader.batch_size must be > 0.")
    if int(loader["num_workers"]) < 0:
        raise ConfigError("data.loader.num_workers must be >= 0.")

    model = config["model"]
    _require_keys(
        model,
        [
            "in_channels",
            "num_classes",
            "base_channels",
            "channel_multipliers",
            "blocks_per_stage",
            "difn",
            "attention",
            "decoder",
            "ablation",
        ],
        "model",
    )
    if int(model["num_classes"]) != 1:
        raise ConfigError("This implementation expects model.num_classes=1 for binary polyp segmentation.")
    if len(model["channel_multipliers"]) != len(model["blocks_per_stage"]):
        raise ConfigError("model.channel_multipliers and model.blocks_per_stage lengths must match.")

    ablation = model["ablation"]
    _require_keys(ablation, ["disable_difn", "disable_attention"], "model.ablation")

    training = config["training"]
    _require_keys(
        training,
        [
            "epochs",
            "amp",
            "gradient_clip_norm",
            "checkpoint_interval",
            "resume_checkpoint",
            "early_stopping",
            "optimizer",
            "scheduler",
            "loss",
            "metrics",
        ],
        "training",
    )
    if int(training["epochs"]) <= 0:
        raise ConfigError("training.epochs must be > 0.")

    optimizer = training["optimizer"]
    _require_keys(optimizer, ["lr", "weight_decay", "betas"], "training.optimizer")
    if float(optimizer["lr"]) <= 0.0:
        raise ConfigError("training.optimizer.lr must be > 0.")
    if len(optimizer["betas"]) != 2:
        raise ConfigError("training.optimizer.betas must contain two values.")

    scheduler = training["scheduler"]
    _require_keys(
        scheduler,
        ["type", "min_lr", "t_max", "step_size", "gamma", "mode", "factor", "patience"],
        "training.scheduler",
    )
    if scheduler["type"] not in {"cosine", "step", "plateau"}:
        raise ConfigError("training.scheduler.type must be one of: cosine, step, plateau.")

    loss_cfg = training["loss"]
    _require_keys(loss_cfg, ["dice_weight", "bce_weight"], "training.loss")
    if float(loss_cfg["dice_weight"]) < 0.0 or float(loss_cfg["bce_weight"]) < 0.0:
        raise ConfigError("training.loss weights must be non-negative.")
    if float(loss_cfg["dice_weight"]) + float(loss_cfg["bce_weight"]) <= 0.0:
        raise ConfigError("At least one loss weight must be > 0.")

    metrics_cfg = training["metrics"]
    _require_keys(metrics_cfg, ["threshold"], "training.metrics")
    if not 0.0 < float(metrics_cfg["threshold"]) < 1.0:
        raise ConfigError("training.metrics.threshold must be in (0, 1).")
    if "mode" in metrics_cfg:
        mode = str(metrics_cfg["mode"]).lower()
        if mode not in {"raw", "postprocess"}:
            raise ConfigError("training.metrics.mode must be one of: raw, postprocess.")

    runtime = config["runtime"]
    _require_keys(runtime, ["device", "inference_only", "cudnn_benchmark"], "runtime")
    if runtime["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("runtime.device must be one of: auto, cpu, cuda.")

    inference = config["inference"]
    _require_keys(inference, ["checkpoint", "split", "save_predictions"], "inference")
    if inference["split"] not in {"train", "val", "test"}:
        raise ConfigError("inference.split must be one of: train, val, test.")


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed across numpy, random, and torch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Deterministically seed each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a deterministic torch Generator instance."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def select_device(runtime_cfg: Dict[str, Any]) -> torch.device:
    """Select runtime device from config and availability."""
    requested = runtime_cfg["device"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def _next_experiment_index(output_root: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_idx = 0
    for item in output_root.iterdir():
        if not item.is_dir():
            continue
        matched = pattern.match(item.name)
        if matched:
            max_idx = max(max_idx, int(matched.group(1)))
    return max_idx + 1


def _build_experiment_paths(exp_root: Path) -> ExperimentPaths:
    checkpoints = exp_root / "checkpoints"
    logs = exp_root / "logs"
    metrics = exp_root / "metrics"
    predictions = exp_root / "predictions"
    figures = exp_root / "figures"
    for directory in [exp_root, checkpoints, logs, metrics, predictions, figures]:
        directory.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        root=exp_root,
        checkpoints=checkpoints,
        logs=logs,
        metrics=metrics,
        predictions=predictions,
        figures=figures,
    )


def create_experiment_paths(output_root: str | Path, prefix: str) -> ExperimentPaths:
    """Create a new numbered experiment directory with standard subfolders."""
    base = Path(output_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    exp_idx = _next_experiment_index(base, prefix)
    exp_root = base / f"{prefix}_{exp_idx:03d}"
    return _build_experiment_paths(exp_root=exp_root)


def create_child_experiment_paths(parent_root: str | Path, name: str) -> ExperimentPaths:
    """Create a child experiment directory inside an existing experiment root."""
    root = Path(parent_root).expanduser().resolve() / name
    return _build_experiment_paths(exp_root=root)


def setup_logger(log_dir: Path, logger_name: str = "dinsnet", enable_console: bool = False) -> logging.Logger:
    """Create a logger that writes to file and optionally to stdout."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    if enable_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def copy_config_to_experiment(config_path: str | Path, destination_dir: Path) -> Path:
    """Copy config YAML into the experiment directory."""
    src = Path(config_path).expanduser().resolve()
    dest = destination_dir / "config.yaml"
    shutil.copy2(src, dest)
    return dest


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def calculate_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int],
    device: torch.device,
) -> Optional[float]:
    """Calculate FLOPs with THOP if available."""
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None

    # THOP attaches bookkeeping buffers (e.g., total_ops/total_params) in-place.
    # Profile on a cloned model so runtime checkpoints remain clean.
    model_for_profile = copy.deepcopy(model).to(device)
    model_was_training = model.training
    model_for_profile.eval()
    dummy = torch.randn(1, *input_shape, device=device)
    with torch.no_grad():
        flops, _ = profile(model_for_profile, inputs=(dummy,), verbose=False)
    if model_was_training:
        model.train()
    return float(flops)


def sanitize_model_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove profiler-injected keys that are not part of real model parameters."""
    filtered: Dict[str, Any] = {}
    for key, value in state_dict.items():
        suffix = key.rsplit(".", maxsplit=1)[-1]
        if suffix in {"total_ops", "total_params"}:
            continue
        filtered[key] = value
    return filtered


def save_model_profile(
    path: Path,
    param_count: int,
    flops: Optional[float],
    input_shape: Tuple[int, int, int],
) -> None:
    """Save model complexity profile as JSON."""
    payload = {
        "input_shape": list(input_shape),
        "trainable_parameters": int(param_count),
        "flops": flops,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(state: Dict[str, Any], checkpoint_path: Path) -> None:
    """Persist checkpoint dictionary to disk."""
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path: str | Path, map_location: torch.device) -> Dict[str, Any]:
    """Load checkpoint dictionary from disk."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach torch tensor to numpy array."""
    return x.detach().cpu().numpy()


def batch_segmentation_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute batch-level Dice, IoU, Precision, and Recall."""
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    preds_flat = preds.reshape(preds.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1.0 - targets_flat)).sum(dim=1)
    fn = ((1.0 - preds_flat) * targets_flat).sum(dim=1)

    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


class MetricTracker:
    """Running average tracker for scalar metrics."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self._totals[key] = self._totals.get(key, 0.0) + float(value) * n
            self._counts[key] = self._counts.get(key, 0) + n

    def averages(self) -> Dict[str, float]:
        return {
            key: self._totals[key] / max(self._counts.get(key, 1), 1)
            for key in sorted(self._totals.keys())
        }


def write_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append row to CSV file, creating header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    """SHA256 hash of a file on disk."""
    data = path.read_bytes()
    return _hash_bytes(data)


def hash_dict(payload: Dict[str, Any]) -> str:
    """Stable SHA256 hash of a JSON-serializable dict."""
    dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _hash_bytes(dumped)


def _safe_package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def collect_package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in packages:
        version = _safe_package_version(name)
        if version is not None:
            versions[name] = version
    return versions


def get_git_metadata(cwd: Path) -> Dict[str, Any]:
    """Return git commit/branch/dirty status if available."""
    metadata: Dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(cwd), text=True).strip()
        dirty_output = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(cwd), text=True)
        metadata["commit"] = commit
        metadata["branch"] = branch
        metadata["dirty"] = bool(dirty_output.strip())
    except Exception:
        return metadata
    return metadata


def capture_pip_freeze() -> List[str]:
    """Capture pip freeze output for reproducibility."""
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return [line for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def collect_torch_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        gpus = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        env["gpus"] = gpus
    try:
        from torch.utils.collect_env import get_pretty_env_info

        env["torch_env_info"] = get_pretty_env_info()
    except Exception:
        env["torch_env_info"] = None
    return env


def export_run_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    config_path: str | Path,
    cli_overrides: Dict[str, Any],
    split_manifest_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
) -> Path:
    """Export experiment metadata for auditability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path).expanduser().resolve()
    package_versions = collect_package_versions(
        [
            "torch",
            "torchvision",
            "numpy",
            "opencv-python-headless",
            "albumentations",
            "PyYAML",
            "tqdm",
            "tensorboard",
            "thop",
            "matplotlib",
        ]
    )
    pip_freeze = capture_pip_freeze()
    pip_freeze_path = output_dir / "pip_freeze.txt"
    if pip_freeze:
        with pip_freeze_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(pip_freeze) + "\n")

    manifests: Dict[str, Any] = {}
    if split_manifest_path is not None and split_manifest_path.exists():
        manifests["split_manifest"] = {
            "path": str(split_manifest_path.expanduser().resolve()),
            "sha256": hash_file(split_manifest_path),
        }
    if dataset_manifest_path is not None and dataset_manifest_path.exists():
        manifests["dataset_manifest"] = {
            "path": str(dataset_manifest_path.expanduser().resolve()),
            "sha256": hash_file(dataset_manifest_path),
        }

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": get_git_metadata(Path.cwd()),
        "packages": package_versions,
        "pip_freeze": pip_freeze,
        "pip_freeze_path": str(pip_freeze_path) if pip_freeze else None,
        "torch_environment": collect_torch_env(),
        "config": {
            "path": str(config_path),
            "sha256": hash_file(config_path) if config_path.exists() else None,
            "effective_sha256": hash_dict(config),
        },
        "manifests": manifests,
        "cli_overrides": cli_overrides,
    }
    path = output_dir / "run_environment.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path
