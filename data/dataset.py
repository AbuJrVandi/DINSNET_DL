"""Dataset and dataloader utilities for multi-domain polyp segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import make_torch_generator, seed_worker


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for a single image-mask pair."""

    image_path: Path
    mask_path: Path
    sample_id: str
    domain: str


LOGGER = logging.getLogger(__name__)


class PolypSegmentationDataset(Dataset):
    """Polyp segmentation dataset with synchronized image-mask transforms."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        image_size: Tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
        resize_mode: str = "pad",
        pad_value: float = 0.0,
        transform: A.Compose | None = None,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.resize_mode = resize_mode
        self.pad_value = float(pad_value)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unable to read image: {sample.image_path}")
        if mask is None:
            raise RuntimeError(f"Unable to read mask: {sample.mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image, mask = _resize_pair(
            image=image,
            mask=mask,
            target_size=self.image_size,
            mode=self.resize_mode,
            pad_value=self.pad_value,
        )

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = (mask >= 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }


def _coerce_pad_value(image: np.ndarray, pad_value: float) -> float:
    if image.dtype.kind == "f":
        max_val = float(np.max(image)) if image.size else 1.0
        if max_val <= 1.0 and pad_value > 1.0:
            return pad_value / 255.0
    return pad_value


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
    mode: str,
    pad_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_size
    if mode == "stretch":
        image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise RuntimeError("Invalid image size for resizing.")

    if mode == "pad":
        scale = min(target_w / w, target_h / h)
    elif mode == "crop":
        scale = max(target_w / w, target_h / h)
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if mode == "crop":
        x0 = max((new_w - target_w) // 2, 0)
        y0 = max((new_h - target_h) // 2, 0)
        image_cropped = image_scaled[y0 : y0 + target_h, x0 : x0 + target_w]
        mask_cropped = mask_scaled[y0 : y0 + target_h, x0 : x0 + target_w]
        return image_cropped, mask_cropped

    pad_val = _coerce_pad_value(image_scaled, pad_value)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = max(pad_h // 2, 0)
    bottom = max(pad_h - top, 0)
    left = max(pad_w // 2, 0)
    right = max(pad_w - left, 0)

    image_padded = cv2.copyMakeBorder(
        image_scaled,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_val, pad_val, pad_val),
    )
    mask_padded = cv2.copyMakeBorder(
        mask_scaled,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return image_padded, mask_padded


def _normalize_extension_set(extensions: Sequence[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def _collect_pairs_from_domain(
    domain_dir: Path,
    domain_name: str,
    image_dir_name: str,
    mask_dir_name: str,
    image_exts: set[str],
    mask_exts: set[str],
) -> List[SampleRecord]:
    image_dir = domain_dir / image_dir_name
    mask_dir = domain_dir / mask_dir_name
    if not image_dir.exists() or not mask_dir.exists():
        return []

    image_map = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_exts
    }
    mask_map = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in mask_exts
    }
    common = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
    if missing_images or missing_masks:
        raise RuntimeError(
            f"Pairing mismatch in domain '{domain_name}': "
            f"missing_images={len(missing_images)} missing_masks={len(missing_masks)}"
        )

    return [
        SampleRecord(
            image_path=image_map[sample_id],
            mask_path=mask_map[sample_id],
            sample_id=sample_id,
            domain=domain_name,
        )
        for sample_id in common
    ]


def discover_samples(
    root_dir: str | Path,
    image_dir_name: str,
    mask_dir_name: str,
    image_extensions: Sequence[str],
    mask_extensions: Sequence[str],
) -> List[SampleRecord]:
    """Discover image-mask pairs across one or multiple domains."""
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    image_exts = _normalize_extension_set(image_extensions)
    mask_exts = _normalize_extension_set(mask_extensions)

    samples: List[SampleRecord] = []

    direct_records = _collect_pairs_from_domain(
        domain_dir=root,
        domain_name="default",
        image_dir_name=image_dir_name,
        mask_dir_name=mask_dir_name,
        image_exts=image_exts,
        mask_exts=mask_exts,
    )
    if direct_records:
        samples.extend(direct_records)
        return samples

    domain_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not domain_dirs:
        raise RuntimeError(f"No dataset domain directories found in: {root}")

    for domain_dir in sorted(domain_dirs):
        domain_name = domain_dir.name
        domain_records = _collect_pairs_from_domain(
            domain_dir=domain_dir,
            domain_name=domain_name,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
            image_exts=image_exts,
            mask_exts=mask_exts,
        )
        if domain_records:
            samples.extend(domain_records)

    if not samples:
        raise RuntimeError(
            "No valid image-mask pairs found. "
            "Expected either root/images+root/masks or root/<domain>/images+root/<domain>/masks."
        )
    return samples


def _load_split_tokens(path: Path) -> List[str]:
    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens.append(line)
    return tokens


def _resolve_split_files(root: Path, split_files_cfg: Dict[str, Any] | None) -> Dict[str, Path] | None:
    split_names = ("train", "val", "test")
    if split_files_cfg:
        resolved: Dict[str, Path] = {}
        for split in split_names:
            value = split_files_cfg.get(split)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            resolved[split] = path
        if resolved:
            missing = [name for name in split_names if name not in resolved or not resolved[name].exists()]
            if missing:
                raise RuntimeError(f"Configured split_files missing: {missing}")
            return resolved

    for candidate in [root / "splits", root]:
        candidate_files = {name: candidate / f"{name}.txt" for name in split_names}
        if all(path.exists() for path in candidate_files.values()):
            return candidate_files
    return None


def _build_sample_index(samples: Sequence[SampleRecord]) -> Dict[str, SampleRecord]:
    index: Dict[str, SampleRecord] = {}
    for sample in samples:
        image_key = str(sample.image_path.expanduser().resolve())
        mask_key = str(sample.mask_path.expanduser().resolve())
        index[image_key] = sample
        index[mask_key] = sample
        index[f"{sample.domain}:{sample.sample_id}"] = sample
        index[f"{sample.domain}/{sample.sample_id}"] = sample
    return index


def _resolve_split_tokens(
    tokens: Sequence[str],
    samples: Sequence[SampleRecord],
    root: Path,
    image_exts: set[str],
    mask_exts: set[str],
) -> List[SampleRecord]:
    index = _build_sample_index(samples)
    sample_id_map: Dict[str, SampleRecord] = {}
    for sample in samples:
        if sample.sample_id not in sample_id_map:
            sample_id_map[sample.sample_id] = sample
        else:
            sample_id_map[sample.sample_id] = None  # type: ignore[assignment]

    resolved: List[SampleRecord] = []
    missing: List[str] = []
    for token in tokens:
        key = token.strip()
        if not key:
            continue

        candidate = None
        path_token = Path(key)
        suffix = path_token.suffix.lower()
        is_path_like = suffix in image_exts or suffix in mask_exts or "/" in key or "\\" in key
        if is_path_like and suffix:
            abs_path = path_token
            if not abs_path.is_absolute():
                abs_path = (root / path_token).expanduser().resolve()
            candidate = index.get(str(abs_path))
            if candidate is None:
                candidate = sample_id_map.get(path_token.stem)
        elif ":" in key:
            candidate = index.get(key)
        elif "/" in key or "\\" in key:
            candidate = index.get(key.replace("\\", "/"))
        else:
            candidate = sample_id_map.get(key)

        if candidate is None:
            missing.append(key)
        else:
            resolved.append(candidate)

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Split file contains unknown samples: {preview} (showing up to 10)")
    return resolved


def _load_split_from_files(
    samples: Sequence[SampleRecord],
    split_files: Dict[str, Path],
    root: Path,
    image_exts: set[str],
    mask_exts: set[str],
) -> Dict[str, List[SampleRecord]]:
    split: Dict[str, List[SampleRecord]] = {}
    for name, path in split_files.items():
        tokens = _load_split_tokens(path)
        split[name] = _resolve_split_tokens(tokens, samples, root, image_exts, mask_exts)

    split_sets = {name: set(id(sample) for sample in records) for name, records in split.items()}
    overlaps = (split_sets["train"] & split_sets["val"]) | (split_sets["train"] & split_sets["test"]) | (
        split_sets["val"] & split_sets["test"]
    )
    if overlaps:
        raise RuntimeError("Split files contain overlapping samples across train/val/test.")

    all_samples = set(id(sample) for sample in samples)
    covered = split_sets["train"] | split_sets["val"] | split_sets["test"]
    missing = all_samples - covered
    if missing:
        raise RuntimeError(
            "Split files do not cover all samples. "
            "Refuse to proceed to avoid silent leakage or data loss."
        )
    return split


def _group_id_from_sample(sample: SampleRecord, group_cfg: Dict[str, Any]) -> str:
    if not group_cfg or not bool(group_cfg.get("enabled", False)):
        return f"{sample.domain}:{sample.sample_id}"

    mode = str(group_cfg.get("mode", "none")).lower()
    source = str(group_cfg.get("source", "sample_id")).lower()
    include_domain = bool(group_cfg.get("include_domain", True))

    if source == "image_path":
        source_value = str(sample.image_path)
    elif source == "mask_path":
        source_value = str(sample.mask_path)
    else:
        source_value = sample.sample_id

    group_raw = None
    if mode == "regex":
        pattern = str(group_cfg.get("regex", ""))
        if not pattern:
            raise RuntimeError("grouping.mode='regex' requires grouping.regex")
        match = re.search(pattern, source_value)
        if match:
            if "group" in match.groupdict():
                group_raw = match.group("group")
            else:
                group_raw = match.group(1) if match.groups() else match.group(0)
    elif mode == "path_parent":
        level = int(group_cfg.get("path_parent_level", 1))
        if level <= 0:
            raise RuntimeError("grouping.path_parent_level must be >= 1")
        path_obj = Path(source_value)
        if len(path_obj.parents) >= level:
            group_raw = path_obj.parents[level - 1].name
    elif mode == "filename_prefix":
        delimiter = str(group_cfg.get("filename_delimiter", "_"))
        parts = int(group_cfg.get("filename_prefix_parts", 1))
        if parts <= 0:
            raise RuntimeError("grouping.filename_prefix_parts must be >= 1")
        tokens = sample.sample_id.split(delimiter)
        group_raw = delimiter.join(tokens[:parts])
    elif mode == "none":
        group_raw = sample.sample_id
    else:
        raise RuntimeError(f"Unsupported grouping.mode: {mode}")

    if not group_raw:
        fallback = str(group_cfg.get("fallback", "sample_id")).lower()
        group_raw = sample.sample_id if fallback == "sample_id" else source_value

    return f"{sample.domain}:{group_raw}" if include_domain else str(group_raw)


def _assign_groups_to_splits(
    groups: Dict[str, List[SampleRecord]],
    split_ratios: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, List[SampleRecord]]:
    split_names = list(split_ratios.keys())
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    total = sum(len(groups[group_id]) for group_id in group_ids)
    targets = {name: float(split_ratios[name]) * total for name in split_names}
    counts = {name: 0 for name in split_names}
    assignments: Dict[str, List[str]] = {name: [] for name in split_names}

    for group_id in group_ids:
        remaining = {name: targets[name] - counts[name] for name in split_names}
        chosen = max(remaining, key=remaining.get)
        assignments[chosen].append(group_id)
        counts[chosen] += len(groups[group_id])

    # Ensure no empty splits when possible.
    for name in split_names:
        if assignments[name]:
            continue
        donor = max(assignments.keys(), key=lambda key: len(assignments[key]))
        if not assignments[donor]:
            continue
        moved = assignments[donor].pop()
        assignments[name].append(moved)

    split_samples: Dict[str, List[SampleRecord]] = {name: [] for name in split_names}
    for name in split_names:
        for group_id in assignments[name]:
            split_samples[name].extend(groups[group_id])
    return split_samples


def _save_split_manifest(
    split: Dict[str, List[SampleRecord]],
    output_dir: Path,
    root: Path,
    group_cfg: Dict[str, Any],
    seed: int,
    source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "seed": int(seed),
        "source": source,
        "grouping": group_cfg or {"enabled": False, "mode": "none"},
        "split_stats": {},
        "splits": {},
    }

    for split_name, samples in split.items():
        records = []
        group_ids = []
        for sample in sorted(samples, key=lambda s: (s.domain, s.sample_id)):
            image_abs = sample.image_path.expanduser().resolve()
            mask_abs = sample.mask_path.expanduser().resolve()
            try:
                image_rel = image_abs.relative_to(root)
            except Exception:
                image_rel = image_abs
            try:
                mask_rel = mask_abs.relative_to(root)
            except Exception:
                mask_rel = mask_abs
            records.append(
                {
                    "domain": sample.domain,
                    "sample_id": sample.sample_id,
                    "image_path": str(image_rel),
                    "mask_path": str(mask_rel),
                    "group_id": _group_id_from_sample(sample, group_cfg),
                }
            )
            group_ids.append(_group_id_from_sample(sample, group_cfg))
        manifest["splits"][split_name] = records
        manifest["split_stats"][split_name] = {
            "num_samples": len(records),
            "num_groups": len(set(group_ids)),
        }

        txt_path = output_dir / f"{split_name}.txt"
        with txt_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record["image_path"] + "\n")

    json_path = output_dir / "split_manifest.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _save_dataset_manifest(samples: Sequence[SampleRecord], output_path: Path, root: Path) -> None:
    records: List[Dict[str, Any]] = []
    for sample in sorted(samples, key=lambda s: (s.domain, s.sample_id)):
        image_abs = sample.image_path.expanduser().resolve()
        mask_abs = sample.mask_path.expanduser().resolve()
        try:
            image_rel = image_abs.relative_to(root)
        except Exception:
            image_rel = image_abs
        try:
            mask_rel = mask_abs.relative_to(root)
        except Exception:
            mask_rel = mask_abs
        records.append(
            {
                "domain": sample.domain,
                "sample_id": sample.sample_id,
                "image_path": str(image_rel),
                "mask_path": str(mask_rel),
                "image_size_bytes": image_abs.stat().st_size if image_abs.exists() else None,
                "mask_size_bytes": mask_abs.stat().st_size if mask_abs.exists() else None,
                "image_mtime": image_abs.stat().st_mtime if image_abs.exists() else None,
                "mask_mtime": mask_abs.stat().st_mtime if mask_abs.exists() else None,
            }
        )
    payload = {
        "root": str(root),
        "num_samples": len(records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def split_samples(
    samples: Sequence[SampleRecord],
    split_cfg: Dict[str, Any],
    domain_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, List[SampleRecord]]:
    """Split samples into train/val/test with group-aware domain generalization support."""
    # Key idea: keep related images (same patient/sequence) in the same split
    # by grouping, so we avoid train/test leakage.
    rng = np.random.default_rng(seed)
    samples = list(samples)
    train_ratio = float(split_cfg["train"])
    val_ratio = float(split_cfg["val"])
    test_ratio = float(split_cfg["test"])

    def group_samples(records: Sequence[SampleRecord]) -> Dict[str, List[SampleRecord]]:
        groups: Dict[str, List[SampleRecord]] = {}
        for record in records:
            group_id = _group_id_from_sample(record, group_cfg)
            groups.setdefault(group_id, []).append(record)
        return groups

    def ensure_group_capacity(groups: Dict[str, List[SampleRecord]], split_names: Iterable[str]) -> None:
        required = [name for name in split_names if split_cfg.get(name, 0) > 0]
        if len(groups) < len(required):
            raise RuntimeError(
                f"Not enough groups ({len(groups)}) to split into {len(required)} splits "
                f"without leakage. Adjust grouping or split ratios."
            )

    if domain_cfg["enabled"] and domain_cfg["mode"] == "leave_one_domain_out":
        # Domain generalization: hold out entire domains for test.
        held_out = domain_cfg["held_out_domains"]
        if isinstance(held_out, str):
            held_out = [held_out]
        held_out_set = set(held_out)
        if not held_out_set:
            raise RuntimeError(
                "Domain generalization mode 'leave_one_domain_out' requires non-empty held_out_domains."
            )

        test_samples = [sample for sample in samples if sample.domain in held_out_set]
        remaining = [sample for sample in samples if sample.domain not in held_out_set]
        if not test_samples:
            raise RuntimeError(f"No test samples found for held_out_domains={sorted(held_out_set)}")
        if len(remaining) < 2:
            raise RuntimeError("Insufficient training samples after held-out split.")

        domain_to_samples: Dict[str, List[SampleRecord]] = {}
        for sample in remaining:
            domain_to_samples.setdefault(sample.domain, []).append(sample)

        train_samples: List[SampleRecord] = []
        val_samples: List[SampleRecord] = []
        for domain_name, domain_samples in sorted(domain_to_samples.items()):
            domain_groups = group_samples(domain_samples)
            ensure_group_capacity(domain_groups, ["train", "val"])
            split_domain = _assign_groups_to_splits(
                domain_groups,
                {"train": train_ratio, "val": val_ratio},
                rng,
            )
            train_samples.extend(split_domain["train"])
            val_samples.extend(split_domain["val"])

        return {"train": train_samples, "val": val_samples, "test": test_samples}

    if domain_cfg["enabled"] and domain_cfg["mode"] == "random":
        # Domain generalization: randomly assign full domains to train/val/test.
        domain_groups: Dict[str, List[SampleRecord]] = {}
        for sample in samples:
            domain_groups.setdefault(sample.domain, []).append(sample)
        if len(domain_groups) < 3:
            raise RuntimeError("Random domain split requires at least 3 domains for train/val/test.")
        ensure_group_capacity(domain_groups, ["train", "val", "test"])
        return _assign_groups_to_splits(
            domain_groups,
            {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            rng,
        )

    stratify_by_domain = bool(split_cfg["stratify_by_domain"])
    if stratify_by_domain:
        # Default: stratify each domain independently to balance domains.
        domain_to_samples: Dict[str, List[SampleRecord]] = {}
        for sample in samples:
            domain_to_samples.setdefault(sample.domain, []).append(sample)

        train_samples: List[SampleRecord] = []
        val_samples: List[SampleRecord] = []
        test_samples: List[SampleRecord] = []
        for _, domain_samples in sorted(domain_to_samples.items()):
            domain_groups = group_samples(domain_samples)
            ensure_group_capacity(domain_groups, ["train", "val", "test"])
            split_domain = _assign_groups_to_splits(
                domain_groups,
                {"train": train_ratio, "val": val_ratio, "test": test_ratio},
                rng,
            )
            train_samples.extend(split_domain["train"])
            val_samples.extend(split_domain["val"])
            test_samples.extend(split_domain["test"])
        return {"train": train_samples, "val": val_samples, "test": test_samples}

    groups = group_samples(samples)
    ensure_group_capacity(groups, ["train", "val", "test"])
    return _assign_groups_to_splits(
        groups,
        {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        rng,
    )


def build_train_transform(aug_cfg: Dict[str, Any]) -> A.Compose | None:
    """Build training augmentation pipeline from config."""
    if not aug_cfg["enable"]:
        return None

    transforms: List[A.BasicTransform] = []
    if float(aug_cfg["horizontal_flip_prob"]) > 0:
        transforms.append(A.HorizontalFlip(p=float(aug_cfg["horizontal_flip_prob"])))
    if float(aug_cfg["vertical_flip_prob"]) > 0:
        transforms.append(A.VerticalFlip(p=float(aug_cfg["vertical_flip_prob"])))
    if float(aug_cfg["rotate_prob"]) > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=float(aug_cfg["shift_limit"]),
                scale_limit=float(aug_cfg["scale_limit"]),
                rotate_limit=int(aug_cfg["rotate_limit"]),
                border_mode=cv2.BORDER_REFLECT_101,
                p=float(aug_cfg["rotate_prob"]),
            )
        )
    if float(aug_cfg["brightness_contrast_prob"]) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(aug_cfg["brightness_limit"]),
                contrast_limit=float(aug_cfg["contrast_limit"]),
                p=float(aug_cfg["brightness_contrast_prob"]),
            )
        )
    if float(aug_cfg["gauss_noise_prob"]) > 0:
        transforms.append(
            A.GaussNoise(
                var_limit=tuple(float(v) for v in aug_cfg["gauss_noise_var_limit"]),
                p=float(aug_cfg["gauss_noise_prob"]),
            )
        )
    return A.Compose(transforms) if transforms else None


def build_dataloaders(
    config: Dict[str, Any],
    seed: int,
    split_output_dir: Path | None = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, List[SampleRecord]]]:
    """Build train/val/test DataLoaders from config."""
    data_cfg = config["data"]
    root_dir = Path(data_cfg["root_dir"]).expanduser().resolve()
    samples = discover_samples(
        root_dir=root_dir,
        image_dir_name=data_cfg["image_dir_name"],
        mask_dir_name=data_cfg["mask_dir_name"],
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
    )
    image_exts = _normalize_extension_set(data_cfg["image_extensions"])
    mask_exts = _normalize_extension_set(data_cfg["mask_extensions"])

    split_cfg = data_cfg.get("split", {})
    allow_image_level = bool(split_cfg.get("allow_image_level_split", False))

    group_cfg = data_cfg.get("grouping")
    if group_cfg is None:
        group_cfg = {
            "enabled": True,
            "mode": "filename_prefix",
            "filename_delimiter": "_",
            "filename_prefix_parts": 1,
            "include_domain": True,
        }
        LOGGER.info(
            "No grouping config provided. Defaulting to filename_prefix grouping for leak-free splits. "
            "Override data.grouping to customize."
        )

    split_files = _resolve_split_files(root=root_dir, split_files_cfg=data_cfg.get("split_files"))
    split_source = "generated"
    group_enabled = bool(group_cfg.get("enabled", False)) and str(group_cfg.get("mode", "none")).lower() != "none"
    if not split_files and not group_enabled:
        # Safety: prevent image-level leakage unless explicitly allowed.
        if not allow_image_level:
            raise RuntimeError(
                "Image-level splitting is disabled by default to prevent leakage. "
                "Either enable data.grouping or set data.split.allow_image_level_split=true."
            )
        LOGGER.warning(
            "Image-level splitting forced via data.split.allow_image_level_split=true. "
            "This can leak signal across splits if frames/patients repeat."
        )
    if split_files:
        if not group_enabled:
            LOGGER.warning(
                "Grouping is disabled while using split files. "
                "Proceeding without group leakage validation."
            )
        split = _load_split_from_files(
            samples=samples,
            split_files=split_files,
            root=root_dir,
            image_exts=image_exts,
            mask_exts=mask_exts,
        )
        split_source = "official_split_files"
        if group_enabled:
            group_to_split: Dict[str, str] = {}
            for split_name, split_records in split.items():
                for record in split_records:
                    group_id = _group_id_from_sample(record, group_cfg)
                    if group_id in group_to_split and group_to_split[group_id] != split_name:
                        raise RuntimeError(
                            f"Group '{group_id}' appears in multiple splits "
                            f"({group_to_split[group_id]}, {split_name})."
                        )
                    group_to_split[group_id] = split_name
    else:
        split = split_samples(
            samples=samples,
            split_cfg=data_cfg["split"],
            domain_cfg=data_cfg["domain_generalization"],
            group_cfg=group_cfg,
            seed=seed,
        )

    split_sets = {name: set(id(sample) for sample in records) for name, records in split.items()}
    overlaps = (split_sets["train"] & split_sets["val"]) | (split_sets["train"] & split_sets["test"]) | (
        split_sets["val"] & split_sets["test"]
    )
    if overlaps:
        raise RuntimeError("Generated splits contain overlapping samples across train/val/test.")
    all_samples = set(id(sample) for sample in samples)
    covered = split_sets["train"] | split_sets["val"] | split_sets["test"]
    if covered != all_samples:
        raise RuntimeError("Generated splits do not cover all samples.")

    split_stats: Dict[str, Dict[str, int]] = {}
    if group_enabled:
        for split_name, split_records in split.items():
            group_ids = {_group_id_from_sample(sample, group_cfg) for sample in split_records}
            split_stats[split_name] = {
                "num_samples": len(split_records),
                "num_groups": len(group_ids),
            }
        LOGGER.info("Split group counts: %s", split_stats)
    else:
        for split_name, split_records in split.items():
            split_stats[split_name] = {"num_samples": len(split_records)}
        LOGGER.info("Split sample counts: %s", split_stats)

    if split_output_dir is not None:
        _save_split_manifest(
            split=split,
            output_dir=split_output_dir,
            root=root_dir,
            group_cfg=group_cfg,
            seed=seed,
            source=split_source,
        )
        dataset_manifest_path = split_output_dir.parent / "dataset_manifest.json"
        _save_dataset_manifest(samples=samples, output_path=dataset_manifest_path, root=root_dir)

    image_size = tuple(int(v) for v in data_cfg["image_size"])
    mean = data_cfg["preprocessing"]["mean"]
    std = data_cfg["preprocessing"]["std"]
    resize_mode = str(data_cfg["preprocessing"].get("resize_mode", "pad")).lower()
    pad_value = float(data_cfg["preprocessing"].get("pad_value", 0.0))
    train_transform = build_train_transform(data_cfg["augmentation"])

    datasets = {
        "train": PolypSegmentationDataset(
            samples=split["train"],
            image_size=image_size,
            mean=mean,
            std=std,
            resize_mode=resize_mode,
            pad_value=pad_value,
            transform=train_transform,
        ),
        "val": PolypSegmentationDataset(
            samples=split["val"],
            image_size=image_size,
            mean=mean,
            std=std,
            resize_mode=resize_mode,
            pad_value=pad_value,
            transform=None,
        ),
        "test": PolypSegmentationDataset(
            samples=split["test"],
            image_size=image_size,
            mean=mean,
            std=std,
            resize_mode=resize_mode,
            pad_value=pad_value,
            transform=None,
        ),
    }

    loader_cfg = data_cfg["loader"]
    batch_size = int(loader_cfg["batch_size"])
    num_workers = int(loader_cfg["num_workers"])
    pin_memory = bool(loader_cfg["pin_memory"])
    persistent_workers = bool(loader_cfg["persistent_workers"]) and num_workers > 0

    generator = make_torch_generator(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        ),
    }
    return loaders, split
