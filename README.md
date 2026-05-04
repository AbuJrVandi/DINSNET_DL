# DINSNet (Simple Guide)

This project trains a polyp segmentation model.

In simple words:
- Input: colonoscopy image
- Output: mask (where the polyp is)

## What This Project Includes
- `main.py`: run training or inference
- `models/`: model code (`DINSNet`, baseline U-Net)
- `data/`: dataset loading and split logic
- `trainer/`: training loop, validation, checkpoint save/load
- `utils.py`: config checks, metrics, experiment folder creation
- `configs/`: run settings (YAML)

## Project Tour (Beginner Friendly)
- Entry point: `main.py` wires config + data + model + trainer.
- Data: `data/dataset.py` discovers images/masks, builds splits, and creates PyTorch DataLoaders.
- Model: `models/dinsnet.py` defines DINSNet (the main model). `models/unet.py` is the baseline.
- Training: `trainer/trainer.py` runs epochs, computes loss/metrics, saves checkpoints.
- Utilities: `utils.py` validates configs, tracks metrics, saves metadata.

## Dataset Format
Each dataset must look like this:

```text
<dataset_root>/
  images/
  masks/
```

In this workspace, original datasets are in:
- `datasets/CVC-300`
- `datasets/CVC-ClinicDB`
- `datasets/CVC-ColonDB`
- `datasets/Kvasir`

## Configs
Store configs under `configs/`. If you use per-dataset debug configs, keep them here as well
and point `data.root_dir` at the matching dataset.

## Quick Start
1. Create and activate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train on one dataset (example: CVC-300):

```bash
python main.py --config configs/config.yaml
```

## Inference (Prediction)
Run inference with a trained checkpoint:

```bash
python main.py --mode inference --config configs/config.yaml --checkpoint /path/to/best.pth
```

## Continue Training on Next Dataset
You can train in sequence (CVC-300, then CVC-ClinicDB, etc.) by passing checkpoint:

```bash
python main.py --config configs/config.yaml --checkpoint /path/to/previous_stage/best.pth
```

## Main CLI Options
- `--config`: choose yaml config
- `--mode train|inference`: run training or inference
- `--checkpoint`: checkpoint path (resume training or inference)
- `--epochs`: override epochs from yaml
- `--batch-size`: override batch size
- `--lr`: override learning rate
- `--num-workers`: override dataloader workers

## Output Structure
Each run creates a new folder in `outputs/`:

```text
outputs/<experiment_name>_001/
  checkpoints/
  logs/
  metrics/
  predictions/
  figures/
```

Important files:
- `checkpoints/best.pth`: best model
- `checkpoints/last.pth`: latest model
- `metrics/training_history.csv`: epoch metrics
- `metrics/data_split.json`: train/val/test counts

## Config File Basics
Most important fields:
- `data.root_dir`: dataset path
- `training.epochs`: number of epochs
- `data.loader.batch_size`: batch size
- `training.optimizer.lr`: learning rate
- `runtime.device`: `auto`, `cpu`, or `cuda`

## Notes
- This repo expects positive train/val/test split ratios in config.
- If CUDA is available and `runtime.device: auto`, it will use GPU.
- Inference mode needs a checkpoint path.
# DINSNET_DL
