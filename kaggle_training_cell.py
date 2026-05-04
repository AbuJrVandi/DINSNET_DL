# ==============================================================================
# DINSNet: Single-Cell Training Code specifically for Kaggle
# ==============================================================================
# INSTRUCTIONS:
# 1. Open Kaggle (kaggle.com) and create a New Notebook.
# 2. Add your dataset: Click "+ Add Data" (top right), go to "Your Datasets", and select your uploaded dataset.
# 3. Make sure GPU is enabled: Settings (right side panel) -> Accelerator -> GPU (P100 or T4x2).
# 4. Turn on Internet: Settings -> Internet -> Turn on (needed to clone GitHub repo).
# 5. Copy and paste ALL the code below into a SINGLE cell and run it.
# ==============================================================================

import os
import yaml
import glob

print("🚀 Step 1: Cloning the DINSNET_DL repository...")
if not os.path.exists("DINSNET_DL"):
    !git clone https://github.com/AbuJrVandi/DINSNET_DL.git
%cd DINSNET_DL

print("\n📦 Step 2: Installing dependencies...")
!pip install -r requirements.txt
!pip install thop  # Required for FLOPs calculation

print("\n📂 Step 3: Finding Kaggle Dataset Path...")
# Kaggle mounts datasets under /kaggle/input/
kaggle_input_dir = "/kaggle/input"
try:
    datasets_in_kaggle = os.listdir(kaggle_input_dir)
    print(f"Datasets found in Kaggle: {datasets_in_kaggle}")
    
    if len(datasets_in_kaggle) > 0:
        # If there's a folder inside the dataset folder, we might need to point to it.
        # usually it's /kaggle/input/dataset-name/
        dataset_path = os.path.join(kaggle_input_dir, datasets_in_kaggle[0])
        
        # Sometimes the dataset itself contains another folder, let's check for 'images' or 'masks'
        # to ensure we are at the right root directory.
        subdirs = os.listdir(dataset_path)
        print(f"Contents of {dataset_path}: {subdirs}")
        
        print(f"✅ Automatically selected dataset path: {dataset_path}")
    else:
        print("⚠️ No datasets found in /kaggle/input/. Did you forget to add the dataset to the notebook?")
        dataset_path = "/kaggle/input/YOUR_DATASET_NAME"
except Exception as e:
    print(f"⚠️ Error reading /kaggle/input/: {e}")
    dataset_path = "/kaggle/input/YOUR_DATASET_NAME"

print("\n⚙️ Step 4: Updating Configuration for the new dataset...")
# Modify the config file dynamically to point to the Kaggle dataset path
config_path = "configs/config.yaml"
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the root_dir to point to the Kaggle dataset
    config['data']['root_dir'] = dataset_path
    
    # Optional: Update batch size or num_workers if needed for Kaggle
    config['data']['loader']['num_workers'] = 2 # Kaggle often prefers 2 or 4 workers
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Config updated successfully! dataset path set to: {dataset_path}")
except Exception as e:
    print(f"⚠️ Could not update config automatically: {e}")

print("\n🚀 Step 5: Starting Training...")
# Run the training script
!python main.py --config configs/config.yaml --mode train

print("\n✅ Training initiated successfully!")
