# ==============================================================================
# DINSNet: Single-Cell Training Code for Google Colab & Kaggle
# ==============================================================================
# INSTRUCTIONS:
# 1. Open Google Colab (colab.research.google.com) or Kaggle.
# 2. Create a new Notebook.
# 3. Make sure GPU is enabled:
#    - Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4)
#    - Kaggle: Settings -> Accelerator -> GPU
# 4. Copy and paste ALL the code below into a SINGLE cell and run it.
# ==============================================================================

import os
import yaml

print("🚀 Step 1: Cloning the DINSNET_DL repository...")
if not os.path.exists("DINSNET_DL"):
    !git clone https://github.com/AbuJrVandi/DINSNET_DL.git
%cd DINSNET_DL

print("\n📦 Step 2: Installing dependencies...")
!pip install -r requirements.txt
# Install thop (for FLOPs calculation) and gdown (for downloading the dataset)
!pip install thop gdown 

print("\n📂 Step 3: Downloading Dataset from Google Drive...")
# The script will download the folder directly using the link you provided.
# ⚠️ IF THIS FAILS: You must go to Google Drive, right-click the folder -> Share -> Change to "Anyone with the link".
folder_url = "https://drive.google.com/drive/folders/1NDKSCJKiMGGRkPhchk4BOCPfOue6Z5_p"
dataset_path = "./datasets/my_training_data"
os.makedirs("./datasets", exist_ok=True)
!gdown --folder {folder_url} -O {dataset_path}
print("\n⚙️ Step 4: Updating Configuration for the new dataset...")
# Modify the config file dynamically to point to the downloaded dataset
config_path = "configs/config.yaml"
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the root_dir to point to our newly downloaded dataset
    config['data']['root_dir'] = dataset_path
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Config updated successfully! dataset path set to: {dataset_path}")
except Exception as e:
    print(f"⚠️ Could not update config automatically: {e}")

print("\n🚀 Step 5: Starting Training...")
# Run the training script
!python main.py --config configs/config.yaml --mode train

print("\n✅ Training initiated successfully!")
