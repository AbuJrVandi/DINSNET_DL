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

print("🚀 Step 1: Cloning the DINSNET_DL repository...")
!git clone https://github.com/AbuJrVandi/DINSNET_DL.git
%cd DINSNET_DL

print("\n📦 Step 2: Installing dependencies...")
!pip install -r requirements.txt
# Install thop for FLOPs calculation (if not already included)
!pip install thop 

print("\n📂 Step 3: Dataset Setup")
# ==============================================================================
# CHOOSE ONE OF THE DATASET METHODS BELOW AND UNCOMMENT IT:
# ==============================================================================

# METHOD A: Google Drive (Colab Only)
# If your dataset is in Google Drive, uncomment these lines to mount your drive
# and copy the dataset into the workspace.
# ------------------------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/Your_Dataset_Folder ./datasets/

# METHOD B: Kaggle Datasets (Kaggle Only)
# In Kaggle, datasets are automatically mounted in the `../input/` directory.
# You will just need to update the config file path below to point to it.
# e.g., --config_dataset_path ../input/your-dataset/

# METHOD C: Direct Download (Wget / Unzip)
# If you have a direct download link for your dataset:
# ------------------------------------------------------------------------------
# !wget "https://your-dataset-link.com/dataset.zip" -O dataset.zip
# !unzip -q dataset.zip -d ./datasets/

print("\n⚙️ Step 4: Starting Training...")
# ==============================================================================
# Run the training script!
# Make sure you edit 'configs/config.yaml' to point to your actual dataset path.
# If your dataset is at './datasets/my_data', update the config.yaml before running.
# ==============================================================================

!python main.py --config configs/config.yaml --mode train

print("\n✅ Training initiated successfully!")
