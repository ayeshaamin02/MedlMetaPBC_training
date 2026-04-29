#!/usr/bin/env bash
set -e

# Simple Vast.AI terminal script: ConvNeXtBase Round3 on GPU X.
# Edit these three lines before running.
HF_USERNAME_OR_ORG="USERNAME"
HF_DATASET_REPO_ID="USERNAME/DATASET_REPOSITORY"
ZIP_FILENAME="DATASET_ARCHIVE.zip"

CUDA_VISIBLE_DEVICES=X python Train_VastAI_HF_Zip_DDP.py \
  --hf_dataset_repo_id "$HF_DATASET_REPO_ID" \
  --hf_dataset_repo_type dataset \
  --zip_filename "$ZIP_FILENAME" \
  --download_dir /dev/shm/hf_downloads \
  --extract_dir /dev/shm/hf_zip_datasets \
  --output_dir /dev/shm/results \
  --hf_results_repo_id "$HF_USERNAME_OR_ORG/RESULTS_REPOSITORY" \
  --hf_results_repo_type model \
  --hf_results_path_in_repo "ConvNeXtBase/Round3" \
  --round_name "PBC_MULTICLASS_ConvNeXtBase_Round3" \
  --model_name convnextv2_base.fcmae_ft_in1k \
  --seed X \
  --epochs X \
  --batch_size X \
  --learning_rate X \
  --num_workers 12 \
  --train_samples 0 \
  --val_samples 0 \
  --test_samples 0 \
  --use_amp true \
  --upload_to_hf true \
  --delete_zip_after_extract true
