#!/usr/bin/env bash
set -e


HF_USERNAME_OR_ORG="USERNAME"
HF_DATASET_REPO_ID="USERNAME/DATASET_REPOSITORY"
ZIP_FILENAME="DATASET_ARCHIVE.zip"

CUDA_VISIBLE_DEVICES=X python Train_VastAI_HF_Zip_DDP.py \
  --hf_dataset_repo_id "USERNAME/DATASET_REPOSITORY" \
  --hf_dataset_repo_type dataset \
  --zip_filename "DATASET_ARCHIVE.zip" \
  --download_dir /dev/shm/hf_downloads \
  --extract_dir /dev/shm/hf_zip_datasets \
  --output_dir /dev/shm/results \
  --hf_results_repo_id "USERNAME/RESULTS_REPOSITORY" \
  --hf_results_repo_type model \
  --hf_results_path_in_repo "EfficientNetV2S/Round3" \
  --round_name "PBC_MULTICLASS_EfficientNetV2S_Round3" \
  --model_name tf_efficientnetv2_s.in21k_ft_in1k \
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
