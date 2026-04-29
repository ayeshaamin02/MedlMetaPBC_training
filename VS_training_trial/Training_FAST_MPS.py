#!/usr/bin/env python3
from __future__ import annotations

import sys

from Training_VS import main


if __name__ == "__main__":
    sys.argv = [
        "Training_VS.py",
        "--data_dir",
        "data/IMAGEFOLDER_SPLIT",
        "--output_dir",
        "results",
        "--round_name",
        "PBC_MULTICLASS_ConvNeXtV2Atto_FAST_MPS",
        "--model_name",
        "convnextv2_atto.fcmae",
        "--seed",
        "101",
        "--epochs",
        "1",
        "--batch_size",
        "16",
        "--lr",
        "0.0001",
        "--stage",
        "all",
        "--train_samples",
        "1000",
        "--val_samples",
        "500",
        "--test_samples",
        "500",
        "--num_workers",
        "0",
        "--upload_to_hf",
        "false",
    ]
    main()
