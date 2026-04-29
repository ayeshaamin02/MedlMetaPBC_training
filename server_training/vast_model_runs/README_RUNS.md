
These are simple terminal scripts for each model to use in jupyter terminal of Vast.AI. The following example follows ConvNeXtBase, replace with EfficientNetV2S and SwinBase for their respective terminal (bash) runs. 

Note: All hyperparameters in the .sh files need to be manually set before running in terminal. 

## Edit This In Each Script

At the top of every `.sh` file, change to HF links:

```bash
HF_USERNAME_OR_ORG="USERNAME"
HF_DATASET_REPO_ID="USERNAME/DATASET_REPOSITORY"
ZIP_FILENAME="DATASET_ARCHIVE.zip"
```

## Hugging Face Save Structure

The script creates/uses this model repo:

```text
YOUR_USERNAME/RESULTS_REPOSITORY
```

Inside it, results upload to:

```text
ConvNeXtBase/Round1
ConvNeXtBase/Round2
ConvNeXtBase/Round3
ConvNeXtBase/Round4
ConvNeXtBase/Round5
```

## Run Four Rounds At The Same Time

From the project folder:

```bash
mkdir -p logs

bash vast_model_runs/ConvNeXtBase/run_round1.sh > logs/round1_gpu0.log 2>&1 &
bash vast_model_runs/ConvNeXtBase/run_round2.sh > logs/round2_gpu1.log 2>&1 &
bash vast_model_runs/ConvNeXtBase/run_round3.sh > logs/round3_gpu2.log 2>&1 &
bash vast_model_runs/ConvNeXtBase/run_round4.sh > logs/round4_gpu3.log 2>&1 &

wait
```

Then run the final fifth round:

```bash
bash vast_model_runs/ConvNeXtBase/run_round5.sh > logs/round5_gpu0.log 2>&1 &

wait
```

## Monitor

```bash
tail -f logs/round1_gpu0.log
nvidia-smi
```

## CUDA Assignment

Each file directly assigns CUDA, which can be changed depending on which GPU is free:

```bash
CUDA_VISIBLE_DEVICES=X
CUDA_VISIBLE_DEVICES=X
CUDA_VISIBLE_DEVICES=X
CUDA_VISIBLE_DEVICES=X
```

So one process uses one GPU. No `torchrun` is needed for these repeated statistical runs.

## Storage Safety On Vast.AI

The ConvNeXtBase scripts write downloads, extracted data, and results to `/dev/shm` instead of `/workspace` to save SSD space on the server:

```bash
--download_dir /dev/shm/hf_downloads
--extract_dir /dev/shm/hf_zip_datasets
--output_dir /dev/shm/results
--delete_zip_after_extract true
```

This avoids filling the small 16GB workspace overlay. Results still upload to Hugging Face.
