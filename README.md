# Peripheral Blood Cell Multiclass Classification

This repository contains the code organization used for a peripheral blood cell image-classification experiment from the MedIMeta dataset (2025). The dataset follows the original train/validation/test splits. Large files such as images, checkpoints, hyperparameters, downloaded archives, local logs, and full server result folders are intentionally not included.

## Repository Layout

```text
preprocessing/
VS_training_trial/
server_training/
statistical_analysis/
XAI/
```


## Hugging Face Resources
- Original Dataset: https://www.woerner.eu/projects/medimeta/
- Preprocessed Dataset: [MedIMeta-PBC ImageFolder split](https://huggingface.co/datasets/ayeshaamin/v4_final_split)
- Trained results and run artifacts: [MedIMetaPBCTraining](https://huggingface.co/ayeshaamin/MedIMetaPBCTraining)

## Training Overview

The final experiments were trained on a Vast.AI server with **4x NVIDIA GeForce RTX 4070 Ti SUPER** GPUs. Runs were launched in parallel from terminal scripts, with one model run assigned to one visible CUDA device. Temporary dataset extraction and intermediate outputs were kept off the small workspace disk during server training.

Three pretrained image-classification model families were evaluated. Each model family was trained across **5 independent runs**, giving 15 final runs for statistical comparison. Exact private repository paths and selected training hyperparameters have been replaced with `X` or generic placeholders in this public copy.

## Statistical Analysis

The statistical analysis folder contains the ANOVA workflow and selected summary outputs. Detailed per-run metadata was intentionally omitted from this public copy. The analysis compared final test metrics across the three model families using the repeated runs as replicates. In the saved ANOVA summary, the primary performance metrics did not show statistically significant differences at `p < 0.05`, but runtime differed across model families.

## XAI

The XAI folder contains Grad-CAM scripts and combined example figures. Model weights and detailed local prediction CSVs are not included. Each figure displays the original image, Grad-CAM heatmap, and overlay for the best run of each model family on representative correctly classified test images.

## Notes

- Hugging Face repository names are placeholders where needed.
- Hyperparameter values are intentionally masked in this public-facing copy.
- The code is meant as an experiment record
