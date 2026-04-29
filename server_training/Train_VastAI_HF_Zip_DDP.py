#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pbc-training")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from timm import create_model, list_models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


DEFAULT_MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"
DEFAULT_ROUND_NAME = "PBC_MULTICLASS_ConvNeXtV2Base_Server_Round1"
NUM_CLASSES = 8
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvNeXtV2 on a Hugging Face ZIP ImageFolder dataset."
    )
    parser.add_argument("--hf_dataset_repo_id", type=str, default="CHANGE_ME/DATASET_REPO")
    parser.add_argument("--hf_dataset_repo_type", type=str, default="dataset")
    parser.add_argument("--zip_filename", type=str, default="")
    parser.add_argument("--download_dir", type=Path, default=Path("data/hf_downloads"))
    parser.add_argument("--extract_dir", type=Path, default=Path("data/hf_zip_datasets"))
    parser.add_argument(
        "--data_subdir",
        type=str,
        default="",
        help="Optional path inside the extracted ZIP that contains train/ val/ test/.",
    )
    parser.add_argument("--force_download", type=str_to_bool, default=False)
    parser.add_argument("--force_extract", type=str_to_bool, default=False)
    parser.add_argument(
        "--delete_zip_after_extract",
        type=str_to_bool,
        default=False,
        help="Useful when SSD is tight. Keeps extracted data but deletes local ZIP copy.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--round_name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--use_amp", type=str_to_bool, default=True)
    parser.add_argument("--upload_to_hf", type=str_to_bool, default=True)
    parser.add_argument("--hf_results_repo_id", type=str, default="CHANGE_ME/RESULTS_REPO")
    parser.add_argument("--hf_results_repo_type", type=str, default="model")
    parser.add_argument(
        "--hf_results_path_in_repo",
        type=str,
        default="",
        help=(
            "Folder inside the Hugging Face results repo. Example: "
            "ConvNeXtBase/Round1. If blank, uses the local output folder name."
        ),
    )
    return parser.parse_args()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def print_main(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def validate_placeholder_args(args: argparse.Namespace) -> None:
    if args.hf_dataset_repo_id.startswith("CHANGE_ME"):
        raise ValueError("Set --hf_dataset_repo_id to your Hugging Face dataset repo.")
    if args.upload_to_hf and args.hf_results_repo_id.startswith("CHANGE_ME"):
        raise ValueError("Set --hf_results_repo_id or run with --upload_to_hf false.")
    for split_name, sample_count in {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }.items():
        if 0 < sample_count < NUM_CLASSES:
            raise ValueError(f"--{split_name}_samples must be 0 or at least {NUM_CLASSES}.")


def choose_zip_file(repo_id: str, repo_type: str, zip_filename: str) -> str:
    if zip_filename.strip():
        return zip_filename.strip()

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    zip_files = sorted(path for path in files if path.lower().endswith(".zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip file found in {repo_type} repo {repo_id}.")
    if len(zip_files) == 1:
        print_main(f"Found ZIP file in repo: {zip_files[0]}")
        return zip_files[0]
    raise ValueError(
        "Multiple ZIP files found. Pass --zip_filename explicitly. Options:\n"
        + "\n".join(f"  - {name}" for name in zip_files)
    )


def download_zip(args: argparse.Namespace, zip_filename: str) -> Path:
    args.download_dir.mkdir(parents=True, exist_ok=True)
    print_main(
        f"Downloading {zip_filename} from {args.hf_dataset_repo_id} "
        f"({args.hf_dataset_repo_type})"
    )
    downloaded = hf_hub_download(
        repo_id=args.hf_dataset_repo_id,
        repo_type=args.hf_dataset_repo_type,
        filename=zip_filename,
        local_dir=args.download_dir,
        force_download=args.force_download,
    )
    return Path(downloaded)


def safe_extract_zip(zip_path: Path, extract_dir: Path, force_extract: bool) -> Path:
    target_dir = extract_dir / zip_path.stem
    complete_marker = target_dir / ".extract_complete"
    lock_dir = extract_dir / f"{zip_path.stem}.extract.lock"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and complete_marker.exists() and not force_extract:
        print_main(f"Using existing extracted dataset: {target_dir}")
        return target_dir

    while True:
        try:
            lock_dir.mkdir()
            lock_acquired = True
            break
        except FileExistsError:
            lock_acquired = False
            if target_dir.exists() and complete_marker.exists() and not force_extract:
                print_main(f"Using existing extracted dataset: {target_dir}")
                return target_dir
            print_main(f"Waiting for dataset extraction lock: {lock_dir}")
            time.sleep(10)

    try:
        if force_extract and target_dir.exists():
            shutil.rmtree(target_dir)
        if target_dir.exists() and complete_marker.exists():
            print_main(f"Using existing extracted dataset: {target_dir}")
            return target_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        print_main(f"Extracting {zip_path} to {target_dir}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if "__MACOSX" in member_path.parts or member_path.name.startswith("._"):
                    continue
                destination = target_dir / member.filename
                resolved_destination = destination.resolve()
                resolved_target = target_dir.resolve()
                if not str(resolved_destination).startswith(str(resolved_target)):
                    raise ValueError(f"Unsafe ZIP path blocked: {member.filename}")
                archive.extract(member, target_dir)
        complete_marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        return target_dir
    finally:
        if lock_acquired:
            try:
                lock_dir.rmdir()
            except OSError:
                pass


def contains_imagefolder_split(path: Path) -> bool:
    if "__MACOSX" in path.parts or path.name.startswith("."):
        return False
    split_dirs = [path / "train", path / "val", path / "test"]
    if not all(split.is_dir() for split in split_dirs):
        return False
    return all(any(child.is_dir() for child in split.iterdir()) for split in split_dirs)


def find_data_dir(extracted_root: Path, data_subdir: str) -> Path:
    if data_subdir.strip():
        candidate = extracted_root / data_subdir.strip()
        if contains_imagefolder_split(candidate):
            return candidate
        raise FileNotFoundError(f"{candidate} does not contain train/, val/, and test/.")

    if contains_imagefolder_split(extracted_root):
        return extracted_root

    matches = [
        path
        for path in sorted(extracted_root.rglob("*"))
        if path.is_dir()
        and "__MACOSX" not in path.parts
        and not path.name.startswith(".")
        and contains_imagefolder_split(path)
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find train/, val/, and test/ inside {extracted_root}.")
    print_main(f"Found ImageFolder split at: {matches[0]}")
    return matches[0]


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, str]:
    zip_filename = choose_zip_file(
        repo_id=args.hf_dataset_repo_id,
        repo_type=args.hf_dataset_repo_type,
        zip_filename=args.zip_filename,
    )
    zip_path = download_zip(args, zip_filename)
    extracted_root = safe_extract_zip(zip_path, args.extract_dir, args.force_extract)
    if args.delete_zip_after_extract and zip_path.exists():
        zip_path.unlink()
        print_main(f"Deleted local ZIP copy after extraction: {zip_path}")
    data_dir = find_data_dir(extracted_root, args.data_subdir)
    return data_dir, zip_filename


def is_valid_image_file(path: str) -> bool:
    image_path = Path(path)
    if image_path.name.startswith("._") or image_path.name.startswith("."):
        return False
    if "__MACOSX" in image_path.parts:
        return False
    return image_path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_balanced_subset(
    dataset: datasets.ImageFolder,
    split_name: str,
    max_samples: int,
    seed: int,
) -> Dataset:
    if max_samples < 0:
        raise ValueError(f"{split_name}: sample count must be 0 or positive.")
    if max_samples == 0:
        print_main(f"{split_name}: using all {len(dataset)} images")
        return dataset
    if max_samples > len(dataset):
        raise ValueError(
            f"{split_name}: requested {max_samples} images, but only found {len(dataset)}."
        )

    rng = random.Random(seed + {"train": 11, "val": 17, "test": 23}[split_name])
    targets = np.array(dataset.targets)
    desired_total = min(max_samples, len(dataset))
    base_per_class = desired_total // NUM_CLASSES
    remainder = desired_total % NUM_CLASSES
    selected: list[int] = []
    unused_by_class: dict[int, list[int]] = {}

    for class_index in range(NUM_CLASSES):
        class_indices = np.where(targets == class_index)[0].tolist()
        rng.shuffle(class_indices)
        quota = base_per_class + (1 if class_index < remainder else 0)
        take = min(quota, len(class_indices))
        selected.extend(class_indices[:take])
        unused_by_class[class_index] = class_indices[take:]

    if len(selected) < desired_total:
        extras: list[int] = []
        for class_index in range(NUM_CLASSES):
            extras.extend(unused_by_class[class_index])
        rng.shuffle(extras)
        selected.extend(extras[: desired_total - len(selected)])

    rng.shuffle(selected)
    print_main(f"{split_name}: using {len(selected)} of {len(dataset)} images")
    return Subset(dataset, selected)


def get_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, datasets.ImageFolder):
        return np.array(dataset.targets)
    if isinstance(dataset, Subset):
        parent_targets = get_targets(dataset.dataset)
        return parent_targets[np.array(dataset.indices)]
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def get_sample_count(dataset: Dataset) -> int:
    return len(dataset)


def compute_class_counts(dataset: Dataset, class_names: list[str]) -> dict[str, int]:
    targets = get_targets(dataset)
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    return {class_name: int(counts[index]) for index, class_name in enumerate(class_names)}


def build_datasets(
    data_dir: Path,
    seed: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> tuple[dict[str, Dataset], list[str]]:
    train_transform, eval_transform = build_transforms()
    base_datasets = {
        "train": datasets.ImageFolder(
            data_dir / "train",
            transform=train_transform,
            is_valid_file=is_valid_image_file,
        ),
        "val": datasets.ImageFolder(
            data_dir / "val",
            transform=eval_transform,
            is_valid_file=is_valid_image_file,
        ),
        "test": datasets.ImageFolder(
            data_dir / "test",
            transform=eval_transform,
            is_valid_file=is_valid_image_file,
        ),
    }
    class_names = base_datasets["train"].classes
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(class_names)}: {class_names}")
    for split in ["val", "test"]:
        if base_datasets[split].classes != class_names:
            raise ValueError(f"{split}/ class folders do not match train/.")

    datasets_by_split = {
        "train": make_balanced_subset(base_datasets["train"], "train", train_samples, seed),
        "val": make_balanced_subset(base_datasets["val"], "val", val_samples, seed),
        "test": make_balanced_subset(base_datasets["test"], "test", test_samples, seed),
    }
    return datasets_by_split, class_names


def build_dataloaders(
    datasets_by_split: dict[str, Dataset],
    batch_size: int,
    num_workers: int,
    seed: int,
    distributed: bool,
) -> tuple[dict[str, DataLoader], DistributedSampler | None]:
    train_sampler = (
        DistributedSampler(
            datasets_by_split["train"],
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=generator if train_sampler is None else None,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
    }
    return dataloaders, train_sampler


def compute_class_weights(train_dataset: Dataset, device: torch.device, class_names: list[str]) -> torch.Tensor:
    targets = get_targets(train_dataset)
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        missing = [class_names[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training split has empty class folder(s): {missing}")
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(model_name: str, device: torch.device) -> nn.Module:
    alternatives = list_models("*", pretrained=True)
    if model_name not in alternatives:
        alternative_text = "\n".join(f"  - {name}" for name in alternatives)
        raise RuntimeError(
            f'Pretrained timm model "{model_name}" is not available.\n'
            f"Available pretrained timm models in this environment:\n{alternative_text}"
        )
    model = create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


def reduce_train_sums(loss_sum: float, correct: int, total: int, device: torch.device) -> tuple[float, int, int]:
    values = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item()), int(values[2].item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    iterator = dataloader
    if is_main_process():
        iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

    for inputs, labels in iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += batch_size

    running_loss, correct, total = reduce_train_sums(running_loss, correct, total, device)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[list[float]] = []

    progress = tqdm(dataloader, desc=f"[{split_name}]", leave=False)
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        probabilities = torch.softmax(outputs.float(), dim=1)
        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        y_proba.extend(probabilities.cpu().numpy().tolist())

    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    y_proba_array = np.array(y_proba)
    return {
        "loss": running_loss / len(dataloader.dataset),
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "macro_f1": f1_score(y_true_array, y_pred_array, average="macro", zero_division=0),
        "y_true": y_true_array,
        "y_pred": y_pred_array,
        "y_proba": y_proba_array,
    }


def model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return model.module.state_dict() if isinstance(model, DDP) else model.state_dict()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    class_names: list[str],
    args_dict: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": args_dict["model_name"],
            "model_state_dict": model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "class_names": class_names,
            "class_to_idx": {name: index for index, name in enumerate(class_names)},
            "args": args_dict,
        },
        path,
    )


def plot_training_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train accuracy")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=300)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    normalize: bool,
) -> None:
    plt.figure(figsize=(11, 9))
    display_matrix = matrix.astype(float) if normalize else matrix
    fmt = ".2f" if normalize else "d"
    plt.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    threshold = np.nanmax(display_matrix) / 2.0 if display_matrix.size else 0.0
    for row in range(display_matrix.shape[0]):
        for col in range(display_matrix.shape[1]):
            value = display_matrix[row, col]
            color = "white" if value > threshold else "black"
            plt.text(col, row, format(value, fmt), ha="center", va="center", color=color, fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def calculate_specificity(cm: np.ndarray, class_names: list[str]) -> tuple[dict[str, float], float]:
    specificity: dict[str, float] = {}
    total = cm.sum()
    for class_index, class_name in enumerate(class_names):
        tp = cm[class_index, class_index]
        fp = cm[:, class_index].sum() - tp
        fn = cm[class_index, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificity[f"specificity_{class_name}"] = float(tn / denom) if denom else float("nan")
    return specificity, float(np.nanmean(list(specificity.values())))


def calculate_roc_auc_and_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> tuple[float | None, str | None]:
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
        roc_auc = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))

        fpr: dict[int, np.ndarray] = {}
        tpr: dict[int, np.ndarray] = {}
        auc_by_class: dict[int, float] = {}
        for class_index in range(NUM_CLASSES):
            fpr[class_index], tpr[class_index], _ = roc_curve(
                y_true_bin[:, class_index], y_proba[:, class_index]
            )
            auc_by_class[class_index] = float(
                roc_auc_score(y_true_bin[:, class_index], y_proba[:, class_index])
            )
        all_fpr = np.unique(np.concatenate([fpr[index] for index in range(NUM_CLASSES)]))
        mean_tpr = np.zeros_like(all_fpr)
        for class_index in range(NUM_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr[class_index], tpr[class_index])
        mean_tpr /= NUM_CLASSES
        macro_auc = float(roc_auc_score(y_true_bin, y_proba, average="macro"))

        plt.figure(figsize=(9, 7))
        for class_index, class_name in enumerate(class_names):
            plt.plot(
                fpr[class_index],
                tpr[class_index],
                linewidth=1.5,
                label=f"{class_name} AUC={auc_by_class[class_index]:.3f}",
            )
        plt.plot(
            all_fpr,
            mean_tpr,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=f"Macro-average AUC={macro_auc:.3f}",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass One-vs-Rest ROC Curves")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return roc_auc, None
    except Exception as exc:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"ROC-AUC could not be calculated:\n{exc}", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return None, str(exc)


def calculate_test_metrics(
    eval_result: dict[str, Any],
    class_names: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]
    y_proba = eval_result["y_proba"]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    specificity_by_class, macro_specificity = calculate_specificity(cm, class_names)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    roc_auc, roc_error = calculate_roc_auc_and_plot(
        y_true, y_proba, class_names, output_dir / "roc_curve.png"
    )

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png", "Confusion Matrix", False)
    plot_confusion_matrix(
        cm_normalized,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix",
        True,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    metrics = {
        "test_loss": float(eval_result["loss"]),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_specificity": float(macro_specificity),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "multiclass_roc_auc_ovr_macro": float(roc_auc) if roc_auc is not None else np.nan,
        "roc_auc_error": roc_error or "",
        "confusion_matrix": json.dumps(cm.tolist()),
        "confusion_matrix_normalized": json.dumps(cm_normalized.tolist()),
    }
    metrics.update(specificity_by_class)
    return metrics, report


def serialize_args(
    args: argparse.Namespace,
    data_dir: Path,
    zip_filename: str,
    class_names: list[str],
    datasets_by_split: dict[str, Dataset],
    world_size: int,
) -> dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "model_name": args.model_name,
        "round": args.round_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size,
        "effective_batch_size": args.batch_size * world_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss with training-set class weights",
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": json.dumps(class_names),
        "class_counts_train": json.dumps(compute_class_counts(datasets_by_split["train"], class_names)),
        "class_counts_val": json.dumps(compute_class_counts(datasets_by_split["val"], class_names)),
        "class_counts_test": json.dumps(compute_class_counts(datasets_by_split["test"], class_names)),
        "train_samples_requested": args.train_samples,
        "val_samples_requested": args.val_samples,
        "test_samples_requested": args.test_samples,
        "train_samples_actual": get_sample_count(datasets_by_split["train"]),
        "val_samples_actual": get_sample_count(datasets_by_split["val"]),
        "test_samples_actual": get_sample_count(datasets_by_split["test"]),
        "hf_dataset_repo_id": args.hf_dataset_repo_id,
        "hf_dataset_repo_type": args.hf_dataset_repo_type,
        "zip_filename": zip_filename,
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
        "use_amp": args.use_amp,
        "world_size": world_size,
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_names": json.dumps(gpu_names),
        "num_workers": args.num_workers,
        "upload_to_hf": args.upload_to_hf,
        "hf_results_repo_id": args.hf_results_repo_id,
        "hf_results_repo_type": args.hf_results_repo_type,
        "hf_results_path_in_repo": args.hf_results_path_in_repo or args.round_name,
    }


def save_args_yaml(args_dict: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "args.yaml").open("w", encoding="utf-8") as handle:
        for key, value in args_dict.items():
            if isinstance(value, bool):
                scalar = "true" if value else "false"
            elif isinstance(value, (int, float)):
                scalar = str(value)
            else:
                scalar = json.dumps(str(value))
            handle.write(f"{key}: {scalar}\n")


def save_text_reports(
    output_dir: Path,
    test_metrics: dict[str, Any],
    classification_report_text: str,
    best_epoch: int,
    args_dict: dict[str, Any],
) -> None:
    with (output_dir / "classification_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("Classification report\n")
        handle.write("=====================\n\n")
        handle.write(classification_report_text)
        handle.write("\n")

    with (output_dir / "testing_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("Testing metrics from best validation checkpoint\n")
        handle.write("===============================================\n\n")
        for key, value in test_metrics.items():
            handle.write(f"{key}: {value}\n")

    with (output_dir / "README_run_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("MedIMeta-PBC ConvNeXtV2 Server Run Summary\n")
        handle.write("==========================================\n\n")
        for key in [
            "model_name",
            "round",
            "seed",
            "epochs",
            "batch_size_per_gpu",
            "effective_batch_size",
            "learning_rate",
            "optimizer",
            "loss",
            "hf_dataset_repo_id",
            "zip_filename",
            "world_size",
            "gpu_names",
        ]:
            handle.write(f"{key}: {args_dict.get(key)}\n")
        handle.write(f"best_epoch_by_validation_macro_f1: {best_epoch}\n\n")
        handle.write("Main test metrics\n")
        handle.write("-----------------\n")
        for key in [
            "test_loss",
            "test_accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "macro_specificity",
            "matthews_correlation_coefficient",
            "cohens_kappa",
            "multiclass_roc_auc_ovr_macro",
        ]:
            handle.write(f"{key}: {test_metrics.get(key)}\n")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def upload_results(output_dir: Path, repo_id: str, repo_type: str, path_in_repo: str) -> None:
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
    )


def main() -> None:
    args = parse_args()
    validate_placeholder_args(args)
    device, rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed + rank)

    try:
        if is_main_process():
            print(f"Using device: {device}")
            print(f"World size: {world_size}")
            if torch.cuda.is_available():
                print(f"Visible CUDA devices: {torch.cuda.device_count()}")
                for index in range(torch.cuda.device_count()):
                    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

        if is_main_process():
            data_dir, zip_filename = prepare_dataset(args)
        else:
            data_dir, zip_filename = None, None

        if is_dist():
            object_list: list[Any] = [data_dir, zip_filename]
            dist.broadcast_object_list(object_list, src=0)
            data_dir, zip_filename = object_list

        assert isinstance(data_dir, Path)
        assert isinstance(zip_filename, str)

        datasets_by_split, class_names = build_datasets(
            data_dir=data_dir,
            seed=args.seed,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
        )
        dataloaders, train_sampler = build_dataloaders(
            datasets_by_split=datasets_by_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            distributed=world_size > 1,
        )

        output_dir = args.output_dir / args.round_name
        weights_dir = output_dir / "weights"
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            weights_dir.mkdir(parents=True, exist_ok=True)

        args_dict = serialize_args(
            args=args,
            data_dir=data_dir,
            zip_filename=zip_filename,
            class_names=class_names,
            datasets_by_split=datasets_by_split,
            world_size=world_size,
        )
        if is_main_process():
            save_args_yaml(args_dict, output_dir)
            print(f"Classes ({len(class_names)}): {class_names}")
            print(f"Output folder: {output_dir}")

        class_weights = compute_class_weights(datasets_by_split["train"], device, class_names)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model = build_model(args.model_name, device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

        history: list[dict[str, Any]] = []
        best_val_macro_f1 = -1.0
        best_epoch = 0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                epochs=args.epochs,
                use_amp=args.use_amp,
            )

            if is_main_process():
                val_result = evaluate(
                    model=model.module if isinstance(model, DDP) else model,
                    dataloader=dataloaders["val"],
                    criterion=criterion,
                    device=device,
                    split_name=f"epoch {epoch}/{args.epochs} val",
                    use_amp=args.use_amp,
                )
                row = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_accuracy),
                    "val_loss": float(val_result["loss"]),
                    "val_accuracy": float(val_result["accuracy"]),
                    "val_macro_f1": float(val_result["macro_f1"]),
                    "epoch_time_minutes": float((time.time() - start_time) / 60.0),
                    "model_name": args.model_name,
                    "round": args.round_name,
                    "seed": args.seed,
                    "batch_size_per_gpu": args.batch_size,
                    "effective_batch_size": args.batch_size * world_size,
                    "learning_rate": args.learning_rate,
                }
                history.append(row)
                pd.DataFrame(history).to_csv(output_dir / "training_validation_metrics.csv", index=False)
                plot_training_curves(history, output_dir)
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
                    f"val_loss={val_result['loss']:.4f} val_acc={val_result['accuracy']:.4f} "
                    f"val_macro_f1={val_result['macro_f1']:.4f}"
                )
                if val_result["macro_f1"] > best_val_macro_f1:
                    best_val_macro_f1 = float(val_result["macro_f1"])
                    best_epoch = epoch
                    save_checkpoint(
                        weights_dir / "best_model.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val_macro_f1=best_val_macro_f1,
                        class_names=class_names,
                        args_dict=args_dict,
                    )
                    print(f"Saved new best model at epoch {epoch}")

            if is_dist():
                dist.barrier()

        if is_main_process():
            save_checkpoint(
                weights_dir / "last_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=args.epochs,
                best_val_macro_f1=best_val_macro_f1,
                class_names=class_names,
                args_dict=args_dict,
            )
            print(f"Saved last model to {weights_dir / 'last_model.pt'}")

            best_checkpoint = load_checkpoint(weights_dir / "best_model.pt", device)
            eval_model = build_model(args.model_name, device)
            eval_model.load_state_dict(best_checkpoint["model_state_dict"])
            test_result = evaluate(
                model=eval_model,
                dataloader=dataloaders["test"],
                criterion=criterion,
                device=device,
                split_name="test",
                use_amp=args.use_amp,
            )
            test_metrics, report = calculate_test_metrics(test_result, class_names, output_dir)
            summary_row = dict(args_dict)
            summary_row["best_epoch"] = best_epoch
            summary_row["best_val_macro_f1"] = best_val_macro_f1
            summary_row["total_runtime_minutes"] = float((time.time() - start_time) / 60.0)
            summary_row.update(test_metrics)
            pd.DataFrame([summary_row]).to_csv(output_dir / "metrics_summary.csv", index=False)
            save_text_reports(output_dir, test_metrics, report, best_epoch, args_dict)

            print("Final test metrics from best validation checkpoint:")
            for key, value in test_metrics.items():
                if key not in {"confusion_matrix", "confusion_matrix_normalized"}:
                    print(f"  {key}: {value}")

            if args.upload_to_hf:
                print(f"Uploading {output_dir} to Hugging Face repo {args.hf_results_repo_id}")
                upload_results(
                    output_dir=output_dir,
                    repo_id=args.hf_results_repo_id,
                    repo_type=args.hf_results_repo_type,
                    path_in_repo=args.hf_results_path_in_repo or args.round_name,
                )
                print("Hugging Face upload complete.")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
