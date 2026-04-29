#!/usr/bin/env python3
"""
Train one ANOVA round for MedIMeta-PBC multiclass image classification.

Model: ConvNeXtV2 Base via timm:
    timm.create_model("convnextv2_base.fcmae", pretrained=True, num_classes=8)

This file is written with VS Code/Jupyter-style "# %%" cells. You can press
"Run Cell" section-by-section in VS Code now, and later run the same file from
the terminal for server training.
"""

# %% Imports and constants
from __future__ import annotations

import argparse
import json
import osj
import random
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import torch.nn as nn
    import torch.optim as optim
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
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import datasets, transforms
    from tqdm.auto import tqdm
except ModuleNotFoundError as exc:
    missing_package = exc.name or "an unknown package"
    raise ModuleNotFoundError(
        f"Missing Python package: {missing_package}. Install the dependencies from "
        "requirements_pbc_round1.txt in your VS Code/Jupyter environment, then rerun."
    ) from exc


MODEL_NAME = "convnextv2_base.fcmae"
NUM_CLASSES = 8
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_ROUND_NAME = "PBC_MULTICLASS_ConvNeXtV2Base_Round1"


# %% Argument parsing
def str_to_bool(value: str | bool) -> bool:
    """Parse common command-line boolean strings."""
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
        description="Train ConvNeXtV2 Base for one MedIMeta-PBC ANOVA round."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/IMAGEFOLDER_SPLIT"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--round_name",
        type=str,
        default=DEFAULT_ROUND_NAME,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "train", "test"],
        default="all",
        help="Run training plus test, training only, or test only from best_model.pt.",
    )
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="0 is safest in VS Code/Jupyter on macOS. Increase later for terminal/server runs.",
    )
    parser.add_argument("--upload_to_hf", type=str_to_bool, default=False)
    parser.add_argument("--hf_repo_id", type=str, default=os.environ.get("HF_REPO_ID", ""))
    parser.add_argument("--hf_repo_type", type=str, default=os.environ.get("HF_REPO_TYPE", "model"))
    args = parser.parse_args()
    args.hf_repo_id = args.hf_repo_id.strip()
    args.hf_repo_type = args.hf_repo_type.strip() or "model"
    return args


# %% Reproducibility and folder checks
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)


def validate_data_dir(data_dir: Path) -> None:
    expected_splits = ["train", "val", "test"]
    missing = [split for split in expected_splits if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing required split folder(s) in {data_dir}: {missing}. "
            "Expected train/, val/, and test/ ImageFolder directories."
        )


# %% Transforms and dataloaders
def build_transforms(image_size: int = IMAGE_SIZE) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
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


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
    samples_by_split: dict[str, int],
    num_workers: int,
) -> tuple[dict[str, DataLoader], dict[str, Dataset], list[str]]:
    train_transform, eval_transform = build_transforms(IMAGE_SIZE)

    base_datasets = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=eval_transform),
        "test": datasets.ImageFolder(data_dir / "test", transform=eval_transform),
    }

    class_names = base_datasets["train"].classes
    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes in train split, found {len(class_names)}: "
            f"{class_names}"
        )

    for split in ["val", "test"]:
        if base_datasets[split].classes != class_names:
            raise ValueError(
                f"Class folders in {split}/ do not match train/. "
                f"train={class_names}, {split}={base_datasets[split].classes}"
            )

    datasets_by_split: dict[str, Dataset] = {}
    for split, dataset in base_datasets.items():
        datasets_by_split[split] = make_balanced_subset(
            dataset=dataset,
            split_name=split,
            max_samples=samples_by_split[split],
            seed=seed,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            generator=generator,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        ),
    }
    return dataloaders, datasets_by_split, class_names


def make_balanced_subset(
    dataset: datasets.ImageFolder,
    split_name: str,
    max_samples: int,
    seed: int,
) -> Dataset:
    """Return a deterministic, roughly class-balanced subset for a small run."""
    if max_samples < 0:
        raise ValueError(f"{split_name}: sample count must be 0 or positive.")
    if max_samples == 0:
        print(f"{split_name}: using all {len(dataset)} images")
        return dataset
    if max_samples > len(dataset):
        raise ValueError(
            f"{split_name}: requested {max_samples} images, but only found "
            f"{len(dataset)} images in the split."
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

    # If a class had fewer images than its quota, fill the remaining slots
    # from other classes while keeping deterministic ordering.
    if len(selected) < desired_total:
        extras: list[int] = []
        for class_index in range(NUM_CLASSES):
            extras.extend(unused_by_class[class_index])
        rng.shuffle(extras)
        selected.extend(extras[: desired_total - len(selected)])

    rng.shuffle(selected)
    print(
        f"{split_name}: using {len(selected)} of {len(dataset)} images "
        f"(requested {max_samples})"
    )
    return Subset(dataset, selected)


def get_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, datasets.ImageFolder):
        return np.array(dataset.targets)
    if isinstance(dataset, Subset):
        parent_targets = get_targets(dataset.dataset)
        return parent_targets[np.array(dataset.indices)]
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)}")


# %% Class weights and model creation
def compute_class_weights(
    train_dataset: Dataset,
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    targets = get_targets(train_dataset)
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        zero_classes = [
            class_names[index] for index, count in enumerate(counts) if count == 0
        ]
        raise ValueError(f"Training split has empty class folder(s): {zero_classes}")

    total = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_convnextv2_pretrained_alternatives() -> list[str]:
    return list_models("*convnextv2*", pretrained=True)


def build_model(device: torch.device, pretrained: bool) -> nn.Module:
    if pretrained:
        alternatives = get_convnextv2_pretrained_alternatives()
        if MODEL_NAME not in alternatives:
            alternative_text = "\n".join(f"  - {name}" for name in alternatives)
            if not alternative_text:
                alternative_text = (
                    "  No pretrained ConvNeXtV2 alternatives found in this timm install."
                )
            raise RuntimeError(
                f'Pretrained timm model "{MODEL_NAME}" is not available in this '
                "environment.\n"
                f"Available pretrained ConvNeXtV2 alternatives:\n{alternative_text}"
            )
    try:
        model = create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)
    except Exception as exc:
        alternatives = list_models("*convnextv2*", pretrained=True)
        alternative_text = "\n".join(f"  - {name}" for name in alternatives)
        if not alternative_text:
            alternative_text = "  No pretrained ConvNeXtV2 alternatives found in this timm install."
        pretrained_text = "pretrained " if pretrained else ""
        raise RuntimeError(
            f'Could not create {pretrained_text}timm model "{MODEL_NAME}".\n'
            f"Available pretrained ConvNeXtV2 alternatives:\n{alternative_text}\n"
            f"Original timm error: {exc}"
        ) from exc
    return model.to(device)


# %% Training and validation loops
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(
            loss=running_loss / max(total, 1),
            acc=correct / max(total, 1),
        )

    return running_loss / total, correct / total


# %% Evaluation helpers
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
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

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
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


# %% Checkpoint saving
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    args_dict = serialize_args(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "class_names": class_names,
            "class_to_idx": {name: index for index, name in enumerate(class_names)},
            "args": args_dict,
        },
        path,
    )


# %% Plotting helpers
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
    normalize: bool = False,
) -> None:
    plt.figure(figsize=(11, 9))
    display_matrix = matrix.astype(float) if normalize else matrix
    fmt = ".2f" if normalize else "d"
    plt.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = np.nanmax(display_matrix) / 2.0 if display_matrix.size else 0.0
    for row in range(display_matrix.shape[0]):
        for col in range(display_matrix.shape[1]):
            value = display_matrix[row, col]
            text = format(value, fmt)
            color = "white" if value > threshold else "black"
            plt.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# %% Test metric calculation
def calculate_specificity(cm: np.ndarray) -> tuple[dict[str, float], float]:
    specificity: dict[str, float] = {}
    total = cm.sum()
    for class_index in range(cm.shape[0]):
        tp = cm[class_index, class_index]
        fp = cm[:, class_index].sum() - tp
        fn = cm[class_index, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificity[f"specificity_class_{class_index}"] = float(tn / denom) if denom else float("nan")
    macro_specificity = float(np.nanmean(list(specificity.values())))
    return specificity, macro_specificity


def calculate_roc_auc_and_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> tuple[float | None, str | None]:
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
        roc_auc = float(
            roc_auc_score(
                y_true_bin,
                y_proba,
                average="macro",
                multi_class="ovr",
            )
        )

        fpr: dict[Any, np.ndarray] = {}
        tpr: dict[Any, np.ndarray] = {}
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
        # Keep the expected artifact present even when ROC is impossible
        # because a test fold lacks a class or probabilities are invalid.
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            f"ROC-AUC could not be calculated:\n{exc}",
            ha="center",
            va="center",
            wrap=True,
        )
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

    specificity_by_index, macro_specificity = calculate_specificity(cm)
    specificity_by_class = {
        f"specificity_{class_name}": specificity_by_index[f"specificity_class_{index}"]
        for index, class_name in enumerate(class_names)
    }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    roc_auc, roc_error = calculate_roc_auc_and_plot(
        y_true=y_true,
        y_proba=y_proba,
        class_names=class_names,
        output_path=output_dir / "roc_curve.png",
    )

    plot_confusion_matrix(
        cm,
        class_names,
        output_dir / "confusion_matrix.png",
        "Confusion Matrix",
        normalize=False,
    )
    plot_confusion_matrix(
        cm_normalized,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix",
        normalize=True,
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


# %% Output files
def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    args_dict = vars(args).copy()
    args_dict["data_dir"] = str(args_dict["data_dir"])
    args_dict["output_dir"] = str(args_dict["output_dir"])
    args_dict["model_name"] = MODEL_NAME
    args_dict["image_size"] = IMAGE_SIZE
    args_dict["optimizer"] = "Adam"
    args_dict["loss"] = "CrossEntropyLoss with training-set class weights"
    args_dict["train_samples"] = args.train_samples
    args_dict["val_samples"] = args.val_samples
    args_dict["test_samples"] = args.test_samples
    return args_dict


def save_args_yaml(args: argparse.Namespace, output_dir: Path) -> None:
    args_dict = serialize_args(args)
    with (output_dir / "args.yaml").open("w", encoding="utf-8") as handle:
        for key, value in args_dict.items():
            handle.write(f"{key}: {to_yaml_scalar(value)}\n")


def to_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return json.dumps(str(value))


def save_text_reports(
    output_dir: Path,
    class_names: list[str],
    test_metrics: dict[str, Any],
    classification_report_text: str,
    best_epoch: int,
    args: argparse.Namespace,
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
        handle.write("MedIMeta-PBC ConvNeXtV2 Base Round Summary\n")
        handle.write("==========================================\n\n")
        handle.write(f"Model: {MODEL_NAME}\n")
        handle.write(f"Round: {args.round_name}\n")
        handle.write(f"Seed: {args.seed}\n")
        handle.write(f"Epochs requested: {args.epochs}\n")
        handle.write(f"Batch size: {args.batch_size}\n")
        handle.write(f"Learning rate: {args.lr}\n")
        handle.write(f"Train images requested: {args.train_samples}\n")
        handle.write(f"Validation images requested: {args.val_samples}\n")
        handle.write(f"Test images requested: {args.test_samples}\n")
        handle.write("Optimizer: Adam\n")
        handle.write("Loss: CrossEntropyLoss with class weights from training split\n")
        handle.write(f"Best epoch by validation macro-F1: {best_epoch}\n")
        handle.write(f"Classes: {', '.join(class_names)}\n\n")
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


# %% Optional Hugging Face upload
def upload_results_to_hf(output_dir: Path, repo_id: str, repo_type: str) -> None:
    if not repo_id:
        raise ValueError("--hf_repo_id is required when --upload_to_hf true.")
    try:
        from huggingface_hub import create_repo, upload_folder
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Install it or run with --upload_to_hf false."
        ) from exc

    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=output_dir.name,
    )


# %% Stage runners
def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load our own training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_training(
    args: argparse.Namespace,
    output_dir: Path,
    weights_dir: Path,
    dataloaders: dict[str, DataLoader],
    datasets_by_split: dict[str, Dataset],
    class_names: list[str],
    device: torch.device,
) -> int:
    class_weights = compute_class_weights(datasets_by_split["train"], class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_model(device, pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
        )
        val_result = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            split_name=f"epoch {epoch}/{args.epochs} val",
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_macro_f1": float(val_result["macro_f1"]),
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
                args=args,
            )
            print(f"Saved new best model at epoch {epoch} (val_macro_f1={best_val_macro_f1:.4f})")

    save_checkpoint(
        weights_dir / "last_model.pt",
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        best_val_macro_f1=best_val_macro_f1,
        class_names=class_names,
        args=args,
    )
    print(f"Saved last model to {weights_dir / 'last_model.pt'}")
    return best_epoch


# %% Main entry point
def select_device() -> torch.device:
    """Use Apple Silicon MPS when available; otherwise fall back to CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_testing(
    args: argparse.Namespace,
    output_dir: Path,
    weights_dir: Path,
    dataloaders: dict[str, DataLoader],
    datasets_by_split: dict[str, Dataset],
    class_names: list[str],
    device: torch.device,
    best_epoch: int | None = None,
) -> None:
    checkpoint_path = weights_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot run test stage because {checkpoint_path} does not exist. "
            "Run with --stage train first, or use --stage all."
        )

    class_weights = compute_class_weights(datasets_by_split["train"], class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_model(device, pretrained=False)

    print(f"Loading best model from {checkpoint_path} for test evaluation")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if best_epoch is None:
        best_epoch = int(checkpoint.get("epoch", 0))

    test_result = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        split_name="test",
    )
    test_metrics, classification_report_text = calculate_test_metrics(
        eval_result=test_result,
        class_names=class_names,
        output_dir=output_dir,
    )

    summary_row = {
        "model_name": MODEL_NAME,
        "round": args.round_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "best_epoch": best_epoch,
    }
    summary_row.update(test_metrics)
    pd.DataFrame([summary_row]).to_csv(output_dir / "metrics_summary.csv", index=False)

    save_text_reports(
        output_dir=output_dir,
        class_names=class_names,
        test_metrics=test_metrics,
        classification_report_text=classification_report_text,
        best_epoch=best_epoch,
        args=args,
    )

    print("Final test metrics from best validation checkpoint:")
    for key, value in test_metrics.items():
        if key not in {"confusion_matrix", "confusion_matrix_normalized"}:
            print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    for split_name, sample_count in {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }.items():
        if 0 < sample_count < NUM_CLASSES:
            raise ValueError(
                f"--{split_name}_samples must be 0 or at least {NUM_CLASSES} "
                "so every class can be represented."
            )
    set_seed(args.seed)

    data_dir = args.data_dir
    output_dir = args.output_dir / args.round_name
    weights_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    validate_data_dir(data_dir)
    save_args_yaml(args, output_dir)

    device = select_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print("MPS is available. Training will use Apple Silicon GPU acceleration.")
    else:
        print("MPS is not available. Training will use CPU.")

    dataloaders, datasets_by_split, class_names = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        samples_by_split={
            "train": args.train_samples,
            "val": args.val_samples,
            "test": args.test_samples,
        },
        num_workers=args.num_workers,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    best_epoch = None
    if args.stage in {"all", "train"}:
        best_epoch = run_training(
            args=args,
            output_dir=output_dir,
            weights_dir=weights_dir,
            dataloaders=dataloaders,
            datasets_by_split=datasets_by_split,
            class_names=class_names,
            device=device,
        )

    if args.stage in {"all", "test"}:
        run_testing(
            args=args,
            output_dir=output_dir,
            weights_dir=weights_dir,
            dataloaders=dataloaders,
            datasets_by_split=datasets_by_split,
            class_names=class_names,
            device=device,
            best_epoch=best_epoch,
        )

    if args.upload_to_hf:
        print(f"Uploading {output_dir} to Hugging Face repo {args.hf_repo_id}")
        upload_results_to_hf(
            output_dir=output_dir,
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
        )
        print("Hugging Face upload complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
