"""
Run_XAI_GradCAM_BestRuns.py

Purpose
-------
Generate one Grad-CAM XAI result for the best run of each trained model family.

This script is designed for the PBC multiclass training project. It reads the
combined ANOVA metrics CSV, selects the best run per model family using highest
test macro-F1 by default, downloads that run's weights/best_model.pt from
Hugging Face, loads the matching timm model, and creates one Grad-CAM overlay
per model.

Default input:
    statistical_analysis/anova/combined_metrics_summary.csv

Default output:
    xai_results/GradCAM_BestRuns/

The script saves one result folder per model family:
    original_image.png
    heatmap.png
    gradcam_overlay.png
    prediction_info.csv
    README_XAI.txt

Notes
-----
- This uses PyTorch only for XAI hooks; no TensorFlow.
- It uses CPU by default on Mac unless MPS is available.
- It supports CUDA if you later run it on a server.
- The best run is selected from final test metrics, not training curves.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from timm import create_model
from torchvision import transforms
from torchvision.datasets import ImageFolder


DEFAULT_HF_REPO_ID = "USERNAME/RESULTS_REPOSITORY"
DEFAULT_MODEL_FAMILIES = ["ConvNeXtBase", "EfficientNetV2S", "SwinBase"]
DEFAULT_METRICS_CSV = "statistical_analysis/anova/combined_metrics_summary.csv"
DEFAULT_DATA_DIR = "data/IMAGEFOLDER_SPLIT"
DEFAULT_OUTPUT_DIR = "xai_results/GradCAM_BestRuns"
NUM_CLASSES = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Grad-CAM XAI result for the best run of each model.")
    parser.add_argument("--metrics_csv", default=DEFAULT_METRICS_CSV)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf_repo_id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf_repo_type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--model_families", nargs="+", default=DEFAULT_MODEL_FAMILIES)
    parser.add_argument("--selection_metric", default="macro_f1")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max_images_to_scan", type=int, default=300)
    parser.add_argument("--prefer_correct", type=str_to_bool, default=True)
    parser.add_argument("--force_download", type=str_to_bool, default=False)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def find_split_dir(data_dir: Path) -> Path:
    if (data_dir / "test").is_dir():
        return data_dir

    candidates = sorted(data_dir.glob("**/test"))
    for test_dir in candidates:
        parent = test_dir.parent
        if (parent / "train").is_dir() and (parent / "val").is_dir():
            return parent

    raise FileNotFoundError(
        f"Could not find ImageFolder split with train/val/test under: {data_dir}"
    )


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {metrics_csv}\n"
            "Run the ANOVA script first, or pass --metrics_csv to the combined_metrics_summary.csv file."
        )
    df = pd.read_csv(metrics_csv)
    if "model_family" not in df.columns:
        raise ValueError("The metrics CSV must contain a model_family column.")
    return df


def select_best_runs(df: pd.DataFrame, model_families: list[str], selection_metric: str) -> pd.DataFrame:
    if selection_metric not in df.columns:
        raise ValueError(f"Selection metric '{selection_metric}' is not in the metrics CSV.")

    rows = []
    for family in model_families:
        subset = df[df["model_family"].astype(str) == family].copy()
        if subset.empty:
            print(f"Warning: no rows found for model family {family}; skipping.")
            continue
        subset[selection_metric] = pd.to_numeric(subset[selection_metric], errors="coerce")
        subset = subset.dropna(subset=[selection_metric])
        if subset.empty:
            print(f"Warning: no numeric {selection_metric} values for {family}; skipping.")
            continue
        rows.append(subset.sort_values(selection_metric, ascending=False).iloc[0])

    if not rows:
        raise ValueError("No best runs could be selected.")
    return pd.DataFrame(rows)


def hf_weights_path(row: pd.Series) -> str:
    if "hf_results_path_in_repo" in row and pd.notna(row["hf_results_path_in_repo"]):
        base = str(row["hf_results_path_in_repo"]).strip("/")
    elif "source_csv" in row and pd.notna(row["source_csv"]):
        base = str(Path(str(row["source_csv"])).parent).strip("/")
    else:
        raise ValueError("Could not infer HF path for weights from metrics row.")
    return f"{base}/weights/best_model.pt"


def download_weights(row: pd.Series, args: argparse.Namespace, weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    filename = hf_weights_path(row)
    local_name = f"{safe_name(str(row['model_family']))}_{safe_name(str(row.get('round', 'best')))}_best_model.pt"
    local_path = weights_dir / local_name

    if local_path.exists() and not args.force_download:
        print(f"Using existing weights: {local_path}")
        return local_path

    print(f"Downloading HF weights: {filename}")
    cached = hf_hub_download(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        filename=filename,
        force_download=args.force_download,
    )
    checkpoint = torch.load(cached, map_location="cpu")
    torch.save(checkpoint, local_path)
    return local_path


def load_model_from_checkpoint(weights_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_name = checkpoint.get("model_name")
    if not model_name:
        model_name = checkpoint.get("args", {}).get("model_name")
    if not model_name:
        raise ValueError(f"Could not find model_name in checkpoint: {weights_path}")

    model = create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def pick_target_layer(model: torch.nn.Module, model_family: str) -> torch.nn.Module:
    family = model_family.lower()

    if "convnext" in family:
        return model.stages[-1].blocks[-1]
    if "efficient" in family:
        return model.blocks[-1]
    if "swin" in family:
        return model.layers[-1].blocks[-1].norm1

    # Fallback: choose the last non-classifier module that has parameters.
    candidates = []
    for name, module in model.named_modules():
        if any(skip in name.lower() for skip in ["head", "classifier", "fc"]):
            continue
        if sum(p.numel() for p in module.parameters(recurse=False)) > 0:
            candidates.append(module)
    if not candidates:
        raise ValueError("Could not infer a target layer for Grad-CAM.")
    return candidates[-1]


def activation_to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert common timm activation layouts to NCHW."""
    if tensor.ndim == 4:
        # Swin often uses NHWC, while CNNs use NCHW.
        if tensor.shape[1] <= 32 and tensor.shape[-1] > 32:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    if tensor.ndim == 3:
        # Token layout: [B, N, C]. Convert to square spatial grid.
        batch, tokens, channels = tensor.shape
        side = int(tokens ** 0.5)
        if side * side != tokens:
            # Drop a class token if present.
            tokens_no_cls = tokens - 1
            side = int(tokens_no_cls ** 0.5)
            if side * side != tokens_no_cls:
                raise ValueError(f"Cannot reshape token activation with shape {tuple(tensor.shape)}")
            tensor = tensor[:, 1:, :]
        return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2)

    raise ValueError(f"Unsupported activation shape for Grad-CAM: {tuple(tensor.shape)}")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activation = output.detach()

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradient = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, target_class].sum()
        score.backward()

        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = activation_to_nchw(self.activation.float())
        gradients = activation_to_nchw(self.gradient.float())
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, logits.detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def tensor_from_path(path: Path, transform: transforms.Compose, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    display_image = image.resize((224, 224))
    return input_tensor, display_image


@torch.no_grad()
def predict(model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple[int, float, np.ndarray]:
    logits = model(input_tensor)
    probabilities = torch.softmax(logits.float(), dim=1).squeeze(0)
    pred_idx = int(probabilities.argmax().item())
    confidence = float(probabilities[pred_idx].item())
    return pred_idx, confidence, probabilities.detach().cpu().numpy()


def find_image_for_xai(
    model: torch.nn.Module,
    dataset: ImageFolder,
    transform: transforms.Compose,
    device: torch.device,
    max_images_to_scan: int,
    prefer_correct: bool,
    seed: int,
) -> dict[str, Any]:
    indices = list(range(len(dataset.samples)))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(max_images_to_scan, len(indices))]

    best_any: dict[str, Any] | None = None
    for index in indices:
        image_path, true_idx = dataset.samples[index]
        input_tensor, display_image = tensor_from_path(Path(image_path), transform, device)
        pred_idx, confidence, probabilities = predict(model, input_tensor)

        record = {
            "image_path": image_path,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
            "input_tensor": input_tensor,
            "display_image": display_image,
        }
        if best_any is None or confidence > best_any["confidence"]:
            best_any = record
        if prefer_correct and pred_idx == true_idx:
            return record

    if best_any is None:
        raise RuntimeError("Could not find an image for XAI.")
    return best_any


def save_gradcam_images(cam: np.ndarray, image: Image.Image, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / "original_image.png"
    heatmap_path = output_dir / "heatmap.png"
    overlay_path = output_dir / "gradcam_overlay.png"

    image.save(original_path)

    cmap = plt.get_cmap("jet")
    heatmap_rgb = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize(image.size)
    heatmap_img.save(heatmap_path)

    image_arr = np.asarray(image).astype(np.float32)
    heatmap_arr = np.asarray(heatmap_img).astype(np.float32)
    overlay = np.clip(0.55 * image_arr + 0.45 * heatmap_arr, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)

    return original_path, heatmap_path, overlay_path


def write_readme(output_dir: Path, row: pd.Series, info: dict[str, Any], class_names: list[str]) -> None:
    readme = output_dir / "README_XAI.txt"
    text = f"""Grad-CAM XAI result
===================

Model family: {row['model_family']}
Model name: {row.get('model_name', 'unknown')}
Selected run: {row.get('round', 'unknown')}
Seed: {row.get('seed', 'unknown')}
Selection metric: macro_f1 = {row.get('macro_f1', 'unknown')}

Image used:
{info['image_path']}

True class:
{class_names[info['true_idx']]}

Predicted class:
{class_names[info['pred_idx']]}

Prediction confidence:
{info['confidence']}

Files:
- original_image.png
- heatmap.png
- gradcam_overlay.png
- prediction_info.csv

Interpretation:
The Grad-CAM overlay highlights image regions that most influenced the model's
predicted class for this selected test image. This is a qualitative explanation,
not a statistical test.
"""
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    output_root = Path(args.output_dir)
    weights_dir = output_root / "downloaded_weights"
    output_root.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(Path(args.metrics_csv))
    best_runs = select_best_runs(metrics, args.model_families, args.selection_metric)
    best_runs.to_csv(output_root / "selected_best_runs.csv", index=False)

    split_dir = find_split_dir(Path(args.data_dir))
    test_dir = split_dir / "test"
    transform = build_transforms(args.image_size)
    dataset = ImageFolder(test_dir, transform=None)
    class_names = dataset.classes

    all_prediction_rows = []
    for _, row in best_runs.iterrows():
        family = str(row["model_family"])
        print(f"\nRunning Grad-CAM for {family}")

        weights_path = download_weights(row, args, weights_dir)
        model, checkpoint = load_model_from_checkpoint(weights_path, device)
        checkpoint_class_names = checkpoint.get("class_names", class_names)

        target_layer = pick_target_layer(model, family)
        image_info = find_image_for_xai(
            model=model,
            dataset=dataset,
            transform=transform,
            device=device,
            max_images_to_scan=args.max_images_to_scan,
            prefer_correct=args.prefer_correct,
            seed=args.seed,
        )

        gradcam = GradCAM(model, target_layer)
        try:
            cam, logits = gradcam(image_info["input_tensor"], target_class=image_info["pred_idx"])
        finally:
            gradcam.close()

        family_dir = output_root / safe_name(family)
        original_path, heatmap_path, overlay_path = save_gradcam_images(
            cam=cam,
            image=image_info["display_image"],
            output_dir=family_dir,
        )

        prediction_row = {
            "model_family": family,
            "model_name": row.get("model_name", checkpoint.get("model_name", "")),
            "round": row.get("round", ""),
            "seed": row.get("seed", ""),
            "selection_metric": args.selection_metric,
            "selection_metric_value": row.get(args.selection_metric, ""),
            "weights_path": str(weights_path),
            "image_path": image_info["image_path"],
            "true_class": checkpoint_class_names[image_info["true_idx"]],
            "predicted_class": checkpoint_class_names[image_info["pred_idx"]],
            "confidence": image_info["confidence"],
            "original_image": str(original_path),
            "heatmap": str(heatmap_path),
            "gradcam_overlay": str(overlay_path),
        }
        for idx, probability in enumerate(image_info["probabilities"]):
            prediction_row[f"probability_{checkpoint_class_names[idx]}"] = float(probability)

        pd.DataFrame([prediction_row]).to_csv(family_dir / "prediction_info.csv", index=False)
        write_readme(family_dir, row, image_info, checkpoint_class_names)
        all_prediction_rows.append(prediction_row)

        print(f"Saved Grad-CAM overlay: {overlay_path}")

    pd.DataFrame(all_prediction_rows).to_csv(output_root / "xai_prediction_summary.csv", index=False)
    print(f"\nXAI complete. Output folder: {output_root.resolve()}")


if __name__ == "__main__":
    main()
