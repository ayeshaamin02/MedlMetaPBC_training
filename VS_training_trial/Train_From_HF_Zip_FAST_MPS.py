#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from Training_VS import main as train_main


DEFAULT_EXTRACT_DIR = Path("data/hf_zip_datasets")
DEFAULT_ROUND_NAME = "PBC_MULTICLASS_HF_ZIP_ConvNeXtV2Atto_FAST_MPS"


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
        description="Download a Hugging Face ZIP dataset and run fast MPS training."
    )
    parser.add_argument("--hf_repo_id", type=str, default=os.environ.get("HF_DATASET_REPO_ID", ""))
    parser.add_argument("--hf_repo_type", type=str, default=os.environ.get("HF_DATASET_REPO_TYPE", "dataset"))
    parser.add_argument("--zip_filename", type=str, default=os.environ.get("HF_DATASET_ZIP", ""))
    parser.add_argument("--extract_dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument(
        "--data_subdir",
        type=str,
        default="",
        help="Optional path inside the extracted ZIP that contains train/ val/ test/.",
    )
    parser.add_argument("--force_download", type=str_to_bool, default=False)
    parser.add_argument("--force_extract", type=str_to_bool, default=False)
    parser.add_argument("--round_name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--model_name", type=str, default="convnextv2_atto.fcmae")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--upload_to_hf", type=str_to_bool, default=False)
    parser.add_argument("--hf_results_repo_id", type=str, default=os.environ.get("HF_RESULTS_REPO_ID", ""))
    return parser.parse_args()


def prompt_if_missing(value: str, prompt: str) -> str:
    value = value.strip()
    if value:
        return value
    return input(prompt).strip()


def choose_zip_file(repo_id: str, repo_type: str, zip_filename: str) -> str:
    zip_filename = zip_filename.strip()
    if zip_filename:
        return zip_filename

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    zip_files = sorted(path for path in files if path.lower().endswith(".zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"No .zip files were found in Hugging Face {repo_type} repo {repo_id}."
        )
    if len(zip_files) == 1:
        print(f"Found ZIP file in repo: {zip_files[0]}")
        return zip_files[0]

    print("Multiple ZIP files found:")
    for index, path in enumerate(zip_files, start=1):
        print(f"{index}. {path}")
    choice = input("Type the exact ZIP filename to download: ").strip()
    if choice not in zip_files:
        raise ValueError(f"{choice!r} is not one of the ZIP files listed above.")
    return choice


def download_zip(
    repo_id: str,
    repo_type: str,
    zip_filename: str,
    force_download: bool,
) -> Path:
    print(f"Downloading {zip_filename} from {repo_id} ({repo_type})")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=zip_filename,
        force_download=force_download,
    )
    return Path(downloaded)


def safe_extract_zip(zip_path: Path, destination: Path, force_extract: bool) -> Path:
    target_dir = destination / zip_path.stem
    if force_extract and target_dir.exists():
        shutil.rmtree(target_dir)
    if target_dir.exists():
        print(f"Using existing extracted dataset: {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if "__MACOSX" in parts or Path(member.filename).name.startswith("._"):
                continue
            member_path = target_dir / member.filename
            resolved_member_path = member_path.resolve()
            resolved_target_dir = target_dir.resolve()
            if not str(resolved_member_path).startswith(str(resolved_target_dir)):
                raise ValueError(f"Unsafe ZIP path blocked: {member.filename}")
            archive.extract(member, target_dir)
    return target_dir


def contains_imagefolder_split(path: Path) -> bool:
    if "__MACOSX" in path.parts or path.name.startswith("."):
        return False
    split_dirs = [path / "train", path / "val", path / "test"]
    if not all(split.is_dir() for split in split_dirs):
        return False
    return all(any(child.is_dir() for child in split.iterdir()) for split in split_dirs)


def find_data_dir(extracted_root: Path, data_subdir: str) -> Path:
    if data_subdir.strip():
        candidate = extracted_root / data_subdir
        if contains_imagefolder_split(candidate):
            return candidate
        raise FileNotFoundError(
            f"--data_subdir was set to {data_subdir!r}, but {candidate} does not "
            "contain train/, val/, and test/ ImageFolder folders."
        )

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
        raise FileNotFoundError(
            f"Could not find a folder inside {extracted_root} containing train/, "
            "val/, and test/. Check the ZIP structure or pass --data_subdir."
        )
    print(f"Found ImageFolder split at: {matches[0]}")
    return matches[0]


def run_training(args: argparse.Namespace, data_dir: Path) -> None:
    sys.argv = [
        "Training_VS.py",
        "--data_dir",
        str(data_dir),
        "--output_dir",
        "results",
        "--round_name",
        args.round_name,
        "--model_name",
        args.model_name,
        "--seed",
        "101",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        "0.0001",
        "--stage",
        "all",
        "--train_samples",
        str(args.train_samples),
        "--val_samples",
        str(args.val_samples),
        "--test_samples",
        str(args.test_samples),
        "--num_workers",
        "0",
        "--upload_to_hf",
        "true" if args.upload_to_hf else "false",
        "--hf_repo_id",
        args.hf_results_repo_id,
        "--hf_repo_type",
        "model",
    ]
    print("Starting fast training run.")
    print(f"Data folder: {data_dir}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    train_main()


def main() -> None:
    args = parse_args()
    repo_id = prompt_if_missing(
        args.hf_repo_id,
        "Enter Hugging Face dataset repo id, e.g. username/repo_name: ",
    )
    repo_type = args.hf_repo_type.strip() or "dataset"
    zip_filename = choose_zip_file(repo_id, repo_type, args.zip_filename)
    zip_path = download_zip(
        repo_id=repo_id,
        repo_type=repo_type,
        zip_filename=zip_filename,
        force_download=args.force_download,
    )
    extracted_root = safe_extract_zip(
        zip_path=zip_path,
        destination=args.extract_dir,
        force_extract=args.force_extract,
    )
    data_dir = find_data_dir(extracted_root, args.data_subdir)
    run_training(args, data_dir)


if __name__ == "__main__":
    main()
