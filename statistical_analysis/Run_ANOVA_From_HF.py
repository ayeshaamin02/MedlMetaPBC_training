"""
Download metrics_summary.csv files from a Hugging Face results repository and run
one-way statistical comparisons across model families.

Use this after training multiple seeds/runs for at least two model families, for
example:
    ConvNeXtBase:      Round1..Round5
    EfficientNetV2S:   Round1..Round5
    SwinBase:          Round1..Round5

"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_HF_REPO_ID = "USERNAME/RESULTS_REPOSITORY"
DEFAULT_MODEL_FAMILIES = ["ConvNeXtBase", "EfficientNetV2S", "SwinBase"]


NON_METRIC_COLUMNS = {
    "run_timestamp_utc",
    "hostname",
    "model_name",
    "model_family",
    "round",
    "round_label",
    "seed",
    "epochs",
    "batch_size",
    "batch_size_per_gpu",
    "effective_batch_size",
    "learning_rate",
    "optimizer",
    "loss",
    "image_size",
    "num_classes",
    "class_names",
    "class_counts_train",
    "class_counts_val",
    "class_counts_test",
    "train_samples_requested",
    "val_samples_requested",
    "test_samples_requested",
    "train_samples_actual",
    "val_samples_actual",
    "test_samples_actual",
    "hf_dataset_repo_id",
    "hf_dataset_repo_type",
    "zip_filename",
    "data_dir",
    "output_dir",
    "use_amp",
    "world_size",
    "gpu_count_visible",
    "gpu_names",
    "num_workers",
    "upload_to_hf",
    "hf_results_repo_id",
    "hf_results_repo_type",
    "hf_results_path_in_repo",
    "best_epoch",
    "hf_repo_id",
    "hf_path",
    "source_csv",
    "local_csv",
    "run_id",
    "roc_auc_error",
    "confusion_matrix",
    "confusion_matrix_normalized",
    "recovered_test_upload",
}


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
    parser = argparse.ArgumentParser(
        description="Download HF metrics_summary.csv files and run ANOVA across model families."
    )
    parser.add_argument(
        "--hf_repo_id",
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face repo containing result folders, e.g. USERNAME/RESULTS_REPOSITORY.",
    )
    parser.add_argument(
        "--hf_repo_type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--model_families",
        nargs="+",
        default=DEFAULT_MODEL_FAMILIES,
        help="Top-level HF folders/model families to include.",
    )
    parser.add_argument(
        "--metrics_filename",
        default="metrics_summary.csv",
        help="CSV filename to collect from HF.",
    )
    parser.add_argument(
        "--output_dir",
        default="statistical_analysis/anova_from_hf",
        help="Local output folder for downloaded CSVs and analysis results.",
    )
    parser.add_argument(
        "--force_download",
        type=str_to_bool,
        default=False,
        help="Force re-download from HF even if cached.",
    )
    parser.add_argument(
        "--make_plots",
        type=str_to_bool,
        default=True,
        help="Save boxplots for each analyzed metric.",
    )
    parser.add_argument(
        "--include_smoketests",
        type=str_to_bool,
        default=False,
        help="Include HF folders/files containing SmokeTest. Default false for final ANOVA.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def infer_model_family(hf_path: str, allowed_families: Iterable[str]) -> str:
    parts = Path(hf_path).parts
    for family in allowed_families:
        if family in parts:
            return family
    return parts[0] if parts else "unknown"


def infer_round(hf_path: str, row: pd.Series | None = None) -> str:
    if row is not None and "round" in row and pd.notna(row["round"]):
        return str(row["round"])
    match = re.search(r"Round\s*[_-]?(\d+)", hf_path, flags=re.IGNORECASE)
    if match:
        return f"Round{match.group(1)}"
    return "unknown"


def download_metrics_from_hf(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    raw_dir = output_dir / "downloaded_metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files from Hugging Face repo: {args.hf_repo_id} ({args.hf_repo_type})")
    repo_files = list_repo_files(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type)

    selected = []
    for file_path in repo_files:
        if not file_path.endswith(args.metrics_filename):
            continue
        if not args.include_smoketests and "smoketest" in file_path.lower():
            continue
        path_parts = Path(file_path).parts
        if args.model_families and not any(family in path_parts for family in args.model_families):
            continue
        selected.append(file_path)

    if not selected:
        raise FileNotFoundError(
            f"No {args.metrics_filename} files found in {args.hf_repo_id} for "
            f"model families: {args.model_families}"
        )

    downloaded_paths = []
    for hf_path in sorted(selected):
        family = infer_model_family(hf_path, args.model_families)
        round_id = infer_round(hf_path)
        local_name = f"{safe_name(family)}_{safe_name(round_id)}_{safe_name(Path(hf_path).parent.name)}_{args.metrics_filename}"
        local_path = raw_dir / local_name

        if local_path.exists() and not args.force_download:
            print(f"Using existing local copy: {local_path}")
            downloaded_paths.append(local_path)
            continue

        print(f"Downloading: {hf_path}")
        cached = hf_hub_download(
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            filename=hf_path,
            force_download=args.force_download,
        )
        df = pd.read_csv(cached)
        df["source_csv"] = hf_path
        df["hf_repo_id"] = args.hf_repo_id
        df.to_csv(local_path, index=False)
        downloaded_paths.append(local_path)

    return downloaded_paths


def load_combined_metrics(csv_paths: list[Path], model_families: list[str]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            print(f"Skipping empty CSV: {path}")
            continue
        if len(df) > 1:
            print(f"Warning: {path} has {len(df)} rows. Keeping all rows as separate runs.")

        source_csv = str(df["source_csv"].iloc[0]) if "source_csv" in df.columns else path.name
        family = infer_model_family(source_csv, model_families)
        round_id = infer_round(source_csv, df.iloc[0])

        df["model_family"] = df.get("model_family", family)
        df["model_family"] = df["model_family"].fillna(family).replace("", family)
        df["round_label"] = round_id
        df["local_csv"] = str(path)
        df["run_id"] = [
            f"{row.get('model_family', family)}_{row.get('round', round_id)}_seed{row.get('seed', 'NA')}"
            for _, row in df.iterrows()
        ]
        frames.append(df)

    if not frames:
        raise ValueError("No usable metrics rows were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["model_family"] = combined["model_family"].astype(str)
    return combined


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = []
    for col in df.columns:
        if col in NON_METRIC_COLUMNS or col.endswith("_error"):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 2:
            numeric_cols.append(col)
            df[col] = values
    return numeric_cols


def group_values(df: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    groups = {}
    for family, group in df.groupby("model_family"):
        values = pd.to_numeric(group[metric], errors="coerce").dropna().astype(float).to_numpy()
        if len(values) > 0:
            groups[str(family)] = values
    return groups


def eta_omega_squared(groups: dict[str, np.ndarray]) -> tuple[float, float]:
    all_values = np.concatenate(list(groups.values()))
    grand_mean = float(np.mean(all_values))

    ss_between = sum(len(v) * (float(np.mean(v)) - grand_mean) ** 2 for v in groups.values())
    ss_within = sum(float(np.sum((v - float(np.mean(v))) ** 2)) for v in groups.values())
    ss_total = ss_between + ss_within

    k = len(groups)
    n = len(all_values)
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total > 0 else np.nan
    return float(eta_sq), float(max(0.0, omega_sq)) if not math.isnan(omega_sq) else np.nan


def run_anova(df: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anova_rows = []
    assumption_rows = []
    descriptive_rows = []

    for metric in metric_cols:
        groups = group_values(df, metric)
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}

        for family, values in groups.items():
            descriptive_rows.append(
                {
                    "metric": metric,
                    "model_family": family,
                    "n": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            )

            if len(values) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(values)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            assumption_rows.append(
                {
                    "metric": metric,
                    "model_family": family,
                    "n": len(values),
                    "shapiro_w": shapiro_stat,
                    "shapiro_p": shapiro_p,
                    "normality_note": "Shapiro needs at least 3 runs" if len(values) < 3 else "",
                }
            )

        if len(valid_groups) < 2:
            anova_rows.append(
                {
                    "metric": metric,
                    "status": "skipped",
                    "reason": "Need at least 2 model families with at least 2 runs each.",
                }
            )
            continue

        arrays = list(valid_groups.values())
        f_stat, anova_p = stats.f_oneway(*arrays)
        h_stat, kruskal_p = stats.kruskal(*arrays)

        if all(len(v) >= 2 for v in arrays):
            levene_stat, levene_p = stats.levene(*arrays, center="median")
        else:
            levene_stat, levene_p = np.nan, np.nan

        eta_sq, omega_sq = eta_omega_squared(valid_groups)
        total_n = sum(len(v) for v in arrays)

        anova_rows.append(
            {
                "metric": metric,
                "status": "ok",
                "model_families": ", ".join(valid_groups.keys()),
                "num_groups": len(valid_groups),
                "total_runs": total_n,
                "anova_f": f_stat,
                "anova_p": anova_p,
                "anova_significant_p_lt_0_05": anova_p < 0.05,
                "kruskal_h": h_stat,
                "kruskal_p": kruskal_p,
                "kruskal_significant_p_lt_0_05": kruskal_p < 0.05,
                "levene_stat": levene_stat,
                "levene_p": levene_p,
                "equal_variance_warning": bool(levene_p < 0.05) if not np.isnan(levene_p) else "",
                "eta_squared": eta_sq,
                "omega_squared": omega_sq,
            }
        )

    return (
        pd.DataFrame(anova_rows),
        pd.DataFrame(assumption_rows),
        pd.DataFrame(descriptive_rows),
    )


def run_tukey_if_available(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except Exception as exc:
        print(f"statsmodels not available; skipping Tukey HSD pairwise tests. ({exc})")
        return pd.DataFrame()

    rows = []
    for metric in metric_cols:
        tmp = df[["model_family", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna()
        counts = tmp.groupby("model_family")[metric].count()
        if len(counts[counts >= 2]) < 2:
            continue

        result = pairwise_tukeyhsd(endog=tmp[metric], groups=tmp["model_family"], alpha=0.05)
        table = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])
        table.insert(0, "metric", metric)
        rows.append(table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_boxplots(df: pd.DataFrame, metric_cols: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for metric in metric_cols:
        tmp = df[["model_family", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna()
        if tmp["model_family"].nunique() < 2:
            continue

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=tmp, x="model_family", y=metric, color="#9ecae1")
        sns.stripplot(data=tmp, x="model_family", y=metric, color="#1f4e79", size=6, jitter=True)
        plt.title(f"{metric} by model family")
        plt.xlabel("Model family")
        plt.ylabel(metric)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{safe_name(metric)}_boxplot.png", dpi=300)
        plt.close()


def write_readme(output_dir: Path, combined: pd.DataFrame, metric_cols: list[str]) -> None:
    readme = output_dir / "README_ANOVA_RESULTS.txt"
    families = sorted(combined["model_family"].dropna().astype(str).unique())
    lines = [
        "ANOVA analysis notes",
        "====================",
        "",
        "Input data:",
        "- The script used metrics_summary.csv files downloaded from Hugging Face.",
        "- Each CSV row is treated as one trained run.",
        "- Model family is the ANOVA grouping variable.",
        "",
        "Model families included:",
        *[f"- {family}" for family in families],
        "",
        "Metrics analyzed:",
        *[f"- {metric}" for metric in metric_cols],
        "",
        "Main output files:",
        "- combined_metrics_summary.csv: all run-level metrics combined.",
        "- anova_results_all_metrics.csv: one-way ANOVA, Kruskal-Wallis, Levene, eta-squared, omega-squared.",
        "- descriptive_stats_by_model.csv: mean, std, median, min, max by model family.",
        "- assumption_checks.csv: Shapiro normality checks by metric and model family.",
        "- tukey_pairwise_results.csv: pairwise Tukey HSD tests if statsmodels is installed.",
        "- plots/: boxplots with individual run points.",
        "",
        "Interpretation reminder:",
        "- p < 0.05 suggests a statistically detectable difference across model families for that metric.",
        "- If Levene p < 0.05, equal variance is questionable; use caution with classical ANOVA.",
        "- Kruskal-Wallis is included as a non-parametric backup.",
        "- With only 5 runs per model, treat results as supportive evidence, not absolute proof.",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = download_metrics_from_hf(args, output_dir)
    combined = load_combined_metrics(csv_paths, args.model_families)
    combined.to_csv(output_dir / "combined_metrics_summary.csv", index=False)

    metric_cols = numeric_metric_columns(combined)
    if not metric_cols:
        raise ValueError("No numeric metric columns found for ANOVA.")

    anova_df, assumptions_df, descriptive_df = run_anova(combined, metric_cols)
    tukey_df = run_tukey_if_available(combined, metric_cols)

    anova_df.to_csv(output_dir / "anova_results_all_metrics.csv", index=False)
    assumptions_df.to_csv(output_dir / "assumption_checks.csv", index=False)
    descriptive_df.to_csv(output_dir / "descriptive_stats_by_model.csv", index=False)
    if not tukey_df.empty:
        tukey_df.to_csv(output_dir / "tukey_pairwise_results.csv", index=False)

    if args.make_plots:
        make_boxplots(combined, metric_cols, output_dir)

    write_readme(output_dir, combined, metric_cols)

    print("\nANOVA analysis complete.")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Runs loaded: {len(combined)}")
    print(f"Model families: {', '.join(sorted(combined['model_family'].unique()))}")
    print(f"Metrics analyzed: {len(metric_cols)}")
    print("\nStart with:")
    print(f"  {output_dir / 'anova_results_all_metrics.csv'}")
    print(f"  {output_dir / 'descriptive_stats_by_model.csv'}")


if __name__ == "__main__":
    main()
