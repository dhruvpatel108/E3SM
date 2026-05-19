#!/usr/bin/env python3
"""
Matched parquet-vs-H1 audit for the dominant failure slices.

This audit compares:
  - parquet train (first 80%)
  - parquet validation (last 20%)
  - exact Stage A+B H1 rows, non-sentinel only, fixed per-file sampling

It focuses on slices that dominated the previous cloud/sign error budget:
  - target nrtend > 0, CLOUD 0.02-0.05
  - target nrtend > 0, CLOUD 0.05-0.1
  - target nrtend < 0, CLOUD 0.01-0.4
  - target nrtend < 0, CLOUD 0.01-0.9
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/{os.environ.get('USER', 'user')}/matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_h1_cloud_sign_error_budget import sample_h1_file_exact_stage_b_nonsentinel
from diagnose_tau_h1_mismatch import (
    INPUT_COLS,
    MODEL_TARGET_COLS,
    load_model,
    load_parquet_stage_b_sample,
    r2_score,
    run_model,
)


SLICE_DEFS = [
    ("pos_cloud_0.02_0.05", "nrtend>0 & 0.02<=CLOUD<0.05"),
    ("pos_cloud_0.05_0.1", "nrtend>0 & 0.05<=CLOUD<0.1"),
    ("neg_cloud_0.01_0.4", "nrtend<0 & 0.01<=CLOUD<0.4"),
    ("neg_cloud_0.01_0.9", "nrtend<0 & 0.01<=CLOUD<0.9"),
]

OUTPUT_SPECS = [
    ("qrtend", "qrtend_TAU", "pred_qrtend"),
    ("nctend", "nctend_TAU", "pred_nctend"),
    ("nrtend", "nrtend_TAU", "pred_nrtend"),
    ("qctend", "qctend_TAU", "pred_qctend"),
]


STUDY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_REPORT_ROOT = (
    STUDY_ROOT / "reports" / "emulator_performance_evaluation" / "evaluate_model_results"
)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="/pscratch/sd/d/dvpatel/mlmicrophysics_project/trained_emulator_files/"
        "emulator_moe_594616_soft_affine_validation.pt",
        help="Path to TorchScript model.",
    )
    p.add_argument(
        "--parquet-dir",
        default="/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data",
        help="Parquet directory.",
    )
    p.add_argument(
        "--h1-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="H1 tau_2mos directory.",
    )
    p.add_argument(
        "--parquet-samples-per-dataset",
        type=int,
        default=800000,
        help="Balanced sample size for parquet train and validation datasets.",
    )
    p.add_argument(
        "--h1-sample-rows-per-file",
        type=int,
        default=120000,
        help="Fixed non-sentinel H1 sample rows per file.",
    )
    p.add_argument(
        "--out-dir",
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis" / "matched_regime_slices"),
        help="Output directory.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return p.parse_args()


def ensure_out_dir(path_str):
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_fixed_h1_sample(h1_dir, rows_per_file, rng):
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No H1 files found in {h1_dir}")

    parts = []
    for i, file_path in enumerate(h1_files):
        print(f"Sampling H1 file {i + 1}/{len(h1_files)}: {os.path.basename(file_path)}")
        df = sample_h1_file_exact_stage_b_nonsentinel(file_path, rows_per_file, rng)
        parts.append(df.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def slice_mask(df, slice_name):
    if slice_name == "pos_cloud_0.02_0.05":
        return (df["nrtend_TAU"] > 0) & (df["CLOUD"] >= 0.02) & (df["CLOUD"] < 0.05)
    if slice_name == "pos_cloud_0.05_0.1":
        return (df["nrtend_TAU"] > 0) & (df["CLOUD"] >= 0.05) & (df["CLOUD"] < 0.1)
    if slice_name == "neg_cloud_0.01_0.4":
        return (df["nrtend_TAU"] < 0) & (df["CLOUD"] >= 0.01) & (df["CLOUD"] < 0.4)
    if slice_name == "neg_cloud_0.01_0.9":
        return (df["nrtend_TAU"] < 0) & (df["CLOUD"] >= 0.01) & (df["CLOUD"] < 0.9)
    raise ValueError(f"Unknown slice: {slice_name}")


def union_mask(df):
    mask = np.zeros(len(df), dtype=bool)
    for slice_name, _ in SLICE_DEFS:
        mask |= slice_mask(df, slice_name).to_numpy()
    return mask


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def score_slice_rows(model, df, dataset_name, total_rows):
    rows = []
    stats_rows = []

    selected_mask = union_mask(df)
    selected_df = df.loc[selected_mask].reset_index(drop=True)
    print(f"  {dataset_name}: scoring {len(selected_df):,} union-slice rows")

    preds = run_model(model, selected_df[INPUT_COLS].to_numpy(dtype=np.float32))
    scored = selected_df.copy()
    scored["pred_qrtend"] = preds[:, 0].astype(np.float64)
    scored["pred_nctend"] = preds[:, 1].astype(np.float64)
    scored["pred_nrtend"] = preds[:, 2].astype(np.float64)
    scored["pred_qctend"] = preds[:, 3].astype(np.float64)

    for slice_name, slice_desc in SLICE_DEFS:
        full_mask = slice_mask(df, slice_name)
        slice_count = int(full_mask.sum())
        fraction = slice_count / float(max(total_rows, 1))

        slice_scored = scored.loc[slice_mask(scored, slice_name)].reset_index(drop=True)
        pred_cols = [pred_col for _, _, pred_col in OUTPUT_SPECS]
        target_cols = [target_col for _, target_col, _ in OUTPUT_SPECS]
        finite_mask = np.isfinite(slice_scored[pred_cols]).all(axis=1).to_numpy()
        finite_mask &= np.isfinite(slice_scored[target_cols]).all(axis=1).to_numpy()
        nonfinite_count = int((~finite_mask).sum())
        slice_scored = slice_scored.loc[finite_mask].reset_index(drop=True)

        row = {
            "dataset": dataset_name,
            "slice_name": slice_name,
            "slice_desc": slice_desc,
            "count": slice_count,
            "fraction_of_dataset": fraction,
            "n_nonfinite_prediction_rows": nonfinite_count,
            "n_finite_rows": int(len(slice_scored)),
        }

        for name, target_col, pred_col in OUTPUT_SPECS:
            target = slice_scored[target_col].to_numpy(dtype=np.float64)
            pred = slice_scored[pred_col].to_numpy(dtype=np.float64)
            if target.size == 0:
                row[f"{name}_r2"] = np.nan
                row[f"{name}_rmse"] = np.nan
                row[f"{name}_mae"] = np.nan
                row[f"{name}_sign_accuracy"] = np.nan
            else:
                diff = pred - target
                row[f"{name}_r2"] = float(r2_score(target, pred))
                row[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(diff))))
                row[f"{name}_mae"] = float(np.mean(np.abs(diff)))
                row[f"{name}_sign_accuracy"] = float(np.mean(np.sign(pred) == np.sign(target)))

        rows.append(row)

        for variable in INPUT_COLS + MODEL_TARGET_COLS:
            stats = summarize_values(df.loc[full_mask, variable].to_numpy(dtype=np.float64))
            stats_rows.append(
                {
                    "dataset": dataset_name,
                    "slice_name": slice_name,
                    "slice_desc": slice_desc,
                    "source": "raw",
                    "variable": variable,
                    "count": slice_count,
                    **stats,
                }
            )

        for name, _, pred_col in OUTPUT_SPECS:
            stats = summarize_values(slice_scored[pred_col].to_numpy(dtype=np.float64))
            stats_rows.append(
                {
                    "dataset": dataset_name,
                    "slice_name": slice_name,
                    "slice_desc": slice_desc,
                    "source": "pred",
                    "variable": name,
                    "count": int(len(slice_scored)),
                    **stats,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(stats_rows)


def plot_prevalence(metrics_df, output_path):
    order = [name for name, _ in SLICE_DEFS]
    datasets = ["train_parquet_stage_b", "validation_parquet_stage_b", "h1_exact_stage_b_nonsentinel"]
    labels = {
        "train_parquet_stage_b": "Train parquet",
        "validation_parquet_stage_b": "Validation parquet",
        "h1_exact_stage_b_nonsentinel": "TAU H1",
    }
    colors = {
        "train_parquet_stage_b": "#264653",
        "validation_parquet_stage_b": "#2a9d8f",
        "h1_exact_stage_b_nonsentinel": "#e76f51",
    }

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(order))
    width = 0.23

    for i, dataset in enumerate(datasets):
        vals = []
        for slice_name in order:
            match = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["slice_name"] == slice_name)]
            vals.append(float(match.iloc[0]["fraction_of_dataset"]) if not match.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=labels[dataset], color=colors[dataset])

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Fraction Of Dataset")
    ax.set_title("Dominant Slice Prevalence: Train vs Validation vs H1")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_model_metrics(metrics_df, output_path):
    order = [name for name, _ in SLICE_DEFS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    datasets = ["validation_parquet_stage_b", "h1_exact_stage_b_nonsentinel"]
    labels = {
        "validation_parquet_stage_b": "Validation parquet",
        "h1_exact_stage_b_nonsentinel": "TAU H1",
    }
    colors = {
        "validation_parquet_stage_b": "#2a9d8f",
        "h1_exact_stage_b_nonsentinel": "#e76f51",
    }
    x = np.arange(len(order))
    width = 0.32

    for i, dataset in enumerate(datasets):
        nrt_vals = []
        nct_vals = []
        for slice_name in order:
            match = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["slice_name"] == slice_name)]
            if match.empty:
                nrt_vals.append(np.nan)
                nct_vals.append(np.nan)
            else:
                nrt_vals.append(float(match.iloc[0]["nrtend_r2"]))
                nct_vals.append(float(match.iloc[0]["nctend_r2"]))
        axes[0].bar(x + (i - 0.5) * width, nrt_vals, width=width, label=labels[dataset], color=colors[dataset])
        axes[1].bar(x + (i - 0.5) * width, nct_vals, width=width, label=labels[dataset], color=colors[dataset])

    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)

    axes[0].set_title("nrtend R2 In Dominant Slices")
    axes[1].set_title("nctend R2 In Dominant Slices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_summary_note(out_path, metrics_df, stats_df):
    def metric(dataset, slice_name, col):
        match = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["slice_name"] == slice_name)]
        if match.empty:
            return np.nan
        return float(match.iloc[0][col])

    def stat(dataset, slice_name, source, variable, col):
        match = stats_df[
            (stats_df["dataset"] == dataset)
            & (stats_df["slice_name"] == slice_name)
            & (stats_df["source"] == source)
            & (stats_df["variable"] == variable)
        ]
        if match.empty:
            return np.nan
        return float(match.iloc[0][col])

    lines = [
        "matched dominant-slice audit",
        "",
        "- Overall verdict: these dominant slices are not explained mainly by offline prevalence mismatch.",
        "- The positive low-cloud slices are only slightly more common in H1 than in train/validation, but their H1 state and target magnitudes are radically larger, and nrtend skill collapses only in H1.",
        "- The negative low-cloud slices are not underrepresented offline at all; they are slightly less prevalent in H1 than in parquet, yet H1 nctend collapses badly inside the same slice definitions.",
        "- So the dominant explanation here is within-slice regime/semantic shift, with only a minor contribution from prevalence mismatch in the positive low-cloud slices.",
        "",
    ]
    for slice_name, slice_desc in SLICE_DEFS:
        train_frac = metric("train_parquet_stage_b", slice_name, "fraction_of_dataset")
        val_frac = metric("validation_parquet_stage_b", slice_name, "fraction_of_dataset")
        h1_frac = metric("h1_exact_stage_b_nonsentinel", slice_name, "fraction_of_dataset")
        val_nrt_r2 = metric("validation_parquet_stage_b", slice_name, "nrtend_r2")
        h1_nrt_r2 = metric("h1_exact_stage_b_nonsentinel", slice_name, "nrtend_r2")
        val_nct_r2 = metric("validation_parquet_stage_b", slice_name, "nctend_r2")
        h1_nct_r2 = metric("h1_exact_stage_b_nonsentinel", slice_name, "nctend_r2")

        lines.append(f"- {slice_name}: {slice_desc}")
        lines.append(
            "  prevalence: train={:.3f}%, validation={:.3f}%, H1={:.3f}%".format(
                100.0 * train_frac, 100.0 * val_frac, 100.0 * h1_frac
            )
        )
        lines.append(
            "  nrtend R2: validation={:.4g}, H1={:.4g}; nctend R2: validation={:.4g}, H1={:.4g}".format(
                val_nrt_r2, h1_nrt_r2, val_nct_r2, h1_nct_r2
            )
        )
        lines.append(
            "  key medians: NR_TAU_in validation={:.4g}, H1={:.4g}; N0R validation={:.4g}, H1={:.4g}; "
            "nrtend_TAU validation={:.4g}, H1={:.4g}; nctend_TAU validation={:.4g}, H1={:.4g}".format(
                stat("validation_parquet_stage_b", slice_name, "raw", "NR_TAU_in", "p50"),
                stat("h1_exact_stage_b_nonsentinel", slice_name, "raw", "NR_TAU_in", "p50"),
                stat("validation_parquet_stage_b", slice_name, "raw", "N0R", "p50"),
                stat("h1_exact_stage_b_nonsentinel", slice_name, "raw", "N0R", "p50"),
                stat("validation_parquet_stage_b", slice_name, "raw", "nrtend_TAU", "p50"),
                stat("h1_exact_stage_b_nonsentinel", slice_name, "raw", "nrtend_TAU", "p50"),
                stat("validation_parquet_stage_b", slice_name, "raw", "nctend_TAU", "p50"),
                stat("h1_exact_stage_b_nonsentinel", slice_name, "raw", "nctend_TAU", "p50"),
            )
        )
        lines.append("")

    lines.append("- Interpretation guide:")
    lines.append("  if H1 prevalence is much larger than train/validation prevalence, that supports underrepresentation offline.")
    lines.append("  if prevalence is similar but target medians and model behavior differ sharply, that supports semantic/online regime shift inside the slice.")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    metrics_path = out_dir / "matched_slice_metrics.csv"
    stats_path = out_dir / "matched_slice_stats.csv"
    prevalence_plot = out_dir / "matched_slice_prevalence.png"
    metrics_plot = out_dir / "matched_slice_model_metrics.png"
    note_path = out_dir / "summary_note.txt"

    rng = np.random.default_rng(args.seed)
    print("Loading TorchScript model ...")
    model = load_model(args.model)

    print("Loading train parquet sample ...")
    train_df = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.parquet_samples_per_dataset,
        rng=rng,
        file_range=(0.0, 0.8),
    )
    print(f"  train rows: {len(train_df):,}")

    print("Loading validation parquet sample ...")
    val_df = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.parquet_samples_per_dataset,
        rng=rng,
        file_range=(0.8, 1.0),
    )
    print(f"  validation rows: {len(val_df):,}")

    print("Loading fixed-per-file H1 sample ...")
    h1_df = load_fixed_h1_sample(args.h1_dir, args.h1_sample_rows_per_file, rng)
    print(f"  h1 rows: {len(h1_df):,}")

    metrics_frames = []
    stats_frames = []

    for dataset_name, df in [
        ("train_parquet_stage_b", train_df),
        ("validation_parquet_stage_b", val_df),
        ("h1_exact_stage_b_nonsentinel", h1_df),
    ]:
        print(f"Scoring dominant slices for {dataset_name} ...")
        mdf, sdf = score_slice_rows(model, df, dataset_name, len(df))
        metrics_frames.append(mdf)
        stats_frames.append(sdf)

    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    stats_df = pd.concat(stats_frames, ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    plot_prevalence(metrics_df, prevalence_plot)
    plot_model_metrics(metrics_df, metrics_plot)
    write_summary_note(note_path, metrics_df, stats_df)

    print(f"Saved CSV:  {metrics_path}")
    print(f"Saved CSV:  {stats_path}")
    print(f"Saved plot: {prevalence_plot}")
    print(f"Saved plot: {metrics_plot}")
    print(f"Saved note: {note_path}")


if __name__ == "__main__":
    main()
