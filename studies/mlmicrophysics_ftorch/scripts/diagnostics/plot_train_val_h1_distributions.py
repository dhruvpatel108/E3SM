#!/usr/bin/env python3
"""
Create matched-sample distribution plots for parquet train, parquet validation,
and one or two TAU/H1 datasets.

Parquet train  : first 80% of files
Parquet val    : last 20% of files
TAU/H1 data    : exact Stage A+B rows, non-sentinel only

Plots are shown in model-relevant transformed spaces:
  - log10 for positive log-transformed inputs
  - signed log10 for qrtend/nctend/qctend
  - arcsinh(x / 1e-3) for nrtend
  - linear for PGAM/RHO_CLUBB/CLOUD/FREQR
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_tau_h1_mismatch import (
    load_h1_exact_stage_b_sample,
    load_parquet_stage_b_sample,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/{os.environ.get('USER', 'user')}/matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_SPECS = [
    ("QC_TAU_in", "log10_pos", "QC_TAU_in"),
    ("QR_TAU_in", "log10_pos", "QR_TAU_in"),
    ("NC_TAU_in", "log10_pos", "NC_TAU_in"),
    ("NR_TAU_in", "log10_pos", "NR_TAU_in"),
    ("PGAM", "linear", "PGAM"),
    ("LAMC", "log10_pos", "LAMC"),
    ("LAMR", "log10_pos", "LAMR"),
    ("N0R", "log10_pos", "N0R"),
    ("RHO_CLUBB", "linear", "RHO_CLUBB"),
    ("CLOUD", "linear", "CLOUD"),
    ("FREQR", "linear", "FREQR"),
    ("qrtend_TAU", "signed_log10", "qrtend_TAU"),
    ("nctend_TAU", "signed_log10", "nctend_TAU"),
    ("nrtend_TAU", "arcsinh_nrtend", "nrtend_TAU"),
    ("qctend_TAU", "signed_log10", "qctend_TAU"),
]

COLORS = {
    "train_parquet_stage_b": "#0072B2",
    "validation_parquet_stage_b": "#009E73",
    "TAU_old": "#D55E00",
    "TAU_new": "#CC79A7",
}
LINESTYLES = {
    "train_parquet_stage_b": "solid",
    "validation_parquet_stage_b": "dashed",
    "TAU_old": "dashdot",
    "TAU_new": "dotted",
}
LABELS = {
    "train_parquet_stage_b": "Train parquet (first 80%)",
    "validation_parquet_stage_b": "Validation parquet (last 20%)",
    "TAU_old": "TAU_old exact Stage A+B, non-sentinel",
    "TAU_new": "TAU_new exact Stage A+B, non-sentinel",
}


STUDY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_REPORT_ROOT = (
    STUDY_ROOT / "reports" / "emulator_performance_evaluation" / "evaluate_model_results"
)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet-dir",
        default="/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data",
        help="Parquet directory.",
    )
    p.add_argument(
        "--h1-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="TAU_old / H1 directory.",
    )
    p.add_argument(
        "--h1-dir-2",
        default=None,
        help="Optional TAU_new / second H1 directory.",
    )
    p.add_argument(
        "--samples-per-dataset",
        type=int,
        default=800000,
        help="Target sample size for each dataset.",
    )
    p.add_argument(
        "--output-stem",
        default="train_val_h1_distribution",
        help="Output filename stem. Produces <stem>_grid.png, <stem>_summary.csv, and <stem>_note.txt.",
    )
    p.add_argument(
        "--out-dir",
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis" / "train_val_h1_distributions"),
        help="Output directory for distribution plots and CSV summaries.",
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


def transform_values(kind, values):
    values = np.asarray(values, dtype=np.float64)
    if kind == "log10_pos":
        return np.log10(np.maximum(values, 1.0e-10))
    if kind == "signed_log10":
        sign = np.sign(values)
        abs_val = np.abs(values) + 1.0e-10
        return sign * np.log10(abs_val)
    if kind == "arcsinh_nrtend":
        return np.arcsinh(values / 1.0e-3)
    return values


def transform_label(kind, raw_name):
    if kind == "log10_pos":
        return f"log10({raw_name})"
    if kind == "signed_log10":
        return f"sign(x) * log10(|{raw_name}| + 1e-10)"
    if kind == "arcsinh_nrtend":
        return f"arcsinh({raw_name} / 1e-3)"
    return raw_name


def summarize_dataset(df, dataset_name):
    rows = []
    for variable, transform_kind, _ in PLOT_SPECS:
        vals = df[variable].to_numpy(dtype=np.float64)
        tvals = transform_values(transform_kind, vals)
        rows.append(
            {
                "dataset": dataset_name,
                "variable": variable,
                "plot_transform": transform_kind,
                "count": int(len(vals)),
                "raw_mean": float(np.mean(vals)),
                "raw_std": float(np.std(vals)),
                "raw_p1": float(np.percentile(vals, 1)),
                "raw_p5": float(np.percentile(vals, 5)),
                "raw_p50": float(np.percentile(vals, 50)),
                "raw_p95": float(np.percentile(vals, 95)),
                "raw_p99": float(np.percentile(vals, 99)),
                "plot_mean": float(np.mean(tvals)),
                "plot_std": float(np.std(tvals)),
                "plot_p1": float(np.percentile(tvals, 1)),
                "plot_p5": float(np.percentile(tvals, 5)),
                "plot_p50": float(np.percentile(tvals, 50)),
                "plot_p95": float(np.percentile(tvals, 95)),
                "plot_p99": float(np.percentile(tvals, 99)),
            }
        )
    return rows


def plot_grid(dataset_frames, output_path):
    fig, axes = plt.subplots(5, 3, figsize=(15, 18))
    axes = axes.ravel()

    for ax, (variable, transform_kind, title) in zip(axes, PLOT_SPECS):
        transformed = {}
        all_vals = []
        for dataset_name, df in dataset_frames.items():
            vals = transform_values(transform_kind, df[variable].to_numpy(dtype=np.float64))
            vals = vals[np.isfinite(vals)]
            transformed[dataset_name] = vals
            all_vals.append(vals)

        all_vals = np.concatenate(all_vals)
        lo = float(np.percentile(all_vals, 0.2))
        hi = float(np.percentile(all_vals, 99.8))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(all_vals))
            hi = float(np.max(all_vals))
        bins = np.linspace(lo, hi, 90)

        for dataset_name, vals in transformed.items():
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.9,
                color=COLORS[dataset_name],
                linestyle=LINESTYLES[dataset_name],
                label=LABELS[dataset_name],
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(transform_label(transform_kind, variable), fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(alpha=0.15)

    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle(
        "Matched-sample train / validation / TAU distributions\n"
        "Parquet sampled evenly by file; TAU sampled evenly by file and timestep",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_note(out_path, dataset_frames):
    lines = [
        "matched-sample distribution comparison",
        "",
        "- Plotting space follows the model-relevant transforms:"
        " log10 for positive transformed inputs,"
        " signed log10 for qrtend/nctend/qctend,"
        " and arcsinh(x / 1e-3) for nrtend.",
        "- TAU datasets are plotted as non-sentinel exact Stage A+B so the visual comparison focuses on physical rows rather than the known -99999 sentinel artifact.",
        "- Colors use a high-contrast Okabe-Ito-style palette, with matching line styles, to keep the four-way overlays easy to distinguish.",
    ]
    for dataset_name, df in dataset_frames.items():
        lines.append(f"- {LABELS[dataset_name]} rows sampled: {len(df):,}.")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    csv_path = out_dir / f"{args.output_stem}_summary.csv"
    plot_path = out_dir / f"{args.output_stem}_grid.png"
    note_path = out_dir / f"{args.output_stem}_note.txt"

    rng = np.random.default_rng(args.seed)

    print("Loading train parquet sample ...")
    train_df = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.samples_per_dataset,
        rng=rng,
        file_range=(0.0, 0.8),
    )
    print(f"  Train rows: {len(train_df):,}")

    print("Loading validation parquet sample ...")
    val_df = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.samples_per_dataset,
        rng=rng,
        file_range=(0.8, 1.0),
    )
    print(f"  Validation rows: {len(val_df):,}")

    print("Loading TAU_old exact Stage A+B non-sentinel sample ...")
    tau_old_df = load_h1_exact_stage_b_sample(
        h1_dir=args.h1_dir,
        target_total_samples=args.samples_per_dataset,
        rng=rng,
        drop_sentinels=True,
    )
    print(f"  TAU_old rows: {len(tau_old_df):,}")

    dataset_frames = {
        "train_parquet_stage_b": train_df,
        "validation_parquet_stage_b": val_df,
        "TAU_old": tau_old_df,
    }

    if args.h1_dir_2:
        print("Loading TAU_new exact Stage A+B non-sentinel sample ...")
        tau_new_df = load_h1_exact_stage_b_sample(
            h1_dir=args.h1_dir_2,
            target_total_samples=args.samples_per_dataset,
            rng=rng,
            drop_sentinels=True,
        )
        print(f"  TAU_new rows: {len(tau_new_df):,}")
        dataset_frames["TAU_new"] = tau_new_df

    rows = []
    for dataset_name, df in dataset_frames.items():
        rows.extend(summarize_dataset(df, dataset_name))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    plot_grid(dataset_frames, plot_path)
    write_note(note_path, dataset_frames)

    print("Saved CSV:  {}".format(csv_path))
    print("Saved plot: {}".format(plot_path))
    print("Saved note: {}".format(note_path))


if __name__ == "__main__":
    main()
