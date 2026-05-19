#!/usr/bin/env python3
"""
Score tau_2mos H1 positive-nrtend performance inside parquet-covered support bands.

The goal is to separate:
  1. failure mainly on clearly out-of-support H1 positive-nrtend rows
  2. failure that persists even inside a parquet-covered region

Support is defined from combined parquet Stage-B rows with target nrtend_TAU > 0.
H1 rows are exact Stage A+B, non-sentinel only, and target nrtend_TAU > 0.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_tau_h1_mismatch import (
    compute_metrics_from_frame,
    load_h1_exact_stage_b_sample,
    load_model,
    load_parquet_stage_b_sample,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/{os.environ.get('USER', 'user')}/matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUPPORT_VARS = ["NR_TAU_in", "N0R", "CLOUD"]
QUANTILE_SETS = {
    "broad_overlap_p01_p99": (1.0, 99.0),
    "core_overlap_p05_p95": (5.0, 95.0),
}


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
        help="Path to the TorchScript model.",
    )
    p.add_argument(
        "--parquet-dir",
        default="/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data",
        help="Parquet directory containing all train+validation files.",
    )
    p.add_argument(
        "--h1-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="Directory containing tau_2mos H1 files.",
    )
    p.add_argument(
        "--support-samples",
        type=int,
        default=800000,
        help="Target Stage-B sample size for parquet support definition.",
    )
    p.add_argument(
        "--h1-samples",
        type=int,
        default=800000,
        help="Target exact Stage A+B H1 sample size.",
    )
    p.add_argument(
        "--out-dir",
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis" / "positive_nrtend_overlap_region"),
        help="Output directory for overlap metrics, plot, and note.",
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


def positive_rows(df):
    return df.loc[df["nrtend_TAU"] > 0].reset_index(drop=True)


def compute_bounds(df, q_lo, q_hi):
    rows = []
    bounds = {}
    for col in SUPPORT_VARS:
        vals = df[col].to_numpy(dtype=np.float64)
        lo = float(np.percentile(vals, q_lo))
        hi = float(np.percentile(vals, q_hi))
        bounds[col] = (lo, hi)
        rows.append(
            {
                "region_name": f"p{int(q_lo):02d}_p{int(q_hi):02d}",
                "variable": col,
                "lower": lo,
                "upper": hi,
            }
        )
    return bounds, rows


def region_mask(df, bounds):
    mask = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in bounds.items():
        vals = df[col].to_numpy(dtype=np.float64)
        mask &= vals >= lo
        mask &= vals <= hi
    return mask


def parquet_bin_edges(df, col):
    vals = df[col].to_numpy(dtype=np.float64)
    raw = np.percentile(vals, [0, 25, 50, 75, 95, 100]).astype(np.float64)
    edges = [raw[0]]
    for value in raw[1:]:
        if value > edges[-1]:
            edges.append(value)
    if len(edges) == 1:
        edges.append(edges[0] + 1.0)
    return np.asarray(edges, dtype=np.float64)


def score_subset(model, df, subset_type, subset_name, total_rows, extra=None):
    row = compute_metrics_from_frame(
        model,
        df.reset_index(drop=True),
        file_name="ALL",
        mode=subset_name,
        target_summary=None,
    )
    row["subset_type"] = subset_type
    row["subset_name"] = subset_name
    row["subset_fraction_of_all_positive_h1"] = len(df) / float(max(total_rows, 1))
    if extra:
        row.update(extra)
    return row


def plot_overlap_summary(region_df, output_path):
    order = [
        "all_positive_h1_nonsentinel",
        "broad_overlap_p01_p99",
        "core_overlap_p05_p95",
        "outside_broad_overlap",
    ]
    plot_df = region_df.set_index("subset_name").reindex(order).reset_index()
    labels = ["All H1 +", "Inside p1-p99", "Inside p5-p95", "Outside p1-p99"]
    x = np.arange(len(plot_df))
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(x, plot_df["n_model_rows"], color=colors)
    axes[0].set_title("Positive H1 Rows")
    axes[0].set_ylabel("Rows")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, plot_df["r2_nrtend_pos"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1].set_title("Positive-nrtend R2")
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(x, 100.0 * plot_df["nrtend_sign_accuracy"], color=colors)
    axes[2].set_title("Positive-nrtend Sign Accuracy")
    axes[2].set_ylabel("Percent")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_note(out_path, region_df, bounds_df):
    region_df = region_df.set_index("subset_name")

    def fmt(col, subset):
        value = float(region_df.loc[subset, col])
        if np.isnan(value):
            return "nan"
        if abs(value) >= 1000 or abs(value) < 1.0e-2:
            return "{:.4e}".format(value)
        return "{:.4f}".format(value)

    broad_frac = 100.0 * float(region_df.loc["broad_overlap_p01_p99", "subset_fraction_of_all_positive_h1"])
    core_frac = 100.0 * float(region_df.loc["core_overlap_p05_p95", "subset_fraction_of_all_positive_h1"])

    lines = [
        "positive-nrtend overlap-region verdict",
        "",
        (
            "- H1 positive non-sentinel rows inside parquet support:"
            f" p1-p99 = {broad_frac:.1f}% of all positive H1 rows,"
            f" p5-p95 = {core_frac:.1f}%."
        ),
        (
            "- Positive-nrtend R2:"
            f" all={fmt('r2_nrtend_pos', 'all_positive_h1_nonsentinel')},"
            f" inside p1-p99={fmt('r2_nrtend_pos', 'broad_overlap_p01_p99')},"
            f" inside p5-p95={fmt('r2_nrtend_pos', 'core_overlap_p05_p95')},"
            f" outside p1-p99={fmt('r2_nrtend_pos', 'outside_broad_overlap')}."
        ),
        (
            "- Positive-nrtend sign accuracy:"
            f" all={100.0 * float(region_df.loc['all_positive_h1_nonsentinel', 'nrtend_sign_accuracy']):.1f}%,"
            f" inside p1-p99={100.0 * float(region_df.loc['broad_overlap_p01_p99', 'nrtend_sign_accuracy']):.1f}%,"
            f" inside p5-p95={100.0 * float(region_df.loc['core_overlap_p05_p95', 'nrtend_sign_accuracy']):.1f}%,"
            f" outside p1-p99={100.0 * float(region_df.loc['outside_broad_overlap', 'nrtend_sign_accuracy']):.1f}%."
        ),
        (
            "- Support variables are NR_TAU_in, N0R, and CLOUD,"
            " using parquet positive-nrtend quantile bands listed in the bounds CSV."
        ),
    ]

    broad_r2 = float(region_df.loc["broad_overlap_p01_p99", "r2_nrtend_pos"])
    all_r2 = float(region_df.loc["all_positive_h1_nonsentinel", "r2_nrtend_pos"])
    core_r2 = float(region_df.loc["core_overlap_p05_p95", "r2_nrtend_pos"])
    if np.isfinite(broad_r2) and np.isfinite(all_r2):
        if np.isfinite(core_r2) and core_r2 > broad_r2 + 100.0 and core_r2 > -1.0:
            lines.append(
                "- The densest parquet-covered core behaves much better than the broader support band:"
                " catastrophic failure is concentrated in edge/sparse regions and clearly out-of-support rows,"
                " not in the center of the parquet-covered regime."
            )
        elif broad_r2 > all_r2 + 1.0:
            lines.append(
                "- This suggests a meaningful part of the failure is concentrated outside parquet-covered support."
            )
        else:
            lines.append(
                "- This suggests the model still fails badly even inside a parquet-covered positive-nrtend region."
            )

    if not bounds_df.empty:
        first = bounds_df[bounds_df["region_name"] == "p01_p99"]
        bounds_txt = ", ".join(
            "{}=[{:.4g}, {:.4g}]".format(row.variable, row.lower, row.upper)
            for row in first.itertuples()
        )
        lines.append(f"- Broad support bounds: {bounds_txt}.")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    bounds_path = out_dir / "positive_nrtend_overlap_bounds.csv"
    metrics_path = out_dir / "positive_nrtend_overlap_metrics.csv"
    plot_path = out_dir / "positive_nrtend_overlap_summary.png"
    note_path = out_dir / "summary_note.txt"

    rng = np.random.default_rng(args.seed)

    print("Loading combined parquet Stage-B sample ...")
    parquet_all = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.support_samples,
        rng=rng,
        file_range=(0.0, 1.0),
    )
    parquet_pos = positive_rows(parquet_all)
    print(f"  Parquet positive-nrtend rows: {len(parquet_pos):,}")

    print("Loading H1 exact Stage A+B non-sentinel sample ...")
    h1_all = load_h1_exact_stage_b_sample(
        h1_dir=args.h1_dir,
        target_total_samples=args.h1_samples,
        rng=rng,
        drop_sentinels=True,
    )
    h1_pos = positive_rows(h1_all)
    print(f"  H1 positive-nrtend non-sentinel rows: {len(h1_pos):,}")

    print("Loading TorchScript model ...")
    model = load_model(args.model)

    bound_rows = []
    region_rows = []
    total_rows = len(h1_pos)
    region_rows.append(score_subset(model, h1_pos, "region", "all_positive_h1_nonsentinel", total_rows))

    broad_mask = None
    for region_name, (q_lo, q_hi) in QUANTILE_SETS.items():
        bounds, rows = compute_bounds(parquet_pos, q_lo=q_lo, q_hi=q_hi)
        for row in rows:
            row["subset_name"] = region_name
        bound_rows.extend(rows)

        mask = region_mask(h1_pos, bounds)
        subset_df = h1_pos.loc[mask].reset_index(drop=True)
        region_rows.append(
            score_subset(
                model,
                subset_df,
                "region",
                region_name,
                total_rows,
            )
        )
        if region_name == "broad_overlap_p01_p99":
            broad_mask = mask

    if broad_mask is None:
        broad_mask = np.zeros(len(h1_pos), dtype=bool)
    outside_broad = h1_pos.loc[~broad_mask].reset_index(drop=True)
    region_rows.append(
        score_subset(
            model,
            outside_broad,
            "region",
            "outside_broad_overlap",
            total_rows,
        )
    )

    bin_rows = []
    for col in SUPPORT_VARS:
        edges = parquet_bin_edges(parquet_pos, col)
        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == len(edges) - 2:
                mask = (h1_pos[col] >= lo) & (h1_pos[col] <= hi)
            else:
                mask = (h1_pos[col] >= lo) & (h1_pos[col] < hi)
            subset = h1_pos.loc[mask].reset_index(drop=True)
            label = f"{col}_bin_{i+1}"
            bin_rows.append(
                score_subset(
                    model,
                    subset,
                    "bin",
                    label,
                    total_rows,
                    extra={
                        "support_variable": col,
                        "bin_index": i + 1,
                        "bin_lower": lo,
                        "bin_upper": hi,
                    },
                )
            )

    bounds_df = pd.DataFrame(bound_rows)
    metrics_df = pd.DataFrame(region_rows + bin_rows)

    bounds_df.to_csv(bounds_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    plot_overlap_summary(metrics_df[metrics_df["subset_type"] == "region"].copy(), plot_path)
    write_note(note_path, metrics_df[metrics_df["subset_type"] == "region"].copy(), bounds_df)

    print("Saved bounds: {}".format(bounds_path))
    print("Saved metrics: {}".format(metrics_path))
    print("Saved plot: {}".format(plot_path))
    print("Saved note: {}".format(note_path))


if __name__ == "__main__":
    main()
