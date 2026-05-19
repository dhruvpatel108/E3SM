#!/usr/bin/env python3
"""
Audit H1 exact Stage A+B, non-sentinel rows by CLOUD bin and nrtend sign.

This script answers:
  1. Which sign/CLOUD slice dominates total H1 nrtend squared error?
  2. Does nctend failure concentrate in the same slice, or is it broader?

Base dataset:
  - exact Stage A+B H1 rows
  - non-sentinel only
  - fixed sampled row count per H1 file
  - deployed TorchScript model
"""

import argparse
import glob
import math
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/{os.environ.get('USER', 'user')}/matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from diagnose_tau_h1_mismatch import (
    CURRENT_SENTINEL_COLS,
    INPUT_COLS,
    MODEL_OUTPUT_NAMES,
    MODEL_TARGET_COLS,
    SENTINEL_VALUE,
    build_rows_from_mask,
    compute_masks,
    load_model,
    r2_score,
    read_timestep_arrays,
    run_model,
    sample_frame,
    to_ndarray,
)


CLOUD_BINS = [
    (0.01, 0.02, "0.01-0.02"),
    (0.02, 0.05, "0.02-0.05"),
    (0.05, 0.10, "0.05-0.1"),
    (0.10, 0.20, "0.1-0.2"),
    (0.20, 0.40, "0.2-0.4"),
    (0.40, 0.70, "0.4-0.7"),
    (0.70, 0.90, "0.7-0.9"),
    (0.90, 0.99, "0.9-0.99"),
    (0.99, 0.999, "0.99-0.999"),
    (0.999, 1.0, "0.999-1.0"),
]

OUTPUT_SPECS = [
    ("qrtend", "qrtend_TAU", "pred_qrtend"),
    ("nctend", "nctend_TAU", "pred_nctend"),
    ("nrtend", "nrtend_TAU", "pred_nrtend"),
    ("qctend", "qctend_TAU", "pred_qctend"),
]

SIGN_SUBSETS = [
    ("all_rows", "ALL"),
    ("target_nrtend_negative", "target nrtend < 0"),
    ("target_nrtend_positive", "target nrtend > 0"),
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
        help="Path to the TorchScript model.",
    )
    p.add_argument(
        "--h1-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="Directory containing tau_2mos H1 files.",
    )
    p.add_argument(
        "--sample-rows-per-file",
        type=int,
        default=120000,
        help="Fixed sampled non-sentinel exact-stage-B rows per H1 file.",
    )
    p.add_argument(
        "--out-dir",
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis" / "cloud_sign_error_budget"),
        help="Output directory for CSVs, plots, and summary note.",
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


def sample_h1_file_exact_stage_b_nonsentinel(file_path, sample_rows_per_file, rng):
    ds = nc.Dataset(file_path)
    nt = len(ds.dimensions["time"])
    nlev = len(ds.dimensions["lev"])
    lev_values = to_ndarray(ds.variables["lev"][:nlev])
    # Slight oversampling per timestep so the final per-file sample can be fixed.
    per_timestep_quota = int(math.ceil(sample_rows_per_file * 1.10 / max(nt, 1)))

    file_parts = []
    for t in range(nt):
        arrays = read_timestep_arrays(ds, t)
        masks = compute_masks(arrays)
        if not np.any(masks["exact_stage_b"]):
            continue
        df = build_rows_from_mask(arrays, masks["exact_stage_b"], lev_values)
        if df.empty:
            continue
        sentinel_mask = (df[CURRENT_SENTINEL_COLS] == SENTINEL_VALUE).any(axis=1)
        df = df.loc[~sentinel_mask].reset_index(drop=True)
        if df.empty:
            continue
        file_parts.append(sample_frame(df, per_timestep_quota, rng))

    ds.close()

    if not file_parts:
        raise RuntimeError(f"No non-sentinel exact-stage-B rows in {file_path}")

    file_df = pd.concat(file_parts, ignore_index=True)
    return sample_frame(file_df, sample_rows_per_file, rng)


def assign_cloud_bin(values):
    values = np.asarray(values, dtype=np.float64)
    labels = np.empty(len(values), dtype=object)
    labels[:] = "OUT_OF_RANGE"

    for i, (lo, hi, label) in enumerate(CLOUD_BINS):
        if i == len(CLOUD_BINS) - 1:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        labels[mask] = label
    return labels


def sign_subset_mask(sign_subset, nrtend_values):
    if sign_subset == "all_rows":
        return np.ones(len(nrtend_values), dtype=bool)
    if sign_subset == "target_nrtend_positive":
        return nrtend_values > 0
    if sign_subset == "target_nrtend_negative":
        return nrtend_values < 0
    raise ValueError(f"Unknown sign subset: {sign_subset}")


def format_interval(label):
    return label


def summarize_array(values, prefix):
    out = {}
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_std"] = np.nan
        out[f"{prefix}_p5"] = np.nan
        out[f"{prefix}_p25"] = np.nan
        out[f"{prefix}_p50"] = np.nan
        out[f"{prefix}_p95"] = np.nan
        out[f"{prefix}_p99"] = np.nan
        return out

    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values))
    out[f"{prefix}_p5"] = float(np.percentile(values, 5))
    out[f"{prefix}_p25"] = float(np.percentile(values, 25))
    out[f"{prefix}_p50"] = float(np.percentile(values, 50))
    out[f"{prefix}_p95"] = float(np.percentile(values, 95))
    out[f"{prefix}_p99"] = float(np.percentile(values, 99))
    return out


def build_scored_h1_sample(h1_dir, model, sample_rows_per_file, rng):
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No H1 files found in {h1_dir}")

    parts = []
    for i, file_path in enumerate(h1_files):
        file_name = os.path.basename(file_path)
        print(f"Sampling file {i + 1}/{len(h1_files)}: {file_name}")
        df = sample_h1_file_exact_stage_b_nonsentinel(
            file_path=file_path,
            sample_rows_per_file=sample_rows_per_file,
            rng=rng,
        )
        print(f"  sampled rows: {len(df):,}")
        preds = run_model(model, df[INPUT_COLS].to_numpy(dtype=np.float32))
        scored = pd.DataFrame(
            {
                "file_name": file_name,
                "CLOUD": df["CLOUD"].to_numpy(dtype=np.float64),
                "cloud_bin": assign_cloud_bin(df["CLOUD"].to_numpy(dtype=np.float64)),
                "qrtend_TAU": df["qrtend_TAU"].to_numpy(dtype=np.float64),
                "nctend_TAU": df["nctend_TAU"].to_numpy(dtype=np.float64),
                "nrtend_TAU": df["nrtend_TAU"].to_numpy(dtype=np.float64),
                "qctend_TAU": df["qctend_TAU"].to_numpy(dtype=np.float64),
                "pred_qrtend": preds[:, 0].astype(np.float64),
                "pred_nctend": preds[:, 1].astype(np.float64),
                "pred_nrtend": preds[:, 2].astype(np.float64),
                "pred_qctend": preds[:, 3].astype(np.float64),
            }
        )
        parts.append(scored)

    full_df = pd.concat(parts, ignore_index=True)
    if (full_df["cloud_bin"] == "OUT_OF_RANGE").any():
        bad = int((full_df["cloud_bin"] == "OUT_OF_RANGE").sum())
        raise RuntimeError(f"Found {bad} rows outside the requested CLOUD bins.")
    return full_df


def compute_global_totals(df):
    finite_mask = np.isfinite(df[[c for _, _, c in OUTPUT_SPECS]]).all(axis=1).to_numpy()
    finite_mask &= np.isfinite(df[[c for _, c, _ in OUTPUT_SPECS]]).all(axis=1).to_numpy()
    finite_df = df.loc[finite_mask].reset_index(drop=True)

    totals = {
        "all_rows": int(len(df)),
        "finite_rows": int(len(finite_df)),
        "positive_rows": int((finite_df["nrtend_TAU"] > 0).sum()),
        "negative_rows": int((finite_df["nrtend_TAU"] < 0).sum()),
        "nrtend_sse_total": float(np.square(finite_df["pred_nrtend"] - finite_df["nrtend_TAU"]).sum()),
        "nrtend_abs_total": float(np.abs(finite_df["pred_nrtend"] - finite_df["nrtend_TAU"]).sum()),
        "nctend_sse_total": float(np.square(finite_df["pred_nctend"] - finite_df["nctend_TAU"]).sum()),
        "nctend_abs_total": float(np.abs(finite_df["pred_nctend"] - finite_df["nctend_TAU"]).sum()),
    }
    return finite_df, totals


def slice_metrics(df, totals, cloud_label, sign_subset):
    nrt = df["nrtend_TAU"].to_numpy(dtype=np.float64)
    mask = (df["cloud_bin"] == cloud_label).to_numpy()
    mask &= sign_subset_mask(sign_subset, nrt)
    slice_df = df.loc[mask].reset_index(drop=True)

    row = {
        "cloud_bin": cloud_label,
        "sign_subset": sign_subset,
        "count": int(len(slice_df)),
        "fraction_of_all_rows": len(slice_df) / float(max(totals["finite_rows"], 1)),
    }

    if sign_subset == "all_rows":
        parent_rows = totals["finite_rows"]
    elif sign_subset == "target_nrtend_positive":
        parent_rows = totals["positive_rows"]
    else:
        parent_rows = totals["negative_rows"]
    row["fraction_of_parent_set"] = len(slice_df) / float(max(parent_rows, 1))

    if slice_df.empty:
        row["n_nonfinite_prediction_rows"] = 0
        row["n_finite_rows"] = 0
        for name, _, _ in OUTPUT_SPECS:
            row[f"{name}_r2"] = np.nan
            row[f"{name}_rmse"] = np.nan
            row[f"{name}_mae"] = np.nan
            row[f"{name}_sign_accuracy"] = np.nan
            row.update(summarize_array([], f"{name}_target"))
            row.update(summarize_array([], f"{name}_pred"))
        row["nrtend_sse_sum"] = 0.0
        row["nrtend_sse_fraction_of_total"] = 0.0
        row["nrtend_abs_sum"] = 0.0
        row["nrtend_abs_fraction_of_total"] = 0.0
        row["nctend_sse_sum"] = 0.0
        row["nctend_sse_fraction_of_total"] = 0.0
        row["nctend_abs_sum"] = 0.0
        row["nctend_abs_fraction_of_total"] = 0.0
        return row

    pred_cols = [pred_col for _, _, pred_col in OUTPUT_SPECS]
    targ_cols = [targ_col for _, targ_col, _ in OUTPUT_SPECS]
    finite_mask = np.isfinite(slice_df[pred_cols]).all(axis=1).to_numpy()
    finite_mask &= np.isfinite(slice_df[targ_cols]).all(axis=1).to_numpy()
    row["n_nonfinite_prediction_rows"] = int((~finite_mask).sum())
    slice_df = slice_df.loc[finite_mask].reset_index(drop=True)
    row["n_finite_rows"] = int(len(slice_df))

    if slice_df.empty:
        for name, _, _ in OUTPUT_SPECS:
            row[f"{name}_r2"] = np.nan
            row[f"{name}_rmse"] = np.nan
            row[f"{name}_mae"] = np.nan
            row[f"{name}_sign_accuracy"] = np.nan
            row.update(summarize_array([], f"{name}_target"))
            row.update(summarize_array([], f"{name}_pred"))
        row["nrtend_sse_sum"] = 0.0
        row["nrtend_sse_fraction_of_total"] = 0.0
        row["nrtend_abs_sum"] = 0.0
        row["nrtend_abs_fraction_of_total"] = 0.0
        row["nctend_sse_sum"] = 0.0
        row["nctend_sse_fraction_of_total"] = 0.0
        row["nctend_abs_sum"] = 0.0
        row["nctend_abs_fraction_of_total"] = 0.0
        return row

    for name, target_col, pred_col in OUTPUT_SPECS:
        target = slice_df[target_col].to_numpy(dtype=np.float64)
        pred = slice_df[pred_col].to_numpy(dtype=np.float64)
        diff = pred - target
        row[f"{name}_r2"] = float(r2_score(target, pred))
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(diff))))
        row[f"{name}_mae"] = float(np.mean(np.abs(diff)))
        row[f"{name}_sign_accuracy"] = float(np.mean(np.sign(pred) == np.sign(target)))
        row.update(summarize_array(target, f"{name}_target"))
        row.update(summarize_array(pred, f"{name}_pred"))

    nrt_diff = slice_df["pred_nrtend"].to_numpy(dtype=np.float64) - slice_df["nrtend_TAU"].to_numpy(dtype=np.float64)
    row["nrtend_sse_sum"] = float(np.square(nrt_diff).sum())
    row["nrtend_sse_fraction_of_total"] = row["nrtend_sse_sum"] / float(max(totals["nrtend_sse_total"], 1.0e-30))
    row["nrtend_abs_sum"] = float(np.abs(nrt_diff).sum())
    row["nrtend_abs_fraction_of_total"] = row["nrtend_abs_sum"] / float(max(totals["nrtend_abs_total"], 1.0e-30))

    nct_diff = slice_df["pred_nctend"].to_numpy(dtype=np.float64) - slice_df["nctend_TAU"].to_numpy(dtype=np.float64)
    row["nctend_sse_sum"] = float(np.square(nct_diff).sum())
    row["nctend_sse_fraction_of_total"] = row["nctend_sse_sum"] / float(max(totals["nctend_sse_total"], 1.0e-30))
    row["nctend_abs_sum"] = float(np.abs(nct_diff).sum())
    row["nctend_abs_fraction_of_total"] = row["nctend_abs_sum"] / float(max(totals["nctend_abs_total"], 1.0e-30))
    return row


def build_slice_metrics(df):
    finite_df, totals = compute_global_totals(df)
    rows = []
    for cloud_label in [label for _, _, label in CLOUD_BINS]:
        for sign_subset, _ in SIGN_SUBSETS:
            rows.append(slice_metrics(finite_df, totals, cloud_label, sign_subset))
    return pd.DataFrame(rows), totals


def format_heat_value(value, kind):
    if not np.isfinite(value):
        return "nan"
    if kind == "fraction":
        return f"{100.0 * value:.1f}%"
    if kind == "percent":
        return f"{100.0 * value:.1f}%"
    if kind == "r2":
        if abs(value) >= 1000 or abs(value) < 1.0e-2:
            return f"{value:.1e}"
        return f"{value:.2f}"
    return f"{value:.2f}"


def heatmap_values(df, sign_subset, value_col):
    sign_rows = ["target_nrtend_negative", "target_nrtend_positive"]
    cloud_labels = [label for _, _, label in CLOUD_BINS]
    arr = np.full((len(sign_rows), len(cloud_labels)), np.nan, dtype=np.float64)
    for i, sign in enumerate(sign_rows):
        for j, cloud in enumerate(cloud_labels):
            match = df[(df["sign_subset"] == sign) & (df["cloud_bin"] == cloud)]
            if not match.empty:
                arr[i, j] = float(match.iloc[0][value_col])
    return arr, sign_rows, cloud_labels


def plot_heatmaps(metrics_df, output_path, spec_list, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 6.5))
    axes = axes.ravel()
    ylabels = ["target nrtend < 0", "target nrtend > 0"]

    for ax, (col, panel_title, cmap, kind, norm) in zip(axes, spec_list):
        arr, _, cloud_labels = heatmap_values(metrics_df, None, col)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(cloud_labels)))
        ax.set_xticklabels(cloud_labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_title(panel_title)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                text = format_heat_value(arr[i, j], kind)
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if kind == "fraction":
            cbar.set_label("Fraction")
        elif kind == "percent":
            cbar.set_label("Percent")
        else:
            cbar.set_label(panel_title)

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def top_slices_text(df, value_col, top_k=5):
    top = df[df["sign_subset"] != "all_rows"].sort_values(value_col, ascending=False).head(top_k)
    lines = []
    for row in top.itertuples():
        sign_label = "nrtend<0" if row.sign_subset == "target_nrtend_negative" else "nrtend>0"
        lines.append(
            f"  {sign_label}, CLOUD {row.cloud_bin}: "
            f"rows={row.count:,}, frac_all={100.0 * row.fraction_of_all_rows:.2f}%, "
            f"{value_col}={100.0 * getattr(row, value_col):.2f}%"
        )
    return lines


def write_summary_note(out_path, metrics_df, totals, sample_rows_per_file, n_files):
    posneg_df = metrics_df[metrics_df["sign_subset"] != "all_rows"].copy()

    top_nrt = posneg_df.sort_values("nrtend_sse_fraction_of_total", ascending=False).iloc[0]
    top_nct = posneg_df.sort_values("nctend_sse_fraction_of_total", ascending=False).iloc[0]

    low_cloud_positive = posneg_df[
        (posneg_df["sign_subset"] == "target_nrtend_positive")
        & (posneg_df["cloud_bin"].isin(["0.01-0.02", "0.02-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", "0.4-0.7", "0.7-0.9"]))
    ]
    high_cloud_positive = posneg_df[
        (posneg_df["sign_subset"] == "target_nrtend_positive")
        & (posneg_df["cloud_bin"].isin(["0.9-0.99", "0.99-0.999", "0.999-1.0"]))
    ]

    def summed(df, col):
        return float(df[col].sum()) if not df.empty else np.nan

    low_pos_rows = int(low_cloud_positive["count"].sum())
    high_pos_rows = int(high_cloud_positive["count"].sum())
    low_neg = posneg_df[
        (posneg_df["sign_subset"] == "target_nrtend_negative")
        & (posneg_df["cloud_bin"].isin(["0.01-0.02", "0.02-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", "0.4-0.7", "0.7-0.9"]))
    ]
    low_neg_rows = int(low_neg["count"].sum())

    lines = [
        "cloud/sign error-budget verdict",
        "",
        f"- Base dataset: exact Stage A+B H1 rows, non-sentinel only, {sample_rows_per_file:,} sampled rows per H1 file across {n_files} files.",
        f"- Total finite scored rows: {totals['finite_rows']:,}; positive nrtend rows: {totals['positive_rows']:,}; negative nrtend rows: {totals['negative_rows']:,}.",
        (
            f"- Largest nrtend SSE slice: "
            f"{'nrtend>0' if top_nrt['sign_subset']=='target_nrtend_positive' else 'nrtend<0'}, "
            f"CLOUD {top_nrt['cloud_bin']}, contributing {100.0 * top_nrt['nrtend_sse_fraction_of_total']:.2f}% "
            f"of total H1 nrtend SSE from only {100.0 * top_nrt['fraction_of_all_rows']:.2f}% of rows."
        ),
        (
            f"- Largest nctend SSE slice: "
            f"{'nrtend>0' if top_nct['sign_subset']=='target_nrtend_positive' else 'nrtend<0'}, "
            f"CLOUD {top_nct['cloud_bin']}, contributing {100.0 * top_nct['nctend_sse_fraction_of_total']:.2f}% "
            f"of total H1 nctend SSE from only {100.0 * top_nct['fraction_of_all_rows']:.2f}% of rows."
        ),
        (
            f"- Positive rows with CLOUD < 0.9 account for {low_pos_rows:,} rows "
            f"({100.0 * low_pos_rows / max(totals['finite_rows'], 1):.2f}% of all rows) and "
            f"{100.0 * summed(low_cloud_positive, 'nrtend_sse_fraction_of_total'):.2f}% of total nrtend SSE "
            f"and {100.0 * summed(low_cloud_positive, 'nrtend_abs_fraction_of_total'):.2f}% of total nrtend absolute error."
        ),
        (
            f"- Positive rows with CLOUD >= 0.9 account for {high_pos_rows:,} rows "
            f"({100.0 * high_pos_rows / max(totals['finite_rows'], 1):.2f}% of all rows) and "
            f"{100.0 * summed(high_cloud_positive, 'nrtend_sse_fraction_of_total'):.2f}% of total nrtend SSE."
        ),
        (
            f"- Negative rows with CLOUD < 0.9 account for {low_neg_rows:,} rows "
            f"({100.0 * low_neg_rows / max(totals['finite_rows'], 1):.2f}% of all rows) and "
            f"{100.0 * summed(low_neg, 'nctend_sse_fraction_of_total'):.2f}% of total nctend SSE."
        ),
        f"- Non-finite predictions on this non-sentinel audit sample: {int(metrics_df.loc[metrics_df['sign_subset'] == 'all_rows', 'n_nonfinite_prediction_rows'].sum()):,}.",
        "- Interpretation: nrtend failure is overwhelmingly a low-cloud positive-formation problem, while nctend failure is dominated by a broader low-cloud negative-nrtend regime rather than the same tiny positive slice.",
        "",
        "- Top nrtend SSE slices:",
        *top_slices_text(posneg_df, "nrtend_sse_fraction_of_total", top_k=5),
        "",
        "- Top nctend SSE slices:",
        *top_slices_text(posneg_df, "nctend_sse_fraction_of_total", top_k=5),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    csv_path = out_dir / "cloud_sign_slice_metrics.csv"
    note_path = out_dir / "summary_note.txt"
    nrt_plot_path = out_dir / "nrtend_cloud_sign_error_budget.png"
    nct_plot_path = out_dir / "nctend_cloud_sign_error_budget.png"

    rng = np.random.default_rng(args.seed)
    print("Loading TorchScript model ...")
    model = load_model(args.model)

    print("Building fixed-per-file exact-stage-B non-sentinel H1 sample ...")
    scored_df = build_scored_h1_sample(
        h1_dir=args.h1_dir,
        model=model,
        sample_rows_per_file=args.sample_rows_per_file,
        rng=rng,
    )
    print(f"Scored rows: {len(scored_df):,}")

    print("Computing slice metrics ...")
    metrics_df, totals = build_slice_metrics(scored_df)
    metrics_df.to_csv(csv_path, index=False)

    frac_norm = Normalize(vmin=0.0, vmax=max(metrics_df["fraction_of_all_rows"].max(), metrics_df["nrtend_sse_fraction_of_total"].max(), metrics_df["nctend_sse_fraction_of_total"].max(), 1.0e-12))
    nrt_specs = [
        ("fraction_of_all_rows", "Row Fraction Of All H1 Rows", "cividis", "fraction", frac_norm),
        ("nrtend_sse_fraction_of_total", "nrtend SSE Fraction Of Total", "magma", "fraction", Normalize(vmin=0.0, vmax=max(metrics_df["nrtend_sse_fraction_of_total"].max(), 1.0e-12))),
        ("nrtend_sign_accuracy", "nrtend Sign Accuracy", "viridis", "percent", Normalize(vmin=0.0, vmax=1.0)),
        ("nrtend_r2", "nrtend R2", "RdYlBu", "r2", TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)),
    ]
    nct_specs = [
        ("fraction_of_all_rows", "Row Fraction Of All H1 Rows", "cividis", "fraction", frac_norm),
        ("nctend_sse_fraction_of_total", "nctend SSE Fraction Of Total", "magma", "fraction", Normalize(vmin=0.0, vmax=max(metrics_df["nctend_sse_fraction_of_total"].max(), 1.0e-12))),
        ("nctend_abs_fraction_of_total", "nctend Absolute Error Fraction", "plasma", "fraction", Normalize(vmin=0.0, vmax=max(metrics_df["nctend_abs_fraction_of_total"].max(), 1.0e-12))),
        ("nctend_r2", "nctend R2", "RdYlBu", "r2", TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)),
    ]
    plot_heatmaps(metrics_df, nrt_plot_path, nrt_specs, "H1 nrtend Error Budget By CLOUD Bin And nrtend Sign")
    plot_heatmaps(metrics_df, nct_plot_path, nct_specs, "H1 nctend Error Budget By CLOUD Bin And nrtend Sign")

    n_files = len(sorted(glob.glob(os.path.join(args.h1_dir, "*.h1.*.nc"))))
    write_summary_note(note_path, metrics_df, totals, args.sample_rows_per_file, n_files)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {nrt_plot_path}")
    print(f"Saved plot: {nct_plot_path}")
    print(f"Saved note: {note_path}")


if __name__ == "__main__":
    main()
