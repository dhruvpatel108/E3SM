#!/usr/bin/env python3
"""
Slice-level semantic and constraint audit across parquet train/validation and
two TAU/H1 runs.

This analysis uses:
  - parquet train (first 80%)
  - parquet validation (last 20%)
  - TAU_old exact Stage A+B, non-sentinel only
  - TAU_new exact Stage A+B, non-sentinel only

For the four dominant slices, it produces:
  - slice metrics with model performance
  - raw target/input summaries
  - prediction summaries
  - simple semantic/constraint checks
  - per-slice overlaid histogram grids
  - a small H1 compatibility summary
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

USER_TMP = Path("/tmp") / os.environ.get("USER", "user")
USER_TMP.mkdir(parents=True, exist_ok=True)
MPL_DIR = USER_TMP / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = USER_TMP / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc

from audit_h1_cloud_sign_error_budget import sample_h1_file_exact_stage_b_nonsentinel
from diagnose_tau_h1_mismatch import (
    CURRENT_SENTINEL_COLS,
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

RAW_VARS = INPUT_COLS + MODEL_TARGET_COLS + ["lev"]

HIST_VAR_ORDER = [
    "QC_TAU_in",
    "QR_TAU_in",
    "NC_TAU_in",
    "NR_TAU_in",
    "PGAM",
    "LAMC",
    "LAMR",
    "N0R",
    "RHO_CLUBB",
    "CLOUD",
    "FREQR",
    "lev",
    "qrtend_TAU",
    "nctend_TAU",
    "nrtend_TAU",
    "qctend_TAU",
]

LOG10_VARS = {
    "QC_TAU_in",
    "QR_TAU_in",
    "NC_TAU_in",
    "NR_TAU_in",
    "LAMC",
    "LAMR",
    "N0R",
}

SIGN_LOG10_VARS = {
    "qrtend_TAU",
    "nctend_TAU",
    "qctend_TAU",
}

NRTEND_ASINH_SCALE = 1.0e-3

DATASET_ORDER = [
    "train_parquet_stage_b",
    "validation_parquet_stage_b",
    "TAU_old",
    "TAU_new",
]

DATASET_LABELS = {
    "train_parquet_stage_b": "Train",
    "validation_parquet_stage_b": "Validation",
    "TAU_old": "TAU_old",
    "TAU_new": "TAU_new",
}

DATASET_COLORS = {
    "train_parquet_stage_b": "#264653",
    "validation_parquet_stage_b": "#2a9d8f",
    "TAU_old": "#e76f51",
    "TAU_new": "#6d597a",
}

REQUIRED_H1_VARS = [
    "P3_qc_in_TAU",
    "P3_qr_in_TAU",
    "P3_nc_in_TAU",
    "P3_nr_in_TAU",
    "P3_mu_c",
    "P3_lamc",
    "P3_lamr",
    "P3_nr",
    "RHO_CLUBB",
    "CLOUD",
    "FREQR",
    "P3_qrtend_TAU_raw",
    "P3_nctend_TAU_raw",
    "P3_nrtend_TAU_raw",
    "P3_qctend_TAU_raw",
    "lev",
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
        "--tau-old-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="Directory for the original TAU/H1 data.",
    )
    p.add_argument(
        "--tau-new-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_emulator2_2mos",
        help="Directory for the new TAU/H1 data.",
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
        "--parquet-max-files",
        type=int,
        default=None,
        help="Optional parquet file cap. Useful for smoke tests.",
    )
    p.add_argument(
        "--h1-max-files",
        type=int,
        default=None,
        help="Optional H1 file cap. Useful for smoke tests.",
    )
    p.add_argument(
        "--compatibility-smoke-rows",
        type=int,
        default=2000,
        help="Rows used for the H1 compatibility smoke sample.",
    )
    p.add_argument(
        "--compatibility-smoke-preds",
        type=int,
        default=256,
        help="Rows used for the H1 inference smoke test.",
    )
    p.add_argument(
        "--out-dir",
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis" / "slice_semantic_constraints_multi_tau"),
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
    (path / "plots").mkdir(parents=True, exist_ok=True)
    (path / "csv").mkdir(parents=True, exist_ok=True)
    return path


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
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "neg_frac": np.nan,
            "zero_frac": np.nan,
            "pos_frac": np.nan,
        }

    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p5": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "neg_frac": float(np.mean(finite < 0)),
        "zero_frac": float(np.mean(finite == 0)),
        "pos_frac": float(np.mean(finite > 0)),
    }


def load_fixed_h1_sample(h1_dir, rows_per_file, rng, max_files=None):
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No H1 files found in {h1_dir}")
    if max_files is not None:
        h1_files = h1_files[:max_files]

    parts = []
    for i, file_path in enumerate(h1_files):
        print(
            f"Sampling H1 file {i + 1}/{len(h1_files)} from {os.path.basename(h1_dir)}: "
            f"{os.path.basename(file_path)}"
        )
        df = sample_h1_file_exact_stage_b_nonsentinel(file_path, rows_per_file, rng)
        parts.append(df.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def normalized_h1_file_names(h1_dir, max_files=None):
    files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if max_files is not None:
        files = files[:max_files]
    normed = []
    base = os.path.basename(h1_dir)
    for path in files:
        name = os.path.basename(path).replace(base, "RUN")
        normed.append(name)
    return normed


def inspect_h1_directory(label, h1_dir, model, smoke_rows, smoke_preds, rng, max_files=None):
    files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not files:
        raise FileNotFoundError(f"No H1 files found in {h1_dir}")
    if max_files is not None:
        files = files[:max_files]

    first_file = files[0]
    last_file = files[-1]
    ds = nc.Dataset(first_file)
    dims = {k: len(v) for k, v in ds.dimensions.items()}
    vars_present = set(ds.variables.keys())
    missing = [name for name in REQUIRED_H1_VARS if name not in vars_present]
    ds.close()

    smoke_df = sample_h1_file_exact_stage_b_nonsentinel(first_file, smoke_rows, rng)
    smoke_inputs = smoke_df[INPUT_COLS].to_numpy(dtype=np.float32)
    smoke_preds = run_model(model, smoke_inputs[:smoke_preds])

    row = {
        "dataset": label,
        "h1_dir": h1_dir,
        "file_count": len(files),
        "first_file": os.path.basename(first_file),
        "last_file": os.path.basename(last_file),
        "dims_json": json.dumps(dims, sort_keys=True),
        "missing_required_vars_json": json.dumps(missing),
        "missing_required_var_count": len(missing),
        "smoke_rows_sampled": int(len(smoke_df)),
        "smoke_pred_rows": int(len(smoke_preds)),
        "smoke_pred_finite_frac": float(np.isfinite(smoke_preds).all(axis=1).mean()),
        "smoke_has_required_columns": int(
            all(col in smoke_df.columns for col in INPUT_COLS + MODEL_TARGET_COLS + ["lev"])
        ),
        "smoke_nrtend_target_p50": float(np.percentile(smoke_df["nrtend_TAU"], 50)),
        "smoke_sentinel_columns_checked": json.dumps(CURRENT_SENTINEL_COLS),
    }
    return row


def transform_for_hist(var_name, values):
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]

    if var_name in LOG10_VARS:
        return np.log10(np.maximum(values, 1.0e-30)), "log10"
    if var_name in SIGN_LOG10_VARS:
        return np.sign(values) * np.log10(np.maximum(np.abs(values), 1.0e-30)), "signlog10"
    if var_name == "nrtend_TAU":
        return np.arcsinh(values / NRTEND_ASINH_SCALE), "asinh(x/1e-3)"
    return values, "raw"


def constraint_metrics(df):
    if df.empty:
        return {
            "target_qr_qc_closure_mean": np.nan,
            "target_qr_qc_closure_abs_p50": np.nan,
            "target_qr_qc_closure_abs_p95": np.nan,
            "target_qr_qc_sign_opposition_frac": np.nan,
            "pred_qr_qc_closure_mean": np.nan,
            "pred_qr_qc_closure_abs_p50": np.nan,
            "pred_qr_qc_closure_abs_p95": np.nan,
            "pred_qr_qc_sign_opposition_frac": np.nan,
            "target_qr_qc_abs_ratio_p50": np.nan,
            "pred_qr_qc_abs_ratio_p50": np.nan,
            "target_nct_negative_frac": np.nan,
            "pred_nct_negative_frac": np.nan,
            "target_nrt_positive_frac": np.nan,
            "pred_nrt_positive_frac": np.nan,
        }

    def closure_stats(a, b, prefix):
        closure = a + b
        abs_closure = np.abs(closure)
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-30)
        ratio = np.abs(a) / np.maximum(np.abs(b), 1.0e-30)
        sign_opp = ((a == 0) | (b == 0) | (np.sign(a) == -np.sign(b)))
        return {
            f"{prefix}_closure_mean": float(np.mean(closure)),
            f"{prefix}_closure_abs_p50": float(np.percentile(abs_closure, 50)),
            f"{prefix}_closure_abs_p95": float(np.percentile(abs_closure, 95)),
            f"{prefix}_sign_opposition_frac": float(np.mean(sign_opp)),
            f"{prefix}_abs_ratio_p50": float(np.percentile(ratio, 50)),
            f"{prefix}_closure_rel_abs_p50": float(np.percentile(abs_closure / denom, 50)),
            f"{prefix}_closure_rel_abs_p95": float(np.percentile(abs_closure / denom, 95)),
        }

    out = {}
    out.update(closure_stats(df["qrtend_TAU"].to_numpy(dtype=np.float64), df["qctend_TAU"].to_numpy(dtype=np.float64), "target_qr_qc"))
    out.update(
        closure_stats(
            df["pred_qrtend"].to_numpy(dtype=np.float64),
            df["pred_qctend"].to_numpy(dtype=np.float64),
            "pred_qr_qc",
        )
    )
    out["target_nct_negative_frac"] = float(np.mean(df["nctend_TAU"].to_numpy(dtype=np.float64) < 0))
    out["pred_nct_negative_frac"] = float(np.mean(df["pred_nctend"].to_numpy(dtype=np.float64) < 0))
    out["target_nrt_positive_frac"] = float(np.mean(df["nrtend_TAU"].to_numpy(dtype=np.float64) > 0))
    out["pred_nrt_positive_frac"] = float(np.mean(df["pred_nrtend"].to_numpy(dtype=np.float64) > 0))
    return out


def analyze_dataset(model, dataset_name, df):
    metrics_rows = []
    raw_rows = []
    pred_rows = []
    semantic_rows = []
    slice_frames = {}

    total_rows = len(df)
    selected = df.loc[union_mask(df)].reset_index(drop=True)
    print(f"  {dataset_name}: scoring {len(selected):,} union-slice rows")
    preds = run_model(model, selected[INPUT_COLS].to_numpy(dtype=np.float32))
    selected = selected.copy()
    selected["pred_qrtend"] = preds[:, 0].astype(np.float64)
    selected["pred_nctend"] = preds[:, 1].astype(np.float64)
    selected["pred_nrtend"] = preds[:, 2].astype(np.float64)
    selected["pred_qctend"] = preds[:, 3].astype(np.float64)

    pred_cols = [pred_col for _, _, pred_col in OUTPUT_SPECS]
    target_cols = [target_col for _, target_col, _ in OUTPUT_SPECS]

    for slice_name, slice_desc in SLICE_DEFS:
        full_mask = slice_mask(df, slice_name)
        slice_count = int(full_mask.sum())
        slice_fraction = slice_count / float(max(total_rows, 1))

        slice_df = selected.loc[slice_mask(selected, slice_name)].reset_index(drop=True)
        finite_mask = np.isfinite(slice_df[pred_cols]).all(axis=1).to_numpy()
        finite_mask &= np.isfinite(slice_df[target_cols]).all(axis=1).to_numpy()
        nonfinite_rows = int((~finite_mask).sum())
        slice_df = slice_df.loc[finite_mask].reset_index(drop=True)
        slice_frames[(dataset_name, slice_name)] = slice_df[RAW_VARS + pred_cols].copy()

        metric_row = {
            "dataset": dataset_name,
            "slice_name": slice_name,
            "slice_desc": slice_desc,
            "count": slice_count,
            "fraction_of_dataset": slice_fraction,
            "n_nonfinite_prediction_rows": nonfinite_rows,
            "n_finite_rows": int(len(slice_df)),
        }

        for name, target_col, pred_col in OUTPUT_SPECS:
            target = slice_df[target_col].to_numpy(dtype=np.float64)
            pred = slice_df[pred_col].to_numpy(dtype=np.float64)
            if target.size == 0:
                metric_row[f"{name}_r2"] = np.nan
                metric_row[f"{name}_rmse"] = np.nan
                metric_row[f"{name}_mae"] = np.nan
                metric_row[f"{name}_sign_accuracy"] = np.nan
            else:
                diff = pred - target
                metric_row[f"{name}_r2"] = float(r2_score(target, pred))
                metric_row[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(diff))))
                metric_row[f"{name}_mae"] = float(np.mean(np.abs(diff)))
                metric_row[f"{name}_sign_accuracy"] = float(np.mean(np.sign(pred) == np.sign(target)))
        metrics_rows.append(metric_row)

        for var_name in RAW_VARS:
            raw_rows.append(
                {
                    "dataset": dataset_name,
                    "slice_name": slice_name,
                    "slice_desc": slice_desc,
                    "source": "raw",
                    "variable": var_name,
                    **summarize_values(slice_df[var_name].to_numpy(dtype=np.float64)),
                }
            )

        for name, _, pred_col in OUTPUT_SPECS:
            pred_rows.append(
                {
                    "dataset": dataset_name,
                    "slice_name": slice_name,
                    "slice_desc": slice_desc,
                    "source": "pred",
                    "variable": name,
                    **summarize_values(slice_df[pred_col].to_numpy(dtype=np.float64)),
                }
            )

        semantic_rows.append(
            {
                "dataset": dataset_name,
                "slice_name": slice_name,
                "slice_desc": slice_desc,
                "count": slice_count,
                "n_finite_rows": int(len(slice_df)),
                **constraint_metrics(slice_df),
            }
        )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(raw_rows),
        pd.DataFrame(pred_rows),
        pd.DataFrame(semantic_rows),
        slice_frames,
    )


def representative_rows(dataset_name, slice_name, df):
    cols = [
        "QC_TAU_in",
        "QR_TAU_in",
        "NC_TAU_in",
        "NR_TAU_in",
        "PGAM",
        "LAMC",
        "LAMR",
        "N0R",
        "CLOUD",
        "FREQR",
        "lev",
        "qrtend_TAU",
        "nctend_TAU",
        "nrtend_TAU",
        "qctend_TAU",
    ]
    if df.empty:
        return pd.DataFrame(columns=["dataset", "slice_name", "representative_rank"] + cols)

    target = df["nrtend_TAU"].to_numpy(dtype=np.float64)
    if np.all(target == target[0]):
        pick_idx = np.array([0, min(len(df) - 1, len(df) // 2), len(df) - 1], dtype=int)
    else:
        quantiles = np.percentile(target, [10, 50, 90])
        pick_idx = []
        for q in quantiles:
            pick_idx.append(int(np.argmin(np.abs(target - q))))
        pick_idx = np.asarray(pick_idx, dtype=int)

    out = df.iloc[pick_idx][cols].copy().reset_index(drop=True)
    out.insert(0, "representative_rank", ["q10_like", "q50_like", "q90_like"])
    out.insert(0, "slice_name", slice_name)
    out.insert(0, "dataset", dataset_name)
    return out


def plot_slice_histograms(slice_name, slice_desc, slice_frames, output_path):
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for ax, var_name in zip(axes, HIST_VAR_ORDER):
        transformed_by_dataset = {}
        mode = None
        pooled = []
        for dataset_name in DATASET_ORDER:
            df = slice_frames[(dataset_name, slice_name)]
            vals, mode_here = transform_for_hist(var_name, df[var_name].to_numpy(dtype=np.float64))
            transformed_by_dataset[dataset_name] = vals
            if vals.size:
                pooled.append(vals)
            if mode is None:
                mode = mode_here

        if not pooled:
            ax.set_visible(False)
            continue

        pooled = np.concatenate(pooled)
        if pooled.size > 100:
            lo = float(np.percentile(pooled, 0.5))
            hi = float(np.percentile(pooled, 99.5))
        else:
            lo = float(np.min(pooled))
            hi = float(np.max(pooled))

        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(pooled))
            hi = float(np.nanmax(pooled))
        if hi <= lo:
            hi = lo + 1.0

        bins = np.linspace(lo, hi, 60)
        for dataset_name in DATASET_ORDER:
            vals = transformed_by_dataset[dataset_name]
            if vals.size == 0:
                continue
            hist = ax.hist(
                np.clip(vals, lo, hi),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=DATASET_COLORS[dataset_name],
                label=DATASET_LABELS[dataset_name],
            )
            if var_name == HIST_VAR_ORDER[0]:
                legend_handles.append(hist[2][0])
                legend_labels.append(DATASET_LABELS[dataset_name])

        title = f"{var_name}"
        if mode != "raw":
            title += f"\n[{mode}]"
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.18)

    fig.suptitle(
        f"{slice_name}: {slice_desc}\nOverlay histograms for Train, Validation, TAU_old, TAU_new",
        fontsize=15,
        y=0.995,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_note(out_path, compatibility_df, metrics_df, raw_df, semantic_df):
    def metric(dataset, slice_name, col):
        match = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["slice_name"] == slice_name)]
        if match.empty:
            return np.nan
        return float(match.iloc[0][col])

    def raw_stat(dataset, slice_name, variable, col):
        match = raw_df[
            (raw_df["dataset"] == dataset)
            & (raw_df["slice_name"] == slice_name)
            & (raw_df["variable"] == variable)
        ]
        if match.empty:
            return np.nan
        return float(match.iloc[0][col])

    def sem(dataset, slice_name, col):
        match = semantic_df[(semantic_df["dataset"] == dataset) & (semantic_df["slice_name"] == slice_name)]
        if match.empty:
            return np.nan
        return float(match.iloc[0][col])

    lines = [
        "multi-TAU slice semantic audit",
        "",
        "- Compatibility check:",
    ]
    for _, row in compatibility_df.iterrows():
        lines.append(
            "  {}: files={}, missing_required_vars={}, smoke_pred_finite_frac={:.3f}, "
            "smoke_nrtend_p50={:.4g}".format(
                row["dataset"],
                int(row["file_count"]),
                int(row["missing_required_var_count"]),
                float(row["smoke_pred_finite_frac"]),
                float(row["smoke_nrtend_target_p50"]),
            )
        )

    lines.extend(
        [
            "",
            "- Interpretation focus:",
            "  compare Validation vs TAU_old vs TAU_new inside the same slice definition.",
            "  if prevalence is similar but within-slice state/target magnitudes differ sharply, that points to regime or semantic shift inside the slice.",
            "",
        ]
    )

    for slice_name, slice_desc in SLICE_DEFS:
        lines.append(f"- {slice_name}: {slice_desc}")
        lines.append(
            "  prevalence: Train={:.3f}%, Validation={:.3f}%, TAU_old={:.3f}%, TAU_new={:.3f}%".format(
                100.0 * metric("train_parquet_stage_b", slice_name, "fraction_of_dataset"),
                100.0 * metric("validation_parquet_stage_b", slice_name, "fraction_of_dataset"),
                100.0 * metric("TAU_old", slice_name, "fraction_of_dataset"),
                100.0 * metric("TAU_new", slice_name, "fraction_of_dataset"),
            )
        )
        lines.append(
            "  nrtend R2: Validation={:.4g}, TAU_old={:.4g}, TAU_new={:.4g}".format(
                metric("validation_parquet_stage_b", slice_name, "nrtend_r2"),
                metric("TAU_old", slice_name, "nrtend_r2"),
                metric("TAU_new", slice_name, "nrtend_r2"),
            )
        )
        lines.append(
            "  nctend R2: Validation={:.4g}, TAU_old={:.4g}, TAU_new={:.4g}".format(
                metric("validation_parquet_stage_b", slice_name, "nctend_r2"),
                metric("TAU_old", slice_name, "nctend_r2"),
                metric("TAU_new", slice_name, "nctend_r2"),
            )
        )
        lines.append(
            "  key medians (Validation / TAU_old / TAU_new): CLOUD={:.4g} / {:.4g} / {:.4g}; "
            "NR_TAU_in={:.4g} / {:.4g} / {:.4g}; N0R={:.4g} / {:.4g} / {:.4g}; "
            "nrtend_TAU={:.4g} / {:.4g} / {:.4g}".format(
                raw_stat("validation_parquet_stage_b", slice_name, "CLOUD", "p50"),
                raw_stat("TAU_old", slice_name, "CLOUD", "p50"),
                raw_stat("TAU_new", slice_name, "CLOUD", "p50"),
                raw_stat("validation_parquet_stage_b", slice_name, "NR_TAU_in", "p50"),
                raw_stat("TAU_old", slice_name, "NR_TAU_in", "p50"),
                raw_stat("TAU_new", slice_name, "NR_TAU_in", "p50"),
                raw_stat("validation_parquet_stage_b", slice_name, "N0R", "p50"),
                raw_stat("TAU_old", slice_name, "N0R", "p50"),
                raw_stat("TAU_new", slice_name, "N0R", "p50"),
                raw_stat("validation_parquet_stage_b", slice_name, "nrtend_TAU", "p50"),
                raw_stat("TAU_old", slice_name, "nrtend_TAU", "p50"),
                raw_stat("TAU_new", slice_name, "nrtend_TAU", "p50"),
            )
        )
        lines.append(
            "  qrt+qct target closure |p95 abs|: Validation={:.4g}, TAU_old={:.4g}, TAU_new={:.4g}".format(
                sem("validation_parquet_stage_b", slice_name, "target_qr_qc_closure_abs_p95"),
                sem("TAU_old", slice_name, "target_qr_qc_closure_abs_p95"),
                sem("TAU_new", slice_name, "target_qr_qc_closure_abs_p95"),
            )
        )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    csv_dir = out_dir / "csv"
    plot_dir = out_dir / "plots"

    rng = np.random.default_rng(args.seed)

    print("Loading TorchScript model ...")
    model = load_model(args.model)

    print("Checking H1 compatibility ...")
    compatibility_rows = []
    compatibility_rows.append(
        inspect_h1_directory(
            "TAU_old",
            args.tau_old_dir,
            model,
            args.compatibility_smoke_rows,
            args.compatibility_smoke_preds,
            rng,
            max_files=args.h1_max_files,
        )
    )
    compatibility_rows.append(
        inspect_h1_directory(
            "TAU_new",
            args.tau_new_dir,
            model,
            args.compatibility_smoke_rows,
            args.compatibility_smoke_preds,
            rng,
            max_files=args.h1_max_files,
        )
    )

    old_names = normalized_h1_file_names(args.tau_old_dir, max_files=args.h1_max_files)
    new_names = normalized_h1_file_names(args.tau_new_dir, max_files=args.h1_max_files)
    if old_names != new_names:
        raise RuntimeError("TAU_old and TAU_new file schedules do not align after normalization.")

    compatibility_df = pd.DataFrame(compatibility_rows)
    compatibility_df["normalized_file_schedule_match"] = int(old_names == new_names)
    compatibility_df.to_csv(csv_dir / "h1_compatibility_summary.csv", index=False)

    print("Loading parquet train sample ...")
    train_df = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.parquet_samples_per_dataset,
        rng=rng,
        file_range=(0.0, 0.8),
        max_files=args.parquet_max_files,
    )
    print(f"  train rows: {len(train_df):,}")

    print("Loading parquet validation sample ...")
    val_df = load_parquet_stage_b_sample(
        parquet_dir=args.parquet_dir,
        target_total_samples=args.parquet_samples_per_dataset,
        rng=rng,
        file_range=(0.8, 1.0),
        max_files=args.parquet_max_files,
    )
    print(f"  validation rows: {len(val_df):,}")

    print("Loading TAU_old sample ...")
    tau_old_df = load_fixed_h1_sample(
        args.tau_old_dir,
        args.h1_sample_rows_per_file,
        rng,
        max_files=args.h1_max_files,
    )
    print(f"  TAU_old rows: {len(tau_old_df):,}")

    print("Loading TAU_new sample ...")
    tau_new_df = load_fixed_h1_sample(
        args.tau_new_dir,
        args.h1_sample_rows_per_file,
        rng,
        max_files=args.h1_max_files,
    )
    print(f"  TAU_new rows: {len(tau_new_df):,}")

    all_metrics = []
    all_raw = []
    all_pred = []
    all_semantic = []
    all_representative = []
    all_slice_frames = {}

    dataset_frames = [
        ("train_parquet_stage_b", train_df),
        ("validation_parquet_stage_b", val_df),
        ("TAU_old", tau_old_df),
        ("TAU_new", tau_new_df),
    ]

    for dataset_name, df in dataset_frames:
        print(f"Analyzing {dataset_name} ...")
        metrics_df, raw_df, pred_df, semantic_df, slice_frames = analyze_dataset(model, dataset_name, df)
        all_metrics.append(metrics_df)
        all_raw.append(raw_df)
        all_pred.append(pred_df)
        all_semantic.append(semantic_df)
        all_slice_frames.update(slice_frames)
        for slice_name, _ in SLICE_DEFS:
            all_representative.append(
                representative_rows(dataset_name, slice_name, slice_frames[(dataset_name, slice_name)])
            )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    raw_df = pd.concat(all_raw, ignore_index=True)
    pred_df = pd.concat(all_pred, ignore_index=True)
    semantic_df = pd.concat(all_semantic, ignore_index=True)
    rep_df = pd.concat(all_representative, ignore_index=True)

    metrics_df.to_csv(csv_dir / "slice_metrics.csv", index=False)
    raw_df.to_csv(csv_dir / "slice_raw_summary.csv", index=False)
    pred_df.to_csv(csv_dir / "slice_prediction_summary.csv", index=False)
    semantic_df.to_csv(csv_dir / "slice_semantic_checks.csv", index=False)
    rep_df.to_csv(csv_dir / "slice_representative_rows.csv", index=False)

    for slice_name, slice_desc in SLICE_DEFS:
        plot_slice_histograms(
            slice_name,
            slice_desc,
            all_slice_frames,
            plot_dir / f"{slice_name}_overlay_histograms.png",
        )

    write_summary_note(
        out_dir / "summary_note.txt",
        compatibility_df,
        metrics_df,
        raw_df,
        semantic_df,
    )

    print(f"Saved CSV:  {csv_dir / 'h1_compatibility_summary.csv'}")
    print(f"Saved CSV:  {csv_dir / 'slice_metrics.csv'}")
    print(f"Saved CSV:  {csv_dir / 'slice_raw_summary.csv'}")
    print(f"Saved CSV:  {csv_dir / 'slice_prediction_summary.csv'}")
    print(f"Saved CSV:  {csv_dir / 'slice_semantic_checks.csv'}")
    print(f"Saved CSV:  {csv_dir / 'slice_representative_rows.csv'}")
    for slice_name, _ in SLICE_DEFS:
        print(f"Saved plot: {plot_dir / f'{slice_name}_overlay_histograms.png'}")
    print(f"Saved note: {out_dir / 'summary_note.txt'}")


if __name__ == "__main__":
    main()
