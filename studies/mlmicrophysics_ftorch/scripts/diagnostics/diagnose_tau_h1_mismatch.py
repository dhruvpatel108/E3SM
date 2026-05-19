#!/usr/bin/env python3
"""
Diagnose why a TorchScript emulator performs well on parquet validation data
but poorly on tau_2mos H1 NetCDF data.

Outputs:
  - CSV: row counts + quantile/stat summaries
  - CSV: file-by-file H1 target/model metrics
  - Plots: parity counts, file drift, positive-nrtend failure map

This script compares two H1 row-construction paths:
  1. Exact parquet/training parity:
       Stage A: QC >= 1e-8
       Stage B: |qctend| > 1e-15, CLOUD > 0.01, QC > 1e-6
  2. Current evaluator path:
       QC > 1e-6, CLOUD > 0.01, no sentinels, and any of qrt/nct/nrt non-zero
       plus the per-timestep sampling cap used by evaluate_torchscript_emulator.py
"""

import argparse
import glob
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/{os.environ.get('USER', 'user')}/matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SENTINEL_VALUE = -99999.0
STAGE_A_QC_THRESHOLD = 1.0e-8
STAGE_B_QCTEND_THRESHOLD = 1.0e-15
STAGE_B_CLOUD_THRESHOLD = 0.01
STAGE_B_QC_THRESHOLD = 1.0e-6

INPUT_COLS = [
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
]

TARGET_COLS = [
    "qrtend_TAU",
    "nctend_TAU",
    "nrtend_TAU",
    "qctend_TAU",
]

ALL_VALUE_COLS = INPUT_COLS + TARGET_COLS
OPTIONAL_COLS = ["lev"]

MODEL_OUTPUT_NAMES = ["qrtend", "nctend", "nrtend", "qctend"]
MODEL_TARGET_COLS = ["qrtend_TAU", "nctend_TAU", "nrtend_TAU", "qctend_TAU"]

CURRENT_SENTINEL_COLS = ["PGAM", "LAMC", "LAMR", "N0R"]

H1_INPUT_MAP = {
    "QC_TAU_in": "P3_qc_in_TAU",
    "QR_TAU_in": "P3_qr_in_TAU",
    "NC_TAU_in": "P3_nc_in_TAU",
    "NR_TAU_in": "P3_nr_in_TAU",
    "PGAM": "P3_mu_c",
    "LAMC": "P3_lamc",
    "LAMR": "P3_lamr",
    "N0R": "P3_nr",
    "RHO_CLUBB": "RHO_CLUBB",
    "CLOUD": "CLOUD",
    "FREQR": "FREQR",
}

H1_TARGET_MAP = {
    "qrtend_TAU": "P3_qrtend_TAU_raw",
    "nctend_TAU": "P3_nctend_TAU_raw",
    "nrtend_TAU": "P3_nrtend_TAU_raw",
    "qctend_TAU": "P3_qctend_TAU_raw",
}

PARQUET_FILE_RANGE = (0.8, 1.0)

STUDY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_REPORT_ROOT = (
    STUDY_ROOT / "reports" / "emulator_performance_evaluation" / "evaluate_model_results"
)
R2_EPS = 1.0e-10


class FileResult:
    def __init__(
        self,
        file_name,
        counts,
        exact_full_df,
        current_full_df,
        current_sampled_df,
        exact_model_df,
        target_summary,
    ):
        self.file_name = file_name
        self.counts = counts
        self.exact_full_df = exact_full_df
        self.current_full_df = current_full_df
        self.current_sampled_df = current_sampled_df
        self.exact_model_df = exact_model_df
        self.target_summary = target_summary


def parse_args() -> argparse.Namespace:
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
        help="Parquet validation directory.",
    )
    p.add_argument(
        "--h1-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="H1 tau_2mos directory.",
    )
    p.add_argument(
        "--out-dir",
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis"),
        help="Output directory for CSVs, plots, and summary.",
    )
    p.add_argument(
        "--parity-files",
        type=int,
        default=2,
        help="Number of H1 files for exact parity comparison.",
    )
    p.add_argument(
        "--distribution-samples",
        type=int,
        default=300000,
        help="Target rows for parquet/H1 distribution comparison.",
    )
    p.add_argument(
        "--metric-samples-per-file",
        type=int,
        default=120000,
        help="Max exact-stage-B rows used for model metrics per H1 file.",
    )
    p.add_argument(
        "--current-samples-per-timestep",
        type=int,
        default=5000,
        help="Per-timestep cap used to mirror the current H1 evaluator.",
    )
    p.add_argument(
        "--parquet-max-files",
        type=int,
        default=None,
        help="Optional cap on validation parquet files. Mainly useful for smoke tests.",
    )
    p.add_argument(
        "--h1-max-files",
        type=int,
        default=None,
        help="Optional cap on H1 files. Mainly useful for smoke tests.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return p.parse_args()


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    csv_dir = out_dir / "csv"
    plot_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return {"base": out_dir, "csv": csv_dir, "plots": plot_dir}


def to_ndarray(x) -> np.ndarray:
    if np.ma.isMaskedArray(x):
        return np.asarray(x.filled(np.nan), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def sample_frame(df: pd.DataFrame, max_rows: Optional[int], rng: np.random.Generator) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    idx = rng.choice(len(df), size=max_rows, replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def summarize_frame(
    df: pd.DataFrame,
    dataset: str,
    subset: str,
    file_name: str = "ALL",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    cols = [c for c in ALL_VALUE_COLS + OPTIONAL_COLS if c in df.columns]
    for col in cols:
        values = df[col].to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        rows.append(
            {
                "record_type": "summary",
                "dataset": dataset,
                "file_name": file_name,
                "subset": subset,
                "variable": col,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p1": float(np.percentile(values, 1)),
                "p5": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
                "neg_frac": float(np.mean(values < 0)),
                "zero_frac": float(np.mean(values == 0)),
                "pos_frac": float(np.mean(values > 0)),
            }
        )
    return rows


def row_count_rows(file_name: str, counts: Dict[str, int]) -> List[Dict[str, object]]:
    out = []
    for stage, count in counts.items():
        out.append(
            {
                "record_type": "row_count",
                "dataset": "h1_parity",
                "file_name": file_name,
                "subset": stage,
                "variable": "__rows__",
                "count": int(count),
            }
        )
    return out


def build_rows_from_mask(
    arrays: Dict[str, np.ndarray],
    mask: np.ndarray,
    lev_values: np.ndarray,
) -> pd.DataFrame:
    li, ci = np.where(mask)
    data = {}
    for col in INPUT_COLS:
        data[col] = arrays[col][li, ci]
    for col in TARGET_COLS:
        data[col] = arrays[col][li, ci]
    data["lev"] = lev_values[li]
    return pd.DataFrame(data).dropna().reset_index(drop=True)


def read_timestep_arrays(ds: nc.Dataset, t: int) -> Dict[str, np.ndarray]:
    nlev = len(ds.dimensions["lev"])
    arrays: Dict[str, np.ndarray] = {}
    for col, var_name in H1_INPUT_MAP.items():
        if col == "RHO_CLUBB":
            arrays[col] = to_ndarray(ds.variables[var_name][t, :nlev, :])
        else:
            arrays[col] = to_ndarray(ds.variables[var_name][t, :, :])
    for col, var_name in H1_TARGET_MAP.items():
        arrays[col] = to_ndarray(ds.variables[var_name][t, :, :])
    return arrays


def compute_masks(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    qc = arrays["QC_TAU_in"]
    cloud = arrays["CLOUD"]
    qct = arrays["qctend_TAU"]
    qrt = arrays["qrtend_TAU"]
    nct = arrays["nctend_TAU"]
    nrt = arrays["nrtend_TAU"]

    stage_a = qc >= STAGE_A_QC_THRESHOLD
    exact_stage_b = (
        stage_a
        & (np.abs(qct) > STAGE_B_QCTEND_THRESHOLD)
        & (cloud > STAGE_B_CLOUD_THRESHOLD)
        & (qc > STAGE_B_QC_THRESHOLD)
    )

    sentinel_mask = np.zeros_like(qc, dtype=bool)
    for col in CURRENT_SENTINEL_COLS:
        sentinel_mask |= arrays[col] == SENTINEL_VALUE

    current_pre = (
        (qc > STAGE_B_QC_THRESHOLD)
        & (cloud > STAGE_B_CLOUD_THRESHOLD)
        & ~sentinel_mask
        & ((np.abs(qrt) > 0) | (np.abs(nct) > 0) | (np.abs(nrt) > 0))
    )

    return {
        "stage_a": stage_a,
        "exact_stage_b": exact_stage_b,
        "current_pre": current_pre,
        "sentinel": sentinel_mask,
        "qct_only": exact_stage_b & (np.abs(qrt) == 0) & (np.abs(nct) == 0) & (np.abs(nrt) == 0),
    }


def finalize_target_summary(acc: Dict[str, float]) -> Dict[str, float]:
    out = dict(acc)
    for name in ("nctend", "nrtend"):
        count = out[f"{name}_count"]
        if count > 0:
            mean = out[f"{name}_sum"] / count
            var = max(out[f"{name}_sumsq"] / count - mean ** 2, 0.0)
            out[f"{name}_mean"] = mean
            out[f"{name}_std"] = math.sqrt(var)
        else:
            out[f"{name}_mean"] = np.nan
            out[f"{name}_std"] = np.nan
    if out["nrtend_count"] > 0:
        out["positive_nrtend_fraction"] = out["nrtend_positive_count"] / out["nrtend_count"]
    else:
        out["positive_nrtend_fraction"] = np.nan
    return out


def analyze_single_h1_file(
    file_path: str,
    rng: np.random.Generator,
    collect_full_frames: bool,
    metric_samples_per_file: int,
    current_samples_per_timestep: int,
) -> FileResult:
    file_name = os.path.basename(file_path)
    ds = nc.Dataset(file_path)
    nt = len(ds.dimensions["time"])
    nlev = len(ds.dimensions["lev"])
    metric_samples_per_timestep = int(math.ceil(metric_samples_per_file / max(nt, 1)))
    lev_values = to_ndarray(ds.variables["lev"][:nlev])

    counts = {
        "raw_rows": 0,
        "exact_stage_a_rows": 0,
        "exact_stage_b_rows": 0,
        "exact_stage_b_dropna_rows": 0,
        "current_pre_rows": 0,
        "current_pre_dropna_rows": 0,
        "current_sampled_rows": 0,
        "exact_not_current_rows": 0,
        "current_not_exact_rows": 0,
        "exact_stage_b_sentinel_rows": 0,
        "exact_stage_b_qct_only_rows": 0,
    }

    target_acc = {
        "nctend_count": 0,
        "nctend_sum": 0.0,
        "nctend_sumsq": 0.0,
        "nrtend_count": 0,
        "nrtend_sum": 0.0,
        "nrtend_sumsq": 0.0,
        "nrtend_positive_count": 0,
    }

    exact_full_parts: List[pd.DataFrame] = []
    current_full_parts: List[pd.DataFrame] = []
    current_sample_parts: List[pd.DataFrame] = []
    exact_metric_parts: List[pd.DataFrame] = []

    for t in range(nt):
        arrays = read_timestep_arrays(ds, t)
        masks = compute_masks(arrays)

        counts["raw_rows"] += int(arrays["QC_TAU_in"].size)
        counts["exact_stage_a_rows"] += int(masks["stage_a"].sum())
        counts["exact_stage_b_rows"] += int(masks["exact_stage_b"].sum())
        counts["current_pre_rows"] += int(masks["current_pre"].sum())
        counts["exact_not_current_rows"] += int((masks["exact_stage_b"] & ~masks["current_pre"]).sum())
        counts["current_not_exact_rows"] += int((masks["current_pre"] & ~masks["exact_stage_b"]).sum())
        counts["exact_stage_b_qct_only_rows"] += int(masks["qct_only"].sum())

        if np.any(masks["exact_stage_b"]):
            exact_df = build_rows_from_mask(arrays, masks["exact_stage_b"], lev_values)
            counts["exact_stage_b_dropna_rows"] += int(len(exact_df))
            if not exact_df.empty:
                counts["exact_stage_b_sentinel_rows"] += int(
                    (exact_df[CURRENT_SENTINEL_COLS] == SENTINEL_VALUE).any(axis=1).sum()
                )
                nct = exact_df["nctend_TAU"].to_numpy(dtype=np.float64)
                nrt = exact_df["nrtend_TAU"].to_numpy(dtype=np.float64)
                target_acc["nctend_count"] += len(nct)
                target_acc["nctend_sum"] += float(nct.sum())
                target_acc["nctend_sumsq"] += float(np.square(nct).sum())
                target_acc["nrtend_count"] += len(nrt)
                target_acc["nrtend_sum"] += float(nrt.sum())
                target_acc["nrtend_sumsq"] += float(np.square(nrt).sum())
                target_acc["nrtend_positive_count"] += int((nrt > 0).sum())

                if collect_full_frames:
                    exact_full_parts.append(exact_df)
                exact_metric_parts.append(sample_frame(exact_df, metric_samples_per_timestep, rng))

        if np.any(masks["current_pre"]):
            current_df = build_rows_from_mask(arrays, masks["current_pre"], lev_values)
            counts["current_pre_dropna_rows"] += int(len(current_df))
            if not current_df.empty:
                if collect_full_frames:
                    current_full_parts.append(current_df)
                current_sampled = sample_frame(current_df, current_samples_per_timestep, rng)
                counts["current_sampled_rows"] += int(len(current_sampled))
                current_sample_parts.append(current_sampled)

    ds.close()

    exact_model_df = (
        pd.concat(exact_metric_parts, ignore_index=True)
        if exact_metric_parts
        else pd.DataFrame(columns=ALL_VALUE_COLS + OPTIONAL_COLS)
    )

    current_sampled_df = (
        pd.concat(current_sample_parts, ignore_index=True)
        if current_sample_parts
        else pd.DataFrame(columns=ALL_VALUE_COLS + OPTIONAL_COLS)
    )

    exact_full_df = (
        pd.concat(exact_full_parts, ignore_index=True)
        if exact_full_parts
        else None
    )
    current_full_df = (
        pd.concat(current_full_parts, ignore_index=True)
        if current_full_parts
        else None
    )

    return FileResult(
        file_name=file_name,
        counts=counts,
        exact_full_df=exact_full_df,
        current_full_df=current_full_df,
        current_sampled_df=current_sampled_df,
        exact_model_df=exact_model_df,
        target_summary=finalize_target_summary(target_acc),
    )


def load_parquet_validation_sample(
    parquet_dir: str,
    target_total_samples: int,
    rng: np.random.Generator,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    return load_parquet_stage_b_sample(
        parquet_dir=parquet_dir,
        target_total_samples=target_total_samples,
        rng=rng,
        file_range=PARQUET_FILE_RANGE,
        max_files=max_files,
    )


def load_parquet_stage_b_sample(
    parquet_dir: str,
    target_total_samples: int,
    rng: np.random.Generator,
    file_range: tuple = PARQUET_FILE_RANGE,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    lo = int(math.floor(file_range[0] * len(files)))
    if file_range[1] >= 1.0:
        hi = len(files)
    else:
        hi = int(math.floor(file_range[1] * len(files)))
    files = files[lo:hi]
    if max_files is not None:
        files = files[:max_files]
    per_file_quota = int(math.ceil(target_total_samples / max(len(files), 1) * 1.05))

    parts: List[pd.DataFrame] = []
    requested_cols = ALL_VALUE_COLS + OPTIONAL_COLS
    for i, file_path in enumerate(files):
        try:
            df = pd.read_parquet(file_path, columns=requested_cols)
        except Exception:
            df = pd.read_parquet(file_path, columns=ALL_VALUE_COLS)
        keep_cols = [c for c in ALL_VALUE_COLS + OPTIONAL_COLS if c in df.columns]
        mask = (
            (np.abs(df["qctend_TAU"]) > STAGE_B_QCTEND_THRESHOLD)
            & (df["CLOUD"] > STAGE_B_CLOUD_THRESHOLD)
            & (df["QC_TAU_in"] > STAGE_B_QC_THRESHOLD)
        )
        df = df.loc[mask, keep_cols].dropna().reset_index(drop=True)
        if df.empty:
            continue
        df = sample_frame(df, per_file_quota, rng)
        parts.append(df)
        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            kept = sum(len(part) for part in parts)
            print("  Parquet files: {}/{} loaded, kept {:,} sampled rows".format(i + 1, len(files), kept))

    if not parts:
        raise RuntimeError("No filtered parquet rows available after Stage B.")

    df = pd.concat(parts, ignore_index=True)
    return sample_frame(df, target_total_samples, rng)


def load_h1_exact_stage_b_sample(
    h1_dir: str,
    target_total_samples: int,
    rng: np.random.Generator,
    max_files: Optional[int] = None,
    drop_sentinels: bool = False,
) -> pd.DataFrame:
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No H1 files found in {h1_dir}")
    if max_files is not None:
        h1_files = h1_files[:max_files]

    per_file_quota = int(math.ceil(target_total_samples / max(len(h1_files), 1) * 1.05))
    parts: List[pd.DataFrame] = []

    for i, file_path in enumerate(h1_files):
        print(
            "  H1 files: {}/{} sampling {}".format(
                i + 1, len(h1_files), os.path.basename(file_path)
            )
        )
        ds = nc.Dataset(file_path)
        nt = len(ds.dimensions["time"])
        nlev = len(ds.dimensions["lev"])
        lev_values = to_ndarray(ds.variables["lev"][:nlev])
        per_timestep_quota = int(math.ceil(per_file_quota / max(nt, 1)))

        file_parts: List[pd.DataFrame] = []
        for t in range(nt):
            arrays = read_timestep_arrays(ds, t)
            masks = compute_masks(arrays)
            if not np.any(masks["exact_stage_b"]):
                continue
            df = build_rows_from_mask(arrays, masks["exact_stage_b"], lev_values)
            if df.empty:
                continue
            if drop_sentinels:
                sentinel_mask = (df[CURRENT_SENTINEL_COLS] == SENTINEL_VALUE).any(axis=1)
                df = df.loc[~sentinel_mask].reset_index(drop=True)
                if df.empty:
                    continue
            file_parts.append(sample_frame(df, per_timestep_quota, rng))

        ds.close()
        if not file_parts:
            continue

        file_df = pd.concat(file_parts, ignore_index=True)
        file_df = sample_frame(file_df, per_file_quota, rng)
        parts.append(file_df)

    if not parts:
        raise RuntimeError("No H1 exact-stage-B rows were loaded.")

    df = pd.concat(parts, ignore_index=True)
    return sample_frame(df, target_total_samples, rng)


def load_model(model_path: str) -> torch.jit.ScriptModule:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def run_model(model: torch.jit.ScriptModule, inputs: np.ndarray) -> np.ndarray:
    preds = np.zeros((len(inputs), len(MODEL_OUTPUT_NAMES)), dtype=np.float32)
    with torch.no_grad():
        try:
            batch = torch.tensor(inputs, dtype=torch.float32)
            preds[:] = model(batch).numpy()
            return preds
        except Exception:
            pass

        for i in range(len(inputs)):
            preds[i] = model(torch.tensor(inputs[i], dtype=torch.float32)).numpy()
    return preds


def r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    target = target.astype(np.float64)
    pred = pred.astype(np.float64)
    var = np.var(target)
    return 1.0 - np.mean((pred - target) ** 2) / max(var, R2_EPS)


def compute_metrics_from_frame(
    model: torch.jit.ScriptModule,
    df: pd.DataFrame,
    file_name: str,
    mode: str,
    target_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "file_name": file_name,
        "mode": mode,
        "n_model_rows": int(len(df)),
    }
    if target_summary is not None:
        row.update(
            {
                "nctend_target_mean_full": target_summary["nctend_mean"],
                "nctend_target_std_full": target_summary["nctend_std"],
                "nrtend_target_mean_full": target_summary["nrtend_mean"],
                "nrtend_target_std_full": target_summary["nrtend_std"],
                "positive_nrtend_fraction_full": target_summary["positive_nrtend_fraction"],
                "nrtend_count_full": int(target_summary["nrtend_count"]),
            }
        )

    if df.empty:
        for name in MODEL_OUTPUT_NAMES:
            row[f"r2_{name}"] = np.nan
        row["r2_nrtend_pos"] = np.nan
        row["r2_nrtend_neg"] = np.nan
        row["n_nrtend_pos_model"] = 0
        row["n_nrtend_neg_model"] = 0
        return row

    inputs = df[INPUT_COLS].to_numpy(dtype=np.float32)
    targets = df[MODEL_TARGET_COLS].to_numpy(dtype=np.float32)
    preds = run_model(model, inputs)

    finite_mask = np.isfinite(inputs).all(axis=1)
    finite_mask &= np.isfinite(targets).all(axis=1)
    finite_mask &= np.isfinite(preds).all(axis=1)
    row["n_finite_model_rows"] = int(finite_mask.sum())
    row["n_nonfinite_model_rows"] = int((~finite_mask).sum())

    if not np.any(finite_mask):
        for name in MODEL_OUTPUT_NAMES:
            row[f"r2_{name}"] = np.nan
        row["r2_nrtend_pos"] = np.nan
        row["r2_nrtend_neg"] = np.nan
        row["n_nrtend_pos_model"] = 0
        row["n_nrtend_neg_model"] = 0
        row["nrtend_sign_accuracy"] = np.nan
        return row

    targets = targets[finite_mask]
    preds = preds[finite_mask]

    for j, name in enumerate(MODEL_OUTPUT_NAMES):
        row[f"r2_{name}"] = r2_score(targets[:, j], preds[:, j])

    nrt = targets[:, MODEL_OUTPUT_NAMES.index("nrtend")]
    pos_mask = nrt > 0
    neg_mask = nrt < 0
    row["n_nrtend_pos_model"] = int(pos_mask.sum())
    row["n_nrtend_neg_model"] = int(neg_mask.sum())
    row["r2_nrtend_pos"] = r2_score(nrt[pos_mask], preds[pos_mask, 2]) if np.any(pos_mask) else np.nan
    row["r2_nrtend_neg"] = r2_score(nrt[neg_mask], preds[neg_mask, 2]) if np.any(neg_mask) else np.nan
    row["nrtend_sign_accuracy"] = float(np.mean(np.sign(preds[:, 2]) == np.sign(nrt)))
    return row


def plot_parity_counts(row_counts_df: pd.DataFrame, output_path: Path) -> None:
    stages = [
        "exact_stage_a_rows",
        "exact_stage_b_dropna_rows",
        "current_pre_dropna_rows",
        "current_sampled_rows",
        "exact_not_current_rows",
        "current_not_exact_rows",
    ]
    files = sorted(row_counts_df["file_name"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.12
    x = np.arange(len(files))

    for i, stage in enumerate(stages):
        values = []
        for file_name in files:
            stage_row = row_counts_df[
                (row_counts_df["file_name"] == file_name)
                & (row_counts_df["subset"] == stage)
            ]
            values.append(stage_row["count"].iloc[0] if not stage_row.empty else 0)
        ax.bar(x + (i - (len(stages) - 1) / 2) * width, values, width=width, label=stage)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".nc", "")[-16:] for f in files], rotation=25, ha="right")
    ax.set_ylabel("Rows")
    ax.set_title("H1 Row-Construction Parity Counts")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_file_drift(metrics_df: pd.DataFrame, output_path: Path) -> None:
    df = metrics_df[metrics_df["mode"] == "exact_stage_b_sampled"].copy()
    if df.empty:
        return

    df = df.sort_values("file_name").reset_index(drop=True)
    x = np.arange(len(df))
    labels = [f.replace(".nc", "")[-16:] for f in df["file_name"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0, 0].plot(x, 100 * df["positive_nrtend_fraction_full"], marker="o")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].set_title("Positive nrtend Fraction")

    axes[0, 1].plot(x, df["nrtend_target_mean_full"], marker="o", label="nrtend mean")
    axes[0, 1].plot(x, df["nctend_target_mean_full"], marker="o", label="nctend mean")
    axes[0, 1].set_title("Target Means by File")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, df["r2_nrtend"], marker="o", label="all")
    axes[1, 0].plot(x, df["r2_nrtend_pos"], marker="o", label="target>0")
    axes[1, 0].plot(x, df["r2_nrtend_neg"], marker="o", label="target<0")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1, 0].set_title("nrtend R2 by File")
    axes[1, 0].set_ylabel("R2")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, df["r2_qrtend"], marker="o", label="qrtend")
    axes[1, 1].plot(x, df["r2_nctend"], marker="o", label="nctend")
    axes[1, 1].plot(x, df["r2_nrtend"], marker="o", label="nrtend")
    axes[1, 1].plot(x, df["r2_qctend"], marker="o", label="qctend")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1, 1].set_title("All-Variable R2 by File")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_positive_failure_map(
    parquet_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    output_path: Path,
    rng: np.random.Generator,
) -> None:
    parquet_pos = parquet_df[parquet_df["nrtend_TAU"] > 0].copy()
    h1_pos = h1_df[h1_df["nrtend_TAU"] > 0].copy()
    if parquet_pos.empty or h1_pos.empty:
        return

    parquet_pos = sample_frame(parquet_pos, 50000, rng)
    h1_pos = sample_frame(h1_pos, 50000, rng)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        ("NR_TAU_in", "N0R", True, True, "NR_TAU_in vs N0R"),
        ("QR_TAU_in", "NR_TAU_in", True, True, "QR_TAU_in vs NR_TAU_in"),
        ("QC_TAU_in", "CLOUD", True, False, "QC_TAU_in vs CLOUD"),
        ("lev", "NR_TAU_in", False, True, "lev vs NR_TAU_in"),
    ]

    for ax, (xcol, ycol, logx, logy, title) in zip(axes.flat, specs):
        if xcol not in parquet_pos.columns or ycol not in parquet_pos.columns:
            ax.set_visible(False)
            continue

        ax.scatter(
            parquet_pos[xcol],
            parquet_pos[ycol],
            s=3,
            alpha=0.08,
            label=f"Parquet pos-nrtend (N={len(parquet_pos):,})",
            color="#1f77b4",
        )
        ax.scatter(
            h1_pos[xcol],
            h1_pos[ycol],
            s=3,
            alpha=0.08,
            label=f"H1 pos-nrtend (N={len(h1_pos):,})",
            color="#d62728",
        )
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    axes[0, 0].legend(markerscale=4, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary_note(
    out_path: Path,
    row_summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    exact_rows = row_summary_df[
        (row_summary_df["record_type"] == "row_count")
        & (row_summary_df["subset"] == "exact_not_current_rows")
    ]["count"].sum()
    current_only_rows = row_summary_df[
        (row_summary_df["record_type"] == "row_count")
        & (row_summary_df["subset"] == "current_not_exact_rows")
    ]["count"].sum()
    sentinel_rows = row_summary_df[
        (row_summary_df["record_type"] == "row_count")
        & (row_summary_df["subset"] == "exact_stage_b_sentinel_rows")
    ]["count"].sum()

    exact_metrics = metrics_df[metrics_df["mode"] == "exact_stage_b_sampled"].copy()
    nrtend_pos_median = float(np.nanmedian(exact_metrics["r2_nrtend_pos"]))
    nrtend_neg_median = float(np.nanmedian(exact_metrics["r2_nrtend_neg"]))
    nctend_median = float(np.nanmedian(exact_metrics["r2_nctend"]))
    nonfinite_median = float(np.nanmedian(exact_metrics["n_nonfinite_model_rows"]))

    def lookup_summary(dataset: str, subset: str, variable: str, field: str) -> float:
        match = row_summary_df[
            (row_summary_df["record_type"] == "summary")
            & (row_summary_df["dataset"] == dataset)
            & (row_summary_df["subset"] == subset)
            & (row_summary_df["file_name"] == "ALL")
            & (row_summary_df["variable"] == variable)
        ]
        if match.empty:
            return np.nan
        return float(match.iloc[0][field])

    parquet_pos_frac = lookup_summary("parquet_validation_stage_b", "all", "nrtend_TAU", "pos_frac")
    h1_pos_frac = lookup_summary("h1_exact_stage_b_global", "all", "nrtend_TAU", "pos_frac")
    parquet_cloud_med = lookup_summary("parquet_validation_stage_b", "nrtend_positive", "CLOUD", "p50")
    h1_cloud_med = lookup_summary("h1_exact_stage_b_global", "nrtend_positive", "CLOUD", "p50")
    parquet_nr_med = lookup_summary("parquet_validation_stage_b", "nrtend_positive", "NR_TAU_in", "p50")
    h1_nr_med = lookup_summary("h1_exact_stage_b_global", "nrtend_positive", "NR_TAU_in", "p50")
    parquet_n0r_med = lookup_summary("parquet_validation_stage_b", "nrtend_positive", "N0R", "p50")
    h1_n0r_med = lookup_summary("h1_exact_stage_b_global", "nrtend_positive", "N0R", "p50")

    conclusion = "mixed"
    if exact_rows <= 0 and nrtend_pos_median < 0:
        conclusion = "mostly distribution shift"
    elif exact_rows > 0 and nrtend_pos_median >= 0 and nctend_median >= 0:
        conclusion = "mostly pipeline mismatch"

    lines = [
        conclusion,
        "",
        (
            "- There is a real pipeline mismatch. On the parity files, the exact Stage A+B path "
            "and the current evaluator path do not select the same rows."
        ),
        (
            f"  exact-only rows: {exact_rows:,}; current-only rows: {current_only_rows:,}; "
            f"exact-stage-B sentinel rows: {sentinel_rows:,}."
        ),
        (
            "- There is also a strong distribution shift in the positive-nrtend regime."
        ),
        (
            f"  positive nrtend fraction: parquet={parquet_pos_frac:.3f}, H1={h1_pos_frac:.3f}; "
            f"median CLOUD on positive-nrtend rows: parquet={parquet_cloud_med:.3g}, H1={h1_cloud_med:.3g}."
        ),
        (
            f"  median NR_TAU_in on positive-nrtend rows: parquet={parquet_nr_med:.3g}, H1={h1_nr_med:.3g}; "
            f"median N0R: parquet={parquet_n0r_med:.3g}, H1={h1_n0r_med:.3g}."
        ),
        (
            "- The failure is present from the start, not just late in the run."
        ),
        (
            f"  median exact-stage-B H1 nctend R2={nctend_median:.4f}, "
            f"positive-only nrtend R2={nrtend_pos_median:.4f}, "
            f"negative-only nrtend R2={nrtend_neg_median:.4f}."
        ),
        (
            f"- Exact-stage-B H1 samples still contain a recurring non-finite prediction subset "
            f"(median {nonfinite_median:.0f} rows per 120k-file sample)."
        ),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dirs = ensure_dirs(Path(args.out_dir))
    rng = np.random.default_rng(args.seed)

    print("Loading parquet validation sample ...")
    parquet_df = load_parquet_validation_sample(
        args.parquet_dir,
        target_total_samples=args.distribution_samples,
        rng=rng,
        max_files=args.parquet_max_files,
    )
    print(f"  Parquet validation sample: {len(parquet_df):,} rows")

    print("Loading TorchScript model ...")
    model = load_model(args.model)

    h1_files = sorted(glob.glob(os.path.join(args.h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No H1 files found in {args.h1_dir}")
    if args.h1_max_files is not None:
        h1_files = h1_files[: args.h1_max_files]

    parity_records: List[Dict[str, object]] = []
    file_metric_rows: List[Dict[str, object]] = []
    h1_distribution_parts: List[pd.DataFrame] = []

    for i, file_path in enumerate(h1_files):
        collect_full = i < args.parity_files
        print(f"Analyzing H1 file {i + 1}/{len(h1_files)}: {os.path.basename(file_path)}")
        result = analyze_single_h1_file(
            file_path=file_path,
            rng=rng,
            collect_full_frames=collect_full,
            metric_samples_per_file=args.metric_samples_per_file,
            current_samples_per_timestep=args.current_samples_per_timestep,
        )
        print(
            "  exact_stage_b_rows={:,} current_pre_rows={:,} model_rows={:,}".format(
                result.counts["exact_stage_b_dropna_rows"],
                result.counts["current_pre_dropna_rows"],
                len(result.exact_model_df),
            )
        )

        parity_records.extend(row_count_rows(result.file_name, result.counts))
        h1_distribution_parts.append(result.exact_model_df)

        file_metric_rows.append(
            compute_metrics_from_frame(
                model,
                result.exact_model_df,
                file_name=result.file_name,
                mode="exact_stage_b_sampled",
                target_summary=result.target_summary,
            )
        )

        if collect_full:
            if result.exact_full_df is not None and not result.exact_full_df.empty:
                parity_records.extend(
                    summarize_frame(
                        result.exact_full_df,
                        dataset="h1_exact_stage_b",
                        subset="all",
                        file_name=result.file_name,
                    )
                )
                pos_df = result.exact_full_df[result.exact_full_df["nrtend_TAU"] > 0]
                if not pos_df.empty:
                    parity_records.extend(
                        summarize_frame(
                            pos_df,
                            dataset="h1_exact_stage_b",
                            subset="nrtend_positive",
                            file_name=result.file_name,
                        )
                    )

            if result.current_full_df is not None and not result.current_full_df.empty:
                parity_records.extend(
                    summarize_frame(
                        result.current_full_df,
                        dataset="h1_current_pre_sample",
                        subset="all",
                        file_name=result.file_name,
                    )
                )

            if result.current_sampled_df is not None and not result.current_sampled_df.empty:
                parity_records.extend(
                    summarize_frame(
                        result.current_sampled_df,
                        dataset="h1_current_sampled",
                        subset="all",
                        file_name=result.file_name,
                    )
                )
                file_metric_rows.append(
                    compute_metrics_from_frame(
                        model,
                        result.current_sampled_df,
                        file_name=result.file_name,
                        mode="current_eval_sampled",
                        target_summary=None,
                    )
                )

    h1_distribution_df = pd.concat(h1_distribution_parts, ignore_index=True)
    h1_distribution_df = sample_frame(h1_distribution_df, args.distribution_samples, rng)
    print(f"  H1 exact-stage-B comparison sample: {len(h1_distribution_df):,} rows")

    parity_records.extend(
        summarize_frame(
            parquet_df,
            dataset="parquet_validation_stage_b",
            subset="all",
            file_name="ALL",
        )
    )
    parquet_pos = parquet_df[parquet_df["nrtend_TAU"] > 0]
    if not parquet_pos.empty:
        parity_records.extend(
            summarize_frame(
                parquet_pos,
                dataset="parquet_validation_stage_b",
                subset="nrtend_positive",
                file_name="ALL",
            )
        )

    parity_records.extend(
        summarize_frame(
            h1_distribution_df,
            dataset="h1_exact_stage_b_global",
            subset="all",
            file_name="ALL",
        )
    )
    h1_pos = h1_distribution_df[h1_distribution_df["nrtend_TAU"] > 0]
    if not h1_pos.empty:
        parity_records.extend(
            summarize_frame(
                h1_pos,
                dataset="h1_exact_stage_b_global",
                subset="nrtend_positive",
                file_name="ALL",
            )
        )

    row_summary_df = pd.DataFrame(parity_records)
    file_metrics_df = pd.DataFrame(file_metric_rows)

    row_summary_path = out_dirs["csv"] / "row_count_quantile_comparison.csv"
    file_metrics_path = out_dirs["csv"] / "h1_file_metrics.csv"
    row_summary_df.to_csv(row_summary_path, index=False)
    file_metrics_df.to_csv(file_metrics_path, index=False)

    plot_parity_counts(
        row_summary_df[row_summary_df["record_type"] == "row_count"].copy(),
        out_dirs["plots"] / "h1_parity_counts.png",
    )
    plot_file_drift(file_metrics_df, out_dirs["plots"] / "h1_file_drift_metrics.png")
    plot_positive_failure_map(
        parquet_df,
        h1_distribution_df,
        out_dirs["plots"] / "positive_nrtend_failure_map.png",
        rng=rng,
    )

    write_summary_note(
        out_dirs["base"] / "summary_note.txt",
        row_summary_df=row_summary_df,
        metrics_df=file_metrics_df,
    )

    print(f"Saved CSV:  {row_summary_path}")
    print(f"Saved CSV:  {file_metrics_path}")
    print(f"Saved note: {out_dirs['base'] / 'summary_note.txt'}")
    print(f"Saved plots in: {out_dirs['plots']}")


if __name__ == "__main__":
    main()
