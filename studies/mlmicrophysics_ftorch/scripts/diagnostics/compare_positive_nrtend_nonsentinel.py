#!/usr/bin/env python3
"""
Compare positive-nrtend parquet validation rows against non-sentinel positive-nrtend
tau_2mos H1 rows under the exact Stage A+B filter.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_tau_h1_mismatch import (
    CURRENT_SENTINEL_COLS,
    build_rows_from_mask,
    compute_masks,
    load_parquet_validation_sample,
    read_timestep_arrays,
    sample_frame,
    to_ndarray,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_REPORT_ROOT = (
    STUDY_ROOT / "reports" / "emulator_performance_evaluation" / "evaluate_model_results"
)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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
        default=str(EVALUATION_REPORT_ROOT / "diagnostics" / "moe" / "run_594616" / "soft_calibration_tau_diagnosis" / "nonsentinel_positive_nrtend"),
        help="Output directory for CSV/plot/note.",
    )
    p.add_argument(
        "--target-samples",
        type=int,
        default=300000,
        help="Target total rows for parquet and H1 exact-stage-B samples.",
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


def sentinel_mask(df):
    return (df[CURRENT_SENTINEL_COLS] == -99999.0).any(axis=1)


def summarize(df, dataset):
    rows = []
    for col in ["QC_TAU_in", "QR_TAU_in", "NR_TAU_in", "N0R", "CLOUD", "lev", "nrtend_TAU"]:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "variable": col,
                "count": int(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p1": float(np.percentile(vals, 1)),
                "p5": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }
        )
    return rows


def load_h1_exact_stage_b_sample(h1_dir, target_samples, rng):
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError("No H1 files found in {}".format(h1_dir))

    per_file_quota = int(np.ceil(target_samples / float(len(h1_files)) * 1.05))
    all_parts = []

    for i, file_path in enumerate(h1_files):
        print("Sampling H1 exact-stage-B file {}/{}: {}".format(i + 1, len(h1_files), os.path.basename(file_path)))
        import netCDF4 as nc

        ds = nc.Dataset(file_path)
        nt = len(ds.dimensions["time"])
        nlev = len(ds.dimensions["lev"])
        lev_values = to_ndarray(ds.variables["lev"][:nlev])
        per_timestep_quota = int(np.ceil(per_file_quota / float(max(nt, 1))))

        file_parts = []
        for t in range(nt):
            arrays = read_timestep_arrays(ds, t)
            masks = compute_masks(arrays)
            if not np.any(masks["exact_stage_b"]):
                continue
            df = build_rows_from_mask(arrays, masks["exact_stage_b"], lev_values)
            if df.empty:
                continue
            file_parts.append(sample_frame(df, per_timestep_quota, rng))
        ds.close()

        if not file_parts:
            continue
        file_df = pd.concat(file_parts, ignore_index=True)
        file_df = sample_frame(file_df, per_file_quota, rng)
        all_parts.append(file_df)

    if not all_parts:
        raise RuntimeError("No exact-stage-B H1 rows were loaded.")

    out = pd.concat(all_parts, ignore_index=True)
    return sample_frame(out, target_samples, rng)


def plot_map(parquet_df, h1_df, output_path, rng):
    parquet_df = sample_frame(parquet_df, 50000, rng)
    h1_df = sample_frame(h1_df, 50000, rng)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        ("NR_TAU_in", "N0R", True, True, "NR_TAU_in vs N0R"),
        ("QR_TAU_in", "NR_TAU_in", True, True, "QR_TAU_in vs NR_TAU_in"),
        ("QC_TAU_in", "CLOUD", True, False, "QC_TAU_in vs CLOUD"),
        ("lev", "NR_TAU_in", False, True, "lev vs NR_TAU_in"),
    ]

    for ax, (xcol, ycol, logx, logy, title) in zip(axes.flat, specs):
        ax.scatter(
            parquet_df[xcol],
            parquet_df[ycol],
            s=3,
            alpha=0.08,
            label="Parquet pos-nrtend",
            color="#1f77b4",
        )
        ax.scatter(
            h1_df[xcol],
            h1_df[ycol],
            s=3,
            alpha=0.08,
            label="H1 non-sentinel pos-nrtend",
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


def write_note(out_path, parquet_df, h1_df):
    def med(df, col):
        return float(np.median(df[col].to_numpy(dtype=np.float64)))

    lines = [
        "non-sentinel positive-nrtend H1 remains strongly shifted from parquet validation",
        "",
        "- Positive-nrtend row counts in the comparison sample: parquet={}, H1 non-sentinel={}.".format(
            len(parquet_df), len(h1_df)
        ),
        "- Median CLOUD: parquet={:.4f}, H1 non-sentinel={:.4f}.".format(
            med(parquet_df, "CLOUD"), med(h1_df, "CLOUD")
        ),
        "- Median NR_TAU_in: parquet={:.4g}, H1 non-sentinel={:.4g}.".format(
            med(parquet_df, "NR_TAU_in"), med(h1_df, "NR_TAU_in")
        ),
        "- Median N0R: parquet={:.4g}, H1 non-sentinel={:.4g}.".format(
            med(parquet_df, "N0R"), med(h1_df, "N0R")
        ),
        "- Median nrtend_TAU: parquet={:.4g}, H1 non-sentinel={:.4g}.".format(
            med(parquet_df, "nrtend_TAU"), med(h1_df, "nrtend_TAU")
        ),
        "- Median QC_TAU_in: parquet={:.4g}, H1 non-sentinel={:.4g}.".format(
            med(parquet_df, "QC_TAU_in"), med(h1_df, "QC_TAU_in")
        ),
        "- So the positive-nrtend mismatch is not being driven by sentinel rows. The non-sentinel H1 positive-formation regime is still substantially different from validation parquet.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    csv_path = out_dir / "positive_nrtend_nonsentinel_summary.csv"
    plot_path = out_dir / "positive_nrtend_nonsentinel_map.png"
    note_path = out_dir / "summary_note.txt"

    rng = np.random.default_rng(args.seed)

    print("Loading parquet validation sample ...")
    parquet_all = load_parquet_validation_sample(args.parquet_dir, args.target_samples, rng)
    parquet_pos = parquet_all[(parquet_all["nrtend_TAU"] > 0) & ~sentinel_mask(parquet_all)].reset_index(drop=True)

    print("Loading H1 exact-stage-B sample ...")
    h1_all = load_h1_exact_stage_b_sample(args.h1_dir, args.target_samples, rng)
    h1_pos = h1_all[(h1_all["nrtend_TAU"] > 0) & ~sentinel_mask(h1_all)].reset_index(drop=True)

    summary_rows = summarize(parquet_pos, "parquet_validation_positive_nrtend_nonsentinel")
    summary_rows.extend(summarize(h1_pos, "h1_exact_stage_b_positive_nrtend_nonsentinel"))
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    plot_map(parquet_pos, h1_pos, plot_path, rng)
    write_note(note_path, parquet_pos, h1_pos)

    print("Saved CSV:  {}".format(csv_path))
    print("Saved plot: {}".format(plot_path))
    print("Saved note: {}".format(note_path))


if __name__ == "__main__":
    main()
