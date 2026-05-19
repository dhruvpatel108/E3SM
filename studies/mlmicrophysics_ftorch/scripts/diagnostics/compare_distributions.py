#!/usr/bin/env python3
"""
compare_distributions.py
========================
Compare input feature and output tendency distributions between:
  1. Validation set (parquet files, last 20%)
  2. E3SM h1 data (from 2-month simulation)

Produces overlapping histograms for each of the 11 input features
and 4 output tendencies (15 plots total).

Usage:
    module load conda && conda activate mlmicrophysics-env
    python compare_distributions.py [options]

Examples:
    python compare_distributions.py
    python compare_distributions.py --max-samples 500000 --output-dir ../../reports/distribution_comparison_plots
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_PARQUET_DIR = "/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data"
DEFAULT_H1_DIR = "/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_emulator_2mos"
STUDY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = str(STUDY_ROOT / "reports" / "distribution_comparison_plots")
SENTINEL_VALUE = -99999.0

# Feature names for display
INPUT_FEATURES = [
    ("QC_TAU_in", "P3_qc_in_TAU", r"$q_c$ (kg/kg)"),
    ("QR_TAU_in", "P3_qr_in_TAU", r"$q_r$ (kg/kg)"),
    ("NC_TAU_in", "P3_nc_in_TAU", r"$N_c$ (#/kg)"),
    ("NR_TAU_in", "P3_nr_in_TAU", r"$N_r$ (#/kg)"),
    ("PGAM", "P3_mu_c", r"$\mu_c$ (PGAM)"),
    ("LAMC", "P3_lamc", r"$\lambda_c$ (LAMC)"),
    ("LAMR", "P3_lamr", r"$\lambda_r$ (LAMR)"),
    ("N0R", "P3_nr", r"$N_0r$ (N0R)"),
    ("RHO_CLUBB", "RHO_CLUBB", r"$\rho$ (RHO_CLUBB)"),
    ("CLOUD", "CLOUD", r"Cloud fraction"),
    ("FREQR", "FREQR", r"Rain fraction (FREQR)"),
]

OUTPUT_TENDENCIES = [
    ("qrtend_TAU", "P3_qrtend_TAU_raw", r"$dq_r/dt$ (qrtend)"),
    ("nctend_TAU", "P3_nctend_TAU_raw", r"$dN_c/dt$ (nctend)"),
    ("nrtend_TAU", "P3_nrtend_TAU_raw", r"$dN_r/dt$ (nrtend)"),
    ("qctend_TAU", "P3_qctend_TAU_raw", r"$dq_c/dt$ (qctend)"),
]


# ============================================================
# DATA LOADERS
# ============================================================

def load_parquet_data(parquet_dir, file_range=(0.8, 1.0), max_samples=500000, seed=42):
    """Load validation data from parquet files (last 20% by default)."""
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No *.parquet files found in {parquet_dir}")

    total_files = len(files)
    lo_idx = int(np.floor(file_range[0] * total_files))
    hi_idx = int(np.ceil(file_range[1] * total_files))
    files = files[lo_idx:hi_idx]
    n_files = len(files)
    print(f"  Parquet: using files [{lo_idx}:{hi_idx}] ({n_files} files) from {total_files} total")

    per_file_quota = int(np.ceil(max_samples / n_files * 1.05)) if max_samples else None

    input_cols = [f[0] for f in INPUT_FEATURES]
    output_cols = [f[0] for f in OUTPUT_TENDENCIES]
    all_cols = input_cols + output_cols

    all_data = []
    total = 0
    for fi, fpath in enumerate(files):
        df = pd.read_parquet(fpath)

        # Same validity filters as evaluation script
        mask = (df["QC_TAU_in"] > 1e-6) & (df["CLOUD"] > 0.01)
        for col in ("PGAM", "LAMC", "LAMR", "N0R"):
            mask &= (df[col] != SENTINEL_VALUE)
        mask &= (
            (df["qrtend_TAU"].abs() > 0)
            | (df["nctend_TAU"].abs() > 0)
            | (df["nrtend_TAU"].abs() > 0)
        )
        df = df[mask]

        if len(df) == 0:
            continue

        if per_file_quota and len(df) > per_file_quota:
            df = df.sample(n=per_file_quota, random_state=rng.integers(2**31))

        all_data.append(df[all_cols].values.astype(np.float32))
        total += len(df)

        if (fi + 1) % 50 == 0 or fi + 1 == n_files:
            print(f"    {fi+1}/{n_files} files ... {total:,} samples", flush=True)

    data = np.concatenate(all_data, axis=0)

    if max_samples and data.shape[0] > max_samples:
        sel = rng.choice(data.shape[0], max_samples, replace=False)
        sel.sort()
        data = data[sel]

    print(f"  Parquet total: {data.shape[0]:,} samples")
    return data[:, :11], data[:, 11:]  # inputs, outputs


def load_h1_data(h1_dir, max_samples=500000, seed=42):
    """Load E3SM h1 simulation data."""
    rng = np.random.default_rng(seed)
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No *.h1.*.nc files found in {h1_dir}")

    print(f"  H1: found {len(h1_files)} files")

    all_inp, all_tgt = [], []
    total = 0
    for fpath in h1_files:
        if max_samples and total >= max_samples:
            break
        fname = os.path.basename(fpath)
        print(f"    {fname} ... ", end="", flush=True)
        ds = nc.Dataset(fpath)
        nt = len(ds.dimensions["time"])
        nlev = len(ds.dimensions["lev"])

        for t in range(nt):
            if max_samples and total >= max_samples:
                break

            qc = ds.variables["P3_qc_in_TAU"][t, :, :]
            qr = ds.variables["P3_qr_in_TAU"][t, :, :]
            nc_in = ds.variables["P3_nc_in_TAU"][t, :, :]
            nr = ds.variables["P3_nr_in_TAU"][t, :, :]
            mu_c = ds.variables["P3_mu_c"][t, :, :]
            lamc = ds.variables["P3_lamc"][t, :, :]
            lamr = ds.variables["P3_lamr"][t, :, :]
            p3_nr = ds.variables["P3_nr"][t, :, :]
            rho = ds.variables["RHO_CLUBB"][t, :nlev, :]
            cloud = ds.variables["CLOUD"][t, :, :]
            freqr = ds.variables["FREQR"][t, :, :]

            qrt = ds.variables["P3_qrtend_TAU_raw"][t, :, :]
            nct = ds.variables["P3_nctend_TAU_raw"][t, :, :]
            nrt = ds.variables["P3_nrtend_TAU_raw"][t, :, :]
            qct = ds.variables["P3_qctend_TAU_raw"][t, :, :]

            sentinel = (
                (mu_c == SENTINEL_VALUE) | (lamc == SENTINEL_VALUE)
                | (lamr == SENTINEL_VALUE) | (p3_nr == SENTINEL_VALUE)
            )
            has_tend = (np.abs(qrt) > 0) | (np.abs(nct) > 0) | (np.abs(nrt) > 0)
            valid = (qc > 1e-6) & (cloud > 0.01) & ~sentinel & has_tend

            li, ci = np.where(valid)
            nv = len(li)
            if nv == 0:
                continue

            inp = np.stack(
                [qc[li, ci], qr[li, ci], nc_in[li, ci], nr[li, ci],
                 mu_c[li, ci], lamc[li, ci], lamr[li, ci], p3_nr[li, ci],
                 rho[li, ci], cloud[li, ci], freqr[li, ci]],
                axis=1,
            ).astype(np.float32)
            tgt = np.stack(
                [qrt[li, ci], nct[li, ci], nrt[li, ci], qct[li, ci]],
                axis=1,
            ).astype(np.float32)

            all_inp.append(inp)
            all_tgt.append(tgt)
            total += inp.shape[0]

        ds.close()
        print(f"{total:,} samples so far")

    inputs = np.concatenate(all_inp, axis=0)
    targets = np.concatenate(all_tgt, axis=0)

    if max_samples and inputs.shape[0] > max_samples:
        sel = rng.choice(inputs.shape[0], max_samples, replace=False)
        sel.sort()
        inputs = inputs[sel]
        targets = targets[sel]

    print(f"  H1 total: {inputs.shape[0]:,} samples")
    return inputs, targets


# ============================================================
# PLOTTING
# ============================================================

def plot_single_comparison(val_data, h1_data, feature_name, display_name, output_path,
                           use_log_x=False, use_log_y=True):
    """Plot overlapping histograms for a single feature."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Compute common bin range
    all_vals = np.concatenate([val_data, h1_data])
    finite = all_vals[np.isfinite(all_vals)]

    if use_log_x:
        # For log-scale x, use log-spaced bins on absolute values
        pos = finite[finite > 0]
        if len(pos) < 100:
            ax.text(0.5, 0.5, "Insufficient positive data for log-x", ha="center",
                    transform=ax.transAxes)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return

        lo = np.log10(np.percentile(pos, 0.5))
        hi = np.log10(np.percentile(pos, 99.5))
        bins = np.logspace(lo, hi, 80)
        ax.set_xscale("log")
    else:
        lo = np.percentile(finite, 0.5)
        hi = np.percentile(finite, 99.5)
        bins = np.linspace(lo, hi, 80)

    ax.hist(val_data, bins=bins, alpha=0.5, color="steelblue", label="Validation (parquet)",
            density=True, edgecolor="none")
    ax.hist(h1_data, bins=bins, alpha=0.5, color="coral", label="E3SM h1 (2-month run)",
            density=True, edgecolor="none")

    if use_log_y:
        ax.set_yscale("log")

    ax.set_xlabel(display_name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution Comparison: {display_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Stats annotation
    val_mean, val_std = np.mean(val_data), np.std(val_data)
    h1_mean, h1_std = np.mean(h1_data), np.std(h1_data)
    val_med = np.median(val_data)
    h1_med = np.median(h1_data)

    stats = (
        f"Validation: N={len(val_data):,}\n"
        f"  mean={val_mean:.3e}, std={val_std:.3e}\n"
        f"  median={val_med:.3e}\n"
        f"E3SM h1: N={len(h1_data):,}\n"
        f"  mean={h1_mean:.3e}, std={h1_std:.3e}\n"
        f"  median={h1_med:.3e}"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_grid_comparison(val_inputs, val_targets, h1_inputs, h1_targets, output_dir):
    """Create a 4x3 grid for inputs and 2x2 grid for outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # --- 11 Input Features (3 pages of 4) ---
    # Decide which features benefit from log-scale x
    log_x_features = {"QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in",
                       "LAMC", "LAMR", "N0R", "FREQR"}

    fig_inputs, axes_inp = plt.subplots(4, 3, figsize=(18, 20))
    axes_inp = axes_inp.flatten()

    for i, (pq_name, h1_name, display) in enumerate(INPUT_FEATURES):
        ax = axes_inp[i]
        v = val_inputs[:, i]
        h = h1_inputs[:, i]

        all_vals = np.concatenate([v, h])
        finite = all_vals[np.isfinite(all_vals)]

        use_log = pq_name in log_x_features
        if use_log:
            pos_v = v[v > 0]
            pos_h = h[h > 0]
            if len(pos_v) < 100 or len(pos_h) < 100:
                use_log = False

        if use_log:
            pos = finite[finite > 0]
            lo = np.log10(np.percentile(pos, 0.5))
            hi = np.log10(np.percentile(pos, 99.5))
            bins = np.logspace(lo, hi, 60)
            ax.set_xscale("log")
        else:
            lo = np.percentile(finite, 0.5)
            hi = np.percentile(finite, 99.5)
            bins = np.linspace(lo, hi, 60)

        ax.hist(v, bins=bins, alpha=0.5, color="steelblue", label="Validation",
                density=True, edgecolor="none")
        ax.hist(h, bins=bins, alpha=0.5, color="coral", label="E3SM h1",
                density=True, edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlabel(display, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{pq_name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Quick stats
        stats = (f"Val: μ={np.mean(v):.2e}, σ={np.std(v):.2e}, min={np.min(v):.2e}, max={np.max(v):.2e}\n"
                 f"E3SM: μ={np.mean(h):.2e}, σ={np.std(h):.2e}, min={np.min(h):.2e}, max={np.max(h):.2e}")
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=6,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Hide extra subplot(s)
    for j in range(len(INPUT_FEATURES), len(axes_inp)):
        axes_inp[j].set_visible(False)

    fig_inputs.suptitle("Input Feature Distribution: Validation vs E3SM h1",
                         fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    input_path = os.path.join(output_dir, "input_features_comparison.png")
    plt.savefig(input_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {input_path}")

    # --- 4 Output Tendencies (2x2 grid) ---
    fig_out, axes_out = plt.subplots(2, 2, figsize=(14, 10))
    axes_out = axes_out.flatten()

    for i, (pq_name, h1_name, display) in enumerate(OUTPUT_TENDENCIES):
        ax = axes_out[i]
        v = val_targets[:, i]
        h = h1_targets[:, i]

        all_vals = np.concatenate([v, h])
        finite = all_vals[np.isfinite(all_vals)]
        lo = np.percentile(finite, 0.5)
        hi = np.percentile(finite, 99.5)
        bins = np.linspace(lo, hi, 80)

        ax.hist(v, bins=bins, alpha=0.5, color="steelblue", label="Validation",
                density=True, edgecolor="none")
        ax.hist(h, bins=bins, alpha=0.5, color="coral", label="E3SM h1",
                density=True, edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlabel(display, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{pq_name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Stats including sign breakdown
        n_pos_v = int(np.sum(v > 0))
        n_neg_v = int(np.sum(v < 0))
        n_pos_h = int(np.sum(h > 0))
        n_neg_h = int(np.sum(h < 0))
        stats = (
            f"Val: μ={np.mean(v):.2e}, σ={np.std(v):.2e}\n"
            f"  pos={n_pos_v:,} ({n_pos_v/len(v)*100:.1f}%), "
            f"neg={n_neg_v:,} ({n_neg_v/len(v)*100:.1f}%)\n"
            f"E3SM: μ={np.mean(h):.2e}, σ={np.std(h):.2e}\n"
            f"  pos={n_pos_h:,} ({n_pos_h/len(h)*100:.1f}%), "
            f"neg={n_neg_h:,} ({n_neg_h/len(h)*100:.1f}%)"
        )
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=6,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig_out.suptitle("Output Tendency Distribution: Validation vs E3SM h1",
                      fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "output_tendencies_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # --- Individual plots for each feature (for detailed inspection) ---
    print("\n  Generating individual feature comparison plots...")
    for i, (pq_name, h1_name, display) in enumerate(INPUT_FEATURES):
        use_log = pq_name in log_x_features
        plot_single_comparison(
            val_inputs[:, i], h1_inputs[:, i],
            pq_name, display,
            os.path.join(output_dir, f"input_{i+1:02d}_{pq_name}.png"),
            use_log_x=use_log,
        )
    for i, (pq_name, h1_name, display) in enumerate(OUTPUT_TENDENCIES):
        plot_single_comparison(
            val_targets[:, i], h1_targets[:, i],
            pq_name, display,
            os.path.join(output_dir, f"output_{i+1:02d}_{pq_name}.png"),
        )
    print(f"  Saved {len(INPUT_FEATURES) + len(OUTPUT_TENDENCIES)} individual plots")

    # --- Print numerical summary ---
    print("\n" + "=" * 100)
    print("  DISTRIBUTION COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Feature':20s} | {'Val Mean':>12s} | {'H1 Mean':>12s} | {'Val Std':>12s} | {'H1 Std':>12s} | {'Val Med':>12s} | {'H1 Med':>12s} | {'Val Min':>12s} | {'H1 Min':>12s} | {'Val Max':>12s} | {'H1 Max':>12s} | {'Val 5th':>12s} | {'H1 5th':>12s} | {'Val 25th':>12s} | {'H1 25th':>12s} | {'Val 75th':>12s} | {'H1 75th':>12s} | {'Val 95th':>12s} | {'H1 95th':>12s} | {'Shift':>8s}")
    print("-" * 100)

    all_features = [(f[0], "input", i) for i, f in enumerate(INPUT_FEATURES)] + \
                   [(f[0], "output", i) for i, f in enumerate(OUTPUT_TENDENCIES)]

    for name, ftype, idx in all_features:
        if ftype == "input":
            v, h = val_inputs[:, idx], h1_inputs[:, idx]
        else:
            v, h = val_targets[:, idx], h1_targets[:, idx]

        vm, hm = np.mean(v), np.mean(h)
        vs, hs = np.std(v), np.std(h)
        vmed, hmed = np.median(v), np.median(h)
        vmin, hmin = np.min(v), np.min(h)
        vmax, hmax = np.max(v), np.max(h)
        v5, h5 = np.percentile(v, 5), np.percentile(h, 5)
        v25, h25 = np.percentile(v, 25), np.percentile(h, 25)
        v75, h75 = np.percentile(v, 75), np.percentile(h, 75)
        v95, h95 = np.percentile(v, 95), np.percentile(h, 95)

        # Relative shift of means
        denom = max(abs(vm), abs(hm), 1e-30)
        shift = abs(vm - hm) / denom * 100

        print(f"{name:20s} | {vm:12.3e} | {hm:12.3e} | {vs:12.3e} | {hs:12.3e} | {vmed:12.3e} | {hmed:12.3e} | {vmin:12.3e} | {hmin:12.3e} | {vmax:12.3e} | {hmax:12.3e} | {v5:12.3e} | {h5:12.3e} | {v25:12.3e} | {h25:12.3e} | {v75:12.3e} | {h75:12.3e} | {v95:12.3e} | {h95:12.3e} | {shift:7.1f}%")

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare distributions between validation parquet and E3SM h1 data"
    )
    p.add_argument("--parquet-dir", default=DEFAULT_PARQUET_DIR,
                    help=f"Directory with parquet files (default: {DEFAULT_PARQUET_DIR})")
    p.add_argument("--h1-dir", default=DEFAULT_H1_DIR,
                    help=f"Directory with h1 NetCDF files (default: {DEFAULT_H1_DIR})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output directory for plots (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--max-samples", type=int, default=500000,
                    help="Max samples per dataset (default: 500000)")
    p.add_argument("--parquet-file-range", default="0.8-1.0",
                    help="Fraction range of parquet files (default: 0.8-1.0 = last 20%%)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    file_range = tuple(float(x) for x in args.parquet_file_range.split("-"))

    print("=" * 80)
    print("  DISTRIBUTION COMPARISON: Validation vs E3SM h1")
    print("=" * 80)
    print(f"  Parquet dir: {args.parquet_dir}")
    print(f"  H1 dir:      {args.h1_dir}")
    print(f"  Max samples: {args.max_samples:,}")
    print(f"  File range:  {file_range}")
    print()

    # Load validation data
    print("Loading validation data (parquet)...")
    val_inputs, val_targets = load_parquet_data(
        args.parquet_dir, file_range=file_range,
        max_samples=args.max_samples, seed=args.seed
    )

    # Load E3SM h1 data
    print("\nLoading E3SM h1 data...")
    h1_inputs, h1_targets = load_h1_data(
        args.h1_dir, max_samples=args.max_samples, seed=args.seed
    )

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_grid_comparison(val_inputs, val_targets, h1_inputs, h1_targets, args.output_dir)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
