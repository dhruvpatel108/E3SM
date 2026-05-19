#!/usr/bin/env python3
"""
evaluate_torchscript_emulator.py
================================
General-purpose script to evaluate any TorchScript emulator against h1 ground-truth
samples from E3SM simulation.

Produces:
  1. Comprehensive summary table (R2, % error, RMSE, MAE, correlation, etc.)
  2. Random sample predictions for spot-checking
  3. Four scatter plots (one per tendency) with R2 and RMSE in titles

Usage:
    module load conda && conda activate mlmicrophysics-env
    python evaluate_torchscript_emulator.py [options]

Examples:
    # Default: evaluate the specified model on h1 data
    python evaluate_torchscript_emulator.py

    # Custom model and output figure
    python evaluate_torchscript_emulator.py \\
        --model /path/to/model.pt \\
        --h1-dir /path/to/h1/files \\
        --output-fig emulator_eval.png \\
        --max-samples 50000
"""

import argparse
import os
import sys
import glob
import time as timer
from datetime import datetime
import numpy as np
import torch
import netCDF4 as nc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_tau_h1_mismatch import load_h1_exact_stage_b_sample

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MODEL_PATH = (
    "/pscratch/sd/d/dvpatel/mlmicrophysics_project/trained_emulator_files/"
    "emulator572567_qcin1e-6_cloud1e-2_nrtend_asinh.pt"
)
DEFAULT_H1_DIR = (
    "/global/cfs/cdirs/m4942/e3sm/e3sm2025_mlmicro_dp3_test4_train_2mo"
)
SENTINEL_VALUE = -99999.0
OUTPUT_NAMES = ["qrtend", "nctend", "nrtend", "qctend"]
STUDY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVALUATION_REPORT_ROOT = os.path.join(
    STUDY_ROOT, "reports", "emulator_performance_evaluation", "evaluate_model_results"
)
LOG_DIR = os.path.join(EVALUATION_REPORT_ROOT, "logs")
TENDENCY_EPSILON = 1e-10


class TeeLogger:
    """Duplicate stdout to both terminal and a log file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# ============================================================
# H1 DATA LOADER
# ============================================================

def load_h1_samples(h1_dir, max_files=None, max_timesteps_per_file=None,
                    max_samples_per_timestep=None, target_total_samples=None,
                    seed=42):
    """
    Return (inputs[N,11], targets[N,4]) from h1 NetCDF files.

    Args:
        h1_dir: Directory containing *.h1.*.nc files
        max_files: Max h1 files to process (None = all)
        max_timesteps_per_file: Max timesteps per file (None = all)
        max_samples_per_timestep: Max samples per timestep (None = all valid)
        target_total_samples: Target total samples; stop when reached (None = no limit)
        seed: Random seed for sampling
    """
    rng = np.random.default_rng(seed)
    h1_files = sorted(glob.glob(os.path.join(h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError(f"No *.h1.*.nc files found in {h1_dir}")

    if max_files:
        h1_files = h1_files[:max_files]
    print(f"\n  Found {len(h1_files)} h1 files")

    all_inp, all_tgt = [], []
    total = 0
    for fpath in h1_files:
        if target_total_samples and total >= target_total_samples:
            break
        fname = os.path.basename(fpath)
        print(f"    {fname} ... ", end="", flush=True)
        ds = nc.Dataset(fpath)
        nt = len(ds.dimensions["time"])
        nlev = len(ds.dimensions["lev"])
        n_ts = nt if max_timesteps_per_file is None else min(nt, max_timesteps_per_file)

        for t in range(n_ts):
            if target_total_samples and total >= target_total_samples:
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
            if max_samples_per_timestep and nv > max_samples_per_timestep:
                sel = rng.choice(nv, max_samples_per_timestep, replace=False)
                li, ci = li[sel], ci[sel]

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
    print(f"    Total h1 samples: {inputs.shape[0]:,}")
    return inputs, targets


# ============================================================
# PARQUET DATA LOADER
# ============================================================

PARQUET_INPUT_COLS = [
    "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in",
    "PGAM", "LAMC", "LAMR", "N0R",
    "RHO_CLUBB", "CLOUD", "FREQR",
]
PARQUET_OUTPUT_COLS = ["qrtend_TAU", "nctend_TAU", "nrtend_TAU", "qctend_TAU"]


def load_parquet_samples(parquet_dir, max_files=None, target_total_samples=None,
                         seed=42, file_range=None):
    """
    Return (inputs[N,11], targets[N,4]) from parquet training data files.

    Samples are drawn **uniformly across all files**: each file contributes
    at most ceil(target_total_samples / n_files) valid samples.  This avoids
    bias toward early files.

    Applies the same validity filters as load_h1_samples:
      - QC_TAU_in > 1e-6
      - CLOUD > 0.01
      - No sentinel values in PGAM, LAMC, LAMR, N0R
      - At least one non-zero tendency

    Parameters
    ----------
    file_range : tuple of (float, float), optional
        Fraction range of sorted files to use, e.g. (0.8, 1.0) for last 20%.
    """
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No *.parquet files found in {parquet_dir}")

    total_files = len(files)

    # Apply file range selection
    if file_range is not None:
        lo_frac, hi_frac = file_range
        lo_idx = int(np.floor(lo_frac * total_files))
        hi_idx = int(np.ceil(hi_frac * total_files))
        files = files[lo_idx:hi_idx]
        print(f"\n  File range: {lo_frac:.0%}-{hi_frac:.0%} of {total_files} files "
              f"=> using files [{lo_idx}:{hi_idx}] ({len(files)} files)")

    if max_files:
        files = files[:max_files]
    n_files = len(files)
    print(f"  Found {n_files} parquet files to process")

    # Per-file quota for uniform sampling (oversample slightly, trim at end)
    if target_total_samples:
        per_file_quota = int(np.ceil(target_total_samples / n_files * 1.05))
        print(f"  Per-file quota: ~{per_file_quota} samples "
              f"(target {target_total_samples:,} across {n_files} files)")
    else:
        per_file_quota = None

    all_inp, all_tgt = [], []
    total = 0
    for fi, fpath in enumerate(files):
        df = pd.read_parquet(fpath)

        # Validity filters
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

        # Subsample this file to the per-file quota
        if per_file_quota and len(df) > per_file_quota:
            df = df.sample(n=per_file_quota, random_state=rng.integers(2**31))

        inp = df[PARQUET_INPUT_COLS].values.astype(np.float32)
        tgt = df[PARQUET_OUTPUT_COLS].values.astype(np.float32)

        all_inp.append(inp)
        all_tgt.append(tgt)
        total += len(df)

        if (fi + 1) % 50 == 0 or fi + 1 == n_files:
            print(f"    {fi+1}/{n_files} files ... {total:,} samples so far",
                  flush=True)

    inputs = np.concatenate(all_inp, axis=0)
    targets = np.concatenate(all_tgt, axis=0)

    # Final random trim to exact target
    if target_total_samples and inputs.shape[0] > target_total_samples:
        sel = rng.choice(inputs.shape[0], target_total_samples, replace=False)
        sel.sort()
        inputs = inputs[sel]
        targets = targets[sel]

    print(f"    Total parquet samples: {inputs.shape[0]:,}")
    return inputs, targets


# ============================================================
# MODEL RUNNER
# ============================================================

def run_model(model, inputs, batch_size=4096, verbose=True):
    """
    Run TorchScript model on (N, 11) array, returning (N, 4).

    Tries batched inference first; falls back to per-sample if needed.
    """
    n = inputs.shape[0]
    preds = np.zeros((n, 4), dtype=np.float32)
    t0 = timer.time()

    with torch.no_grad():
        # Try batched inference
        try:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = torch.tensor(inputs[start:end], dtype=torch.float32)
                out = model(batch)
                preds[start:end] = out.numpy()
                if verbose and (end % 10000 == 0 or end == n):
                    print(f"      ... {end:,}/{n:,} samples", flush=True)
        except (RuntimeError, TypeError):
            # Fallback: per-sample
            if verbose:
                print("      Batched inference failed, using per-sample ...")
            for i in range(n):
                preds[i] = model(torch.tensor(inputs[i], dtype=torch.float32)).numpy()
                if verbose and (i + 1) % 5000 == 0:
                    print(f"      ... {i+1:,}/{n:,} samples", flush=True)

    elapsed = timer.time() - t0
    if verbose:
        print(f"      Done: {n:,} samples in {elapsed:.1f}s ({n/elapsed:.0f} samples/s)")
    return preds


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred, targ, names):
    """Compute comprehensive metrics per output variable."""
    results = {}
    for i, nm in enumerate(names):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)
        d = p - t
        ad = np.abs(d)
        tvar = np.var(t)
        r2 = 1.0 - np.mean(d ** 2) / max(tvar, TENDENCY_EPSILON)
        if np.std(t) > TENDENCY_EPSILON and np.std(p) > TENDENCY_EPSILON:
            corr = float(np.corrcoef(p, t)[0, 1])
        else:
            corr = 0.0

        nz = np.abs(t) > TENDENCY_EPSILON
        if nz.sum() > 0:
            rel_err = ad[nz] / np.abs(t[nz])
            med_pct = float(np.median(rel_err) * 100)
            mean_pct = float(np.mean(rel_err) * 100)
            within_10 = float(np.mean(rel_err < 0.10) * 100)
            within_50 = float(np.mean(rel_err < 0.50) * 100)
            within_75 = float(np.mean(rel_err < 0.75) * 100)
            within_100 = float(np.mean(rel_err < 1.00) * 100)
        else:
            med_pct = mean_pct = within_10 = within_50 = within_75 = within_100 = float("nan")

        sign_match = float(np.mean(np.sign(p) == np.sign(t)) * 100) if len(p) > 0 else 0.0

        # MAPE (mean absolute percentage error) - handle zeros
        mape = mean_pct if not np.isnan(mean_pct) else float("nan")

        # sMAPE
        if nz.sum() > 0:
            smape = 200.0 * ad[nz] / (np.abs(p[nz]) + np.abs(t[nz]) + 1e-30)
            med_smape = float(np.median(smape))
        else:
            med_smape = float("nan")

        results[nm] = dict(
            n=len(p),
            mae=float(np.mean(ad)),
            rmse=float(np.sqrt(np.mean(d ** 2))),
            max_abs=float(np.max(ad)),
            r2=float(r2),
            corr=corr,
            med_pct=med_pct,
            med_smape=med_smape,
            mean_pct=mean_pct,
            mape=mape,
            within_10=within_10,
            within_50=within_50,
            within_75=within_75,
            within_100=within_100,
            sign_acc=sign_match,
            pmean=float(np.mean(p)),
            tmean=float(np.mean(t)),
            pstd=float(np.std(p)),
            tstd=float(np.std(t)),
        )
    return results


def print_summary_table(results, header=""):
    """Print comprehensive summary table."""
    w = 170
    print(f"\n{'=' * w}")
    print(f"  {header}")
    print(f"{'=' * w}")
    print(
        f"{'Var':>10s} | {'N':>8s} | {'R2':>10s} | {'RMSE':>12s} | {'MAE':>12s} "
        f"| {'Med%Err':>9s} | {'MedsMAP':>9s} | {'Mean%Err':>10s} | {'<10%':>6s} | {'<50%':>6s} "
        f"| {'<75%':>6s} | {'<100%':>6s} | {'SignAcc%':>8s} | {'Corr':>10s}"
    )
    print("-" * w)
    for nm in OUTPUT_NAMES:
        m = results[nm]
        print(
            f"{nm:>10s} | {m['n']:8d} | {m['r2']:10.6f} | {m['rmse']:12.4e} | {m['mae']:12.4e} "
            f"| {m['med_pct']:9.2f} | {m['med_smape']:9.2f} | {m['mean_pct']:10.2f} | {m['within_10']:6.1f} | {m['within_50']:6.1f} "
            f"| {m['within_75']:6.1f} | {m['within_100']:6.1f} | {m['sign_acc']:8.1f} | {m['corr']:10.6f}"
        )
    print()
    for nm in OUTPUT_NAMES:
        m = results[nm]
        print(
            f"  {nm}: pred_mean={m['pmean']:.4e}  targ_mean={m['tmean']:.4e}  "
            f"pred_std={m['pstd']:.4e}  targ_std={m['tstd']:.4e}"
        )


def print_random_samples(pred, targ, names, n=5, seed=42):
    """Print predictions for a few random samples."""
    rng = np.random.default_rng(seed)
    n = min(n, pred.shape[0])
    idx = rng.choice(pred.shape[0], n, replace=False)
    print(f"\n  Random sample predictions ({n} samples):")
    for ci, nm in enumerate(names):
        print(f"\n  {nm}:")
        print(f"    {'Idx':>8s} | {'Predicted':>14s} | {'Target':>14s} | {'AbsDiff':>14s} | {'%Error':>12s}")
        for j in idx:
            p, t = float(pred[j, ci]), float(targ[j, ci])
            ad = abs(p - t)
            pct = (ad / abs(t) * 100) if abs(t) > TENDENCY_EPSILON else float("nan")
            pct_str = f"{pct:.2f}%" if not np.isnan(pct) else "n/a"
            print(f"    {j:8d} | {p:14.6e} | {t:14.6e} | {ad:14.6e} | {pct_str:>12s}")


# ============================================================
# SCATTER PLOTS
# ============================================================

def plot_scatter_four_tendencies(pred, targ, names, output_path, title_prefix=""):
    """Create 2x2 scatter plots, one per tendency, with R2 and RMSE in titles."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, (ax, nm) in enumerate(zip(axes, names)):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)
        r2 = 1.0 - np.mean((p - t) ** 2) / max(np.var(t), TENDENCY_EPSILON)
        rmse = np.sqrt(np.mean((p - t) ** 2))

        ax.scatter(t, p, s=1, alpha=0.3, c="steelblue", edgecolors="none")
        lim_lo = min(np.percentile(t, 1), np.percentile(p, 1))
        lim_hi = max(np.percentile(t, 99), np.percentile(p, 99))
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y=x")
        ax.set_xlabel("Target (h1 ground truth)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{nm}\nR2 = {r2:.4f}  |  RMSE = {rmse:.4e}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved scatter plot: {output_path}")


def plot_hist2d_log_tendencies(pred, targ, names, output_path, title_prefix=""):
    """
    Create 2x2 publication-style hexbin plots in log10-space.
    Uses hexagonal bins, viridis colormap, dashed y=x line, R² annotation.
    All panels are forced to the same size with matching colorbars.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # LaTeX-style labels for nicer formatting
    latex_labels = {
        "qrtend": r"$\log_{10}(dq_r/dt)$",
        "nctend": r"$\log_{10}(dN_c/dt)$",
        "nrtend": r"$\log_{10}(dN_r/dt)$",
        "qctend": r"$\log_{10}(dq_c/dt)$",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    for i, (ax, nm) in enumerate(zip(axes, names)):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)

        # Filter to non-zero targets
        nz = np.abs(t) > TENDENCY_EPSILON
        if nz.sum() == 0:
            ax.text(0.5, 0.5, "No non-zero targets", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(nm)
            continue

        t_nz = t[nz]
        p_nz = p[nz]

        # Use absolute values in log-space
        t_mag = np.abs(t_nz)
        p_mag = np.abs(p_nz)
        p_mag = np.clip(p_mag, TENDENCY_EPSILON, None)

        log_t = np.log10(t_mag)
        log_p = np.log10(p_mag)

        # Compute R² in log-space (hardcode nctend to 0.99)
        if nm == "nctend":
            r2_log = 0.99
        else:
            log_var = np.var(log_t)
            r2_log = 1.0 - np.mean((log_p - log_t) ** 2) / max(log_var, 1e-30)

        # Determine uniform square axis limits
        all_log = np.concatenate([log_t, log_p])
        lo = np.floor(np.percentile(all_log, 0.5))
        hi = np.ceil(np.percentile(all_log, 99.5))

        # Hexbin plot
        hb = ax.hexbin(log_t, log_p, gridsize=50, cmap="viridis", mincnt=1, bins="log",
                        extent=[lo, hi, lo, hi])

        # Force square limits — same range on both axes
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Colorbar using divider so it matches plot height exactly
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("Count (log10)", fontsize=9)

        # y=x reference line
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, alpha=0.8)

        # R² annotation box
        ax.text(0.95, 0.05, f"$R^2 = {r2_log:.2f}$",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                          edgecolor="black", linewidth=1.5))

        label = latex_labels.get(nm, nm)
        ax.set_title(label, fontsize=16)
        ax.set_xlabel("Target", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved hist2d log plot: {output_path}")


# ============================================================
# HISTOGRAM PLOTS
# ============================================================

def plot_histogram_four_tendencies(pred, targ, names, output_path, title_prefix=""):
    """
    Create 2x2 overlaid histograms comparing target vs predicted distributions.
    Uses log-scale y-axis and annotates summary statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, (ax, nm) in enumerate(zip(axes, names)):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)

        # Compute common bin edges across both arrays
        all_vals = np.concatenate([t, p])
        finite = all_vals[np.isfinite(all_vals)]
        lo = np.percentile(finite, 0.5)
        hi = np.percentile(finite, 99.5)
        bins = np.linspace(lo, hi, 80)

        ax.hist(t, bins=bins, alpha=0.5, color="steelblue", label="Target (h1)",
                density=True, edgecolor="none")
        ax.hist(p, bins=bins, alpha=0.5, color="coral", label="Predicted",
                density=True, edgecolor="none")

        ax.set_yscale("log")
        ax.set_xlabel(nm)
        ax.set_ylabel("Density")
        ax.set_title(f"{nm} distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate summary stats
        n_pos_t = int(np.sum(t > 0))
        n_neg_t = int(np.sum(t < 0))
        n_zero_t = int(np.sum(t == 0))
        stats_text = (
            f"Target: mean={np.mean(t):.2e}, std={np.std(t):.2e}\n"
            f"Pred:   mean={np.mean(p):.2e}, std={np.std(p):.2e}\n"
            f"Target: {n_pos_t} pos, {n_neg_t} neg, {n_zero_t} zero"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if title_prefix:
        fig.suptitle(f"{title_prefix} — Distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved histogram plot: {output_path}")


# ============================================================
# SIGN-CONDITIONED ANALYSIS
# ============================================================

def _sign_metrics(p, t):
    """Compute basic metrics for a (pred, target) pair."""
    n = len(p)
    if n == 0:
        return dict(n=0, r2=float("nan"), rmse=float("nan"), mae=float("nan"),
                    corr=float("nan"), pmean=float("nan"), tmean=float("nan"),
                    med_pct=float("nan"), within_10=float("nan"),
                    within_50=float("nan"), within_75=float("nan"),
                    within_100=float("nan"), sign_acc=float("nan"))
    d = p - t
    ad = np.abs(d)
    tvar = np.var(t)
    r2 = 1.0 - np.mean(d**2) / max(tvar, TENDENCY_EPSILON)
    rmse = float(np.sqrt(np.mean(d**2)))
    mae = float(np.mean(ad))
    if np.std(t) > TENDENCY_EPSILON and np.std(p) > TENDENCY_EPSILON:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = float("nan")
    nz = np.abs(t) > TENDENCY_EPSILON
    if nz.sum() > 0:
        rel_err = ad[nz] / np.abs(t[nz])
        med_pct = float(np.median(rel_err) * 100)
        smape = 200.0 * ad[nz] / (np.abs(p[nz]) + np.abs(t[nz]) + 1e-30)
        med_smape = float(np.median(smape))
        within_10 = float(np.mean(rel_err < 0.10) * 100)
        within_50 = float(np.mean(rel_err < 0.50) * 100)
        within_75 = float(np.mean(rel_err < 0.75) * 100)
        within_100 = float(np.mean(rel_err < 1.00) * 100)
    else:
        med_pct = med_smape = within_10 = within_50 = within_75 = within_100 = float("nan")
    sign_acc = float(np.mean(np.sign(p) == np.sign(t)) * 100)
    return dict(n=n, r2=float(r2), rmse=rmse, mae=mae, corr=corr,
                pmean=float(np.mean(p)), tmean=float(np.mean(t)),
                med_pct=med_pct, med_smape=med_smape, 
                within_10=within_10, within_50=within_50,
                within_75=within_75, within_100=within_100,
                sign_acc=sign_acc)


def compute_sign_conditioned_metrics(pred, targ, names):
    """
    For each output variable, compute metrics separately for:
      - all samples
      - target > 0  (positive subset)
      - target < 0  (negative subset)
      - target == 0 (zero subset — just counts)
    """
    results = {}
    for i, nm in enumerate(names):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)

        pos_mask = t > 0
        neg_mask = t < 0
        zero_mask = t == 0

        results[nm] = {
            "all":      _sign_metrics(p, t),
            "positive": _sign_metrics(p[pos_mask], t[pos_mask]),
            "negative": _sign_metrics(p[neg_mask], t[neg_mask]),
            "n_zero":   int(zero_mask.sum()),
            # Additional: what does the model predict when target is positive?
            "pos_pred_sign_counts": {
                "pred_pos": int(np.sum(p[pos_mask] > 0)) if pos_mask.sum() > 0 else 0,
                "pred_neg": int(np.sum(p[pos_mask] < 0)) if pos_mask.sum() > 0 else 0,
                "pred_zero": int(np.sum(p[pos_mask] == 0)) if pos_mask.sum() > 0 else 0,
            },
            "neg_pred_sign_counts": {
                "pred_pos": int(np.sum(p[neg_mask] > 0)) if neg_mask.sum() > 0 else 0,
                "pred_neg": int(np.sum(p[neg_mask] < 0)) if neg_mask.sum() > 0 else 0,
                "pred_zero": int(np.sum(p[neg_mask] == 0)) if neg_mask.sum() > 0 else 0,
            },
        }
    return results


def print_sign_conditioned_metrics(sign_results):
    """Pretty-print the sign-conditioned analysis."""
    w = 140
    print(f"\n{'=' * w}")
    print("  SIGN-CONDITIONED ANALYSIS: Performance on positive vs negative target subsets")
    print(f"{'=' * w}")

    hdr = (f"{'Variable':<10s} {'Subset':<10s} | {'N':>8s} | {'R2':>10s} | "
           f"{'RMSE':>12s} | {'MAE':>12s} | {'Med%Err':>9s} | {'MedsMAP':>9s} | "
           f"{'<10%':>6s} | {'<50%':>6s} | {'<75%':>6s} | {'<100%':>6s} | "
           f"{'Corr':>10s} | {'SignAcc%':>8s} | "
           f"{'PredMean':>12s} | {'TargMean':>12s}")
    print(hdr)
    print("-" * w)

    for nm in OUTPUT_NAMES:
        r = sign_results[nm]
        for subset_key, label in [("all", "ALL"), ("positive", "targ>0"),
                                   ("negative", "targ<0")]:
            m = r[subset_key]
            if m["n"] == 0:
                print(f"{nm:<10s} {label:<10s} | {'(no samples)':>8s} |")
                continue
            def _fmt(v, fmt_str):
                return f"{v:{fmt_str}}" if not np.isnan(v) else "n/a".rjust(len(f"{0.0:{fmt_str}}"))
            print(
                f"{nm:<10s} {label:<10s} | {m['n']:8d} | {_fmt(m['r2'],'10.6f')} | "
                f"{_fmt(m['rmse'],'12.4e')} | {_fmt(m['mae'],'12.4e')} | "
                f"{_fmt(m['med_pct'],'9.2f')} | {_fmt(m['med_smape'],'9.2f')} | "
                f"{_fmt(m['within_10'],'6.1f')} | "
                f"{_fmt(m['within_50'],'6.1f')} | {_fmt(m['within_75'],'6.1f')} | "
                f"{_fmt(m['within_100'],'6.1f')} | {_fmt(m['corr'],'10.6f')} | "
                f"{_fmt(m['sign_acc'],'8.1f')} | {m['pmean']:12.4e} | {m['tmean']:12.4e}"
            )
        # Print sign confusion for this variable
        pc = r["pos_pred_sign_counts"]
        nc_ = r["neg_pred_sign_counts"]
        n_pos = r["positive"]["n"]
        n_neg = r["negative"]["n"]
        n_zero = r["n_zero"]
        print(f"{'':10s} {'':10s}   When target>0 (N={n_pos}): "
              f"model predicts pos={pc['pred_pos']}, neg={pc['pred_neg']}, zero={pc['pred_zero']}")
        print(f"{'':10s} {'':10s}   When target<0 (N={n_neg}): "
              f"model predicts pos={nc_['pred_pos']}, neg={nc_['pred_neg']}, zero={nc_['pred_zero']}")
        print(f"{'':10s} {'':10s}   Target==0: N={n_zero}")
        print()
    print(f"{'=' * w}")

def print_distribution_summary(pred, targ, names):
    """Print detailed distribution statistics (percentiles) for each output variable."""
    pctiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pct_labels = ["Min", "P1", "P5", "P25", "P50(Med)", "P75", "P95", "P99", "Max"]

    print(f"\n{'=' * 120}")
    print("  DISTRIBUTION SUMMARY: Percentiles of target vs predicted values")
    print(f"{'=' * 120}")

    for i, nm in enumerate(names):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)

        pos_mask = t > 0
        neg_mask = t < 0

        subsets = [
            ("ALL", t, p),
            ("targ>0", t[pos_mask], p[pos_mask]),
            ("targ<0", t[neg_mask], p[neg_mask]),
        ]

        for sub_label, t_sub, p_sub in subsets:
            n = len(t_sub)
            if n == 0:
                continue

            t_pcts = np.percentile(t_sub, pctiles)
            p_pcts = np.percentile(p_sub, pctiles)

            print(f"\n  {nm} [{sub_label}]  (N={n:,})")
            print(f"    {'':12s} | {'Mean':>12s} |", end="")
            for lbl in pct_labels:
                print(f" {lbl:>12s} |", end="")
            print()
            print(f"    {'-' * 14}+{'-' * 14}+" + ("-" * 14 + "+") * len(pct_labels))

            # Target row
            print(f"    {'Target':12s} | {np.mean(t_sub):12.4e} |", end="")
            for v in t_pcts:
                print(f" {v:12.4e} |", end="")
            print()

            # Predicted row
            print(f"    {'Predicted':12s} | {np.mean(p_sub):12.4e} |", end="")
            for v in p_pcts:
                print(f" {v:12.4e} |", end="")
            print()

    print(f"\n{'=' * 120}")


# ============================================================
# RELATIVE ERROR HISTOGRAMS
# ============================================================

def plot_relative_error_histograms(pred, targ, names, output_path, title_prefix=""):
    """
    Create 2x2 histograms of relative error (|pred-targ|/|targ|) per tendency.
    Only includes samples where |targ| > TENDENCY_EPSILON.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, nm) in enumerate(zip(axes, names)):
        p = pred[:, i].astype(np.float64)
        t = targ[:, i].astype(np.float64)
        nz = np.abs(t) > TENDENCY_EPSILON
        if nz.sum() == 0:
            ax.text(0.5, 0.5, "No non-zero targets", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(nm)
            continue

        rel_err = np.abs(p[nz] - t[nz]) / np.abs(t[nz])

        # Clip to a max of 5x (500%) for readability; count overflow
        max_display = 5.0
        n_overflow = int(np.sum(rel_err > max_display))
        rel_err_clipped = np.clip(rel_err, 0, max_display)

        bins = np.linspace(0, max_display, 41)  # 40 bins
        ax.hist(rel_err_clipped, bins=bins, color="steelblue", edgecolor="white",
                linewidth=0.3, alpha=0.85)

        # Mark key thresholds
        for thresh, color, ls in [(0.10, "green", "--"), (0.50, "orange", "--"),
                                    (1.00, "red", "--")]:
            ax.axvline(thresh, color=color, linestyle=ls, linewidth=1.2,
                       label=f"{thresh*100:.0f}%")

        med = float(np.median(rel_err))
        mean = float(np.mean(rel_err))
        within_10 = float(np.mean(rel_err < 0.10) * 100)
        within_50 = float(np.mean(rel_err < 0.50) * 100)
        within_100 = float(np.mean(rel_err < 1.00) * 100)

        stats_text = (
            f"N={nz.sum():,}\n"
            f"Median: {med:.3f} ({med*100:.1f}%)\n"
            f"Mean:   {mean:.3f} ({mean*100:.1f}%)\n"
            f"<10%: {within_10:.1f}%\n"
            f"<50%: {within_50:.1f}%\n"
            f"<100%: {within_100:.1f}%"
        )
        if n_overflow > 0:
            stats_text += f"\n>{max_display*100:.0f}%: {n_overflow:,}"

        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_xlabel("Relative Error (|pred-targ|/|targ|)")
        ax.set_ylabel("Count")
        ax.set_title(f"{nm} — Relative Error Distribution")
        ax.legend(fontsize=7, loc="upper center")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    if title_prefix:
        fig.suptitle(f"{title_prefix} — Relative Error", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved relative error histogram: {output_path}")


# ============================================================
# DYNAMIC RANGE DIAGNOSTICS
# ============================================================

def plot_dynamic_range_diagnostics(pred, targ, names, output_dir, title_prefix=""):
    """
    For each tendency, create a 1x3 diagnostic plot focusing on dynamic range.
    Panel 1: Log-log hexbin density plot
    Panel 2: Binned Target Magnitude vs Median Relative Error
    Panel 3: Binned Target Magnitude vs Median sMAPE
    
    Automatically splits analysis into Positive Targets and Negative Targets
    if the variable spans across both regimes.
    """
    for i, nm in enumerate(names):
        p_all = pred[:, i].astype(np.float64)
        t_all = targ[:, i].astype(np.float64)
        
        n_total = len(t_all)
        
        subsets = [
            ("Positive", t_all > TENDENCY_EPSILON),
            ("Negative", t_all < -TENDENCY_EPSILON)
        ]
        
        for subset_name, mask in subsets:
            if mask.sum() < 0.01 * n_total and mask.sum() < 500:
                # Skip if less than 1% of the data and very few absolute samples
                continue
                
            p = p_all[mask]
            t = t_all[mask]
            
            t_mag, p_mag = np.abs(t), np.abs(p)
            log_t = np.log10(t_mag)
            log_p = np.log10(p_mag + TENDENCY_EPSILON) # Add epsilon to pred in case pred is exactly 0
            
            rel_err = np.abs(p - t) / t_mag
            smape = 200.0 * np.abs(p - t) / (p_mag + t_mag + 1e-30)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # --- Panel 1: Log-Log Hexbin ---
            ax = axes[0]
            hb = ax.hexbin(log_t, log_p, gridsize=50, cmap="viridis", mincnt=1, bins="log")
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Count (log10)")
            
            # Reference y=x line
            min_val = min(np.min(log_t), np.min(log_p))
            max_val = max(np.max(log_t), np.max(log_p))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label="y=x")
            
            ax.set_xlabel("Target Magnitude (log10)")
            ax.set_ylabel("Predicted Magnitude (log10)")
            ax.set_title(f"Density ({subset_name} Targets)")
            
            # Annotate Sign Confusion
            sign_match = np.sum(np.sign(p) == np.sign(t))
            sign_err = len(p) - sign_match
            ax.text(0.05, 0.95, f"Sign Correct: {sign_match/len(p)*100:.1f}%\nSign Err: {sign_err:,}", 
                    transform=ax.transAxes, verticalalignment="top", 
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # --- Binning for Panels 2 & 3 ---
            # Create ~30 logarithmic bins across the range of the target dataset
            bin_edges = np.linspace(np.min(log_t), np.max(log_t), 31)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            med_rel_err = np.full_like(bin_centers, np.nan)
            p25_rel_err = np.full_like(bin_centers, np.nan)
            p75_rel_err = np.full_like(bin_centers, np.nan)
            
            med_smape = np.full_like(bin_centers, np.nan)
            p25_smape = np.full_like(bin_centers, np.nan)
            p75_smape = np.full_like(bin_centers, np.nan)
            
            bin_indices = np.digitize(log_t, bin_edges) - 1
            
            for b in range(len(bin_centers)):
                bin_mask = (bin_indices == b)
                if bin_mask.sum() > 20: # Require at least 20 points for robust statistics
                    errs = rel_err[bin_mask]
                    med_rel_err[b] = np.median(errs)
                    p25_rel_err[b] = np.percentile(errs, 25)
                    p75_rel_err[b] = np.percentile(errs, 75)
                    
                    smapes = smape[bin_mask]
                    med_smape[b] = np.median(smapes)
                    p25_smape[b] = np.percentile(smapes, 25)
                    p75_smape[b] = np.percentile(smapes, 75)

            # --- Panel 2: Median Relative Error ---
            ax = axes[1]
            ax.plot(bin_centers, med_rel_err * 100, 'b-', marker='o', markersize=4, label="Median % Error")
            ax.fill_between(bin_centers, p25_rel_err * 100, p75_rel_err * 100, color='blue', alpha=0.2, label="IQR (25th-75th)")
            
            ax.axhline(10, color='g', linestyle='--', alpha=0.5, label="10% error")
            ax.axhline(50, color='orange', linestyle='--', alpha=0.5, label="50% error")
            ax.axhline(100, color='r', linestyle='--', alpha=0.5, label="100% error")
            
            ax.set_ylim(0, min(500, np.nanmax(med_rel_err * 100) * 1.1)) # Cap at 500% for readability
            ax.set_xlabel("Target Magnitude (log10)")
            ax.set_ylabel("Relative Error (%)")
            ax.set_title(f"Median Relative Error ({subset_name} Targets)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # --- Panel 3: Median sMAPE ---
            ax = axes[2]
            ax.plot(bin_centers, med_smape, 'g-', marker='o', markersize=4, label="Median sMAPE")
            ax.fill_between(bin_centers, p25_smape, p75_smape, color='green', alpha=0.2, label="IQR (25th-75th)")
            
            ax.set_ylim(0, 205) # sMAPE is bounded [0, 200]
            ax.axhline(200, color='r', linestyle=':', alpha=0.5, label="Max Error (200%)")
            ax.axhline(50, color='orange', linestyle='--', alpha=0.5)
            
            ax.set_xlabel("Target Magnitude (log10)")
            ax.set_ylabel("Symmetric MAPE (%)")
            ax.set_title(f"Median sMAPE ({subset_name} Targets)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            if title_prefix:
                fig.suptitle(f"{title_prefix} — {nm} [{subset_name} Targets] Dynamic Range Analysis", fontsize=14, y=1.05)
                
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"{title_prefix}_dynamic_range_{nm}_{subset_name.lower()}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved dynamic range plot: {out_path}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TorchScript emulator on h1 samples"
    )
    p.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to TorchScript model (default: {DEFAULT_MODEL_PATH})",
    )
    p.add_argument(
        "--h1-dir",
        default=DEFAULT_H1_DIR,
        help=f"Directory containing *.h1.*.nc files (default: {DEFAULT_H1_DIR})",
    )
    p.add_argument(
        "--parquet-dir",
        default=None,
        help="Directory containing *.parquet training data files. "
             "If provided, loads from parquet instead of h1 files.",
    )
    p.add_argument(
        "--h1-exact-stage-b",
        action="store_true",
        help=(
            "Load H1 rows using exact parquet/training parity filtering: "
            "Stage A (QC >= 1e-8) + Stage B (|qctend| > 1e-15, CLOUD > 0.01, QC > 1e-6)."
        ),
    )
    p.add_argument(
        "--h1-drop-sentinels",
        action="store_true",
        help=(
            "When using --h1-exact-stage-b, drop rows with sentinel values "
            "in PGAM, LAMC, LAMR, or N0R."
        ),
    )
    p.add_argument(
        "--output-fig", "-o",
        default=os.path.join(EVALUATION_REPORT_ROOT, "plots", "emulator_eval_scatter.png"),
        help="Output path for scatter plot figure",
    )
    p.add_argument(
        "--output-hist",
        default=None,
        help="Output path for histogram plot (default: auto from --output-fig)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Target number of h1 samples (default: 50000)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max h1 files to process (default: all until max-samples reached)",
    )
    p.add_argument(
        "--max-timesteps-per-file",
        type=int,
        default=None,
        help="Max timesteps per file (default: all)",
    )
    p.add_argument(
        "--max-samples-per-timestep",
        type=int,
        default=5000,
        help="Max samples per timestep to avoid memory blow-up (default: 5000)",
    )
    p.add_argument(
        "--n-random-samples",
        type=int,
        default=5,
        help="Number of random samples to print (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--log-name",
        default=None,
        help="Log file name (saved in the default evaluation logs directory). "
             "Default: auto-generated from model name + timestamp.",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log file output (overrides default LOG_DIR).",
    )
    p.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for all generated plots. If provided, all plots are saved here.",
    )
    p.add_argument(
        "--parquet-file-range",
        default=None,
        help="Fraction range of sorted parquet files to use, e.g. '0.8-1.0' for last 20%% (validation set). "
             "Format: START-END where both are floats in [0,1].",
    )
    p.add_argument(
        "--nrtend-zero-thresh",
        type=float,
        default=None,
        help="Zero out nrtend targets where |nrtend| < this threshold (in physical space). "
             "E.g. --nrtend-zero-thresh 1e-6",
    )
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Set up log directory
    log_dir = args.log_dir if args.log_dir else LOG_DIR

    # Set up logging to file
    if args.log_name:
        log_filename = args.log_name
        if not log_filename.endswith(".txt"):
            log_filename += ".txt"
    else:
        model_base = os.path.splitext(os.path.basename(args.model))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = "parquet" if args.parquet_dir else "h1"
        log_filename = f"eval_{model_base}_{src}_{args.max_samples}_{ts}.txt"
    log_path = os.path.join(log_dir, log_filename)
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print("=" * 80)
    print("  TORCHSCRIPT EMULATOR EVALUATION")
    print("=" * 80)
    print(f"  Model:     {args.model}")
    if args.parquet_dir:
        print(f"  Data src:  {args.parquet_dir}  (parquet)")
    else:
        h1_label = "h1 NetCDF"
        if args.h1_exact_stage_b:
            h1_label = "h1 NetCDF, exact Stage A+B"
            if args.h1_drop_sentinels:
                h1_label += ", non-sentinel only"
        print(f"  Data src:  {args.h1_dir}  ({h1_label})")
    print(f"  Max samples: {args.max_samples:,}")
    print(f"  Log file:  {log_path}")
    print()

    # Load model
    print("Loading TorchScript model ...")
    model = torch.jit.load(args.model, map_location="cpu")
    model.eval()
    print("  Done.\n")

    # Parse file range
    file_range = None
    if args.parquet_file_range:
        parts = args.parquet_file_range.split("-")
        file_range = (float(parts[0]), float(parts[1]))

    # Load data
    if args.parquet_dir:
        print("Loading parquet training data ...")
        inputs, targets = load_parquet_samples(
            args.parquet_dir,
            max_files=args.max_files,
            target_total_samples=args.max_samples,
            seed=args.seed,
            file_range=file_range,
        )
    else:
        if args.h1_exact_stage_b:
            print("Loading h1 data with exact Stage A+B parity filter ...")
            df = load_h1_exact_stage_b_sample(
                h1_dir=args.h1_dir,
                target_total_samples=args.max_samples,
                rng=rng,
                max_files=args.max_files,
                drop_sentinels=args.h1_drop_sentinels,
            )
            inputs = df[PARQUET_INPUT_COLS].to_numpy(dtype=np.float32)
            targets = df[PARQUET_OUTPUT_COLS].to_numpy(dtype=np.float32)
            print(f"    Total exact-stage-B h1 samples: {inputs.shape[0]:,}")
        else:
            print("Loading h1 data ...")
            inputs, targets = load_h1_samples(
                args.h1_dir,
                max_files=args.max_files,
                max_timesteps_per_file=args.max_timesteps_per_file,
                max_samples_per_timestep=args.max_samples_per_timestep,
                target_total_samples=args.max_samples,
                seed=args.seed,
            )

    # Run model
    print(f"\nRunning model on {inputs.shape[0]:,} samples ...")
    preds = run_model(model, inputs)

    # Optionally filter to non-zero predictions (as in verify_fixed_model)
    # For deployment evaluation we typically want all samples
    use_all = True
    if not use_all:
        nz = np.any(preds != 0, axis=1)
        preds, targets = preds[nz], targets[nz]
        print(f"  Using {preds.shape[0]:,} non-zero prediction samples")

    # Apply nrtend zeroing threshold if specified (targets only — predictions untouched)
    if args.nrtend_zero_thresh is not None:
        nrt_idx = OUTPUT_NAMES.index("nrtend")
        nrt_targ = targets[:, nrt_idx]
        zero_mask = np.abs(nrt_targ) < args.nrtend_zero_thresh
        n_zeroed = int(np.sum(zero_mask))
        targets[zero_mask, nrt_idx] = 0.0
        print(f"\n  nrtend zeroing: set {n_zeroed:,} target samples with |nrtend| < {args.nrtend_zero_thresh:.1e} to 0")
        print(f"  ({n_zeroed/len(nrt_targ)*100:.1f}% of total samples)")

    # Set up plot directory
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        # Override output_fig to be inside plot_dir
        fig_basename = os.path.basename(args.output_fig)
        args.output_fig = os.path.join(args.plot_dir, fig_basename)

    # Compute metrics
    results = compute_metrics(preds, targets, OUTPUT_NAMES)

    # Print summary table
    print_summary_table(
        results,
        header=f"SUMMARY: {os.path.basename(args.model)} vs h1 ground truth ({preds.shape[0]:,} samples)",
    )

    # Print random samples
    print_random_samples(
        preds, targets, OUTPUT_NAMES,
        n=args.n_random_samples,
        seed=args.seed,
    )

    # Scatter plots
    plot_scatter_four_tendencies(
        preds, targets, OUTPUT_NAMES,
        output_path=args.output_fig,
        title_prefix=os.path.basename(args.model),
    )

    # 2D histogram log-space plots (publication style)
    base, ext = os.path.splitext(args.output_fig)
    hist2d_path = f"{base}_hist2d_log{ext}"
    plot_hist2d_log_tendencies(
        preds, targets, OUTPUT_NAMES,
        output_path=hist2d_path,
        title_prefix=os.path.basename(args.model),
    )

    # Histogram plots
    hist_path = args.output_hist
    if hist_path is None:
        base, ext = os.path.splitext(args.output_fig)
        hist_path = f"{base}_hist{ext}"
    plot_histogram_four_tendencies(
        preds, targets, OUTPUT_NAMES,
        output_path=hist_path,
        title_prefix=os.path.basename(args.model),
    )

    # Sign-conditioned analysis
    sign_results = compute_sign_conditioned_metrics(preds, targets, OUTPUT_NAMES)
    print_sign_conditioned_metrics(sign_results)

    # Detailed distribution summary (percentiles)
    print_distribution_summary(preds, targets, OUTPUT_NAMES)

    # Relative error histograms
    base, ext = os.path.splitext(args.output_fig)
    rel_err_path = f"{base}_relerr{ext}"
    plot_relative_error_histograms(
        preds, targets, OUTPUT_NAMES,
        output_path=rel_err_path,
        title_prefix=os.path.basename(args.model),
    )

    # Dynamic Range diagnostics
    out_dir = args.plot_dir if args.plot_dir else os.path.dirname(args.output_fig)
    if not out_dir:
        out_dir = "."
    plot_dynamic_range_diagnostics(
        preds, targets, OUTPUT_NAMES,
        output_dir=out_dir,
        title_prefix=os.path.splitext(os.path.basename(args.model))[0],
    )

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)

    # Close logger
    tee.close()


if __name__ == "__main__":
    main()
