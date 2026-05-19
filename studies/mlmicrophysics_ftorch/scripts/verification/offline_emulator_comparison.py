#!/usr/bin/env python3
"""
Offline Emulator Comparison Script
===================================
Reads raw inputs from Andrew's h1 files, runs them through the TorchScript emulator,
and compares predicted outputs against the raw outputs stored in the h1 files.

This verifies whether the trained emulator produces the same outputs offline
as when plugged into E3SM via FTorch.

Usage:
    conda activate mlmicrophysics-env
    python offline_emulator_comparison.py
"""

import os
import glob
import numpy as np
import torch
import netCDF4 as nc
import time as timer

# =========================================================================
# CONFIGURATION
# =========================================================================

H1_DIR = '/global/cfs/cdirs/m4942/e3sm/e3sm2025_mlmicro_dp3_test4_train_2mo'
MODEL_PATH = '/pscratch/sd/d/dvpatel/mlmicrophysics_project/trained_emulator_files/emulator107666_torchscript_qctauin1e-6_cloud1e-2.pt'

# Variable name mapping: h1 file name -> emulator input order (index)
# Emulator input order (11 features):
#   [0] QC_TAU_in  -> P3_qc_in_TAU
#   [1] QR_TAU_in  -> P3_qr_in_TAU
#   [2] NC_TAU_in  -> P3_nc_in_TAU
#   [3] NR_TAU_in  -> P3_nr_in_TAU
#   [4] PGAM       -> P3_mu_c
#   [5] LAMC       -> P3_lamc
#   [6] LAMR       -> P3_lamr
#   [7] N0R        -> P3_nr
#   [8] RHO_CLUBB  -> RHO_CLUBB (note: on ilev grid, 81 levels vs 80)
#   [9] CLOUD      -> CLOUD
#   [10] FREQR     -> FREQR

INPUT_VAR_MAP = [
    ('P3_qc_in_TAU', 'lev'),    # 0: QC_TAU_in
    ('P3_qr_in_TAU', 'lev'),    # 1: QR_TAU_in
    ('P3_nc_in_TAU', 'lev'),    # 2: NC_TAU_in
    ('P3_nr_in_TAU', 'lev'),    # 3: NR_TAU_in
    ('P3_mu_c',      'lev'),    # 4: PGAM
    ('P3_lamc',      'lev'),    # 5: LAMC
    ('P3_lamr',      'lev'),    # 6: LAMR
    ('P3_nr',        'lev'),    # 7: N0R
    ('RHO_CLUBB',    'ilev'),   # 8: RHO_CLUBB (on ilev)
    ('CLOUD',        'lev'),    # 9: CLOUD
    ('FREQR',        'lev'),    # 10: FREQR
]

# Emulator output order (4 outputs):
#   [0] qrtend_TAU  -> P3_qrtend_TAU_raw
#   [1] nctend_TAU  -> P3_nctend_TAU_raw
#   [2] nrtend_TAU  -> P3_nrtend_TAU_raw
#   [3] qctend_TAU  -> P3_qctend_TAU_raw (= -qrtend)

OUTPUT_VARS = [
    'P3_qrtend_TAU_raw',   # 0
    'P3_nctend_TAU_raw',   # 1
    'P3_nrtend_TAU_raw',   # 2
    'P3_qctend_TAU_raw',   # 3
]

OUTPUT_NAMES = ['qrtend', 'nctend', 'nrtend', 'qctend']

SENTINEL_VALUE = -99999.0

# How many time steps to process (None = all)
MAX_TIMESTEPS = None
# How many files to process (None = all)
MAX_FILES = None


def load_model(model_path):
    """Load TorchScript model."""
    print(f"Loading model from: {model_path}")
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    return model


def extract_inputs_outputs(ds, t_idx):
    """
    Extract input features and output targets for a single time step.
    
    Returns:
        inputs: np.array of shape (n_valid, 11)
        targets: np.array of shape (n_valid, 4)
        valid_mask: boolean mask of shape (lev, ncol)
        n_sentinel_in_valid: count of sentinel values in valid points
    """
    nlev = len(ds.dimensions['lev'])
    ncol = len(ds.dimensions['ncol'])
    
    # Read all input variables
    input_arrays = []
    for var_name, grid_type in INPUT_VAR_MAP:
        arr = ds.variables[var_name][t_idx, :, :]  # (lev or ilev, ncol)
        if grid_type == 'ilev':
            # RHO_CLUBB is on ilev (81 levels) - take first 80 to match lev
            arr = arr[:nlev, :]
        input_arrays.append(arr)
    
    # Stack inputs: shape (nlev, ncol, 11)
    input_stack = np.stack(input_arrays, axis=-1)
    
    # Read output variables
    output_arrays = []
    for var_name in OUTPUT_VARS:
        arr = ds.variables[var_name][t_idx, :, :]  # (lev, ncol)
        output_arrays.append(arr)
    output_stack = np.stack(output_arrays, axis=-1)  # (nlev, ncol, 4)
    
    # Apply validity filter (same as in the TorchScript model)
    qc_tau_in = input_stack[:, :, 0]  # QC_TAU_in
    cloud = input_stack[:, :, 9]       # CLOUD
    valid_mask = (qc_tau_in > 1e-6) & (cloud > 0.01)
    
    # Check for sentinel values in valid points
    p3_nr_vals = input_stack[:, :, 7]  # P3_nr (N0R)
    p3_mu_c_vals = input_stack[:, :, 4]  # P3_mu_c (PGAM)
    sentinel_mask = (p3_nr_vals == SENTINEL_VALUE) | (p3_mu_c_vals == SENTINEL_VALUE)
    n_sentinel_in_valid = int(np.sum(sentinel_mask & valid_mask))
    
    # Exclude sentinel values from valid mask for clean comparison
    clean_valid_mask = valid_mask & ~sentinel_mask
    
    # Extract valid points
    inputs = input_stack[clean_valid_mask]   # (n_valid, 11)
    targets = output_stack[clean_valid_mask]  # (n_valid, 4)
    
    return inputs, targets, clean_valid_mask, n_sentinel_in_valid


def run_emulator(model, inputs):
    """
    Run the TorchScript emulator on input features.
    
    The TorchScript model processes one sample at a time (expects shape [11]).
    
    Args:
        model: TorchScript model
        inputs: np.array of shape (n_samples, 11)
    
    Returns:
        predictions: np.array of shape (n_samples, 4)
    """
    n_samples = inputs.shape[0]
    predictions = np.zeros((n_samples, 4), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(n_samples):
            x = torch.tensor(inputs[i], dtype=torch.float32)
            y = model(x)
            predictions[i] = y.numpy()
    
    return predictions


def compute_metrics(predictions, targets, name=""):
    """Compute comparison metrics between emulator predictions and h1 outputs."""
    metrics = {}
    
    for i, out_name in enumerate(OUTPUT_NAMES):
        pred = predictions[:, i]
        targ = targets[:, i]
        
        # Basic statistics
        diff = pred - targ
        abs_diff = np.abs(diff)
        
        # Avoid division by zero
        nonzero_mask = np.abs(targ) > 1e-30
        
        metrics[out_name] = {
            'n_samples': len(pred),
            'pred_mean': float(np.mean(pred)),
            'targ_mean': float(np.mean(targ)),
            'mae': float(np.mean(abs_diff)),
            'rmse': float(np.sqrt(np.mean(diff**2))),
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_abs_diff': float(np.mean(abs_diff)),
            'exact_match_frac': float(np.mean(abs_diff < 1e-30)),
            'close_match_1pct': float(np.mean(abs_diff[nonzero_mask] / np.abs(targ[nonzero_mask]) < 0.01)) if nonzero_mask.sum() > 0 else 0.0,
            'r_squared': float(1 - np.sum(diff**2) / max(np.sum((targ - np.mean(targ))**2), 1e-30)),
            'correlation': float(np.corrcoef(pred, targ)[0, 1]) if np.std(targ) > 0 and np.std(pred) > 0 else 0.0,
        }
    
    return metrics


def print_metrics(metrics, header=""):
    """Pretty-print comparison metrics."""
    print(f"\n{'='*90}")
    print(f"  {header}")
    print(f"{'='*90}")
    
    # Header
    print(f"{'Variable':>12s} | {'N':>8s} | {'MAE':>12s} | {'RMSE':>12s} | {'MaxAbsDiff':>12s} | {'R²':>8s} | {'Corr':>8s} | {'Exact%':>7s}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
    
    for out_name in OUTPUT_NAMES:
        m = metrics[out_name]
        print(f"{out_name:>12s} | {m['n_samples']:8d} | {m['mae']:12.4e} | {m['rmse']:12.4e} | "
              f"{m['max_abs_diff']:12.4e} | {m['r_squared']:8.5f} | {m['correlation']:8.5f} | "
              f"{m['exact_match_frac']*100:6.2f}%")
    
    print()
    for out_name in OUTPUT_NAMES:
        m = metrics[out_name]
        print(f"  {out_name}: pred_mean={m['pred_mean']:.6e}, targ_mean={m['targ_mean']:.6e}")


def main():
    print("=" * 90)
    print("  OFFLINE EMULATOR COMPARISON")
    print("  Comparing TorchScript emulator predictions vs E3SM h1 raw outputs")
    print("=" * 90)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Find h1 files
    h1_files = sorted(glob.glob(os.path.join(H1_DIR, '*.h1.*.nc')))
    if MAX_FILES:
        h1_files = h1_files[:MAX_FILES]
    print(f"\nFound {len(h1_files)} h1 files")
    
    # Aggregated results
    all_predictions = []
    all_targets = []
    total_valid = 0
    total_sentinel_in_valid = 0
    total_timesteps = 0
    
    for fi, fpath in enumerate(h1_files):
        fname = os.path.basename(fpath)
        ds = nc.Dataset(fpath)
        nt = len(ds.dimensions['time'])
        
        if MAX_TIMESTEPS and total_timesteps >= MAX_TIMESTEPS:
            ds.close()
            break
        
        print(f"\n--- File {fi+1}/{len(h1_files)}: {fname} ({nt} time steps) ---")
        
        for t in range(nt):
            if MAX_TIMESTEPS and total_timesteps >= MAX_TIMESTEPS:
                break
            
            t0 = timer.time()
            
            # Extract data
            inputs, targets, valid_mask, n_sentinel = extract_inputs_outputs(ds, t)
            n_valid = inputs.shape[0]
            total_valid += n_valid
            total_sentinel_in_valid += n_sentinel
            
            # Run emulator
            predictions = run_emulator(model, inputs)
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            total_timesteps += 1
            
            elapsed = timer.time() - t0
            date = ds.variables['date'][t]
            datesec = ds.variables['datesec'][t]
            print(f"  t={t:3d} date={date} sec={datesec:5d} | valid={n_valid:6d} sentinel_skipped={n_sentinel:5d} | {elapsed:.1f}s")
        
        ds.close()
    
    # Aggregate and compute metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"  Total time steps processed:    {total_timesteps}")
    print(f"  Total valid points:            {total_valid:,}")
    print(f"  Total sentinel-contaminated:   {total_sentinel_in_valid:,} (excluded)")
    print(f"  Points compared:               {all_predictions.shape[0]:,}")
    
    # Compute and print metrics
    metrics = compute_metrics(all_predictions, all_targets)
    print_metrics(metrics, header="AGGREGATED COMPARISON: Emulator Predictions vs E3SM h1 Raw Outputs")
    
    # Check if outputs are exactly identical (bitwise)
    exact_match = np.allclose(all_predictions, all_targets, atol=0, rtol=0)
    close_match = np.allclose(all_predictions, all_targets, atol=1e-6, rtol=1e-4)
    print(f"\n  Exact bitwise match:      {exact_match}")
    print(f"  Close match (atol=1e-6):  {close_match}")
    
    # Per-output detailed check
    for i, out_name in enumerate(OUTPUT_NAMES):
        pred = all_predictions[:, i]
        targ = all_targets[:, i]
        abs_diff = np.abs(pred - targ)
        pct_within = [
            (1e-10, np.mean(abs_diff < 1e-10) * 100),
            (1e-8,  np.mean(abs_diff < 1e-8) * 100),
            (1e-6,  np.mean(abs_diff < 1e-6) * 100),
            (1e-4,  np.mean(abs_diff < 1e-4) * 100),
            (1e-2,  np.mean(abs_diff < 1e-2) * 100),
        ]
        print(f"\n  {out_name} - % of points within absolute tolerance:")
        for tol, pct in pct_within:
            print(f"    |diff| < {tol:.0e}: {pct:7.3f}%")
    
    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
