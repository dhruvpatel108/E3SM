#!/usr/bin/env python3
"""
verify_emulator_bugs.py
========================
Numerical verification of two suspected bugs in the TorchScript emulator's
postprocessing (export_model_with_preprocessing.py):

  Bug A (Primary): Sign-preserving log inverse broken for |x| < 1
    Current:  sign(y) * (10^|y| - eps)
    Correct:  per-variable formula using known physical sign

  Bug B: Output scaler index mismatch
    Scaler fit order (config output_cols): [qctend_TAU, nctend_TAU, nrtend_TAU, qrtend_TAU]
    Tensor concat order (OUTPUT_TENSOR_ORDER): [qrtend, nctend, nrtend, qctend]
    => tensor[0]=qrtend uses scaler[0]=qctend params (indices 0 & 3 swapped)

Tests:
  0: h1 round-trip (TorchScript as-is vs h1 ground truth)
  1: Confirm scaler index mismatch numerically from training data
  2: Verify DNN outputs are in standardized space
  3: Buggy pipeline baseline on training data
  4: Fix inverse log ONLY (keep original scaler indices)
  5: Fix index swap ONLY (keep buggy inverse log)
  6: Fix BOTH bugs

Usage:
    module load conda && conda activate mlmicrophysics-env
    python verify_emulator_bugs.py
"""

import numpy as np
import torch
import pandas as pd
import netCDF4 as nc
import glob
import os
import sys
import time as timer

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "/pscratch/sd/d/dvpatel/mlmicrophysics_project/trained_emulator_files/"
    "emulator107666_torchscript_qctauin1e-6_cloud1e-2.pt"
)
TRAINING_DATA_DIR = (
    "/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data"
)
H1_DIR = (
    "/global/cfs/cdirs/m4942/e3sm/"
    "e3sm2025_mlmicro_dp3_test4_train_2mo"
)

# ============================================================
# CONSTANTS
# ============================================================

LOG_EPSILON = 1e-10
SENTINEL_VALUE = -99999.0

INPUT_COLS = [
    "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in",
    "PGAM", "LAMC", "LAMR", "N0R",
    "RHO_CLUBB", "CLOUD", "FREQR",
]
LOG_INPUT_INDICES = [0, 1, 2, 3, 5, 6, 7]

OUTPUT_SCALER_MEAN = np.array(
    [8.74489395, -3.68680888, 2.37256747, -8.74489395], dtype=np.float64
)
OUTPUT_SCALER_SCALE = np.array(
    [1.34873917, 1.20987077, 2.6218439, 1.34873917], dtype=np.float64
)

INPUT_SCALER_MEAN = np.array(
    [-4.24440123, -7.42833105, 7.16416675, 0.083601,
     11.21180435, 5.82658226, 3.44377318, 4.00202219,
     0.46366718, 0.65588051, 0.35180502], dtype=np.float64
)
INPUT_SCALER_SCALE = np.array(
    [0.74795709, 1.78301471, 0.78666628, 2.8632883,
     1.29954431, 0.21027716, 2.16248501, 2.80776632,
     0.45069757, 0.38438506, 0.47520882], dtype=np.float64
)

# Tensor order: [qrtend, nctend, nrtend, qctend]  (indices 0,1,2,3)
# Scaler order: [qctend, nctend, nrtend, qrtend]  (indices 0,1,2,3)
# Correct mapping: tensor[i] should use scaler[CORRECTED[i]]
CORRECTED_SCALER_INDICES = [3, 1, 2, 0]

# Physical sign of each output in tensor order
# +1 = always positive, -1 = always negative, 0 = ambiguous (use sign(y) fallback)
PHYSICAL_SIGNS = [+1, -1, 0, -1]  # qrtend>=0, nctend<=0, nrtend=?, qctend<=0

OUTPUT_NAMES = ["qrtend", "nctend", "nrtend", "qctend"]

N_TRAINING_SAMPLES = 10000
MAX_H1_SAMPLES_PER_TIMESTEP = 3000
MAX_H1_FILES = 5
MAX_H1_TIMESTEPS_PER_FILE = 10


# ============================================================
# TRANSFORM HELPERS
# ============================================================

def forward_log(x, eps=LOG_EPSILON):
    """sign(x) * log10(|x| + eps)"""
    return np.sign(x) * np.log10(np.abs(x) + eps)


def buggy_inverse_log(y, eps=LOG_EPSILON):
    """sign(y) * (10^|y| - eps)  -- the current TorchScript formula."""
    return np.sign(y) * (10.0 ** np.abs(y) - eps)


def correct_inverse_log(y, physical_signs, eps=LOG_EPSILON):
    """
    Per-variable correct inverse of  f(x) = sign(x) * log10(|x| + eps).

    For x > 0:  f(x) = log10(x + eps)         =>  x = 10^y - eps
    For x < 0:  f(x) = -log10(-x + eps)        =>  x = -(10^{-y} - eps)
    Ambiguous:  fall back to sign(y)*(10^|y|-eps)  [valid when |x| >> 1]
    """
    result = np.empty_like(y)
    for i, s in enumerate(physical_signs):
        col = y[:, i] if y.ndim == 2 else y[i]
        if s > 0:
            out = 10.0 ** col - eps
        elif s < 0:
            out = -(10.0 ** (-col) - eps)
        else:
            out = np.sign(col) * (10.0 ** np.abs(col) - eps)
        if y.ndim == 2:
            result[:, i] = out
        else:
            result[i] = out
    return result


def reverse_buggy_postprocess(y_phys, mean, scale, eps=LOG_EPSILON):
    """Invert the TorchScript postprocessing to recover standardised DNN output."""
    y_log = forward_log(y_phys, eps)
    y_std = (y_log - mean) / scale
    return y_std


# ============================================================
# MODEL RUNNER
# ============================================================

def run_model_batch(model, inputs):
    """Run TorchScript model on (N, 11) array, returning (N, 4)."""
    n = inputs.shape[0]
    preds = np.zeros((n, 4), dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            preds[i] = model(torch.tensor(inputs[i], dtype=torch.float32)).numpy()
            if (i + 1) % 2000 == 0:
                print(f"      ... {i+1}/{n} samples", flush=True)
    return preds


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred, targ, names):
    results = {}
    for i, nm in enumerate(names):
        p, t = pred[:, i], targ[:, i]
        d = p - t
        ad = np.abs(d)
        tvar = np.var(t)
        r2 = 1.0 - np.mean(d ** 2) / max(tvar, 1e-30)
        if np.std(t) > 1e-30 and np.std(p) > 1e-30:
            corr = float(np.corrcoef(p, t)[0, 1])
        else:
            corr = 0.0
        results[nm] = dict(
            n=len(p),
            mae=float(np.mean(ad)),
            rmse=float(np.sqrt(np.mean(d ** 2))),
            max_abs=float(np.max(ad)),
            r2=float(r2),
            corr=corr,
            pmean=float(np.mean(p)),
            tmean=float(np.mean(t)),
            pstd=float(np.std(p)),
            tstd=float(np.std(t)),
        )
    return results


def print_table(results, header):
    w = 105
    print(f"\n{'=' * w}")
    print(f"  {header}")
    print(f"{'=' * w}")
    fmt_h = (
        f"{'Var':>10s} | {'N':>7s} | {'MAE':>12s} | {'RMSE':>12s} "
        f"| {'MaxAbsDiff':>12s} | {'R^2':>12s} | {'Corr':>10s}"
    )
    print(fmt_h)
    print("-" * w)
    for nm in OUTPUT_NAMES:
        m = results[nm]
        print(
            f"{nm:>10s} | {m['n']:7d} | {m['mae']:12.4e} | {m['rmse']:12.4e} "
            f"| {m['max_abs']:12.4e} | {m['r2']:12.6f} | {m['corr']:10.6f}"
        )
    print()
    for nm in OUTPUT_NAMES:
        m = results[nm]
        print(
            f"  {nm}: pred_mean={m['pmean']:.4e}  targ_mean={m['tmean']:.4e}  "
            f"pred_std={m['pstd']:.4e}  targ_std={m['tstd']:.4e}"
        )


def print_examples(pred, targ, names, n=5):
    n = min(n, pred.shape[0])
    idx = np.linspace(0, pred.shape[0] - 1, n, dtype=int)
    print(f"\n  Example comparisons ({n} evenly-spaced samples):")
    for ci, nm in enumerate(names):
        print(f"\n  {nm}:")
        print(f"    {'Idx':>7s} | {'Predicted':>14s} | {'Target':>14s} | {'AbsDiff':>14s}")
        for j in idx:
            p, t = pred[j, ci], targ[j, ci]
            print(f"    {j:7d} | {p:14.6e} | {t:14.6e} | {abs(p - t):14.6e}")


# ============================================================
# DATA LOADERS
# ============================================================

def load_training_data():
    """Return (df, inputs[N,11], targets_physical[N,4]) from parquet files."""
    print(f"\n  Loading training data from {TRAINING_DATA_DIR} ...")
    files = sorted(glob.glob(os.path.join(TRAINING_DATA_DIR, "*.parquet")))
    print(f"    {len(files)} parquet files found")

    chunks = []
    total = 0
    for fp in files:
        df = pd.read_parquet(fp)
        df = df[(df["QC_TAU_in"] > 1e-6) & (df["CLOUD"] > 0.01)]
        for col in ("PGAM", "LAMC", "LAMR", "N0R"):
            df = df[df[col] != SENTINEL_VALUE]
        df = df[(df["qrtend_TAU"].abs() > 0) | (df["nctend_TAU"].abs() > 0)]
        chunks.append(df)
        total += len(df)
        if total >= N_TRAINING_SAMPLES * 5:
            break

    df = pd.concat(chunks, ignore_index=True)
    if len(df) > N_TRAINING_SAMPLES:
        df = df.sample(n=N_TRAINING_SAMPLES, random_state=42)
    print(f"    Using {len(df)} filtered samples")

    inputs = df[INPUT_COLS].values.astype(np.float32)
    targets = df[
        ["qrtend_TAU", "nctend_TAU", "nrtend_TAU", "qctend_TAU"]
    ].values.astype(np.float32)
    return df, inputs, targets


def load_h1_samples():
    """Return (inputs[N,11], targets[N,4]) from h1 NetCDF files."""
    h1_files = sorted(glob.glob(os.path.join(H1_DIR, "*.h1.*.nc")))
    print(f"\n  Found {len(h1_files)} h1 files; using first {MAX_H1_FILES}")

    all_inp, all_tgt = [], []
    for fpath in h1_files[:MAX_H1_FILES]:
        fname = os.path.basename(fpath)
        print(f"    {fname} ... ", end="", flush=True)
        ds = nc.Dataset(fpath)
        nt = len(ds.dimensions["time"])
        nlev = len(ds.dimensions["lev"])

        for t in range(min(nt, MAX_H1_TIMESTEPS_PER_FILE)):
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
                (mu_c == SENTINEL_VALUE)
                | (lamc == SENTINEL_VALUE)
                | (lamr == SENTINEL_VALUE)
                | (p3_nr == SENTINEL_VALUE)
            )
            has_tend = (np.abs(qrt) > 0) | (np.abs(nct) > 0) | (np.abs(nrt) > 0)
            valid = (qc > 1e-6) & (cloud > 0.01) & ~sentinel #& has_tend

            li, ci = np.where(valid)
            nv = len(li)
            if nv == 0:
                continue
            if nv > MAX_H1_SAMPLES_PER_TIMESTEP:
                sel = np.random.choice(nv, MAX_H1_SAMPLES_PER_TIMESTEP, replace=False)
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

        ds.close()
        print(f"{sum(a.shape[0] for a in all_inp)} samples so far")

    inputs = np.concatenate(all_inp, axis=0)
    targets = np.concatenate(all_tgt, axis=0)
    print(f"    Total h1 samples: {inputs.shape[0]}")
    return inputs, targets


# ============================================================
# TESTS
# ============================================================

def test_0(model):
    """h1 round-trip: deployed TorchScript model vs h1 ground truth."""
    print("\n" + "#" * 105)
    print("# TEST 0: h1 ROUND-TRIP  (TorchScript model as-is  vs  h1 ground truth)")
    print("#" * 105)

    h1_inp, h1_tgt = load_h1_samples()
    print(f"\n  Running TorchScript model on {h1_inp.shape[0]} h1 samples ...")
    t0 = timer.time()
    preds = run_model_batch(model, h1_inp)
    print(f"    Done in {timer.time() - t0:.1f}s")

    nz = np.any(preds != 0, axis=1)
    print(f"    Non-zero predictions: {nz.sum()} / {preds.shape[0]}")
    preds, tgt = preds[nz], h1_tgt[nz]

    m = compute_metrics(preds, tgt, OUTPUT_NAMES)
    print_table(m, "TEST 0  —  h1 round-trip (deployed model vs h1 outputs)")
    print_examples(preds, tgt, OUTPUT_NAMES)
    return m


def test_1(training_df):
    """Confirm scaler index mismatch from training data statistics."""
    print("\n" + "#" * 105)
    print("# TEST 1: SCALER INDEX MISMATCH CONFIRMATION")
    print("#" * 105)

    scaler_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
    tensor_names = ["qrtend", "nctend", "nrtend", "qctend"]

    print(f"\n  Computing log-transformed output statistics ({len(training_df)} samples)")
    print(f"\n  {'Scaler[i]':>12s} {'Config col':>14s} | {'DataLogMean':>12s} {'ScalerMean':>12s} "
          f"{'MeanDiff':>10s} | {'DataLogStd':>11s} {'ScalerScale':>12s} {'StdDiff':>10s}")
    print("  " + "-" * 100)

    for j, col in enumerate(scaler_cols):ƒ
        vals = training_df[col].values.astype(np.float64)
        lv = np.sign(vals) * np.log10(np.abs(vals) + LOG_EPSILON)
        dm, ds_ = np.mean(lv), np.std(lv)
        sm, ss = OUTPUT_SCALER_MEAN[j], OUTPUT_SCALER_SCALE[j]
        print(
            f"  scaler[{j}]    {col:>14s} | {dm:+12.5f} {sm:+12.5f} {abs(dm - sm):10.5f} "
            f"| {ds_:11.5f} {ss:12.5f} {abs(ds_ - ss):10.5f}"
        )

    print(f"\n  Tensor-to-scaler alignment check:")
    for i, (tn, sn) in enumerate(zip(tensor_names, scaler_cols)):
        expected = tn + "_TAU"
        ok = "OK" if expected == sn else "MISMATCH"
        print(f"    tensor[{i}] = {tn:>10s}  <->  scaler[{i}] = {sn:>14s}  =>  {ok}")

    print("\n  CONCLUSION: indices 0 (qrtend<->qctend_TAU) and 3 (qctend<->qrtend_TAU) are swapped.")


def test_2(y_std_all):
    """Check whether raw DNN output lives in standardised (mean~0, std~1) space."""
    print("\n" + "#" * 105)
    print("# TEST 2: DNN OUTPUT STANDARDISATION CHECK")
    print("#" * 105)

    print(f"\n  Samples: {y_std_all.shape[0]}")
    print(f"\n  Using ORIGINAL (buggy) scaler indices to reverse postprocessing:")
    print(f"  {'Var':>10s} | {'Mean':>10s} | {'Std':>10s} | {'Expected':>20s}")
    print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 20}")
    for i, nm in enumerate(OUTPUT_NAMES):
        print(f"  {nm:>10s} | {np.mean(y_std_all[:, i]):+10.4f} | "
              f"{np.std(y_std_all[:, i]):10.4f} | mean~0, std~1")

    print(f"\n  Now using CORRECTED scaler indices:")
    cm = OUTPUT_SCALER_MEAN[CORRECTED_SCALER_INDICES]
    cs = OUTPUT_SCALER_SCALE[CORRECTED_SCALER_INDICES]
    # re-standardise from the buggy de-standardised log values
    # buggy: y_log_buggy = y_std_buggy * orig_scale + orig_mean
    # y_log_buggy is what it is; now re-standardise with corrected params
    y_log = y_std_all * OUTPUT_SCALER_SCALE + OUTPUT_SCALER_MEAN  # de-standardise buggy
    y_std_corr = (y_log - cm) / cs
    print(f"  {'Var':>10s} | {'Mean':>10s} | {'Std':>10s} | {'Expected':>20s}")
    print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 20}")
    for i, nm in enumerate(OUTPUT_NAMES):
        print(f"  {nm:>10s} | {np.mean(y_std_corr[:, i]):+10.4f} | "
              f"{np.std(y_std_corr[:, i]):10.4f} | mean~0, std~1")

    print("\n  Whichever set of indices gives values closer to (0, 1) "
          "is the correct alignment.")


def run_tests_3_to_6(preds_buggy, targets_phys):
    """Tests 3-6 on training data, sharing the single model run."""
    nz = np.any(preds_buggy != 0, axis=1) & np.any(targets_phys != 0, axis=1)
    pb = preds_buggy[nz].astype(np.float64)
    tgt = targets_phys[nz].astype(np.float64)
    print(f"\n  Non-zero sample pairs: {pb.shape[0]}")

    # Recover standardised DNN output via reversing buggy postprocess
    y_std = reverse_buggy_postprocess(pb, OUTPUT_SCALER_MEAN, OUTPUT_SCALER_SCALE)

    # Corrected scaler params
    cm = OUTPUT_SCALER_MEAN[CORRECTED_SCALER_INDICES]
    cs = OUTPUT_SCALER_SCALE[CORRECTED_SCALER_INDICES]

    # ---------- Test 3: buggy baseline ----------
    print("\n" + "#" * 105)
    print("# TEST 3: BUGGY PIPELINE BASELINE (training data)")
    print("#" * 105)
    m3 = compute_metrics(pb, tgt, OUTPUT_NAMES)
    print_table(m3, "TEST 3  —  Buggy pipeline (as deployed)")
    print_examples(pb, tgt, OUTPUT_NAMES)

    # ---------- Test 4: fix inverse log only ----------
    print("\n" + "#" * 105)
    print("# TEST 4: FIX INVERSE LOG ONLY (keep original scaler indices)")
    print("#" * 105)
    y_log4 = y_std * OUTPUT_SCALER_SCALE + OUTPUT_SCALER_MEAN
    p4 = correct_inverse_log(y_log4, PHYSICAL_SIGNS)
    m4 = compute_metrics(p4, tgt, OUTPUT_NAMES)
    print_table(m4, "TEST 4  —  Fix inverse log only")
    print_examples(p4, tgt, OUTPUT_NAMES)

    # ---------- Test 5: fix index swap only ----------
    print("\n" + "#" * 105)
    print("# TEST 5: FIX INDEX SWAP ONLY (keep buggy inverse log)")
    print("#" * 105)
    y_log5 = y_std * cs + cm
    p5 = buggy_inverse_log(y_log5)
    m5 = compute_metrics(p5, tgt, OUTPUT_NAMES)
    print_table(m5, "TEST 5  —  Fix index swap only")
    print_examples(p5, tgt, OUTPUT_NAMES)

    # ---------- Test 6: fix both ----------
    print("\n" + "#" * 105)
    print("# TEST 6: FIX BOTH BUGS")
    print("#" * 105)
    y_log6 = y_std * cs + cm
    p6 = correct_inverse_log(y_log6, PHYSICAL_SIGNS)
    m6 = compute_metrics(p6, tgt, OUTPUT_NAMES)
    print_table(m6, "TEST 6  —  Fix both (corrected indices + corrected inverse log)")
    print_examples(p6, tgt, OUTPUT_NAMES)

    return m3, m4, m5, m6


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)
    W = 105

    print("=" * W)
    print("  EMULATOR BUG VERIFICATION")
    print("=" * W)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Train:    {TRAINING_DATA_DIR}")
    print(f"  h1 data:  {H1_DIR}")
    print()

    # ------ load model ------
    print("Loading TorchScript model ...")
    t0 = timer.time()
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print(f"  Loaded in {timer.time() - t0:.1f}s\n")

    # ------ TEST 0: h1 round-trip ------
    m0 = test_0(model)

    # ------ load training data ------
    training_df, train_inp, train_tgt = load_training_data()

    # ------ TEST 1 ------
    test_1(training_df)

    # ------ run model once on training data (shared for tests 2-6) ------
    print(f"\n  Running TorchScript model on {train_inp.shape[0]} training samples ...")
    t0 = timer.time()
    train_preds_buggy = run_model_batch(model, train_inp)
    print(f"    Done in {timer.time() - t0:.1f}s")

    nz = np.any(train_preds_buggy != 0, axis=1)
    print(f"    Non-zero predictions: {nz.sum()} / {train_preds_buggy.shape[0]}")

    # ------ TEST 2 ------
    y_std_nz = reverse_buggy_postprocess(
        train_preds_buggy[nz].astype(np.float64),
        OUTPUT_SCALER_MEAN,
        OUTPUT_SCALER_SCALE,
    )
    test_2(y_std_nz)

    # ------ TESTS 3-6 ------
    m3, m4, m5, m6 = run_tests_3_to_6(train_preds_buggy, train_tgt)

    # ------ SUMMARY ------
    print("\n" + "=" * W)
    print("  SUMMARY: MAE COMPARISON")
    print("=" * W)
    rows = [
        ("3: Buggy baseline", m3),
        ("4: Fix inv-log only", m4),
        ("5: Fix idx-swap only", m5),
        ("6: Fix both", m6),
    ]
    print(f"\n  {'Test':>25s} | {'qrtend':>12s} | {'nctend':>12s} | {'nrtend':>12s} | {'qctend':>12s}")
    print(f"  {'-' * 25}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
    for label, m in rows:
        print(
            f"  {label:>25s} | {m['qrtend']['mae']:12.4e} | {m['nctend']['mae']:12.4e} | "
            f"{m['nrtend']['mae']:12.4e} | {m['qctend']['mae']:12.4e}"
        )

    print(f"\n  {'Test':>25s} | {'qrtend':>12s} | {'nctend':>12s} | {'nrtend':>12s} | {'qctend':>12s}")
    print(f"  {'-' * 25}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
    for label, m in rows:
        print(
            f"  {label:>25s} | {m['qrtend']['corr']:12.6f} | {m['nctend']['corr']:12.6f} | "
            f"{m['nrtend']['corr']:12.6f} | {m['qctend']['corr']:12.6f}"
        )

    print(f"\n  {'Test':>25s} | {'qrtend':>12s} | {'nctend':>12s} | {'nrtend':>12s} | {'qctend':>12s}")
    print(f"  {'-' * 25}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
    for label, m in rows:
        print(
            f"  {label:>25s} | {m['qrtend']['r2']:12.6f} | {m['nctend']['r2']:12.6f} | "
            f"{m['nrtend']['r2']:12.6f} | {m['qctend']['r2']:12.6f}"
        )

    print("\n" + "=" * W)
    print("  DONE")
    print("=" * W)


if __name__ == "__main__":
    main()
