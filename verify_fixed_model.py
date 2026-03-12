#!/usr/bin/env python3
"""
verify_fixed_model.py
======================
Side-by-side verification of the OLD (buggy) and NEW (fixed) TorchScript
emulator against h1 ground-truth data from Andrew's E3SM simulation.

Tests:
  A: h1 round-trip — old vs new model, full metrics including %error
  B: Spot-check — verify the new model embeds the correct postprocessing
  C: nrtend breakdown — per-sign / per-magnitude-regime analysis

Usage:
    module load conda && conda activate mlmicrophysics-env
    python verify_fixed_model.py
"""

import numpy as np
import torch
import netCDF4 as nc
import glob
import os
import time as timer

# ============================================================
# PATHS
# ============================================================

OLD_MODEL_PATH = (
    "/pscratch/sd/d/dvpatel/mlmicrophysics_project/trained_emulator_files/"
    "emulator107666_torchscript_qctauin1e-6_cloud1e-2.pt"
)
NEW_MODEL_PATH = (
    "/pscratch/sd/d/dvpatel/mlmicrophysics_project/trained_emulator_files/"
    "emulator_standalone_v2_fixed_inverse.pt"
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

OUTPUT_NAMES = ["qrtend", "nctend", "nrtend", "qctend"]

# Scaler params baked into the OLD model (scaler fit order: qctend, nctend, nrtend, qrtend)
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

LOG_INPUT_INDICES = [0, 1, 2, 3, 5, 6, 7]
CORRECTED_SCALER_INDICES = [3, 1, 2, 0]
PHYSICAL_SIGNS = [+1, -1, 0, -1]  # qrtend>=0, nctend<=0, nrtend=?, qctend<=0

MAX_H1_FILES = 3
MAX_H1_TIMESTEPS_PER_FILE = 8
MAX_H1_SAMPLES_PER_TIMESTEP = 5000


# ============================================================
# H1 DATA LOADER
# ============================================================

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
                (mu_c == SENTINEL_VALUE) | (lamc == SENTINEL_VALUE)
                | (lamr == SENTINEL_VALUE) | (p3_nr == SENTINEL_VALUE)
            )
            has_tend = (np.abs(qrt) > 0) | (np.abs(nct) > 0) | (np.abs(nrt) > 0)
            valid = (qc > 1e-6) & (cloud > 0.01) & ~sentinel & has_tend

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
# MODEL RUNNER
# ============================================================

def run_model_batch(model, inputs, label=""):
    """Run TorchScript model on (N, 11) array, returning (N, 4)."""
    n = inputs.shape[0]
    preds = np.zeros((n, 4), dtype=np.float32)
    t0 = timer.time()
    with torch.no_grad():
        for i in range(n):
            preds[i] = model(torch.tensor(inputs[i], dtype=torch.float32)).numpy()
            if (i + 1) % 5000 == 0:
                print(f"      {label} ... {i+1}/{n}", flush=True)
    elapsed = timer.time() - t0
    print(f"      {label} done: {n} samples in {elapsed:.1f}s")
    return preds


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred, targ, names):
    results = {}
    for i, nm in enumerate(names):
        p, t = pred[:, i].astype(np.float64), targ[:, i].astype(np.float64)
        d = p - t
        ad = np.abs(d)
        tvar = np.var(t)
        r2 = 1.0 - np.mean(d ** 2) / max(tvar, 1e-30)
        if np.std(t) > 1e-30 and np.std(p) > 1e-30:
            corr = float(np.corrcoef(p, t)[0, 1])
        else:
            corr = 0.0

        nz = np.abs(t) > 1e-30
        if nz.sum() > 0:
            rel_err = ad[nz] / np.abs(t[nz])
            med_pct = float(np.median(rel_err) * 100)
            mean_pct = float(np.mean(rel_err) * 100)
            within_10 = float(np.mean(rel_err < 0.10) * 100)
            within_50 = float(np.mean(rel_err < 0.50) * 100)
        else:
            med_pct = mean_pct = within_10 = within_50 = float("nan")

        sign_match = float(np.mean(np.sign(p) == np.sign(t)) * 100) if len(p) > 0 else 0.0

        results[nm] = dict(
            n=len(p),
            mae=float(np.mean(ad)),
            rmse=float(np.sqrt(np.mean(d ** 2))),
            max_abs=float(np.max(ad)),
            r2=float(r2),
            corr=corr,
            med_pct=med_pct,
            mean_pct=mean_pct,
            within_10=within_10,
            within_50=within_50,
            sign_acc=sign_match,
            pmean=float(np.mean(p)),
            tmean=float(np.mean(t)),
            pstd=float(np.std(p)),
            tstd=float(np.std(t)),
        )
    return results


def print_table(results, header):
    w = 140
    print(f"\n{'=' * w}")
    print(f"  {header}")
    print(f"{'=' * w}")
    print(
        f"{'Var':>10s} | {'N':>7s} | {'MAE':>12s} | {'RMSE':>12s} "
        f"| {'Med%Err':>9s} | {'Mean%Err':>10s} | {'<10%':>6s} | {'<50%':>6s} "
        f"| {'SignAcc%':>8s} | {'R^2':>12s} | {'Corr':>10s}"
    )
    print("-" * w)
    for nm in OUTPUT_NAMES:
        m = results[nm]
        print(
            f"{nm:>10s} | {m['n']:7d} | {m['mae']:12.4e} | {m['rmse']:12.4e} "
            f"| {m['med_pct']:9.2f} | {m['mean_pct']:10.2f} | {m['within_10']:6.1f} | {m['within_50']:6.1f} "
            f"| {m['sign_acc']:8.1f} | {m['r2']:12.6f} | {m['corr']:10.6f}"
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
        print(f"    {'Idx':>7s} | {'Predicted':>14s} | {'Target':>14s} | {'AbsDiff':>14s} | {'%Error':>12s}")
        for j in idx:
            p, t = float(pred[j, ci]), float(targ[j, ci])
            ad = abs(p - t)
            pct = (ad / abs(t) * 100) if abs(t) > 1e-30 else float("nan")
            print(f"    {j:7d} | {p:14.6e} | {t:14.6e} | {ad:14.6e} | {pct:11.2f}%")


# ============================================================
# TEST A: SIDE-BY-SIDE H1 ROUND-TRIP
# ============================================================

def test_a(old_model, new_model, h1_inp, h1_tgt):
    print("\n" + "#" * 140)
    print("# TEST A: SIDE-BY-SIDE h1 ROUND-TRIP  (old buggy model  vs  new fixed model  vs  h1 ground truth)")
    print("#" * 140)

    print(f"\n  Running OLD model on {h1_inp.shape[0]} h1 samples ...")
    old_preds = run_model_batch(old_model, h1_inp, label="OLD")

    print(f"  Running NEW model on {h1_inp.shape[0]} h1 samples ...")
    new_preds = run_model_batch(new_model, h1_inp, label="NEW")

    old_nz = np.any(old_preds != 0, axis=1)
    new_nz = np.any(new_preds != 0, axis=1)
    print(f"\n  OLD non-zero: {old_nz.sum()} / {h1_inp.shape[0]}")
    print(f"  NEW non-zero: {new_nz.sum()} / {h1_inp.shape[0]}")

    both_nz = old_nz & new_nz
    op = old_preds[both_nz]
    np_ = new_preds[both_nz]
    tgt = h1_tgt[both_nz]
    print(f"  Both non-zero: {both_nz.sum()}")

    m_old = compute_metrics(op, tgt, OUTPUT_NAMES)
    m_new = compute_metrics(np_, tgt, OUTPUT_NAMES)

    print_table(m_old, "TEST A — OLD (buggy) model vs h1 ground truth")
    print_examples(op, tgt, OUTPUT_NAMES)

    print_table(m_new, "TEST A — NEW (fixed) model vs h1 ground truth")
    print_examples(np_, tgt, OUTPUT_NAMES)

    # Side-by-side summary
    w = 140
    print(f"\n{'=' * w}")
    print("  SIDE-BY-SIDE SUMMARY (OLD vs NEW)")
    print(f"{'=' * w}")
    print(
        f"  {'Var':>10s} | {'OLD MAE':>12s} {'NEW MAE':>12s} {'Ratio':>8s} "
        f"| {'OLD Med%':>9s} {'NEW Med%':>9s} "
        f"| {'OLD Corr':>10s} {'NEW Corr':>10s} "
        f"| {'OLD R^2':>12s} {'NEW R^2':>12s} "
        f"| {'OLD Sign%':>9s} {'NEW Sign%':>9s}"
    )
    print("  " + "-" * 136)
    for nm in OUTPUT_NAMES:
        mo, mn = m_old[nm], m_new[nm]
        ratio = mo["mae"] / mn["mae"] if mn["mae"] > 1e-30 else float("inf")
        print(
            f"  {nm:>10s} | {mo['mae']:12.4e} {mn['mae']:12.4e} {ratio:8.1f}x "
            f"| {mo['med_pct']:9.2f} {mn['med_pct']:9.2f} "
            f"| {mo['corr']:10.6f} {mn['corr']:10.6f} "
            f"| {mo['r2']:12.6f} {mn['r2']:12.6f} "
            f"| {mo['sign_acc']:9.1f} {mn['sign_acc']:9.1f}"
        )

    return m_old, m_new, np_, tgt


# ============================================================
# TEST B: SPOT-CHECK POSTPROCESSING
# ============================================================

def forward_log(x, eps=LOG_EPSILON):
    return np.sign(x) * np.log10(np.abs(x) + eps)


def correct_inverse_log_vec(y_log, physical_signs, eps=LOG_EPSILON):
    result = np.empty_like(y_log)
    for i, s in enumerate(physical_signs):
        col = y_log[:, i]
        if s > 0:
            result[:, i] = 10.0 ** col - eps
        elif s < 0:
            result[:, i] = -(10.0 ** (-col) - eps)
        else:
            result[:, i] = np.sign(col) * (10.0 ** np.abs(col) - eps)
    return result


def test_b(old_model, new_model, h1_inp):
    """Spot-check: manually replicate corrected postprocessing and compare to new model."""
    print("\n" + "#" * 140)
    print("# TEST B: SPOT-CHECK — verify new model embeds the correct postprocessing")
    print("#" * 140)

    n = min(200, h1_inp.shape[0])
    subset = h1_inp[:n]

    with torch.no_grad():
        old_out = np.array([old_model(torch.tensor(subset[i], dtype=torch.float32)).numpy() for i in range(n)])
        new_out = np.array([new_model(torch.tensor(subset[i], dtype=torch.float32)).numpy() for i in range(n)])

    # Manually replicate the CORRECT pipeline using the OLD model's buggy output
    old_out_f64 = old_out.astype(np.float64)
    y_log_buggy = forward_log(old_out_f64)
    y_std = (y_log_buggy - OUTPUT_SCALER_MEAN) / OUTPUT_SCALER_SCALE

    corrected_mean = OUTPUT_SCALER_MEAN[CORRECTED_SCALER_INDICES]
    corrected_scale = OUTPUT_SCALER_SCALE[CORRECTED_SCALER_INDICES]
    y_log_fixed = y_std * corrected_scale + corrected_mean
    manual_fixed = correct_inverse_log_vec(y_log_fixed, PHYSICAL_SIGNS).astype(np.float32)

    new_out_f = new_out.astype(np.float64)
    manual_f = manual_fixed.astype(np.float64)

    print(f"\n  Comparing new model output vs manually-corrected pipeline on {n} samples:")
    for i, nm in enumerate(OUTPUT_NAMES):
        diff = np.abs(new_out_f[:, i] - manual_f[:, i])
        max_d = np.max(diff)
        med_d = np.median(diff)
        # relative to magnitude of prediction
        rel = diff / (np.abs(manual_f[:, i]) + 1e-30)
        max_rel = np.max(rel)
        print(f"    {nm:>10s}: max|diff|={max_d:.4e}  median|diff|={med_d:.4e}  max_rel_diff={max_rel:.4e}")

    overall_max = np.max(np.abs(new_out_f - manual_f))
    if overall_max < 1e-2:
        print(f"\n  PASS: New model output matches manually-corrected pipeline (max diff = {overall_max:.4e})")
    else:
        print(f"\n  WARNING: Max diff = {overall_max:.4e} — investigate whether the export applied fixes correctly")
        print("  (Some discrepancy is expected from float32 vs float64 precision in the log/exp round-trip)")

    # Also verify the new model is DIFFERENT from the old model
    diff_old_new = np.max(np.abs(old_out.astype(np.float64) - new_out.astype(np.float64)))
    print(f"\n  Sanity check: max|old_model - new_model| = {diff_old_new:.4e}")
    if diff_old_new < 1e-6:
        print("  FAIL: Old and new models produce identical output — the file may not be updated!")
    else:
        print("  PASS: Old and new models produce different output (fixes are active)")


# ============================================================
# TEST C: NRTEND BREAKDOWN
# ============================================================

def test_c(new_preds, h1_tgt):
    """Breakdown of nrtend by sign and magnitude regime."""
    print("\n" + "#" * 140)
    print("# TEST C: NRTEND BREAKDOWN — by sign and magnitude regime")
    print("#" * 140)

    nrt_idx = 2
    pred = new_preds[:, nrt_idx].astype(np.float64)
    targ = h1_tgt[:, nrt_idx].astype(np.float64)

    categories = [
        ("ALL",                  np.ones(len(targ), dtype=bool)),
        ("targ > 0, |targ| < 1", (targ > 0) & (np.abs(targ) < 1)),
        ("targ > 0, |targ| >= 1", (targ > 0) & (np.abs(targ) >= 1)),
        ("targ < 0, |targ| < 1", (targ < 0) & (np.abs(targ) < 1)),
        ("targ < 0, |targ| >= 1", (targ < 0) & (np.abs(targ) >= 1)),
        ("targ == 0",            targ == 0),
    ]

    w = 140
    print(f"\n{'=' * w}")
    print("  nrtend: NEW model predictions vs h1 ground truth, by category")
    print(f"{'=' * w}")
    print(
        f"  {'Category':>25s} | {'N':>8s} {'%Total':>7s} "
        f"| {'MAE':>12s} | {'Med%Err':>9s} | {'Sign%':>7s} "
        f"| {'<10%':>6s} | {'<50%':>6s} | {'Corr':>10s}"
    )
    print("  " + "-" * 110)

    for label, mask in categories:
        n = mask.sum()
        if n == 0:
            print(f"  {label:>25s} | {0:8d} {0:7.2f}% | {'n/a':>12s} | {'n/a':>9s} | {'n/a':>7s} | {'n/a':>6s} | {'n/a':>6s} | {'n/a':>10s}")
            continue

        p, t = pred[mask], targ[mask]
        d = p - t
        ad = np.abs(d)
        mae = float(np.mean(ad))

        nz = np.abs(t) > 1e-30
        if nz.sum() > 0:
            rel = ad[nz] / np.abs(t[nz])
            med_pct = float(np.median(rel) * 100)
            w10 = float(np.mean(rel < 0.10) * 100)
            w50 = float(np.mean(rel < 0.50) * 100)
        else:
            med_pct = w10 = w50 = float("nan")

        sign_ok = float(np.mean(np.sign(p) == np.sign(t)) * 100)

        if np.std(t) > 1e-30 and np.std(p) > 1e-30:
            corr = float(np.corrcoef(p, t)[0, 1])
        else:
            corr = float("nan")

        pct_total = n / len(targ) * 100
        print(
            f"  {label:>25s} | {n:8d} {pct_total:7.2f}% "
            f"| {mae:12.4e} | {med_pct:9.2f} | {sign_ok:7.1f} "
            f"| {w10:6.1f} | {w50:6.1f} | {corr:10.6f}"
        )

    # Show some example failures from the problematic quadrant
    mask_problem = (targ < 0) & (np.abs(targ) < 1) & (np.abs(targ) > 1e-30)
    if mask_problem.sum() > 0:
        idx_problem = np.where(mask_problem)[0]
        n_show = min(5, len(idx_problem))
        chosen = idx_problem[np.linspace(0, len(idx_problem)-1, n_show, dtype=int)]
        print(f"\n  Example failures (targ < 0, |targ| < 1):")
        print(f"    {'Idx':>7s} | {'Predicted':>14s} | {'Target':>14s} | {'AbsDiff':>14s} | {'%Error':>12s}")
        for j in chosen:
            p, t = float(pred[j]), float(targ[j])
            ad = abs(p - t)
            pct = ad / abs(t) * 100 if abs(t) > 1e-30 else float("nan")
            print(f"    {j:7d} | {p:14.6e} | {t:14.6e} | {ad:14.6e} | {pct:11.2f}%")

    mask_good = (targ < 0) & (np.abs(targ) >= 1)
    if mask_good.sum() > 0:
        idx_good = np.where(mask_good)[0]
        n_show = min(5, len(idx_good))
        chosen = idx_good[np.linspace(0, len(idx_good)-1, n_show, dtype=int)]
        print(f"\n  Example successes (targ < 0, |targ| >= 1):")
        print(f"    {'Idx':>7s} | {'Predicted':>14s} | {'Target':>14s} | {'AbsDiff':>14s} | {'%Error':>12s}")
        for j in chosen:
            p, t = float(pred[j]), float(targ[j])
            ad = abs(p - t)
            pct = ad / abs(t) * 100 if abs(t) > 1e-30 else float("nan")
            print(f"    {j:7d} | {p:14.6e} | {t:14.6e} | {ad:14.6e} | {pct:11.2f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)
    W = 140

    print("=" * W)
    print("  FIXED MODEL VERIFICATION")
    print("=" * W)
    print(f"  OLD model: {OLD_MODEL_PATH}")
    print(f"  NEW model: {NEW_MODEL_PATH}")
    print(f"  h1 data:   {H1_DIR}")
    print()

    print("Loading OLD TorchScript model ...")
    old_model = torch.jit.load(OLD_MODEL_PATH, map_location="cpu")
    old_model.eval()

    print("Loading NEW TorchScript model ...")
    new_model = torch.jit.load(NEW_MODEL_PATH, map_location="cpu")
    new_model.eval()

    print("\nLoading h1 data ...")
    h1_inp, h1_tgt = load_h1_samples()

    # Test A
    m_old, m_new, new_preds_nz, tgt_nz = test_a(old_model, new_model, h1_inp, h1_tgt)

    # Test B
    test_b(old_model, new_model, h1_inp)

    # Test C
    test_c(new_preds_nz, tgt_nz)

    # Final verdict
    print(f"\n{'=' * W}")
    print("  FINAL VERDICT")
    print(f"{'=' * W}")
    for nm in OUTPUT_NAMES:
        mo, mn = m_old[nm], m_new[nm]
        status = "FIXED" if mn["med_pct"] < mo["med_pct"] * 0.5 else (
            "IMPROVED" if mn["med_pct"] < mo["med_pct"] else "UNCHANGED/WORSE"
        )
        if nm == "nrtend":
            status += " (known limitation: non-invertible transform for ambiguous-sign variable)"
        print(f"  {nm:>10s}: {status}")
        print(f"    OLD: MAE={mo['mae']:.4e}  Med%Err={mo['med_pct']:.2f}%  Corr={mo['corr']:.6f}  R^2={mo['r2']:.6f}")
        print(f"    NEW: MAE={mn['mae']:.4e}  Med%Err={mn['med_pct']:.2f}%  Corr={mn['corr']:.6f}  R^2={mn['r2']:.6f}")

    print(f"\n{'=' * W}")
    print("  DONE")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
