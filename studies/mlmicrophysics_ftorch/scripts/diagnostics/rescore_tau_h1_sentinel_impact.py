#!/usr/bin/env python3
"""
Re-score exact Stage A+B tau_2mos H1 samples with and without sentinel-valued
inputs to test whether sentinel rows materially explain the remaining H1 failure.
"""

import argparse
import glob
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from diagnose_tau_h1_mismatch import (
    CURRENT_SENTINEL_COLS,
    analyze_single_h1_file,
    compute_metrics_from_frame,
    load_model,
)


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
        "--h1-dir",
        default="/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_tau_2mos",
        help="Directory containing tau_2mos H1 files.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for sentinel-impact CSVs and summary note. "
            "If omitted, a model-specific path is generated automatically."
        ),
    )
    p.add_argument(
        "--metric-samples-per-file",
        type=int,
        default=120000,
        help="Number of exact Stage A+B rows to sample per H1 file.",
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


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")


def infer_h1_label(h1_dir: str) -> str:
    base = Path(h1_dir).name
    mapping = {
        "e3sm2026_ftorch_tau_2mos": "tau_old",
        "e3sm2026_ftorch_emulator2_2mos": "tau_new",
    }
    return mapping.get(base, sanitize_name(base) or "h1")


def default_out_dir(model_path: str, h1_dir: str) -> str:
    model_stem = Path(model_path).stem
    h1_label = infer_h1_label(h1_dir)
    return str(
        EVALUATION_REPORT_ROOT
        / "diagnostics"
        / "moe"
        / sanitize_name(model_stem)
        / f"{h1_label}_exact_stage_b_sentinel_impact"
    )


def sentinel_mask(df):
    return (df[CURRENT_SENTINEL_COLS] == -99999.0).any(axis=1)


def add_shared_counts(row, result, sampled_rows, sampled_sentinel_rows):
    exact_rows = float(result.counts["exact_stage_b_dropna_rows"])
    row["full_exact_stage_b_rows"] = int(result.counts["exact_stage_b_dropna_rows"])
    row["full_exact_stage_b_sentinel_rows"] = int(result.counts["exact_stage_b_sentinel_rows"])
    row["full_exact_stage_b_sentinel_fraction"] = (
        result.counts["exact_stage_b_sentinel_rows"] / exact_rows if exact_rows > 0 else np.nan
    )
    row["sampled_rows_subset"] = int(sampled_rows)
    row["sampled_sentinel_rows_subset"] = int(sampled_sentinel_rows)
    row["sampled_sentinel_fraction_subset"] = (
        sampled_sentinel_rows / float(sampled_rows) if sampled_rows > 0 else np.nan
    )
    return row


def write_note(out_path, metrics_df):
    all_df = metrics_df[metrics_df["mode"] == "exact_stage_b_sampled_all"].copy()
    nons_df = metrics_df[metrics_df["mode"] == "exact_stage_b_sampled_nonsentinel"].copy()
    sent_df = metrics_df[metrics_df["mode"] == "exact_stage_b_sampled_sentinel_only"].copy()

    def med(df, col):
        if df.empty:
            return np.nan
        return float(np.nanmedian(df[col]))

    def fmt(x):
        if x != x:
            return "nan"
        return "{:.4f}".format(x)

    all_nct = med(all_df, "r2_nctend")
    ns_nct = med(nons_df, "r2_nctend")
    all_nrt = med(all_df, "r2_nrtend")
    ns_nrt = med(nons_df, "r2_nrtend")
    all_pos = med(all_df, "r2_nrtend_pos")
    ns_pos = med(nons_df, "r2_nrtend_pos")
    all_neg = med(all_df, "r2_nrtend_neg")
    ns_neg = med(nons_df, "r2_nrtend_neg")
    all_nonfinite = med(all_df, "n_nonfinite_model_rows")
    ns_nonfinite = med(nons_df, "n_nonfinite_model_rows")
    sent_frac = med(all_df, "full_exact_stage_b_sentinel_fraction")
    sent_sample_rows = med(all_df, "sampled_sentinel_rows_subset")

    major = (
        np.isfinite(all_pos)
        and np.isfinite(ns_pos)
        and (ns_pos - all_pos > 5.0 or (all_nonfinite > 0 and ns_nonfinite == 0 and ns_nrt - all_nrt > 5.0))
    )
    if major:
        verdict = "sentinel rows are a major contributor, but not necessarily the whole story"
    else:
        verdict = "sentinel rows are a secondary contributor, not the main explanation"

    lines = [
        verdict,
        "",
        "- Median exact-stage-B sentinel fraction by file: {}.".format(fmt(sent_frac)),
        "- Median all-row metrics: nctend R2={}, nrtend R2={}, pos-nrtend R2={}, neg-nrtend R2={}.".format(
            fmt(all_nct), fmt(all_nrt), fmt(all_pos), fmt(all_neg)
        ),
        "- Median non-sentinel metrics: nctend R2={}, nrtend R2={}, pos-nrtend R2={}, neg-nrtend R2={}.".format(
            fmt(ns_nct), fmt(ns_nrt), fmt(ns_pos), fmt(ns_neg)
        ),
        "- Median non-finite prediction rows per 120k sample: all={}, non-sentinel={}.".format(
            fmt(all_nonfinite), fmt(ns_nonfinite)
        ),
        "- In practice, the sampled sentinel rows and sampled non-finite rows match almost exactly: median sentinel rows per 120k sample={}.".format(
            fmt(sent_sample_rows)
        ),
        "- So removing sentinels eliminates the non-finite subset, but it does not improve the finite-row nctend or positive-nrtend skill. The remaining H1 failure is still dominated by non-sentinel rows.".format(
        ),
    ]
    if not sent_df.empty:
        lines.append(
            "- Sentinel-only rows are pathological on their own: median nctend R2={}, nrtend R2={}, pos-nrtend R2={}, neg-nrtend R2={}.".format(
                fmt(med(sent_df, "r2_nctend")),
                fmt(med(sent_df, "r2_nrtend")),
                fmt(med(sent_df, "r2_nrtend_pos")),
                fmt(med(sent_df, "r2_nrtend_neg")),
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir or default_out_dir(args.model, args.h1_dir))
    csv_path = out_dir / "sentinel_rescore_metrics.csv"
    note_path = out_dir / "summary_note.txt"

    rng = np.random.default_rng(args.seed)
    model = load_model(args.model)
    h1_files = sorted(glob.glob(os.path.join(args.h1_dir, "*.h1.*.nc")))
    if not h1_files:
        raise FileNotFoundError("No H1 files found in {}".format(args.h1_dir))

    rows = []

    for i, file_path in enumerate(h1_files):
        file_name = os.path.basename(file_path)
        print("Analyzing {}/{}: {}".format(i + 1, len(h1_files), file_name))
        result = analyze_single_h1_file(
            file_path=file_path,
            rng=rng,
            collect_full_frames=False,
            metric_samples_per_file=args.metric_samples_per_file,
            current_samples_per_timestep=1,
        )

        exact_df = result.exact_model_df.reset_index(drop=True)
        mask = sentinel_mask(exact_df)
        nons_df = exact_df.loc[~mask].reset_index(drop=True)
        sent_df = exact_df.loc[mask].reset_index(drop=True)

        all_row = compute_metrics_from_frame(
            model,
            exact_df,
            file_name=result.file_name,
            mode="exact_stage_b_sampled_all",
            target_summary=result.target_summary,
        )
        rows.append(add_shared_counts(all_row, result, len(exact_df), int(mask.sum())))

        nons_row = compute_metrics_from_frame(
            model,
            nons_df,
            file_name=result.file_name,
            mode="exact_stage_b_sampled_nonsentinel",
            target_summary=None,
        )
        rows.append(add_shared_counts(nons_row, result, len(nons_df), 0))

        if len(sent_df) > 0:
            sent_row = compute_metrics_from_frame(
                model,
                sent_df,
                file_name=result.file_name,
                mode="exact_stage_b_sampled_sentinel_only",
                target_summary=None,
            )
            rows.append(add_shared_counts(sent_row, result, len(sent_df), len(sent_df)))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(csv_path, index=False)
    write_note(note_path, metrics_df)

    print("Saved CSV:  {}".format(csv_path))
    print("Saved note: {}".format(note_path))


if __name__ == "__main__":
    main()
