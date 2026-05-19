#!/usr/bin/env python3
"""Compare E3SM history/restart outputs and timing archives for matched cases."""

from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np


DEFAULT_VARS = [
    "P3_qctend_TAU",
    "P3_nctend_TAU",
    "P3_qrtend_TAU",
    "P3_nrtend_TAU",
    "P3_qctend_TAU_raw",
    "P3_nctend_TAU_raw",
    "P3_qrtend_TAU_raw",
    "P3_nrtend_TAU_raw",
    "CLDLIQ",
    "RAINQM",
    "NUMLIQ",
    "NUMRAI",
    "CLOUD",
    "PRECL",
    "PRECT",
]

TIMERS = [
    "a:microp_p3_tend",
    "a:micro_p3_tend_loop",
    "a:microp_tend",
    "a:bc_physics",
    "a:physpkg_st1",
    "a_i:microp_p3_tend",
    "a_i:micro_p3_tend_loop",
]


def latest_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} under {run_dir}")
    return matches[-1]


def numeric_array(ds: netCDF4.Dataset, name: str) -> np.ndarray | None:
    var = ds.variables[name]
    try:
        arr = np.asarray(var[:])
    except Exception:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.asarray(arr, dtype=np.float64)


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": a.shape, "shape_b": b.shape, "shape_match": False}

    diff = b - a
    abs_diff = np.abs(diff)
    finite = np.isfinite(a) & np.isfinite(b)
    nonfinite_mismatch = int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))

    if np.any(finite):
        denom = np.maximum(np.abs(a[finite]), 1.0e-300)
        rel = np.abs(diff[finite]) / denom
        max_abs = float(np.max(abs_diff[finite]))
        rms = float(math.sqrt(np.mean(diff[finite] ** 2)))
        max_rel = float(np.max(rel))
    else:
        max_abs = float("nan")
        rms = float("nan")
        max_rel = float("nan")

    return {
        "shape": list(a.shape),
        "shape_match": True,
        "max_abs": max_abs,
        "rms": rms,
        "max_rel": max_rel,
        "allclose_0_0": bool(np.array_equal(a, b)),
        "nonzero_diff_count": int(np.count_nonzero(abs_diff > 0.0)),
        "nonfinite_mismatch_count": nonfinite_mismatch,
    }


def compare_netcdf(path_a: Path, path_b: Path, requested: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"file_a": str(path_a), "file_b": str(path_b), "variables": {}}
    with netCDF4.Dataset(path_a) as ds_a, netCDF4.Dataset(path_b) as ds_b:
        names = [name for name in requested if name in ds_a.variables and name in ds_b.variables]
        missing = sorted(set(requested) - set(names))
        out["missing_requested"] = missing
        for name in names:
            arr_a = numeric_array(ds_a, name)
            arr_b = numeric_array(ds_b, name)
            if arr_a is None or arr_b is None:
                continue
            out["variables"][name] = compare_arrays(arr_a, arr_b)
    return out


def parse_timing_archive(run_dir: Path) -> dict[str, Any]:
    archive = latest_file(run_dir, "timing.*.tar.gz")
    timers: dict[str, Any] = {}
    line_re = re.compile(
        r'^"(?P<name>[^"]+)"\s+\S+\s+\S+\s+\S+\s+'
        r'(?P<count>\S+)\s+(?P<walltotal>\S+)\s+(?P<wallmax>\S+)'
    )
    with tarfile.open(archive) as tf:
        stats_members = [m for m in tf.getmembers() if m.name.endswith("model_timing_stats")]
        if not stats_members:
            return {"archive": str(archive), "timers": timers}
        text = tf.extractfile(stats_members[-1]).read().decode("utf-8", "replace")
    for line in text.splitlines():
        match = line_re.match(line)
        if not match:
            continue
        name = match.group("name")
        if name not in TIMERS:
            continue
        timers[name] = {
            "count": float(match.group("count")),
            "walltotal": float(match.group("walltotal")),
            "wallmax": float(match.group("wallmax")),
        }
    return {"archive": str(archive), "timers": timers}


def case_run_dir(case_root: Path, run_config: str) -> Path:
    return case_root / "tests" / run_config / "run"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-a", type=Path, required=True)
    parser.add_argument("--case-b", type=Path, required=True)
    parser.add_argument("--label-a", default="case_a")
    parser.add_argument("--label-b", default="case_b")
    parser.add_argument("--run-config", default="XS_1x1_ndays")
    parser.add_argument("--vars", nargs="*", default=DEFAULT_VARS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_a = case_run_dir(args.case_a, args.run_config)
    run_b = case_run_dir(args.case_b, args.run_config)
    hist_a = latest_file(run_a, "*.eam.h1.*.nc")
    hist_b = latest_file(run_b, "*.eam.h1.*.nc")
    rest_a = latest_file(run_a, "*.eam.r.*.nc")
    rest_b = latest_file(run_b, "*.eam.r.*.nc")

    report = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "history_compare": compare_netcdf(hist_a, hist_b, args.vars),
        "restart_compare": compare_netcdf(rest_a, rest_b, args.vars),
        "timing": {
            args.label_a: parse_timing_archive(run_a),
            args.label_b: parse_timing_archive(run_b),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
