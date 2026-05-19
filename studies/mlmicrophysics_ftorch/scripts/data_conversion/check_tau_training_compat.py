#!/usr/bin/env python3
"""Check whether an E3SM run directory can feed the TAU parquet converter."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import netCDF4 as nc

try:
    import yaml
except ImportError:  # pragma: no cover - optional at runtime
    yaml = None


DEFAULT_REQUIRED_VARS = [
    "T",
    "RHO_CLUBB",
    "CLOUD",
    "FREQR",
    "P3_qc_in_TAU",
    "P3_nc_in_TAU",
    "P3_qr_in_TAU",
    "P3_nr_in_TAU",
    "P3_qc_out_TAU",
    "P3_nc_out_TAU",
    "P3_qr_out_TAU",
    "P3_nr_out_TAU",
    "P3_qctend_TAU_raw",
    "P3_nctend_TAU_raw",
    "P3_qrtend_TAU_raw",
    "P3_nrtend_TAU_raw",
    "P3_mu_c",
    "P3_lamc",
    "P3_lamr",
    "P3_nr",
]

STREAM_RE = re.compile(r"\.eam\.(h\d+|rs)\.")
FILE_START_RE = re.compile(r"^(.*\.eam\.(?:h\d+|rs))\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an E3SM run directory and report whether any EAM history stream "
            "contains the variables required by the TAU parquet converter."
        )
    )
    parser.add_argument("run_dir", help="Run directory containing E3SM history NetCDF files")
    parser.add_argument(
        "--config",
        help=(
            "Optional converter YAML. When provided, required variables are loaded from "
            "its out_variables plus subset_variable entries."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    return parser.parse_args()


def load_required_vars(config_path: Path | None) -> list[str]:
    required = list(DEFAULT_REQUIRED_VARS)
    if config_path is None:
        return required
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --config but is not installed")
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise TypeError(f"Config {config_path} did not parse to a dictionary")
    ordered = []
    for key in ("out_variables", "subset_variable"):
        values = config.get(key, [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            if value and value not in ordered:
                ordered.append(value)
    return ordered or required


def discover_streams(run_dir: Path) -> dict[str, list[Path]]:
    streams: dict[str, list[Path]] = defaultdict(list)
    candidates = list(run_dir.glob("*.eam.h*.nc")) + list(run_dir.glob("*.eam.rs.*.nc"))
    for path in sorted(set(candidates)):
        match = STREAM_RE.search(path.name)
        if not match:
            continue
        streams[match.group(1)].append(path)
    return dict(streams)


def infer_file_start(path: Path) -> str:
    match = FILE_START_RE.match(path.name)
    if match:
        return match.group(1)
    return path.stem


def inspect_sample(path: Path, required_vars: list[str]) -> dict[str, Any]:
    with nc.Dataset(path) as ds:
        available = set(ds.variables)
        present = [var for var in required_vars if var in available]
        missing = [var for var in required_vars if var not in available]
        p3_variables = sorted(var for var in ds.variables if var.startswith("P3_"))
        dims = {name: len(dim) for name, dim in ds.dimensions.items()}
    return {
        "sample_file": str(path),
        "model_file_start": infer_file_start(path),
        "present_required_vars": present,
        "missing_required_vars": missing,
        "n_present_required_vars": len(present),
        "n_missing_required_vars": len(missing),
        "compatible": not missing,
        "available_p3_variables": p3_variables,
        "dimensions": dims,
    }


def build_report(run_dir: Path, required_vars: list[str]) -> dict[str, Any]:
    streams = discover_streams(run_dir)
    if not streams:
        raise FileNotFoundError(f"No EAM history or restart NetCDF files found under {run_dir}")

    stream_reports = {}
    compatible_streams = []
    total_files = 0

    for stream, files in streams.items():
        total_files += len(files)
        sample = inspect_sample(files[0], required_vars)
        sample["stream"] = stream
        sample["n_files"] = len(files)
        sample["first_file"] = files[0].name
        sample["last_file"] = files[-1].name
        stream_reports[stream] = sample
        if sample["compatible"]:
            compatible_streams.append(stream)

    return {
        "run_dir": str(run_dir),
        "required_vars": required_vars,
        "n_required_vars": len(required_vars),
        "n_streams": len(stream_reports),
        "n_files": total_files,
        "compatible_streams": compatible_streams,
        "compatible": bool(compatible_streams),
        "streams": stream_reports,
    }


def render_text(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Run directory: {report['run_dir']}")
    lines.append(
        f"Discovered {report['n_streams']} EAM stream groups across {report['n_files']} files"
    )
    lines.append(
        f"Required variables ({report['n_required_vars']}): {', '.join(report['required_vars'])}"
    )
    if report["compatible"]:
        lines.append(
            "Compatible streams: " + ", ".join(report["compatible_streams"])
        )
    else:
        lines.append("Compatible streams: none")

    for stream in sorted(report["streams"]):
        data = report["streams"][stream]
        lines.append("")
        lines.append(f"[{stream}] {data['n_present_required_vars']}/{report['n_required_vars']} required vars present")
        lines.append(f"  Files: {data['n_files']}  sample: {data['first_file']}")
        lines.append(f"  Suggested model_file_start: {data['model_file_start']}")
        if data["present_required_vars"]:
            lines.append("  Present required vars: " + ", ".join(data["present_required_vars"]))
        if data["missing_required_vars"]:
            lines.append("  Missing required vars: " + ", ".join(data["missing_required_vars"]))
        if data["available_p3_variables"]:
            lines.append("  Available P3 vars: " + ", ".join(data["available_p3_variables"]))
        else:
            lines.append("  Available P3 vars: none")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    config_path = Path(args.config).resolve() if args.config else None
    required_vars = load_required_vars(config_path)
    report = build_report(run_dir, required_vars)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
