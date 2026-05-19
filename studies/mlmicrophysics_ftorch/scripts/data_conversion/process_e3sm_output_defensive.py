#!/usr/bin/env python3
"""Defensive E3SM-to-parquet conversion for microphysics training data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml


RENAME_MAP = {
    "P3_qc_in_TAU": "QC_TAU_in",
    "P3_nc_in_TAU": "NC_TAU_in",
    "P3_qr_in_TAU": "QR_TAU_in",
    "P3_nr_in_TAU": "NR_TAU_in",
    "P3_qc_out_TAU": "QC_TAU_out",
    "P3_nc_out_TAU": "NC_TAU_out",
    "P3_qr_out_TAU": "QR_TAU_out",
    "P3_nr_out_TAU": "NR_TAU_out",
    "P3_mu_c": "PGAM",
    "P3_lamc": "LAMC",
    "P3_lamr": "LAMR",
    "P3_nr": "N0R",
    "P3_qctend_TAU_raw": "qctend_TAU",
    "P3_nctend_TAU_raw": "nctend_TAU",
    "P3_qrtend_TAU_raw": "qrtend_TAU",
    "P3_nrtend_TAU_raw": "nrtend_TAU",
}


@dataclass
class FileSummary:
    filename: str
    n_time: int
    time_start: float
    time_end: float
    hour_start: int
    hour_end: int
    output_hours: list[int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration yaml file")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run full preflight validation without writing parquet output",
    )
    parser.add_argument(
        "--manifest-name",
        default="conversion_manifest.json",
        help="Manifest filename to write into the output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial output directory by skipping completed parquet files",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise TypeError(f"Config {path} did not parse to a dictionary")
    required = [
        "model_path",
        "model_file_start",
        "model_file_end",
        "time_var",
        "out_variables",
        "subset_variable",
        "subset_threshold",
        "out_path",
        "out_start",
        "out_format",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Config is missing required keys: {missing}")
    return config


def discover_input_files(config: dict[str, Any]) -> list[Path]:
    root = Path(config["model_path"])
    pattern = f"{config['model_file_start']}*.{config['model_file_end']}"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern!r} under {root}")
    return files


def normalize_subset_variables(subset_variable: Any, subset_threshold: Any) -> tuple[list[str], list[float]]:
    if subset_variable in (None, ""):
        return [], []
    if isinstance(subset_variable, str):
        variables = [subset_variable]
    else:
        variables = list(subset_variable)
    if isinstance(subset_threshold, (int, float)):
        thresholds = [float(subset_threshold)] * len(variables)
    else:
        thresholds = [float(x) for x in subset_threshold]
    if len(thresholds) != len(variables):
        raise ValueError("subset_threshold length must match subset_variable length")
    return variables, thresholds


def compute_output_hours(times: np.ndarray) -> list[int]:
    return [int(float(t) * 24.0) for t in times]


def inspect_file(
    path: Path,
    time_var: str,
    out_variables: list[str],
    subset_variables: list[str],
) -> FileSummary:
    ds = xr.open_dataset(path, decode_times=False, chunks={})
    try:
        if time_var not in ds.variables and time_var not in ds.coords:
            raise KeyError(f"{path.name} is missing required time variable {time_var!r}")
        missing_vars = [var for var in out_variables if var not in ds.variables]
        if missing_vars:
            raise KeyError(f"{path.name} is missing required out_variables: {missing_vars}")
        missing_subset = [var for var in subset_variables if var not in ds.variables]
        if missing_subset:
            raise KeyError(f"{path.name} is missing subset variables: {missing_subset}")

        times = np.asarray(ds[time_var].values, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError(f"{path.name} has invalid time coordinate shape {times.shape}")
        if not np.all(np.diff(times) >= 0):
            raise ValueError(f"{path.name} has non-monotonic {time_var} values")

        hours = compute_output_hours(times)
        if len(set(hours)) != len(hours):
            raise ValueError(f"{path.name} would generate duplicate output hour filenames within the same file")

        return FileSummary(
            filename=path.name,
            n_time=len(times),
            time_start=float(times[0]),
            time_end=float(times[-1]),
            hour_start=hours[0],
            hour_end=hours[-1],
            output_hours=hours,
        )
    finally:
        ds.close()


def build_manifest(
    *,
    config_path: Path,
    config: dict[str, Any],
    files: list[Path],
    summaries: list[FileSummary],
) -> dict[str, Any]:
    all_hours = [hour for summary in summaries for hour in summary.output_hours]
    hour_diffs = sorted({b - a for a, b in zip(all_hours[:-1], all_hours[1:])})
    return {
        "generated_at_utc": utc_now(),
        "config_path": str(config_path),
        "source_root": config["model_path"],
        "file_glob_start": config["model_file_start"],
        "file_extension": config["model_file_end"],
        "out_path": config["out_path"],
        "out_start": config["out_start"],
        "out_format": config["out_format"],
        "n_source_files": len(files),
        "n_output_time_steps": len(all_hours),
        "n_unique_output_hours": len(set(all_hours)),
        "min_output_hour": min(all_hours),
        "max_output_hour": max(all_hours),
        "output_hour_step_values": hour_diffs,
        "subset_variable": config["subset_variable"],
        "subset_threshold": config["subset_threshold"],
        "out_variables": config["out_variables"],
        "source_files": [str(path) for path in files],
        "file_summaries": [
            {
                "filename": summary.filename,
                "n_time": summary.n_time,
                "time_start": summary.time_start,
                "time_end": summary.time_end,
                "hour_start": summary.hour_start,
                "hour_end": summary.hour_end,
            }
            for summary in summaries
        ],
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def output_name(out_start: str, output_hour: int, out_format: str) -> str:
    return f"{out_start}_{output_hour:06d}.{out_format}"


def prepare_output_dir(
    *,
    output_dir: Path,
    suffix: str,
    expected_output_names: set[str],
    resume: bool,
) -> set[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_files = sorted(output_dir.glob(f"*.{suffix}.tmp"))
    if tmp_files:
        raise RuntimeError(
            f"Found {len(tmp_files)} temporary files in {output_dir}. "
            "Please inspect and remove them before continuing."
        )

    existing = sorted(output_dir.glob(f"*.{suffix}"))
    existing_names = {path.name for path in existing}
    unexpected = sorted(existing_names - expected_output_names)
    if unexpected:
        raise RuntimeError(
            f"Output directory {output_dir} contains {len(unexpected)} unexpected *.{suffix} files, "
            f"for example: {unexpected[:5]}"
        )

    if existing and not resume:
        raise RuntimeError(
            f"Refusing to write into non-empty output directory {output_dir} "
            f"because it already contains {len(existing)} *.{suffix} files. "
            "Use --resume to continue a partial conversion."
        )
    return existing_names


def build_subset_mask(
    frame: pd.DataFrame,
    subset_variables: list[str],
    subset_thresholds: list[float],
) -> np.ndarray:
    if not subset_variables:
        return np.ones(frame.shape[0], dtype=bool)
    valid = np.zeros(frame.shape[0], dtype=bool)
    for variable, threshold in zip(subset_variables, subset_thresholds):
        valid |= frame[variable] >= threshold
    return valid


def convert_file(
    path: Path,
    *,
    config: dict[str, Any],
    subset_variables: list[str],
    subset_thresholds: list[float],
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    ds = xr.open_dataset(path, decode_times=False, chunks={})
    try:
        times = np.asarray(ds[config["time_var"]].values, dtype=float)
        output_hours = compute_output_hours(times)
        written = []
        skipped = []

        for time_value, output_hour in zip(times, output_hours):
            final_path = output_dir / output_name(config["out_start"], output_hour, config["out_format"])
            if final_path.exists():
                if resume:
                    skipped.append(
                        {
                            "output_file": final_path.name,
                            "time_value_days": float(time_value),
                            "output_hour": output_hour,
                        }
                    )
                    continue
                raise FileExistsError(f"Output file already exists: {final_path}")

            frame = ds[config["out_variables"]].sel(**{config["time_var"]: time_value}).to_dataframe()
            mask = build_subset_mask(frame, subset_variables, subset_thresholds)
            subset = frame.loc[mask].reset_index()
            subset.rename(columns=RENAME_MAP, inplace=True)

            missing_after_rename = [var for var in config["out_variables"] if var not in frame.columns]
            if missing_after_rename:
                raise KeyError(f"Missing columns after dataframe conversion in {path.name}: {missing_after_rename}")

            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            if config["out_format"] == "parquet":
                subset.to_parquet(tmp_path)
            elif config["out_format"] == "csv":
                subset.to_csv(tmp_path, index=False)
            else:
                raise ValueError(f"Unsupported out_format {config['out_format']!r}")
            tmp_path.replace(final_path)

            written.append(
                {
                    "output_file": final_path.name,
                    "time_value_days": float(time_value),
                    "output_hour": output_hour,
                    "rows_written": int(len(subset)),
                }
            )

        return {
            "source_file": path.name,
            "n_time": len(times),
            "outputs": written,
            "skipped_outputs": skipped,
            "n_written": len(written),
            "n_skipped": len(skipped),
            "rows_written_total": int(sum(item["rows_written"] for item in written)),
        }
    finally:
        ds.close()


def write_progress_manifest(
    *,
    base_manifest: dict[str, Any],
    output_dir: Path,
    per_file_outputs: list[dict[str, Any]],
    existing_output_names: set[str],
) -> Path:
    progress_manifest = dict(base_manifest)
    progress_manifest["progress_updated_utc"] = utc_now()
    progress_manifest["per_file_outputs"] = per_file_outputs
    progress_manifest["completed_output_files"] = len(existing_output_names)
    progress_manifest["remaining_output_files"] = (
        progress_manifest["n_output_time_steps"] - len(existing_output_names)
    )
    progress_manifest["total_rows_written_this_manifest"] = int(
        sum(result["rows_written_total"] for result in per_file_outputs)
    )
    return write_manifest(progress_manifest, output_dir, "conversion_progress.json")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    subset_variables, subset_thresholds = normalize_subset_variables(
        config["subset_variable"],
        config["subset_threshold"],
    )
    input_files = discover_input_files(config)

    summaries = [
        inspect_file(
            path,
            time_var=config["time_var"],
            out_variables=list(config["out_variables"]),
            subset_variables=subset_variables,
        )
        for path in input_files
    ]

    all_hours = [hour for summary in summaries for hour in summary.output_hours]
    if len(set(all_hours)) != len(all_hours):
        raise ValueError("Duplicate output hour filenames detected across input files")

    manifest = build_manifest(
        config_path=config_path,
        config=config,
        files=input_files,
        summaries=summaries,
    )
    expected_output_names = {
        output_name(config["out_start"], output_hour, config["out_format"])
        for output_hour in all_hours
    }

    output_dir = Path(config["out_path"]).resolve()
    if args.validate_only:
        manifest["mode"] = "validate_only"
        manifest_path = write_manifest(
            manifest,
            config_path.parent,
            args.manifest_name.replace(".json", "_validate_only.json"),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"\nWrote validation manifest to {manifest_path}")
        return

    existing_output_names = prepare_output_dir(
        output_dir=output_dir,
        suffix=config["out_format"],
        expected_output_names=expected_output_names,
        resume=args.resume,
    )
    manifest["mode"] = "resume" if args.resume else "convert"
    manifest["existing_output_files_found"] = len(existing_output_names)
    manifest["remaining_output_files_at_start"] = (
        manifest["n_output_time_steps"] - len(existing_output_names)
    )
    manifest["conversion_started_utc"] = utc_now()
    per_file_outputs = []

    for index, (path, summary) in enumerate(zip(input_files, summaries), start=1):
        file_expected_names = {
            output_name(config["out_start"], output_hour, config["out_format"])
            for output_hour in summary.output_hours
        }
        if args.resume and file_expected_names.issubset(existing_output_names):
            print(f"[{index}/{len(input_files)}] Skipping {path.name} (all outputs already present)")
            result = {
                "source_file": path.name,
                "n_time": summary.n_time,
                "outputs": [],
                "skipped_outputs": [
                    {"output_file": name}
                    for name in sorted(file_expected_names)
                ],
                "n_written": 0,
                "n_skipped": len(file_expected_names),
                "rows_written_total": 0,
            }
            per_file_outputs.append(result)
            write_progress_manifest(
                base_manifest=manifest,
                output_dir=output_dir,
                per_file_outputs=per_file_outputs,
                existing_output_names=existing_output_names,
            )
            continue

        print(f"[{index}/{len(input_files)}] Processing {path.name}")
        result = convert_file(
            path,
            config=config,
            subset_variables=subset_variables,
            subset_thresholds=subset_thresholds,
            output_dir=output_dir,
            resume=args.resume,
        )
        per_file_outputs.append(result)
        existing_output_names.update(item["output_file"] for item in result["outputs"])
        print(
            f"  wrote {result['n_written']} parquet files, skipped {result['n_skipped']}, "
            f"and wrote {result['rows_written_total']} rows from {path.name}"
        )
        write_progress_manifest(
            base_manifest=manifest,
            output_dir=output_dir,
            per_file_outputs=per_file_outputs,
            existing_output_names=existing_output_names,
        )

    manifest["conversion_finished_utc"] = utc_now()
    manifest["per_file_outputs"] = per_file_outputs
    manifest["completed_output_files"] = len(existing_output_names)
    manifest["remaining_output_files"] = manifest["n_output_time_steps"] - len(existing_output_names)
    manifest["total_rows_written"] = int(
        sum(result["rows_written_total"] for result in per_file_outputs)
    )
    manifest_path = write_manifest(manifest, output_dir, args.manifest_name)
    print(f"\nConversion complete. Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
