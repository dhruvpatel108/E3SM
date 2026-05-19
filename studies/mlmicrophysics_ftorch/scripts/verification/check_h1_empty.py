#!/usr/bin/env python3
"""
Check whether an E3SM h1 netCDF file contains any time samples.

Usage:
  python check_h1_empty.py /path/to/file.h1....nc
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report whether a .h1 netCDF file has any time records."
    )
    parser.add_argument("h1_file", help="Absolute path to the .h1 netCDF file")
    args = parser.parse_args()

    try:
        from netCDF4 import Dataset  # type: ignore
    except Exception as exc:
        print(f"ERROR: netCDF4 import failed: {exc}")
        print("Hint: load a Python environment with netCDF4 available.")
        return 2

    try:
        ds = Dataset(args.h1_file, "r")
    except Exception as exc:
        print(f"ERROR: Could not open file: {args.h1_file}")
        print(f"Details: {exc}")
        return 2

    try:
        if "time" not in ds.dimensions:
            print("ERROR: No 'time' dimension found in file.")
            return 1

        ntime = len(ds.dimensions["time"])
        print(f"File: {args.h1_file}")
        print(f"time dimension length: {ntime}")

        if ntime == 0:
            print("RESULT: EMPTY (no instantaneous samples stored).")
            return 1

        print("RESULT: NOT EMPTY (contains instantaneous samples).")
        if "date" in ds.variables:
            print(f"date[:]: {ds.variables['date'][:]}")
        if "datesec" in ds.variables:
            print(f"datesec[:]: {ds.variables['datesec'][:]}")
        if "time" in ds.variables:
            print(f"time[:]: {ds.variables['time'][:]}")
        return 0
    finally:
        ds.close()


if __name__ == "__main__":
    sys.exit(main())
