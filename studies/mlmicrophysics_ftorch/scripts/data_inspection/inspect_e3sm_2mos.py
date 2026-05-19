#!/usr/bin/env python3
"""Inspect the 2-month E3SM FTorch emulator output data."""
import netCDF4 as nc
import numpy as np
import os

DATA_DIR = "/global/cfs/cdirs/m4942/e3sm/e3sm2026_ftorch_emulator_2mos"

# List all files
files = sorted(os.listdir(DATA_DIR))
print("=== Files in directory ===")
for f in files:
    fpath = os.path.join(DATA_DIR, f)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")

# Inspect one h1 file
h1_files = [f for f in files if ".h1." in f]
if h1_files:
    print(f"\n=== Inspecting first h1 file: {h1_files[0]} ===")
    ds = nc.Dataset(os.path.join(DATA_DIR, h1_files[0]))
    print("\nDimensions:")
    for d in ds.dimensions:
        print(f"  {d}: {len(ds.dimensions[d])}")
    print(f"\nVariables ({len(ds.variables)} total):")
    for v in sorted(ds.variables):
        var = ds.variables[v]
        ln = getattr(var, 'long_name', '')
        units = getattr(var, 'units', '')
        print(f"  {v:35s} shape={str(var.shape):25s} {ln} [{units}]")
    ds.close()

# Also check h0 file
h0_files = [f for f in files if ".h0." in f]
if h0_files:
    print(f"\n=== Inspecting first h0 file: {h0_files[0]} ===")
    ds = nc.Dataset(os.path.join(DATA_DIR, h0_files[0]))
    print("\nDimensions:")
    for d in ds.dimensions:
        print(f"  {d}: {len(ds.dimensions[d])}")
    print(f"\nVariables ({len(ds.variables)} total):")
    for v in sorted(ds.variables):
        var = ds.variables[v]
        ln = getattr(var, 'long_name', '')
        print(f"  {v:35s} shape={str(var.shape):25s} {ln}")
    ds.close()
