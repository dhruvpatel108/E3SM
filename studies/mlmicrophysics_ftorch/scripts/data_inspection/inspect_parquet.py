#!/usr/bin/env python3
"""Quick script to inspect parquet schema and sample data."""
import pyarrow.parquet as pq
import pandas as pd

path = '/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/e3sm_mp_data_000000.parquet'

meta = pq.read_metadata(path)
print(f"num_rows: {meta.num_rows}")
print(f"num_columns: {meta.num_columns}")

schema = pq.read_schema(path)
print("\nColumns:")
for i, name in enumerate(schema.names):
    print(f"  [{i:2d}] {name}: {schema.field(name).type}")

# Read a small sample
df = pd.read_parquet(path, engine='pyarrow')
print(f"\nShape: {df.shape}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
print(f"\nDescribe:")
print(df.describe().to_string())
