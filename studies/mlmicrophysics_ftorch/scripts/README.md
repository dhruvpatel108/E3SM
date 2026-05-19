# Scripts

Run scripts from the repository root or by absolute path unless a script documents otherwise. Defaults for newly patched diagnostics now write under `studies/mlmicrophysics_ftorch/reports/` to avoid recreating clutter at repository root.

## Categories

- `diagnostics/`: H1/parquet mismatch investigation, slice semantics, sentinel impact, distribution plots, and TorchScript emulator scoring.
- `verification/`: standalone checks for fixed TorchScript models and known emulator bugs.
- `case_comparison/`: pairwise E3SM output/timing comparison.
- `data_conversion/`: defensive NetCDF-to-parquet conversion plus compatibility checks and configs.
- `data_inspection/`: quick manual inspection utilities.
- `job_scripts/`: Slurm/CIME launch helpers.
- `deception_legacy/`: older scripts/configs kept as historical context.

## Common Examples

```bash
python studies/mlmicrophysics_ftorch/scripts/case_comparison/compare_e3sm_case_outputs.py --help
python studies/mlmicrophysics_ftorch/scripts/diagnostics/diagnose_tau_h1_mismatch.py --help
python studies/mlmicrophysics_ftorch/scripts/verification/verify_fixed_model.py --help
```
