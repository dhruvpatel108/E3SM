# File Index

## Scripts

- `scripts/diagnostics/`: emulator/H1 distribution checks, tau mismatch diagnosis, slice audits, sentinel analysis, plotting, and general TorchScript evaluation.
- `scripts/verification/`: focused model verification scripts and quick H1/model checks.
- `scripts/case_comparison/`: E3SM case-to-case history/restart/timing comparison tools.
- `scripts/data_conversion/`: E3SM NetCDF-to-parquet conversion utilities and configs.
- `scripts/data_inspection/`: small data-inspection helpers.
- `scripts/job_scripts/`: Slurm/CIME job helpers.
- `scripts/deception_legacy/`: older Deception-side processing/training support scripts retained for provenance.

## Reports And Artifacts

- `reports/2026-05-02-vectorized-batch-v3/`: TAU, scalar FTorch, and vectorized batch FTorch comparison JSONs.
- `reports/emulator_performance_evaluation/`: evaluation logs, plots, CSVs, and diagnostics from emulator performance studies.
- `reports/legacy-root-logs/`: logs that previously lived at repository root.
- `reports/data_conversion/logs/`: Slurm stdout/stderr target for the parquet conversion submit script.

## Other

- `notebooks/analysis.ipynb`: exploratory notebook moved from repository root.
- `manifests/conversion_manifest_validate_only.json`: validation-only conversion manifest.
- `docs/prompts/deception_batch_torchscript_prompt.md`: preserved prompt/notes for Deception batch TorchScript work.
