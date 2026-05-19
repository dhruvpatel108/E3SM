# Vectorized Batch V3 Comparison

Generated: 2026-05-02 around 00:13 local filesystem time.

## Purpose

Compare three E3SM microphysics runs:

- TAU baseline: `e3sm2025_mlmicro_tau_cmp_v3`
- Scalar FTorch: `e3sm2025_mlmicro_ftorch_scalar_cmp_v3`
- Vectorized batch FTorch: `e3sm2025_mlmicro_ftorch_batch_cmp_v3`

## Generator

The JSON reports were generated with:

```bash
python studies/mlmicrophysics_ftorch/scripts/case_comparison/compare_e3sm_case_outputs.py
```

The script compares latest history files, restart files, and timing archives for two case roots, then writes a JSON report when `--output` is provided.

## Files

- `json/scalar_vs_vectorized_batch_v3.json`
- `json/scalar_vs_vectorized_batch_v3_p3_diagnostics.json`
- `json/tau_vs_scalar_v3.json`
- `json/tau_vs_vectorized_batch_v3.json`

## Notes

The vectorized batch run was faster in the `micro_p3_tend_loop` timer, but the scalar-vs-vectorized-batch comparison was not numerically equivalent. The targeted P3 diagnostics were added afterward to help separate input/output-field differences from broader trajectory drift.
