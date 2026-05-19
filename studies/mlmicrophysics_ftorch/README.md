# MLMicrophysics FTorch Study Area

This directory contains project-specific emulator auditing, verification, and comparison work layered on top of the E3SM checkout. It is intentionally separate from upstream E3SM directories such as `components/`, `cime/`, `externals/`, and `share/`.

## Layout

- `scripts/`: runnable analysis, verification, conversion, comparison, and job scripts.
- `reports/`: generated JSON, plots, CSVs, notes, and logs from completed runs.
- `notebooks/`: exploratory notebooks.
- `manifests/`: conversion or validation manifests that document generated data.
- `docs/prompts/`: prompts, notes, and other supporting text artifacts.

## Working Convention

Keep upstream E3SM source changes in their original E3SM locations. Put new one-off or study-specific Python, shell, notebook, JSON, plot, and log artifacts here instead.

When adding a new result, prefer:

```text
reports/YYYY-MM-DD-short-description/
  README.md
  json/
  logs/
  plots/
  csv/
```

Each report `README.md` should record the input cases/data, the script command, and a short conclusion.
