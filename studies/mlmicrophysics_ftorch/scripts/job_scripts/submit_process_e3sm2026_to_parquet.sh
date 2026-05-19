#!/bin/bash
#SBATCH --job-name=e3sm2026_to_parquet
#SBATCH --account=m4942
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --licenses=cfs,SCRATCH
#SBATCH --output=/global/homes/d/dvpatel/e3sm_ftorch/studies/mlmicrophysics_ftorch/reports/data_conversion/logs/process_e3sm2026_to_parquet_%j.out
#SBATCH --error=/global/homes/d/dvpatel/e3sm_ftorch/studies/mlmicrophysics_ftorch/reports/data_conversion/logs/process_e3sm2026_to_parquet_%j.err


set -euo pipefail

export OMP_NUM_THREADS=4
export DASK_ARRAY__SLICING__SPLIT_LARGE_CHUNKS=True

PYTHON_BIN="/global/homes/d/dvpatel/.conda/envs/mlmicrophysics-env/bin/python"
SCRIPT_PATH="/global/homes/d/dvpatel/e3sm_ftorch/studies/mlmicrophysics_ftorch/scripts/data_conversion/process_e3sm_output_defensive.py"
CONFIG_PATH="/global/homes/d/dvpatel/e3sm_ftorch/studies/mlmicrophysics_ftorch/scripts/data_conversion/configs/e3sm2026_mlmicro_tau_train_to_parquet.yml"

echo "Starting conversion at $(date -u)"
echo "Python: ${PYTHON_BIN}"
echo "Script: ${SCRIPT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Resume mode: enabled"

"${PYTHON_BIN}" "${SCRIPT_PATH}" "${CONFIG_PATH}" --resume

echo "Finished conversion at $(date -u)"
