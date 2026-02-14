#!/bin/bash
#SBATCH --error=logs/%A/openpi_eval.err
#SBATCH --output=logs/%A/openpi_eval.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

OPENPI_ROOT="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/nigeln/JHU/11_openpi_surgical"
GR00T_ROOT="/lustre/fsw/portfolios/healthcareeng/users/nigeln/JHU/8_groot_n1.6"

: "${EVAL_CONFIG_PATH:?ERROR: EVAL_CONFIG_PATH not set. Use launch_open_loop_eval_1gpu.sh.}"

LOG_DIR="logs/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "GPU Eval Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Log dir: ${LOG_DIR}"
echo "Stdout log: ${LOG_DIR}/openpi_eval.out"
echo "Stderr log: ${LOG_DIR}/openpi_eval.err"
echo "Config: ${EVAL_CONFIG_PATH}"
echo "=========================================="

srun --container-image="/lustre/fsw/portfolios/healthcareeng/users/nigeln/images/cosmos-predict2-container.sqsh" \
     --container-mounts="/lustre/fsw/portfolios/healthcareeng/users/nigeln:/lustre/fsw/portfolios/healthcareeng/users/nigeln,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan" \
     bash -c '
          set -euo pipefail
          OPENPI_ROOT="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/nigeln/JHU/11_openpi_surgical"
          GR00T_ROOT="/lustre/fsw/portfolios/healthcareeng/users/nigeln/JHU/8_groot_n1.6"
          export PYTHONPATH="${OPENPI_ROOT}/src:${GR00T_ROOT}"
          cd "${OPENPI_ROOT}"
          uv pip install --force-reinstall pandas matplotlib
          uv run python eval_scripts/rollout_eval.py "${EVAL_CONFIG_PATH}"
     '
