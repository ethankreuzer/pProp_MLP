#!/usr/bin/env bash
#
# Launch the pProp MLP regression wandb Bayesian sweep on SLURM.
# Mirrors launch_sweep.sh exactly; only the job name, log paths, and SWEEP_ID differ.
#
# ONE-TIME setup before the first launch:
#   1. cd /home/ethan2/pProp_MLP
#      .venv/bin/wandb sweep sweeps/sweep_regression.yaml
#      -> prints "Creating sweep with ID: xxxx"
#   2. put that <sweep_id> below (or pass it: SWEEP_ID=xxxx sbatch launch_sweep_regression.sh)
#
# Then:  sbatch launch_sweep_regression.sh

#SBATCH --job-name=pprop_mlp_regression_sweep
#SBATCH --output=/home/ethan2/logs/pprop_mlp_regression_sweep_%A_%a.out
#SBATCH --error=/home/ethan2/logs/pprop_mlp_regression_sweep_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=1000000000:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=mps:20
#SBATCH --array=0-4

set -e

ENTITY="${WANDB_ENTITY:-ethan_personal}"
PROJECT="pprop-mlp-minimol"
SWEEP_ID="${SWEEP_ID:-19n7o53o}"    # <-- from `wandb sweep sweeps/sweep_regression.yaml`

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

cd /home/ethan2/pProp_MLP

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
source /home/ethan2/pProp_MLP/.venv/bin/activate

python -m wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}"

sleep 20s
