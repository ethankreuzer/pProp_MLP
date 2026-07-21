#!/usr/bin/env bash
#
# Launch the TWO-TOWER (MiniMol + ECFP) dual-head pProp MLP wandb Bayesian sweep
# on SLURM, packing several agents onto GPUs via NVIDIA MPS (each task gets a
# slice of one GPU).
#
# ONE-TIME setup before the first launch:
#   1. wandb login                          # writes ~/.netrc
#   2. cd /home/ethan2/pProp_MLP
#      .venv/bin/python src/featurize_ecfp.py --radii 2 3 4   # ECFP caches
#      .venv/bin/wandb sweep sweeps/sweep.yaml
#      -> prints "Creating sweep with ID: xxxx"
#   3. put that <sweep_id> in SWEEP_ID below (or: SWEEP_ID=xxxx sbatch launch_sweep.sh)
#
# Then:  sbatch launch_sweep.sh

#SBATCH --job-name=pprop_mlp_twotower_sweep
#SBATCH --output=/home/ethan2/logs/pprop_mlp_twotower_sweep_%A_%a.out
#SBATCH --error=/home/ethan2/logs/pprop_mlp_twotower_sweep_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=1000000000:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=mps:20
#SBATCH --array=0-4

set -e

ENTITY="${WANDB_ENTITY:-ethan_personal}"
PROJECT="pprop-mlp-minimol-ecfp-twotower"
SWEEP_ID="${SWEEP_ID:-53yxmxkg}"   # <-- replace with ID from `wandb sweep`

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

cd /home/ethan2/pProp_MLP

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
source /home/ethan2/pProp_MLP/.venv/bin/activate

python -m wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}"

sleep 20s
