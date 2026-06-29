#!/usr/bin/env bash
#
# Launch the pProp MLP wandb Bayesian sweep on SLURM, packing several agents
# onto GPUs via NVIDIA MPS (each task gets a slice of one GPU). Mirrors the
# GrowthNet launch pattern.
#
# ONE-TIME setup before the first launch:
#   1. wandb login                          # writes ~/.netrc (shared on the cluster home)
#   2. cd /home/ethan2/pProp_MLP
#      .venv/bin/wandb sweep sweeps/sweep.yaml
#      -> prints "Creating sweep with ID: xxxx" and a full path
#         <entity>/pprop-mlp-minimol/<sweep_id>
#   3. put that <sweep_id> below (or pass it in: SWEEP_ID=xxxx sbatch launch_sweep.sh)
#
# Then:  sbatch launch_sweep.sh

## Name of your SLURM job
#SBATCH --job-name=pprop_mlp_sweep

## Logs (stdout + stderr); %A = job id, %a = array task id
#SBATCH --output=/home/ethan2/logs/pprop_mlp_sweep_%A_%a.out
#SBATCH --error=/home/ethan2/logs/pprop_mlp_sweep_%A_%a.err
#SBATCH --open-mode=append

## Time limit
#SBATCH --time=1000000000:00:00

## CPUs per task (data is pre-featurized; training is light)
#SBATCH --cpus-per-task=8

## Memory per task in MB
#SBATCH --mem=32000

## Request 20% of a GPU per task via MPS, and run 5 agents in parallel.
## Raise the array upper bound to run more agents at once.
#SBATCH --gres=mps:20
#SBATCH --array=0-4

set -e

# ---- wandb sweep to attach to (entity/project/sweep_id) -------------------
ENTITY="${WANDB_ENTITY:-ethan_personal}"      # <-- your wandb entity
PROJECT="pprop-mlp-minimol"                    # auto-created on first run
SWEEP_ID="${SWEEP_ID:-dd3nnk84}"    # <-- from `wandb sweep`

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# Run from the project root so the sweep's relative paths resolve
# (program: src/sweep_train.py, split_dir: data/split_3).
cd /home/ethan2/pProp_MLP

# Sanitize any venv inherited from the submitting shell, then use the project venv.
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
source /home/ethan2/pProp_MLP/.venv/bin/activate

# Use "python -m wandb agent" so wandb spawns runs via this venv's interpreter.
# (sweep.yaml also pins .venv/bin/python in its `command:` for belt-and-braces.)
python -m wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}"

# Brief pause so the job is visible in `squeue` right after submit.
sleep 20s
