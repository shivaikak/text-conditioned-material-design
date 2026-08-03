#!/bin/bash
#SBATCH --job-name=smiles_gpt
#SBATCH --account=torch_pr_927_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/sk10731/logs/smiles_gpt_%j.out
#SBATCH --error=/scratch/sk10731/logs/smiles_gpt_%j.err
#SBATCH --mail-user=shiveshkumar3000@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/scratch/sk10731

PYTHON=/share/apps/anaconda3/2025.06/bin/python3.13

mkdir -p /scratch/sk10731/logs
mkdir -p /scratch/sk10731/checkpoints_gpt

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $($PYTHON --version)"

# WandB offline mode so the job doesn't need outbound auth mid-run;
# sync manually after with: wandb sync wandb/offline-run-*
export WANDB_MODE=offline

$PYTHON train_smiles_gpt.py \
    --max-epochs 300 \
    --batch-size 128 \
    --lr 3e-4 \
    --n-embd 256 \
    --n-head 4 \
    --n-layer 6 \
    --dropout 0.1 \
    --n-sample 1000

echo "Job finished: $(date)"
