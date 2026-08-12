#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=torch_pr_927_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/sk10731/text-generated-material-design/text-conditioned-material-design/experiments/baseline/slurm-%j.out
#SBATCH --error=/scratch/sk10731/text-generated-material-design/text-conditioned-material-design/experiments/baseline/slurm-%j.err
#SBATCH --mail-user=sk10731@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/scratch/sk10731/text-generated-material-design/text-conditioned-material-design

set -euo pipefail
export AUTOLAB_PYTHON=/scratch/sk10731/text-generated-material-design/text-conditioned-material-design/.venv_shared/bin/python
bash /scratch/sk10731/text-generated-material-design/text-conditioned-material-design/slurm/experiment.slurm baseline
