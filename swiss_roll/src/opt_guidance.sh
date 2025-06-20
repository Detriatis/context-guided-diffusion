#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=serial
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/job_%A_%a.out
#SBATCH --error=../logs/job_%A_%a.err

set -e  # fail fully on first line failure
module purge
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate swiss_roll

INDEX=$SLURM_ARRAY_TASK_ID

python swiss_roll/guidance.py --index ${INDEX}