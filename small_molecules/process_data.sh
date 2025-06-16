#!/bin/bash
#SBATCH --job-name="mood"
#SBATCH --partition=himem
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G

nvidia-smi
nvidia-smi -L

set -e  # fail fully on first line failure
module purge 

source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate mood

module load cuda/12.6.2

python -u data/preprocess.py --dataset "ZINC250k"
python -u data/preprocess.py --dataset "ZINC500k"