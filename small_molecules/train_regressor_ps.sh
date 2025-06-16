#!/bin/bash
#SBATCH --job-name="mood"
#SBATCH --partition=himem
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G

nvidia-smi
nvidia-smi -L

set -e  # fail fully on first line failure
module purge 

source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate mood
module load cuda/12.6.2

for i in {0..8}
do
    echo "Running job $i"
    python -u main.py --type train --config prop_train_ps/prop_train_2_${i}
done

for j in {0..2}
do
    echo "Running sample $j"
    python -u main.py --type retrain_best --config prop_train_ps/sample_2_${j}
done