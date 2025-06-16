#!/bin/bash
#SBATCH --job-name="mood"
#SBATCH --partition=himem
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G

nvidia-smi
nvidia-smi -L

set -e  # fail fully on first line failure

module load Anaconda3/2022.10
module load CUDA/11.3.1 
conda activate mood

for i in {0..57}
do
    echo "Running training $i"
    python -u main.py --type train --config prop_train_fseb/prop_train_2_${i}
done

for j in {0..2}
do
    echo "Running sample $j"
    python -u main.py --type retrain_best --config prop_train_fseb/sample_2_${j}
done
