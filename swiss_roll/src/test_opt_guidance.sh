#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=serial
#SBATCH --mem=1G
#SBATCH --time=1:00:00
#SBATCH --output=../logs/job_%A_%a.out

module purge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate swiss_roll

TMPDIR=${SLURM_TMPDIR:-/tmp}

BATCHCONFIGS=/mnt/iusers01/mace01/e56218md/scratch/context-guided-diffusion/swiss_roll/conf/batched_configs
# Use a test batch tarball
cp ${BATCHCONFIGS}/batch_000.tar.gz $TMPDIR
mkdir -p ${TMPDIR}/batch_000/conf
tar -xzf ${TMPDIR}/batch_000.tar.gz -C ${TMPDIR}/batch_000/conf

echo "Starting run 1"
# Run just one config manually
python swiss_roll/guidance.py \
    --conf ${TMPDIR}/batch_000/conf/0001.yaml \
    --writeout ${TMPDIR}/batch_000/output/run_0001

echo "Starting run 2"
python swiss_roll/guidance.py \
    --conf ${TMPDIR}/batch_000/conf/0002.yaml \
    --writeout ${TMPDIR}/batch_000/output/run_0002



cp -r $TMPDIR/batch_000/output /mnt/iusers01/mace01/e56218md/scratch/context-guided-diffusion/swiss_roll/runs/guidance_models/batch_000
rm -rf ${TMPDIR}/batch_000
rm -rf ${TMPDIR}/batch_000.tar.gz