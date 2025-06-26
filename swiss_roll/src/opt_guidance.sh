#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --array=0-158
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=serial
#SBATCH --mem=2G
#SBATCH --time=10:00:00
#SBATCH --output=../logs/job_%A_%a.out

module purge
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate swiss_roll

TMPDIR=${SLURM_TMPDIR:-/tmp}
BATCH_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

BATCHCONFIGS=/mnt/iusers01/mace01/e56218md/scratch/context-guided-diffusion/swiss_roll/conf/batched_configs

BATCH_TARBALL=${BATCHCONFIGS}/batch_${BATCH_ID}.tar.gz
echo $BATCH_TARBALL

cp $BATCH_TARBALL $TMPDIR
mkdir -p $TMPDIR/conf
tar -xzf $TMPDIR/batch_${BATCH_ID}.tar.gz -C $TMPDIR/conf

for CONF in $TMPDIR/conf/*.yaml; do
    RUN_ID=$(basename "$CONF" .yaml)
    echo "Starting run: $RUN_ID"
    WRITEOUT="$TMPDIR/output/run_${RUN_ID}"
    mkdir -p "$WRITEOUT"

    python swiss_roll/guidance.py \
        --conf "$CONF" \
        --writeout "$WRITEOUT"
done

cp -r $TMPDIR/output /mnt/iusers01/mace01/e56218md/scratch/context-guided-diffusion/swiss_roll/runs/guidance_models/batch_${BATCH_ID}
