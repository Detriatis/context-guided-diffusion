#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --array=2-52
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=10:00:00
#SBATCH --output=../logs/job_%A_%a.out

module purge
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate swiss_roll

# SCRIPTDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
# SLURM_ARRAY_TASK_ID=1

BATCH_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
SCRIPTDIR="$SLURM_SUBMIT_DIR"
TMPDIR=${SLURM_TMPDIR:-/tmp}
BATCHCONFIGS=${SCRIPTDIR}/../conf/batched_configs
BATCH_TARBALL=${BATCHCONFIGS}/batch_${BATCH_ID}.tar.gz

echo $TMPDIR

cp $BATCH_TARBALL $TMPDIR
mkdir -p $TMPDIR/batch_$BATCH_ID/conf
tar -xzf $TMPDIR/batch_${BATCH_ID}.tar.gz -C $TMPDIR/batch_$BATCH_ID/conf

for CONF in ${TMPDIR}/batch_${BATCH_ID}/conf/*.yaml; do
    RUN_ID=$(basename "$CONF" .yaml)
    WRITEOUT="$TMPDIR/batch_$BATCH_ID/run_${RUN_ID}"
    LOGGER="${SCRIPTDIR}/../runs/guidance_models/batch_$BATCH_ID/run_${RUN_ID}/logger"
    
    mkdir -p "$WRITEOUT"

    echo "Starting run: $RUN_ID"
    python ${SCRIPTDIR}/swiss_roll/guidance.py \
        --conf "$CONF" \
        --writeout "$WRITEOUT" \
        --logger "$LOGGER"
done

cp -r $TMPDIR/batch_$BATCH_ID/* ${SCRIPTDIR}/../runs/guidance_models/batch_${BATCH_ID}

rm -r ${TMPDIR}/batch_${BATCH_ID}
rm -r ${TMPDIR}/batch_${BATCH_ID}.tar.gz
