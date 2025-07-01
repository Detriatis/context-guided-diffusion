#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --array=0-11
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/job_%A_%a.out

module purge
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate swiss_roll

# SLURM_ARRAY_TASK_ID=1

RUN_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
SCRIPTDIR="$SLURM_SUBMIT_DIR"
SCRIPTDIR="./"
CONF=${SCRIPTDIR}../conf/bayesopt_conf/${RUN_ID}.yaml
RESULTDIR=${SCRIPTDIR}../runs/bayesopt/

mkdir -p $RESULTDIR

WRITEOUT=${RESULTDIR}/
LOGGER=${RESULTDIR}/log_${RUN_ID}.log

echo $CONF
echo "Starting run: $RUN_ID"
python ${SCRIPTDIR}/swiss_roll/hyperopt.py \
    --conf "$CONF" \
    --writeout "$WRITEOUT" \
    --logger "$LOGGER"
