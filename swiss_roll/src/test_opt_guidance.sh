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

SCRIPTDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
TMPDIR=${SLURM_TMPDIR:-/tmp}
BATCHCONFIGS=${SCRIPTDIR}/../conf/batched_configs

# Use a test batch tarball
cp ${BATCHCONFIGS}/batch_000.tar.gz ${TMPDIR}/batch_000.tar.gz
mkdir -p ${TMPDIR}/batch_000/conf
echo ${TMPDIR}/batch_000/conf
tar -xzf ${TMPDIR}/batch_000.tar.gz -C ${TMPDIR}/batch_000/conf

echo "Starting run 1"
# Run just one config manually
python ${SCRIPTDIR}/swiss_roll/guidance.py \
    --conf ${TMPDIR}/batch_000/conf/0001.yaml \
    --writeout ${TMPDIR}/batch_000/output/run_0001 \
    --logger ${TMPDIR}/batch_000/output/run_0001/logger_0001

echo "Starting run 2"
python ${SCRIPTDIR}/swiss_roll/guidance.py \
    --conf ${TMPDIR}/batch_000/conf/0002.yaml \
    --writeout ${TMPDIR}/batch_000/output/run_0002 \
    --logger ${TMPDIR}/batch_000/output/run_0002/logger_0002



cp -r $TMPDIR/batch_000/output ${SCRIPTDIR}/../runs/guidance_models/batch_000

rm -r ${TMPDIR}/batch_000
rm -r ${TMPDIR}/batch_000.tar.gz