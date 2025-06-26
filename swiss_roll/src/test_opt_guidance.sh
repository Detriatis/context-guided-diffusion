module purge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate swiss_roll

BATCHCONFIGS=/mnt/iusers01/mace01/e56218md/scratch/context-guided-diffusion/swiss_roll/conf/batched_configs
# Use a test batch tarball
cp ${BATCHCONFIGS}/batch_000.tar.gz /tmp
mkdir /tmp/conf
tar -xzf /tmp/batch_000.tar.gz -C /tmp/conf

# Run just one config manually
python swiss_roll/guidance.py \
    --conf /tmp/conf/0001.yaml \
    --writeout /tmp/output/run_0001
