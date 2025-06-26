rm -r batched_configs
mkdir batched_configs
find guidance_conf -name "*.yaml" | sort | split -a 3 -l 20 -d - batched_configs/batch_

cd batched_configs
for f in batch_*; do
  mkdir $f.dir
  xargs -a $f -I{} cp ../{} $f.dir/
  tar -czf ${f}.tar.gz -C $f.dir .
  rm -r $f.dir
done