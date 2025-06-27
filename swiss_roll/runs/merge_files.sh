for d in ./guidance_models/*; do
    batchdir=$(basename "$d")
    target_dir="./guidance_models/$batchdir"

    copieddir=$target_dir/$batchdir
    
    for r in $copieddir/*; do
        runfile=$(basename "$r") 
        fromthis=$copieddir/$runfile 
        tothis=$target_dir/$runfile
        cp -r $fromthis/* $tothis
    done
done