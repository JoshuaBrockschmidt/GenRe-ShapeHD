#!/usr/bin/env bash

# Evaluates GenRe against our validation split of the ShapeNet dataset.

out_dir='./output/eval'
fullmodel='./downloads/models/full_model.pt'


if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu[ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments

set -e

source activate shaperecon

python 'eval.py' \
    --net genre_full_model \
    --net_file "$fullmodel" \
    --input_rgb "$rgb_pattern" \
    --input_mask "$mask_pattern" \
    --output_dir "$out_dir" \
    --suffix '{net}' \
    --overwrite \
    --workers 0 \
    --batch_size 1 \
    --vis_workers 4 \
    --logdir "$out_dir" \
    --gpu "$gpu" \
    $*

source deactivate
