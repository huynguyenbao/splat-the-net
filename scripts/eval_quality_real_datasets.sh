#!/bin/bash
conda activate splat_the_net

# Set number of neurons
neurons=8

# List of scene categories
scenes=(
  bicycle bonsai counter flowers garden kitchen room stump
  treehill train truck playroom drjohnson
)

# Loop through each scene
for scene in "${scenes[@]}"; do
    base_path="output/$scene"
    # Check if scene directory exists
    if [ -d "$base_path" ]; then
        # Loop through each subfolder in the scene directory
        for subfolder in "$base_path"/*; do
            if [ -d "$subfolder" ]; then
                echo "Processing: $subfolder"

                python3 render.py \
                --model_path "$subfolder" \
                --iteration best \
                --skip_train \
                --render_backend inria_cuda_mlp \
                --n_neurons "$neurons"

                python3 metrics.py \
                --model_path "$subfolder"
            fi
        done
    fi
done

python3 merge_results.py
