#!/bin/bash
conda activate splat_the_net

python3 train_mlp_hb_max.py \
                -s data/360_v2/bicycle \
                --eval \
                --num_cams -1 \
                -m output/bicycle/ \
                --resolution 4 \

python3 train_mlp_hb_max.py \
                -s data/360_v2/bonsai \
                --eval \
                --num_cams -1 \
                -m output/bonsai/ \
                --resolution 2 \


python3 train_mlp_hb_max.py \
                -s data/360_v2/counter \
                --eval \
                --num_cams -1 \
                -m output/counter/ \
                --resolution 2 \

python3 train_mlp_hb_max.py \
                -s data/db/drjohnson \
                --eval \
                --num_cams -1 \
                -m output/drjohnson/ \
                --resolution 1 \


python3 train_mlp_hb_max.py \
                -s data/360_v2/flowers \
                --eval \
                --num_cams -1 \
                -m output/flowers/ \
                --resolution 4 \

python3 train_mlp_hb_max.py \
                -s data/360_v2/garden \
                --eval \
                --num_cams -1 \
                -m output/garden/ \
                --resolution 4 \

python3 train_mlp_hb_max.py \
                -s data/360_v2/kitchen \
                --eval \
                --num_cams -1 \
                -m output/kitchen/ \
                --resolution 2 \

 
python3 train_mlp_hb_max.py \
                -s data/db/playroom \
                --eval \
                --num_cams -1 \
                -m output/playroom/ \
                --resolution 1 \

python3 train_mlp_hb_max.py \
                -s data/360_v2/room \
                --eval \
                --num_cams -1 \
                -m output/room/ \
                --resolution 2 \

python3 train_mlp_hb_max.py \
                -s data/360_v2/stump \
                --eval \
                --num_cams -1 \
                -m output/stump/ \
                --resolution 4 \


python3 train_mlp_hb_max.py \
                -s data/tandt/train \
                --eval \
                --num_cams -1 \
                -m output/train/ \
                --resolution 1 \

python3 train_mlp_hb_max.py \
                -s data/360_v2/treehill \
                --eval \
                --num_cams -1 \
                -m output/treehill/ \
                --resolution 4 \


python3 train_mlp_hb_max.py \
                -s data/tandt/truck \
                --eval \
                --num_cams -1 \
                -m output/truck/ \
                --resolution 1 \
    