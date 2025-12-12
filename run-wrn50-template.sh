#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: $0 <only-a|only-b|...>}"
GPU="${2:-1}"

datapath=/workspace/AnomalyDetection/paper-revision/dataset/SimpleNet-Dataset-100

python3 -u main.py \
  --gpu "${GPU}" \
  --seed 0 \
  --log_group simplenet_StitchingNet \
  --log_project StitchingNet_Results \
  --results_path results \
  --run_name "wrn50_${DATASET}_gan2" \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 4000 \
    --embedding_size 256 \
    --gan_epochs 2 \
    --noise_std 0.015 \
    --dsc_hidden 1024 \
    --dsc_layers 2 \
    --dsc_margin .5 \
    --pre_proj 1 \
  dataset \
    --batch_size 4 \
    --resize 224 \
    --imagesize 224 \
    -d "${DATASET}" \
    StitchingNet "${datapath}"
