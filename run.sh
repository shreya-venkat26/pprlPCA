#!/bin/bash

if (( $# != 1 )); then
  echo "error, need gpu ID"
  exit
fi

gpu_id=$1

echo "gpu_id: $gpu_id"

ulimit -n 4096

CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train_sac.py env=push_chair model=pointgpt_rl algo=aux_sac > train_$(date + %Y%m%d_%H%M%S).log 2>&1 &
