#!/bin/bash

if (( $# != 2 )); then
  echo "Error, usage: bash run.sh [gpu_id] \"log_filename\" "
  exit
fi

gpu_id=$1
log_filename=$2

echo "gpu_id: $gpu_id"

ulimit -n 4096

CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train_sac.py env=thread_in_hole model=pointgpt_rl algo=aux_sac > "./logs/$log_filename.log" 2>&1 &
