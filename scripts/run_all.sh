#!/bin/bash
ENVS=(open_cabinet_door open_cabinet_drawer push_chair turn_faucet)
MISMATCHES=(roll15 roll30 x50 y50 z50 view50)
SEED=0
i=0
for env in "${ENVS[@]}"; do
  for mis in "${MISMATCHES[@]}"; do
    gpu=$((i % 2))
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_sac.py \
      algo=aux_sac model=ppt env="$env" camera_mismatch="$mis" \
      seed=$SEED device="cuda:$gpu" wandb.group_name="pprl_aux_${env}_${mis}" &
    ((i++))
    if (( i % 2 == 0 )); then wait; fi
  done
done
wait
