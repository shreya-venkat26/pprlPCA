#!/bin/bash

ulimit -n 4096
CUDA_VISIBLE_DEVICES=0 python scripts/train_sac.py env=thread_in_hole model=pointgpt_rl algo=aux_sac platform=debug
