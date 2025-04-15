#!/bin/bash

ulimit -n 4096
nohup python scripts/train_sac.py env=push_chair model=ppt > output.log 2>&1 &
