#!/bin/bash

ulimit -n 4096
python scripts/train_sac.py env=push_chair model=ppt
