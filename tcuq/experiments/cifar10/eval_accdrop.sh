#!/usr/bin/env bash
set -e
python -m tcuq.scripts.eval_accdrop_vision \
  --dataset cifar10 \
  --ckpt runs/cifar10/best.pt \
  --data_root data \
  --cifar10c_root data/cifar10-c \
  --batch_size 128 \
  --window 50 \
  --severity 1
