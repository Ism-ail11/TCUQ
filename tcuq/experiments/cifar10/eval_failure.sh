#!/usr/bin/env bash
set -e
python -m tcuq.scripts.eval_failure_vision \
  --dataset cifar10 \
  --ckpt runs/cifar10/best.pt \
  --data_root data \
  --use_tcuq \
  --window 32 \
  --lags 1 2 4 \
  --w_mix 0.5 0.3 0.2 \
  --alpha_margin 0.5 \
  --head_path runs/cifar10/tcuq_head.json \
  --ood_dataset cifar100
