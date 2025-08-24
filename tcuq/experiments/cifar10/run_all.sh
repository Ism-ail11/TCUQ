#!/usr/bin/env bash
set -e
python -m tcuq.scripts.train_backbone --cfg configs/cifar10.yaml
python -m tcuq.scripts.fit_tcuq_head --cfg configs/tcuq_head.yaml
python -m tcuq.scripts.eval_accdrop --cfg configs/eval_accdrop.yaml
python -m tcuq.scripts.eval_failure --cfg configs/eval_failure.yaml
python -m tcuq.scripts.eval_calibration --cfg configs/cifar10.yaml
