#!/usr/bin/env bash
set -e
python -m tcuq.scripts.eval_accdrop --cfg configs/eval_accdrop.yaml
python -m tcuq.scripts.eval_failure --cfg configs/eval_failure.yaml
python -m tcuq.scripts.eval_calibration --cfg configs/cifar10.yaml
