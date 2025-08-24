#!/usr/bin/env bash
set -e
python -m tcuq.scripts.train_backbone --cfg configs/mnist.yaml
python -m tcuq.scripts.fit_tcuq_head --cfg configs/tcuq_head.yaml
python -m tcuq.scripts.eval_calibration --cfg configs/mnist.yaml
