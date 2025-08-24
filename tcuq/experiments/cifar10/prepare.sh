#!/usr/bin/env bash
python -m tcuq.scripts.train_backbone --config configs/cifar10.yaml
python -m tcuq.scripts.fit_tcuq_head --config configs/tcuq_head.yaml
