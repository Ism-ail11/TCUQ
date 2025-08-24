#!/usr/bin/env bash
set -e
python -m tcuq.scripts.eval_accdrop_audio \
  --ckpt runs/speechcmd/best.pt \
  --data_root data \
  --batch_size 64 \
  --window 50 \
  --corruptions gaussian_noise air_absorption band_pass band_stop high_pass high_shelf low_pass low_shelf peaking_filter tanh_distortion time_mask time_stretch \
  --severities 1 2 3 4 5
