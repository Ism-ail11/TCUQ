tcuq/

├─ README.md

├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ pyproject.toml
├─ Makefile
├─ Dockerfile
├─ .gitignore
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ configs/
│  ├─ mnist.yaml
│  ├─ cifar10.yaml
│  ├─ speechcmd.yaml           # (stub pipeline note)
│  ├─ tinyimagenet.yaml        # (stub pipeline note)
│  ├─ tcuq_head.yaml
│  ├─ eval_accdrop.yaml
│  ├─ eval_failure.yaml
│  └─ logging.yaml
├─ tcuq/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ seed.py
│  │  ├─ logging.py
│  │  └─ io.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ mnist.py              # working
│  │  ├─ cifar10.py            # working
│  │  ├─ speechcmd.py          # stub (torchaudio hook noted)
│  │  ├─ tinyimagenet.py       # stub (download structure noted)
│  │  ├─ corruptions_vision.py # stub hooks
│  │  └─ corruptions_audio.py  # stub hooks
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ common.py
│  │  ├─ cnn4_mnist.py
│  │  ├─ resnet8_tiny.py
│  │  └─ mobilenetv2_tiny.py   # (minimal placeholder if you want later)
│  ├─ streaming/
│  │  ├─ __init__.py
│  │  ├─ ring_buffer.py
│  │  ├─ signals.py
│  │  ├─ logistic_head.py
│  │  ├─ quantile.py
│  │  ├─ abstain.py
│  │  └─ state.py
│  ├─ baselines/
│  │  ├─ __init__.py
│  │  ├─ mc_dropout.py
│  │  ├─ deep_ensemble.py
│  │  └─ ee_ensemble.py        # placeholder interface
│  ├─ eval/
│  │  ├─ __init__.py
│  │  ├─ metrics.py
│  │  ├─ calibration.py
│  │  ├─ failure_detection.py
│  │  └─ accdrop_detection.py
│  └─ scripts/
│     ├─ train_backbone.py     # MNIST/CIFAR10 end-to-end
│     ├─ fit_tcuq_head.py      # offline head fit over temporal signals
│     ├─ run_stream_eval.py
│     ├─ eval_failure.py
│     ├─ eval_calibration.py
│     ├─ eval_accdrop.py
│     ├─ export_mcu_profile.py
│     └─ plot_curves.py
├─ experiments/
│  ├─ mnist/
│  │  ├─ prepare.sh
│  │  ├─ run_all.sh
│  │  └─ eval_all.sh
│  └─ cifar10/
│     ├─ prepare.sh
│     ├─ run_all.sh
│     └─ eval_all.sh
├─ notebooks/
│  └─ README.md
├─ ckpts/        # .gitkeep
├─ data/         # .gitkeep
├─ outputs/      # .gitkeep
└─ tests/
   ├─ test_quantile.py
   ├─ test_signals.py
   └─ test_state_step.py
