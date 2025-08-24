<!-- ====================================================================== -->
<!--                         T C U Q   (TinyML)                              -->
<!-- ====================================================================== -->
<div align="center">

████████╗ ██████╗██╗ ██╗ ██████╗
╚══██╔══╝██╔════╝██║ ██║██╔═══██╗
██║ ██║ ██║ ██║██║ ██║ T C U Q
██║ ██║ ██║ ██║██║ ██║ Temporal-Consistency Uncertainty
██║ ╚██████╗╚██████╔╝╚██████╔╝ with Streaming Conformal Calibration
╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ for TinyML (single pass)


**Single-pass, label-free, streaming uncertainty for TinyML**

</div>

---

## Table of Contents
- [Overview](#overview)
- [Key Ideas](#key-ideas)
- [Repo Layout](#repo-layout)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Datasets](#datasets)
- [Configurations](#configurations)
- [Typical Workflows](#typical-workflows)
- [Reproducing Tables & Figures](#reproducing-tables--figures)
- [Profiling on Desktop (MCU Proxy)](#profiling-on-desktop-mcu-proxy)
- [Extending the Codebase](#extending-the-codebase)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Makefile & Docker Shortcuts](#makefile--docker-shortcuts)
- [Testing](#testing)
- [License](#license)

---

## Overview
**TCUQ** converts **short-horizon temporal consistency** (from features and posteriors) into a **calibrated, streaming risk** signal using:
- a tiny \(O(W)\) **ring buffer**,  
- \(O(1)\) **per-step updates**,  
- and a **streaming conformal** layer for **budgeted accept/abstain** decisions.

**No extra forward passes**, no multi-exit averaging, and no labels at deployment. Designed for **kilobyte-scale MCUs**; runs on desktop for training/evaluation and to proxy footprint/latency/energy.

---

## Key Ideas
- **Single-pass** inference (no MC-Dropout, no deep ensembles at runtime).
- **Temporal signals**: multi-lag posterior divergence, feature stability, decision persistence, and a confidence proxy (margin/confidence blend).
- **Logistic TCUQ head**: learned once offline on a small dev set; then frozen for deployment.
- **Streaming calibration**: online quantile tracker -> calibrated threshold -> **accept/abstain** under a rate budget.
- **Tiny state**: \(W\) last steps; constant-time updates.

---

## Repo Layout
tcuq/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ pyproject.toml
├─ Makefile
├─ Dockerfile
├─ .gitignore
├─ .github/workflows/ci.yml
├─ configs/
│ ├─ mnist.yaml # backbone training
│ ├─ cifar10.yaml
│ ├─ speechcmd.yaml
│ ├─ tinyimagenet.yaml
│ ├─ tcuq_head.yaml # fit TCUQ head (W, lags, alpha, lambda)
│ ├─ eval_accdrop.yaml # streaming accuracy-drop detection
│ ├─ eval_failure.yaml # failure detection (ID✓|ID×, ID✓|OOD proxy)
│ └─ logging.yaml
├─ tcuq/
│ ├─ init.py
│ ├─ utils/ # seeds, logging, I/O helpers
│ ├─ data/ # loaders + corruptions (vision/audio)
│ ├─ models/ # compact backbones (CNN4, ResNet-8, MobileNetV2)
│ ├─ streaming/ # ring buffer, signals, quantile, abstain policy
│ ├─ baselines/ # MC Dropout, Deep Ensemble, EE-ensemble (opt.)
│ ├─ eval/ # metrics + evaluation routines
│ └─ scripts/ # CLI entry points (training/eval/export/plots)
├─ experiments/
│ ├─ mnist/ {prepare.sh, run_all.sh, eval_all.sh}
│ └─ cifar10/ {prepare.sh, run_all.sh, eval_all.sh}
├─ notebooks/ # optional exploratory notebooks
├─ ckpts/ (git-ignored)
├─ data/ (git-ignored)
├─ outputs/ (git-ignored)
└─ tests/ # minimal unit tests



---

## Quickstart
```bash
# 0) Environment (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt    # or: pip install -e .

# 1) Train a compact backbone (MNIST example)
python -m tcuq.scripts.train_backbone \
  --config configs/mnist.yaml \
  --outdir outputs/mnist_base

# 2) Fit the TCUQ head on a small dev split
python -m tcuq.scripts.fit_tcuq_head \
  --backbone_ckpt outputs/mnist_base/ckpt_best.pt \
  --config configs/tcuq_head.yaml \
  --out outputs/mnist_head.pt

# 3) Streamed evaluation (e.g., accuracy-drop AUPRC on corruptions)
python -m tcuq.scripts.run_stream_eval \
  --backbone_ckpt outputs/mnist_base/ckpt_best.pt \
  --tcuq_head outputs/mnist_head.pt \
  --config configs/eval_accdrop.yaml \
  --out outputs/mnist_stream_eval.json


Installation

Local

Installation
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# optional: pip install -e .

