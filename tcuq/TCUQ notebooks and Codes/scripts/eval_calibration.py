#!/usr/bin/env python3
"""
Compute calibration metrics (F1, Brier Score, NLL, ECE) on ID test set.

Usage:
  python -m tcuq.scripts.eval_calibration \
    --backbone_ckpt outputs/mnist_base/ckpt_best.pt \
    --config configs/mnist.yaml \
    --out outputs/mnist_calib.json
"""
import argparse, json, yaml, torch
from pathlib import Path
import numpy as np
import torch.nn.functional as F

from tcuq.scripts.train_backbone import build_loaders, build_model

def ece_score(probs, labels, n_bins=15):
    # probs: [N,C], labels: [N]
    conf, pred = probs.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (conf > lo) & (conf <= hi)
        if not np.any(sel): continue
        acc = (pred[sel] == labels[sel]).mean()
        avg_conf = conf[sel].mean()
        ece += (sel.mean()) * abs(avg_conf - acc)
    return float(ece)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(args.backbone_ckpt)
    model.to(device).eval()

    _, _, test_loader, _ = build_loaders(cfg)

    y_true, y_pred, y_prob, nll_acc = [], [], [], 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        nll_acc += F.nll_loss(torch.log(probs + 1e-9), y, reduction="sum").item()
        y_true.append(y.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    n = len(y_true)

    # F1 macro
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average="macro")
    # Brier: multiclass (mean prob mass on incorrect classes)
    onehot = np.eye(y_prob.shape[1])[y_true]
    bs = float(((y_prob - onehot) ** 2).sum(axis=1).mean())
    # NLL
    nll = float(nll_acc / n)
    # ECE
    ece = ece_score(y_prob, y_true)

    out = {"F1": float(f1), "BS": bs, "NLL": nll, "ECE": ece, "n": int(n)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
