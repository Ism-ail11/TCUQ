#!/usr/bin/env python3
"""
Accuracy-drop (CID) detection via sliding-window confidence & accuracy.

Usage:
  python -m tcuq.scripts.eval_accdrop \
    --backbone_ckpt outputs/cifar_base/ckpt_best.pt \
    --config configs/eval_accdrop.yaml \
    --out outputs/cifar_accdrop.json

Config expects:
  dataset: {name, root, num_workers}
  accdrop:
    window: int (m)
    thresholds: [0.5, 0.55, ..., 0.95]  # confidence thresholds
"""
import argparse, json, yaml, torch, numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

from tcuq.scripts.train_backbone import build_loaders, build_model

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

    # Build ID loader and a CID loader according to your corruption builder (here we reuse test as stream)
    _, _, stream_loader, _ = build_loaders(cfg)

    m = int(cfg["accdrop"].get("window", 50))
    conf_hist, acc_hist = [], []

    y_true, conf_seq, acc_seq = [], [], []
    for x, y in stream_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); post = torch.softmax(logits, dim=-1)
        pred = post.argmax(dim=1)
        conf = post.max(dim=1).values
        for i in range(x.size(0)):
            conf_hist.append(float(conf[i]))
            acc_hist.append(1.0 if int(pred[i]) == int(y[i]) else 0.0)
            # sliding averages
            c_sw = float(np.mean(conf_hist[-m:]))
            a_sw = float(np.mean(acc_hist[-m:]))
            conf_seq.append(c_sw)
            acc_seq.append(a_sw)
            y_true.append(1 if a_sw <= 0.0 else 0)  # harsh drop indicator (toy); replace with μ_ID - 3σ_ID if you pre-compute

    scores = 1.0 - np.array(conf_seq)
    y_true = np.array(y_true)
    P, R, _ = precision_recall_curve(y_true, scores)
    auprc = auc(R, P)

    out = {"AUPRC": float(auprc), "n": int(len(y_true))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
