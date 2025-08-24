#!/usr/bin/env python3
"""
Plot PR curves / severity curves from JSON metrics (Matplotlib).

Usage:
  python -m tcuq.scripts.plot_curves --inputs outputs/*.json --out figures/curve.png
"""
import argparse, json, glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xs, ys = [], []
    for pat in args.inputs:
        for f in glob.glob(pat):
            with open(f, "r") as fh:
                d = json.load(fh)
            if "recall" in d and "precision" in d:
                xs.append(np.array(d["recall"]))
                ys.append(np.array(d["precision"]))

    if not xs:
        print("No PR curves found in inputs; expecting JSON with keys 'recall' and 'precision'.")
        return

    plt.figure(figsize=(4,3))
    for r, p in zip(xs, ys):
        plt.plot(r, p, lw=1.5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, ls="--", alpha=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
