#!/usr/bin/env python3
"""
Approximate model footprint & latency; optionally estimate energy.

Usage:
  python -m tcuq.scripts.export_mcu_profile \
    --backbone_ckpt outputs/speech_base/ckpt_best.pt \
    --out outputs/profile_speech.json \
    --power_mw 120
"""
import argparse, json, time, yaml, torch
from pathlib import Path
import numpy as np

from tcuq.scripts.train_backbone import build_model

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--power_mw", type=float, default=120.0,
                    help="Assumed board power (mW) to get Energy=Power*Latency")
    ap.add_argument("--input_shape", type=str, default="", help="Override e.g. 1,1,28,28")
    args = ap.parse_args()

    model, cfg = build_model(args.backbone_ckpt)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    size_bytes = params * 4  # fp32

    if args.input_shape:
        ishape = tuple(map(int, args.input_shape.split(",")))
    else:
        ds = cfg["dataset"]["name"].lower()
        if ds == "mnist":
            ishape = (1, 1, 28, 28)
        elif ds == "cifar10":
            ishape = (1, 3, 32, 32)
        elif ds == "tinyimagenet":
            ishape = (1, 3, 64, 64)
        elif ds == "speechcmd":
            ishape = (1, 1, 49, 10)  # mel-spec as HxW
        else:
            ishape = (1, 3, 32, 32)
    x = torch.randn(*ishape)

    # Warmup
    for _ in range(10):
        _ = model(x)

    # Timing
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) / iters * 1000.0
    energy_mj = (args.power_mw / 1000.0) * (latency_ms / 1000.0)

    out = {"params": int(params),
           "size_bytes": int(size_bytes),
           "latency_ms_cpu": float(latency_ms),
           "energy_mj_cpu": float(energy_mj),
           "assumed_power_mw": float(args.power_mw),
           "input_shape": ishape,
           "dataset": cfg["dataset"]["name"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
