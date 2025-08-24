#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Toy streaming demo: shows how to turn U_t into accept/abstain decisions
with streaming conformal (QuantileTracker) and a budget controller.
This script reads precomputed per-step (U_t, conf) from numpy (or simulates).
"""
import argparse, os
import numpy as np
from tcuq.core.conformal import QuantileTracker, Nonconformity, NonconformityConfig, BudgetController

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1, help="target exceedance")
    ap.add_argument("--lam", type=float, default=0.6, help="blend for nonconformity")
    ap.add_argument("--budget", type=float, default=0.1, help="allowed abstention rate")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Simulate U_t and conf with a drift in the middle
    U = np.clip(0.2 + 0.15*rng.standard_normal(args.steps), 0, 1)
    conf = np.clip(0.85 + 0.1*rng.standard_normal(args.steps), 0, 1)
    U[250:] += 0.2; conf[250:] -= 0.2

    q = QuantileTracker(alpha=args.alpha)
    nc = Nonconformity(NonconformityConfig(lam=args.lam))
    bc = BudgetController(budget_b=args.budget)

    abstain_flags = []
    for t in range(args.steps):
        r_t = nc.score(float(U[t]), float(conf[t]))
        q_t = q.update(r_t)
        want_abstain = r_t >= q_t
        abst = bc.decide(want_abstain)
        abstain_flags.append(1 if abst else 0)
        if t % 50 == 0:
            print(f"t={t:03d}  U={U[t]:.2f}  conf={conf[t]:.2f}  r={r_t:.2f}  q={q_t:.2f}  abstain={abst}")

    print(f"Final abstention EWMA≈{bc.rate_ewma:.3f}, empirical={np.mean(abstain_flags):.3f}")

if __name__ == "__main__":
    main()
