# -*- coding: utf-8 -*-
"""
TCUQ temporal signals and small ring buffer.
Computes:
 - D_t : multi-lag Jensen-Shannon divergence over posteriors
 - S_t : feature stability (cosine similarity; uses 1 - S_t)
 - c_t : decision persistence; uses 1 - c_t
 - m_t : blended proxy (1 - confidence) vs (1 - margin)
Aggregates via logistic: U_t = σ(w^T s + b).

Weights (w,b) can be loaded from a JSON path; if missing, defaults are used.
"""
from dataclasses import dataclass
import json
import math
import numpy as np

EPS = 1e-8

def _safe_prob(p: np.ndarray, eps=1e-6):
    p = np.clip(p, eps, 1.0)
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0 / len(p))
    return p / s

def jsd(p: np.ndarray, q: np.ndarray, eps=1e-6) -> float:
    p = _safe_prob(p, eps); q = _safe_prob(q, eps)
    m = 0.5 * (p + q)
    def _kl(a,b):
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))
    return 0.5 * (_kl(p, m) + _kl(q, m))

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u) + EPS)
    nv = float(np.linalg.norm(v) + EPS)
    return float(np.dot(u, v) / (nu * nv))

@dataclass
class SignalConfig:
    window: int = 32
    lags: tuple = (1, 2, 4)
    w_mixture: tuple = (0.5, 0.3, 0.2)  # weights for JSD over lags
    alpha_margin: float = 0.5          # blend for proxy m_t
    use_features: bool = True          # if False, skips S_t (sets S_t=1)
    head_path: str = ""                # optional path to JSON with {"w":[...], "b":...}

class RingBuffer:
    def __init__(self, W: int, feat_dim: int = 0, num_classes: int = 10):
        self.W = int(W)
        self.post = np.zeros((W, num_classes), dtype=np.float32)
        self.pred = -np.ones(W, dtype=np.int32)
        self.conf = np.zeros(W, dtype=np.float32)
        self.idx = 0
        self.count = 0
        self.has_feat = feat_dim > 0
        self.feat = np.zeros((W, feat_dim), dtype=np.float32) if self.has_feat else None

    def push(self, prob: np.ndarray, pred: int, conf: float, feat: np.ndarray | None):
        j = self.idx % self.W
        self.post[j] = prob
        self.pred[j] = int(pred)
        self.conf[j] = float(conf)
        if self.has_feat and feat is not None:
            self.feat[j] = feat.astype(np.float32, copy=False)
        self.idx += 1
        self.count = min(self.count + 1, self.W)

    def ready(self) -> bool:
        return self.count >= 2

    def get(self, lag: int):
        """Return (prob, pred, conf, feat) at t-lag; assumes lag < count."""
        j = (self.idx - 1 - lag) % self.W
        return self.post[j], self.pred[j], self.conf[j], (self.feat[j] if self.has_feat else None)

class TCUQSignals:
    def __init__(self, cfg: SignalConfig, feat_dim: int, num_classes: int):
        self.cfg = cfg
        self.R = RingBuffer(cfg.window, feat_dim=feat_dim if cfg.use_features else 0,
                            num_classes=num_classes)
        # head weights (w,b) for logistic aggregation
        if cfg.head_path:
            try:
                obj = json.load(open(cfg.head_path, "r"))
                self.w = np.array(obj.get("w", [0.9, 0.5, 0.5, 0.9]), dtype=np.float32)
                self.b = float(obj.get("b", 0.0))
            except Exception:
                self.w = np.array([0.9, 0.5, 0.5, 0.9], dtype=np.float32)
                self.b = 0.0
        else:
            self.w = np.array([0.9, 0.5, 0.5, 0.9], dtype=np.float32)
            self.b = 0.0

    def step(self, prob_t: np.ndarray, feat_t: np.ndarray | None) -> dict:
        """
        Update buffers with current prob/feat, then compute signals & U_t.
        Returns dict with keys: U_t, D_t, S_t, c_t, m_t, conf, pred.
        """
        prob_t = _safe_prob(prob_t)
        pred_t = int(np.argmax(prob_t))
        conf_t = float(np.max(prob_t))
        # push AFTER grabbing past values (so 't-ℓ' refers to previous indices)
        # but we need current features in signals; so we push then read
        self.R.push(prob_t, pred_t, conf_t, feat_t)

        # If not enough history, return a safe low U_t
        if self.R.count <= max(self.cfg.lags) + 1:
            return dict(U_t=0.0, D_t=0.0, S_t=1.0, c_t=1.0, m_t=1.0,
                        conf=conf_t, pred=pred_t)

        # ---- 1) Multi-lag JSD ----
        D_vals = []
        for w, lag in zip(self.cfg.w_mixture, self.cfg.lags):
            p_lag, _, _, _ = self.R.get(lag)
            D_vals.append(float(w) * jsd(prob_t, p_lag))
        D_t = float(sum(D_vals))

        # ---- 2) Feature stability S_t (cosine) ----
        if self.cfg.use_features and self.R.has_feat:
            S_vals = []
            for lag in self.cfg.lags:
                _, _, _, f_lag = self.R.get(lag)
                S_vals.append(cosine(feat_t, f_lag))
            S_t = float(np.mean(S_vals))
        else:
            S_t = 1.0  # neutral (no penalty)

        # ---- 3) Decision persistence c_t ----
        c_vals = []
        for lag in self.cfg.lags:
            _, ylag, _, _ = self.R.get(lag)
            c_vals.append(1.0 if ylag == pred_t else 0.0)
        c_t = float(np.mean(c_vals))

        # ---- 4) Proxy margin / confidence ----
        # margin = p1 - p2
        p_sorted = np.sort(prob_t)[::-1]
        margin = float(p_sorted[0] - p_sorted[1] if len(p_sorted) > 1 else p_sorted[0])
        alpha = float(self.cfg.alpha_margin)
        m_t = alpha * (1.0 - conf_t) + (1.0 - alpha) * (1.0 - margin)

        # ---- Aggregate (logistic) ----
        s = np.array([D_t, (1.0 - S_t), (1.0 - c_t), m_t], dtype=np.float32)
        z = float(np.dot(self.w, s) + self.b)
        U_t = 1.0 / (1.0 + math.exp(-z))

        return dict(U_t=U_t, D_t=D_t, S_t=S_t, c_t=c_t, m_t=m_t,
                    conf=conf_t, pred=pred_t)
