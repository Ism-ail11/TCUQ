# -*- coding: utf-8 -*-
"""
Streaming conformal components for TCUQ:
 - QuantileTracker: online (label-free) (1-α)-quantile via stochastic approx.
 - Nonconformity score wrapper (lambda blend).
 - BudgetController: simple rate-limiter for abstentions.
"""
from dataclasses import dataclass
import math

class QuantileTracker:
    """
    Tracks the (1-α) quantile online with O(1) state.
    Update rule (stochastic approx):
        q_{t+1} = q_t + η_t * (I[x_t > q_t] - α)
    where η_t is a diminishing step (or small constant).
    """
    def __init__(self, alpha: float = 0.1, eta0: float = 0.05, min_eta: float = 1e-3):
        assert 0.0 < alpha < 1.0
        self.alpha = alpha
        self.q = 0.0
        self.t = 0
        self.eta0 = eta0
        self.min_eta = min_eta
        self._init = True  # whether we have seen any data

    def update(self, x: float) -> float:
        self.t += 1
        if self._init:
            self.q = float(x)
            self._init = False
            return self.q
        eta = max(self.min_eta, self.eta0 / math.sqrt(self.t))
        step = (1.0 if x > self.q else 0.0) - self.alpha
        self.q += eta * step
        return self.q

    def current(self) -> float:
        return float(self.q)

@dataclass
class NonconformityConfig:
    lam: float = 0.6  # blend between U_t and (1 - confidence)

class Nonconformity:
    def __init__(self, cfg: NonconformityConfig):
        self.lam = float(cfg.lam)

    def score(self, U_t: float, conf_t: float) -> float:
        # r_t = lambda*U_t + (1-lambda)*(1 - conf)
        return float(self.lam * U_t + (1.0 - self.lam) * (1.0 - conf_t))

class BudgetController:
    """
    Simple rate controller to respect an abstention budget b in the long run.
    Maintains an EWMA of the abstention rate and disallows abstain when rate>b.
    """
    def __init__(self, budget_b: float = 0.10, beta: float = 0.98):
        assert 0.0 <= budget_b < 1.0
        self.b = float(budget_b)
        self.beta = float(beta)
        self.rate_ewma = 0.0

    def decide(self, want_abstain: bool) -> bool:
        """
        Returns True if we will abstain, False otherwise.
        Suppresses abstain when ewma already exceeds budget b.
        """
        if want_abstain and self.rate_ewma <= self.b:
            did_abstain = True
        else:
            did_abstain = False
        # update EWMA after the decision
        y = 1.0 if did_abstain else 0.0
        self.rate_ewma = self.beta * self.rate_ewma + (1.0 - self.beta) * y
        return did_abstain
