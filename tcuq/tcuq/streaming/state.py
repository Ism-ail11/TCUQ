import torch
from .ring_buffer import RingBuffer
from .signals import build_signal_vector
from .logistic_head import LogisticHead
from .quantile import P2Quantile
from .abstain import BudgetController

class TCUQMonitor:
    def __init__(self, head: LogisticHead, W=16, lags=(1,2,4), w_lags=(0.5,0.3,0.2),
                 alpha=0.5, q_alpha=0.1, warmup=256, budget=0.1, device="cpu"):
        self.device = device
        self.buf = RingBuffer(W)
        self.head = head.to(device)
        self.lags = list(lags); self.w_lags = torch.tensor(w_lags, dtype=torch.float)
        self.alpha = alpha
        self.quant = P2Quantile(q=1.0 - q_alpha)
        self.warmup = warmup
        self.ctrl = BudgetController(budget)

    @torch.no_grad()
    def step(self, f_t, p_t, yhat_t):
        # compute signals
        s_t = build_signal_vector(p_t, f_t, yhat_t, self.buf, self.lags, self.w_lags)
        U_t = self.head(s_t.to(self.head.w.weight.device)).cpu()
        conf = p_t.max(dim=-1).values.cpu()
        r_t = self.alpha * U_t + (1 - self.alpha) * (1 - conf)

        # update quantile (warmup conservative)
        self.quant.update(r_t.mean().item())
        q = self.quant.value() if self.quant.n > self.warmup else 1.0  # conservative high threshold

        abstain = False
        if r_t.mean().item() >= q and self.ctrl.allow():
            abstain = True
        self.ctrl.update(abstain)

        # push to buffer
        self.buf.push(f_t.cpu(), p_t.cpu(), yhat_t.cpu())
        return U_t, r_t, q, abstain
