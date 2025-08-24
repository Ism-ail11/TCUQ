import collections, torch

class RingBuffer:
    """Stores last W steps of features (f), posteriors (p), and predictions (yhat)."""
    def __init__(self, W: int):
        self.W = W
        self.f = collections.deque(maxlen=W)
        self.p = collections.deque(maxlen=W)
        self.yhat = collections.deque(maxlen=W)

    def push(self, f_t: torch.Tensor, p_t: torch.Tensor, yhat_t: torch.Tensor):
        self.f.append(f_t.detach().cpu())
        self.p.append(p_t.detach().cpu())
        self.yhat.append(yhat_t.detach().cpu())

    def get(self, lag: int):
        if lag <= 0 or lag > len(self.p): return None
        return self.f[-lag], self.p[-lag], self.yhat[-lag]

    def ready(self, lag: int) -> bool:
        return len(self.p) >= lag
