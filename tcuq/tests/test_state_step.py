import torch
from tcuq.streaming.state import TCUQMonitor
from tcuq.streaming.logistic_head import LogisticHead

def test_state_step():
    B, C, d = 4, 10, 64
    head = LogisticHead(4)
    mon = TCUQMonitor(head, W=4, lags=(1,2), w_lags=(0.6,0.4))
    f = torch.randn(B, d)
    p = torch.softmax(torch.randn(B, C), dim=-1)
    y = p.argmax(dim=-1)
    U, r, q, abst = mon.step(f, p, y)
    assert U.shape == (B,)
