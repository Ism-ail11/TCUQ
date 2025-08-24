import torch
from tcuq.streaming.ring_buffer import RingBuffer
from tcuq.streaming.signals import build_signal_vector

def test_signal_shapes():
    B, C, d = 8, 10, 64
    p_t = torch.softmax(torch.randn(B, C), dim=-1)
    f_t = torch.randn(B, d)
    yhat = p_t.argmax(dim=-1)
    buf = RingBuffer(W=4)
    s = build_signal_vector(p_t, f_t, yhat, buf, lags=[1,2], w_lags=torch.tensor([0.6,0.4]))
    assert s.shape == (B, 4)
