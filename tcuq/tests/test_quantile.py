from tcuq.streaming.quantile import P2Quantile
import random

def test_quantile_basic():
    q = P2Quantile(q=0.9)
    xs = [random.random() for _ in range(1000)]
    for v in xs: q.update(v)
    est = q.value()
    assert 0.7 <= est <= 1.0
