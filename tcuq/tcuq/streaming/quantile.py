# P^2 algorithm for online quantile tracking (approximate)
class P2Quantile:
    def __init__(self, q=0.9):
        self.q = q
        self.n = 0
        self.buffer = []

    def update(self, x):
        # Very small, robust fallback: keep small buffer for demo.
        # For production, implement full P^2 marker updates.
        self.buffer.append(float(x))
        if len(self.buffer) > 2048:
            self.buffer.pop(0)
        self.n += 1

    def value(self):
        if not self.buffer: return 0.0
        s = sorted(self.buffer)
        k = int(self.q * (len(s)-1))
        return s[k]
