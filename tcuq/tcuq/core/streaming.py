import numpy as np

def sliding_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple causal sliding mean (uniform window), right-aligned at each step."""
    arr = np.asarray(arr, dtype=np.float32)
    if window <= 1 or window >= len(arr):
        w = min(max(1, window), len(arr))
        csum = np.cumsum(arr, dtype=np.float64)
        out = csum / np.arange(1, len(arr)+1)
        return out.astype(np.float32)
    pad = np.pad(arr, (window-1, 0), mode="constant", constant_values=0)
    csum = np.cumsum(pad, dtype=np.float64)
    out = (csum[window:] - csum[:-window]) / float(window)
    return out.astype(np.float32)
