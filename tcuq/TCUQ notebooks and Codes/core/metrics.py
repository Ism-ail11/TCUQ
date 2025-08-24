import numpy as np

def average_precision_binary(y_true, y_score):
    """
    AP for binary labels (0/1). Equivalent to sklearn's average_precision_score
    with 'interp(step)' convention.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    # sort by score descending
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    # cumulative TPs at each rank
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(n_pos)
    # AP = sum (r_i - r_{i-1}) * p_i
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)

def auroc_binary(y_true, y_score):
    """
    Fast AUROC via Mann-Whitney U (no sklearn required).
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # ranks of pooled scores
    pooled = np.concatenate([pos, neg])
    order = np.argsort(pooled)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(pooled) + 1)
    r_pos = ranks[:n_pos].sum()
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
