import torch
import torch.nn.functional as F

def _normalize(p, eps=1e-8):
    p = torch.clamp(p, min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    return p

def js_divergence(p, q, eps=1e-8):
    p = _normalize(p, eps); q = _normalize(q, eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=-1)
    kl_qm = (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def cosine_similarity(f, g, eps=1e-8):
    f = F.normalize(f, dim=-1, eps=eps)
    g = F.normalize(g, dim=-1, eps=eps)
    return (f * g).sum(dim=-1)

def decision_persistence(yhat_t, yhat_tmL):
    return (yhat_t == yhat_tmL).float()

def margin_entropy_proxy(p):
    # p: [B, C]
    conf, argmax = p.max(dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return 0.5 * (1.0 - conf) + 0.5 * (1.0 - margin)  # m_t in paper

def build_signal_vector(p_t, f_t, yhat_t, buffer, lags, w_lags):
    # Multi-lag JSD
    jsd_vals, stab_vals, persist_vals = [], [], []
    for w, L in zip(w_lags, lags):
        if not buffer.ready(L): continue
        f_L, p_L, y_L = buffer.get(L)
        jsd_vals.append(w * js_divergence(p_t, p_L))
        stab_vals.append(cosine_similarity(f_t, f_L))
        persist_vals.append(decision_persistence(yhat_t, y_L))
    if len(jsd_vals)==0:
        D_t = torch.zeros(p_t.shape[0])
        S_t = torch.ones(p_t.shape[0])
        c_t = torch.ones(p_t.shape[0])
    else:
        D_t = torch.stack(jsd_vals, dim=0).sum(dim=0)
        S_t = torch.stack(stab_vals, dim=0).mean(dim=0)
        c_t = torch.stack(persist_vals, dim=0).mean(dim=0)
    m_t = margin_entropy_proxy(p_t)
    s_vec = torch.stack([D_t, 1.0 - S_t, 1.0 - c_t, m_t], dim=-1)  # [B,4]
    return s_vec
