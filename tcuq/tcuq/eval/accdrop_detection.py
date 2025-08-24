import torch, random
from sklearn.metrics import average_precision_score
from ..data.corruptions_vision import apply_corruption

def stream_mixer(loader, corruption_types, severities, id_fraction=0.5):
    for x, y in loader:
        B = x.size(0)
        mask_id = torch.rand(B) < id_fraction
        x_out = x.clone()
        # corrupt non-ID
        for i in range(B):
            if not mask_id[i]:
                c = random.choice(corruption_types)
                s = random.choice(severities)
                x_out[i] = apply_corruption(x_out[i], c, s)
        yield x_out, y, mask_id

def eval_accuracy_drop_auprc(model, loader, device="cpu",
                             corruption_types=("gaussian_noise","motion_blur","fog","jpeg","brightness"),
                             severities=(1,2,3,4,5), window=50):
    model.eval(); model.to(device)
    probs_hist, acc_hist = [], []
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y, mask_id in stream_mixer(loader, corruption_types, severities):
            x, y = x.to(device), y.to(device)
            probs = torch.softmax(model(x), dim=-1)
            conf  = probs.max(dim=-1).values
            pred  = probs.argmax(dim=-1)
            correct = (pred == y).float()
            probs_hist.extend(conf.cpu().tolist())
            acc_hist.extend(correct.cpu().tolist())
            if len(probs_hist) >= window:
                csw = sum(probs_hist[-window:]) / window
                asw = sum(acc_hist[-window:]) / window
                y_score.append(1.0 - csw)   # lower confidence => positive for drop
                # Label "drop" if below mean-3sigma computed online (rough proxy)
                arr = torch.tensor(acc_hist[-window:])
                mu, sd = arr.mean().item(), arr.std().item()
                y_true.append(1 if asw <= (mu - 3*sd) else 0)
    if len(set(y_true)) < 2:  # edge case
        return 0.5
    return average_precision_score(y_true, y_score)
