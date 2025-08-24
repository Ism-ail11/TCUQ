import torch
from .metrics import ece_score, brier_score, nll_score, f1_micro

def evaluate_calibration(model, loader, num_classes=10, device="cpu"):
    model.eval(); model.to(device)
    all_probs, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(y)
    probs = torch.cat(all_probs); logits = torch.cat(all_logits); labels = torch.cat(all_labels)
    preds = probs.argmax(dim=-1)
    ece = ece_score(probs, labels, n_bins=15)
    bs  = brier_score(probs, labels, num_classes)
    nll = nll_score(logits, labels)
    f1  = f1_micro(preds, labels, num_classes)
    return {"ECE": ece, "BS": bs, "NLL": nll, "F1": f1}
