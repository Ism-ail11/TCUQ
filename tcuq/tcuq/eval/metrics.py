import torch, numpy as np

def ece_score(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) * (confidences <= bins[i+1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece.item()

def brier_score(probs, labels, num_classes):
    onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    return ((probs - onehot) ** 2).sum(dim=1).mean().item()

def nll_score(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels, reduction='mean').item()

def f1_micro(preds, labels, num_classes):
    preds = preds.cpu().numpy(); labels = labels.cpu().numpy()
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average="micro")
