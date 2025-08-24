import torch
from sklearn.metrics import roc_auc_score

def auroc_id_correct_incorrect(model, loader, device="cpu"):
    model.eval(); model.to(device)
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values
            pred = probs.argmax(dim=-1)
            correct = (pred == y).float()
            y_true.extend(correct.cpu().numpy().tolist())
            y_score.extend(conf.cpu().numpy().tolist())
    return roc_auc_score(y_true, y_score)

def auroc_id_ood(model, id_loader, ood_loader, device="cpu"):
    model.eval(); model.to(device)
    scores, labels = [], []
    with torch.no_grad():
        for x, _ in id_loader:
            x = x.to(device)
            conf = torch.softmax(model(x), dim=-1).max(dim=-1).values
            scores.extend(conf.cpu().numpy().tolist())
            labels.extend([1]*conf.shape[0])
        for x, _ in ood_loader:
            x = x.to(device)
            conf = torch.softmax(model(x), dim=-1).max(dim=-1).values
            scores.extend(conf.cpu().numpy().tolist())
            labels.extend([0]*conf.shape[0])
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)
