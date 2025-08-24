#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Failure detection on vision:
 - ID✓|ID×: AUROC where positives = misclassifications on ID.
 - ID✓|OOD : AUROC where positives = OOD samples.
Supports TCUQ uncertainty (temporal signals) or 1 - confidence.

OOD options for CIFAR-10:
  --ood_dataset cifar100 | svhn | folder (with images + labels.txt)

Usage:
  python -m tcuq.scripts.eval_failure_vision --dataset cifar10 \
    --ckpt runs/cifar10/best.pt --data_root data \
    --use_tcuq --window 32 --lags 1 2 4 --alpha_margin 0.5
"""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tcuq.models.resnet8_tiny import ResNet8Tiny
from tcuq.models.mobilenetv2_tiny import MobileNetV2Tiny
from tcuq.core.metrics import auroc_binary
from tcuq.core.tcuq_signals import TCUQSignals, SignalConfig

def _softmax(x): return torch.softmax(x, dim=1)

def _get_features(model, x):
    # Preferred: model.forward_features(x). Fallback: last hidden via hook.
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    # fallback: use penultimate activations if available; else zeros
    with torch.no_grad():
        f = model.avgpool(model.features(x)) if hasattr(model, "features") else None
        if f is None:
            return torch.zeros((x.size(0), 128), device=x.device)
        return torch.flatten(f, 1)

def _id_loader(dataset, data_root, batch_size):
    if dataset == "cifar10":
        tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914,0.4822,0.4465),
                                                     (0.2023,0.1994,0.2010))])
        ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == "tinyimagenet":
        from tcuq.data.tinyimagenet import TinyImageNetDataset
        ds = TinyImageNetDataset(root=os.path.join(data_root,"tinyimagenet"), split="val")
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        raise ValueError("Unsupported dataset.")

def _ood_loader(args, dataset, batch_size):
    name = args.ood_dataset.lower()
    if dataset == "cifar10":
        if name == "cifar100":
            tf = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071,0.4867,0.4408),
                                                         (0.2675,0.2565,0.2761))])
            ds = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tf)
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        elif name == "svhn":
            tf = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4377,0.4438,0.4728),
                                                         (0.1980,0.2010,0.1970))])
            ds = datasets.SVHN(root=args.data_root, split="test", download=True, transform=tf)
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        elif name == "folder":
            from torchvision.datasets import ImageFolder
            tf = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914,0.4822,0.4465),
                                                          (0.2023,0.1994,0.2010))])
            ds = ImageFolder(root=args.ood_folder, transform=tf)
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == "tinyimagenet":
        if name != "folder":
            raise ValueError("For TinyImageNet, set --ood_dataset folder and --ood_folder path.")
        from torchvision.datasets import ImageFolder
        tf = transforms.Compose([transforms.Resize((64,64)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485,0.456,0.406),
                                                      (0.229,0.224,0.225))])
        ds = ImageFolder(root=args.ood_folder, transform=tf)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    raise ValueError("Unsupported OOD setting.")

@torch.no_grad()
def _collect_scores_ID(model, device, loader, use_tcuq=False, tcuq_cfg=None, feat_dim=128, num_classes=10):
    y_true, y_pred, scores = [], [], []
    if use_tcuq:
        tcuq = TCUQSignals(tcuq_cfg, feat_dim=feat_dim, num_classes=num_classes)
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = _softmax(logits).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        conf = prob.max(axis=1)
        if use_tcuq:
            feats = _get_features(model, x).cpu().numpy()
            for i in range(len(y)):
                out = tcuq.step(prob[i], feats[i])
                scores.append(out["U_t"])
        else:
            scores.extend(list(1.0 - conf))  # uncertainty
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred); scores = np.array(scores, dtype=np.float32)
    labels_err = (y_true != y_pred).astype(np.int32)  # positives = misclassified
    return labels_err, scores

@torch.no_grad()
def _collect_scores_OOD(model, device, id_loader, ood_loader, use_tcuq=False, tcuq_cfg=None, feat_dim=128, num_classes=10):
    # concatenate ID then OOD; positives are OOD (1)
    scores, labels = [], []
    if use_tcuq:
        tcuq = TCUQSignals(tcuq_cfg, feat_dim=feat_dim, num_classes=num_classes)
    # ID (label 0)
    for x, _ in id_loader:
        x = x.to(device)
        logits = model(x)
        prob = _softmax(logits).cpu().numpy()
        conf = prob.max(axis=1)
        if use_tcuq:
            feats = _get_features(model, x).cpu().numpy()
            for i in range(len(conf)):
                out = tcuq.step(prob[i], feats[i])
                scores.append(out["U_t"])
        else:
            scores.extend(list(1.0 - conf))
        labels.extend([0] * len(conf))
    # OOD (label 1)
    for x, _ in ood_loader:
        x = x.to(device)
        logits = model(x)
        prob = _softmax(logits).cpu().numpy()
        conf = prob.max(axis=1)
        if use_tcuq:
            feats = _get_features(model, x).cpu().numpy()
            for i in range(len(conf)):
                out = tcuq.step(prob[i], feats[i])
                scores.append(out["U_t"])
        else:
            scores.extend(list(1.0 - conf))
        labels.extend([1] * len(conf))
    return np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","tinyimagenet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--use_tcuq", action="store_true")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--lags", nargs="+", type=int, default=[1,2,4])
    ap.add_argument("--w_mix", nargs="+", type=float, default=[0.5,0.3,0.2])
    ap.add_argument("--alpha_margin", type=float, default=0.5)
    ap.add_argument("--head_path", type=str, default="")
    # OOD:
    ap.add_argument("--ood_dataset", type=str, default="cifar100")
    ap.add_argument("--ood_folder", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "cifar10":
        model = ResNet8Tiny(num_classes=10).to(device)
        feat_dim, num_classes = 128, 10
    else:
        model = MobileNetV2Tiny(num_classes=200).to(device)
        feat_dim, num_classes = 1280, 200  # typical penultimate dims

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    id_loader = _id_loader(args.dataset, args.data_root, args.batch_size)
    tcuq_cfg = SignalConfig(window=args.window, lags=tuple(args.lags),
                            w_mixture=tuple(args.w_mix), alpha_margin=args.alpha_margin,
                            use_features=True, head_path=args.head_path)

    # ---- ID✓|ID× ----
    y_err, scores_err = _collect_scores_ID(model, device, id_loader,
                                           use_tcuq=args.use_tcuq,
                                           tcuq_cfg=tcuq_cfg,
                                           feat_dim=feat_dim,
                                           num_classes=num_classes)
    auroc_err = auroc_binary(y_err, scores_err)
    print(f"[ID✓|ID×] AUROC = {auroc_err:.3f}")

    # ---- ID✓|OOD ----
    ood_loader = _ood_loader(args, args.dataset, args.batch_size)
    y_ood, scores_ood = _collect_scores_OOD(model, device, id_loader, ood_loader,
                                            use_tcuq=args.use_tcuq,
                                            tcuq_cfg=tcuq_cfg,
                                            feat_dim=feat_dim,
                                            num_classes=num_classes)
    auroc_ood = auroc_binary(y_ood, scores_ood)
    print(f"[ID✓|OOD] AUROC = {auroc_ood:.3f}")

if __name__ == "__main__":
    main()
