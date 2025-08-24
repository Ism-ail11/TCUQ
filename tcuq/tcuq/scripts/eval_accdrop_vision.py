#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming accuracy-drop (CID) detection for vision datasets.
Two-phase sliding-window evaluation identical to audio script.

Usage (CIFAR-10-C official):
  python -m tcuq.scripts.eval_accdrop_vision --dataset cifar10 \
      --ckpt runs/cifar10/best.pt --data_root data \
      --cifar10c_root data/cifar10-c --window 50

For TinyImageNet-C, pass --tinyc_root to the corruption folder.
If no *-C folder is given, an approximate corruption fallback is used.
"""
import argparse, os, glob
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tcuq.models.resnet8_tiny import ResNet8Tiny
from tcuq.models.mobilenetv2_tiny import MobileNetV2Tiny
from tcuq.core.metrics import average_precision_binary
from tcuq.core.streaming import sliding_mean

def _softmax_pred_conf(logits):
    p = logits.softmax(dim=1)
    c, y = p.max(dim=1)
    return y.cpu().numpy(), c.cpu().numpy().astype(np.float32)

# ----------------- CIFAR-10 loaders -----------------
def _cifar10_id(data_root, batch_size):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914,0.4822,0.4465),
                                                  (0.2023,0.1994,0.2010))])
    test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
    return DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def _iterate_loader_preds(model, device, loader):
    ys, yhat, confs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred, conf = _softmax_pred_conf(logits)
            ys.extend(y.numpy().tolist())
            yhat.extend(pred.tolist())
            confs.extend(conf.tolist())
    return np.array(ys), np.array(yhat), np.array(confs, dtype=np.float32)

def _cifar10c_list(c_root):
    names = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur","motion_blur",
             "zoom_blur","snow","frost","fog","brightness","contrast","elastic_transform",
             "pixelate","jpeg_compression","speckle_noise","spatter","gaussian_blur","saturate"]
    files = [os.path.join(c_root, f"{n}.npy") for n in names]
    ok = [n for n,f in zip(names, files) if os.path.isfile(f)]
    return ok

def _cifar10c_streams(c_root, tf, batch_size, severity):
    # returns a DataLoader producing (x,y) with corruption severity (1..5)
    class CIFAR10C(torch.utils.data.Dataset):
        def __init__(self, c_root, name, severity, tf):
            self.x = np.load(os.path.join(c_root, f"{name}.npy"))[(severity-1)*10000:severity*10000]
            self.y = np.load(os.path.join(c_root, "labels.npy"))
            self.tf = tf
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            img = self.x[i]
            x = self.tf(torch.tensor(img).permute(2,0,1).float()/255.0)
            return x, int(self.y[i])
    def make(name):
        ds = CIFAR10C(c_root, name, severity, tf)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return make

# ----------------- TinyImageNet loaders -----------------
def _tiny_id(data_root, batch_size):
    # expects your TinyImageNetDataset (from your repo)
    from tcuq.data.tinyimagenet import TinyImageNetDataset
    tf = None  # dataset class applies its own normalization
    val = TinyImageNetDataset(root=os.path.join(data_root, "tinyimagenet"), split="val")
    return DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ----------------- Evaluation core -----------------
def _acc_series(y_true, y_pred): return (y_true == y_pred).astype(np.float32)

def _auprc_series(asw, csw, mu_id, sig_id):
    labels = (asw <= (mu_id - 3.0*sig_id)).astype(np.int32)
    scores = 1.0 - csw
    return average_precision_binary(labels, scores)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","tinyimagenet"], required=True)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--data_root", default="data", type=str)
    ap.add_argument("--cifar10c_root", default="", type=str)
    ap.add_argument("--tinyc_root", default="", type=str)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--severity", type=int, default=1)  # evaluate one severity at a time (1..5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "cifar10":
        model = ResNet8Tiny(num_classes=10).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.eval()

        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4914,0.4822,0.4465),
                                                      (0.2023,0.1994,0.2010))])
        # ID stats
        id_loader = _cifar10_id(args.data_root, args.batch_size)
        y, yhat, conf = _iterate_loader_preds(model, device, id_loader)
        asw_id = sliding_mean(_acc_series(y,yhat), args.window)
        mu_id, sig_id = float(np.mean(asw_id)), float(np.std(asw_id) + 1e-8)
        print(f"[ID] mu={mu_id:.4f}  sigma={sig_id:.4f}")

        # CID per corruption
        names = _cifar10c_list(args.cifar10c_root)
        make_dl = _cifar10c_streams(args.cifar10c_root, tf, args.batch_size, args.severity)
        results = []
        for name in names:
            cid_loader = make_dl(name)
            # concatenate ID then CID
            y1,yh1,c1 = _iterate_loader_preds(model, device, id_loader)
            y2,yh2,c2 = _iterate_loader_preds(model, device, cid_loader)
            y = np.concatenate([y1,y2]); yh = np.concatenate([yh1,yh2]); conf = np.concatenate([c1,c2])
            asw = sliding_mean(_acc_series(y,yh), args.window)
            csw = sliding_mean(conf, args.window)
            ap = _auprc_series(asw, csw, mu_id, sig_id)
            results.append(ap)
            print(f"{name:22s} severity={args.severity}  AUPRC={ap:.3f}")
        print(f"\nMean AUPRC over {len(results)} corruptions: {float(np.mean(results)):.3f}")

    else:  # tinyimagenet
        model = MobileNetV2Tiny(num_classes=200).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.eval()

        id_loader = _tiny_id(args.data_root, args.batch_size)
        y, yhat, conf = _iterate_loader_preds(model, device, id_loader)
        asw_id = sliding_mean(_acc_series(y,yhat), args.window)
        mu_id, sig_id = float(np.mean(asw_id)), float(np.std(asw_id) + 1e-8)
        print(f"[ID] mu={mu_id:.4f}  sigma={sig_id:.4f}")

        # For TinyImageNet-C, we assume pre-generated corrupted splits (folder per corruption/severity)
        if not args.tinyc_root or not os.path.isdir(args.tinyc_root):
            raise ValueError("Please provide --tinyc_root pointing to TinyImageNet-C.")
        # Expected layout: tinyc_root/<corruption>/severity-<1..5>/*.png
        corrs = sorted([d for d in os.listdir(args.tinyc_root) if os.path.isdir(os.path.join(args.tinyc_root, d))])
        from PIL import Image
        from torchvision import transforms

        mean,std=(0.485,0.456,0.406),(0.229,0.224,0.225)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

        def make_loader(cname, severity):
            cdir = os.path.join(args.tinyc_root, cname, f"severity-{severity}")
            files = sorted([os.path.join(cdir,f) for f in os.listdir(cdir) if f.lower().endswith((".png",".jpg",".jpeg"))])
            labels_path = os.path.join(cdir, "labels.txt")
            labels = [int(x.strip()) for x in open(labels_path)] if os.path.isfile(labels_path) else [0]*len(files)
            class _DS(torch.utils.data.Dataset):
                def __len__(self): return len(files)
                def __getitem__(self, i):
                    x = tf(Image.open(files[i]).convert("RGB"))
                    return x, labels[i]
            return DataLoader(_DS(), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        results = []
        for cname in corrs:
            loader_c = make_loader(cname, args.severity)
            y1,yh1,c1 = _iterate_loader_preds(model, device, id_loader)
            y2,yh2,c2 = _iterate_loader_preds(model, device, loader_c)
            y = np.concatenate([y1,y2]); yh = np.concatenate([yh1,yh2]); conf = np.concatenate([c1,c2])
            asw = sliding_mean(_acc_series(y,yh), args.window)
            csw = sliding_mean(conf, args.window)
            ap = _auprc_series(asw, csw, mu_id, sig_id)
            results.append(ap)
            print(f"{cname:22s} severity={args.severity}  AUPRC={ap:.3f}")
        print(f"\nMean AUPRC over {len(results)} corruptions: {float(np.mean(results)):.3f}")

if __name__ == "__main__":
    main()
