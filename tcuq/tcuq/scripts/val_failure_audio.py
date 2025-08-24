#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Failure detection for SpeechCommands:
 - ID✓|ID×: positives = misclassified ID samples.
 - ID✓|OOD: positives = non-target words treated as OOD.
Supports TCUQ uncertainty (temporal signals) or 1 - confidence.
"""
import argparse, os
import numpy as np
import torch, torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from tcuq.models.dscnn_speech import DSCNN_Speech
from tcuq.data.speechcmd import TARGET_CMDS
from tcuq.core.metrics import auroc_binary
from tcuq.core.tcuq_signals import TCUQSignals, SignalConfig

def _mel_49x10(sample_rate=16000):
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=64
    )
    amp2db = torchaudio.transforms.AmplitudeToDB(stype="power")
    def f(wav):
        mel = melspec(wav)           # [1,64,T]
        mel = amp2db(mel)
        mel = F.adaptive_avg_pool2d(mel, (49, 10))
        return mel
    return f

def _iter_subset(root, subset="testing", sample_rate=16000):
    ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=True)
    for i in range(len(ds)):
        wav, sr, label, _, _ = ds[i]
        label = label.lower()
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if wav.ndim == 2: wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[-1] < sample_rate:
            wav = F.pad(wav, (0, sample_rate - wav.shape[-1]))
        elif wav.shape[-1] > sample_rate:
            wav = wav[..., :sample_rate]
        yield wav, label

@torch.no_grad()
def _collect(model, device, feats_fn, seq, targets, use_tcuq=False, tcuq_cfg=None, feat_dim=64*49*10, num_classes=10):
    y_true, scores, preds = [], [], []
    if use_tcuq:
        tcuq = TCUQSignals(tcuq_cfg, feat_dim=feat_dim, num_classes=num_classes)
    for wav, label in seq:
        y = (TARGET_CMDS.index(label) if label in TARGET_CMDS else -1)
        x = feats_fn(wav)  # [1,49,10]
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        conf = float(prob.max())
        pred = int(prob.argmax())
        if use_tcuq:
            # flatten mel as a proxy feature (or replace with model feature if available)
            feat = x.cpu().numpy().reshape(-1).astype(np.float32)
            out = tcuq.step(prob, feat)
            scores.append(out["U_t"])
        else:
            scores.append(1.0 - conf)
        y_true.append(y)
        preds.append(pred)
    return np.array(y_true), np.array(preds), np.array(scores, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--use_tcuq", action="store_true")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--lags", nargs="+", type=int, default=[1,2,4])
    ap.add_argument("--w_mix", nargs="+", type=float, default=[0.5,0.3,0.2])
    ap.add_argument("--alpha_margin", type=float, default=0.5)
    ap.add_argument("--head_path", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSCNN_Speech(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    feats_fn = _mel_49x10()
    # ID stream: only target words
    id_seq = [(w,l) for (w,l) in _iter_subset(args.data_root, "testing") if l.lower() in TARGET_CMDS]
    # OOD stream: non-target words as OOD
    ood_seq = [(w,l) for (w,l) in _iter_subset(args.data_root, "testing") if l.lower() not in TARGET_CMDS]

    tcuq_cfg = SignalConfig(window=args.window, lags=tuple(args.lags),
                            w_mixture=tuple(args.w_mix), alpha_margin=args.alpha_margin,
                            use_features=True, head_path=args.head_path)
    # ---- ID✓|ID× ----
    y, pred, s_id = _collect(model, device, feats_fn, id_seq, TARGET_CMDS,
                             use_tcuq=args.use_tcuq, tcuq_cfg=tcuq_cfg,
                             feat_dim=49*10, num_classes=10)
    labels_err = (y != pred).astype(np.int32)
    au_iderr = auroc_binary(labels_err, s_id)
    print(f"[ID✓|ID×] AUROC = {au_iderr:.3f}")

    # ---- ID✓|OOD ----
    _, _, s_id2 = _collect(model, device, feats_fn, id_seq, TARGET_CMDS,
                           use_tcuq=args.use_tcuq, tcuq_cfg=tcuq_cfg,
                           feat_dim=49*10, num_classes=10)
    _, _, s_ood = _collect(model, device, feats_fn, ood_seq, TARGET_CMDS,
                           use_tcuq=args.use_tcuq, tcuq_cfg=tcuq_cfg,
                           feat_dim=49*10, num_classes=10)
    y_bin = np.concatenate([np.zeros_like(s_id2, dtype=np.int32),
                            np.ones_like(s_ood, dtype=np.int32)])
    s_all = np.concatenate([s_id2, s_ood]).astype(np.float32)
    au_idood = auroc_binary(y_bin, s_all)
    print(f"[ID✓|OOD] AUROC = {au_idood:.3f}")

if __name__ == "__main__":
    main()
