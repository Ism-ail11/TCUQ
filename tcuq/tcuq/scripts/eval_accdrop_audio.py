#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming accuracy-drop (CID) detection for SpeechCommands.
Implements the sliding-window method from the paper:
 - First pass on ID only -> estimate μ_ID, σ_ID of ASW (sliding accuracy).
 - Second pass on ID+CID -> compute CSW (sliding confidence), use score = 1 - CSW.
 - Event is positive when ASW <= μ_ID - 3 σ_ID; compute AUPRC over time.
Supports per-corruption/per-severity reporting and an average summary.

Usage:
  python -m tcuq.scripts.eval_accdrop_audio --ckpt runs/speechcmd/best.pt \
      --data_root data --batch_size 64 --window 50 \
      --corruptions gaussian_noise air_absorption band_pass band_stop \
      --severities 1 2 3 4 5
"""
import argparse, os, math, time
import numpy as np
import torch, torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from tcuq.models.dscnn_speech import DSCNN_Speech
from tcuq.data.speechcmd import TARGET_CMDS
from tcuq.data.corruptions_audio import apply_audio_corruption
from tcuq.core.metrics import average_precision_binary
from tcuq.core.streaming import sliding_mean

# ---------- helpers to iterate raw SpeechCommands (10-class subset) ----------
def _label_from_path(path: str) -> str:
    return os.path.split(os.path.dirname(path))[-1].lower()

def _iter_sc_wav_subset(root, subset="testing", sample_rate=16000):
    ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)
    for i in range(len(ds)):
        wav, sr, label, _, _ = ds[i]
        label = label.lower()
        if label not in TARGET_CMDS:
            continue
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        # ensure mono 1sec
        if wav.ndim == 2: wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[-1] < sample_rate:
            pad = sample_rate - wav.shape[-1]
            wav = F.pad(wav, (0, pad))
        elif wav.shape[-1] > sample_rate:
            wav = wav[..., :sample_rate]
        y = TARGET_CMDS.index(label)
        yield wav, sample_rate, y

# log-mel to [1,49,10] consistent with training
class _Mel49x10:
    def __init__(self, sample_rate=16000):
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=64
        )
        self.amp2db = torchaudio.transforms.AmplitudeToDB(stype="power")
    @torch.no_grad()
    def __call__(self, wav):
        mel = self.melspec(wav)         # [1,64,T]
        mel = self.amp2db(mel)
        mel = F.adaptive_avg_pool2d(mel, (49, 10))
        return mel  # [1,49,10]

@torch.no_grad()
def _predict_stream(model, device, feats_iter):
    """Iterate (x,y) features -> return arrays of y_true, y_pred, conf (max softmax)."""
    ys, yhat, confs = [], [], []
    for x, y in feats_iter:
        x = x.to(device)
        logits = model(x)
        prob = logits.softmax(dim=1)
        c, pred = prob.max(dim=1)
        ys.append(y)
        yhat.append(pred.item())
        confs.append(c.item())
    return np.array(ys), np.array(yhat), np.array(confs, dtype=np.float32)

def _iter_feats_from_wav(seq_wav_sr_y, featurizer, device):
    for wav, sr, y in seq_wav_sr_y:
        x = featurizer(wav)           # [1,49,10]
        yield x.to(device), y

def _make_id_stream(args, device, featurizer):
    return list(_iter_feats_from_wav(_iter_sc_wav_subset(args.data_root, "testing"), featurizer, device))

def _make_cid_stream(args, device, featurizer, corruption, severity):
    seq = _iter_sc_wav_subset(args.data_root, "testing")
    out = []
    for wav, sr, y in seq:
        wav_np = wav.squeeze(0).cpu().numpy()
        wav_c = apply_audio_corruption(wav_np, sr, corruption, severity)
        wav_c = torch.tensor(wav_c, dtype=torch.float32).unsqueeze(0)
        x = featurizer(wav_c)    # [1,49,10]
        out.append((x.to(device), y))
    return out

def _accuracy_series(y_true, y_pred):
    return (y_true == y_pred).astype(np.float32)

def _auprc_from_series(asw, csw, mu_id, sig_id):
    # positives when accuracy sliding window under μ_ID - 3σ_ID
    thresh = mu_id - 3.0 * sig_id
    labels = (asw <= thresh).astype(np.int32)
    scores = 1.0 - csw  # lower confidence => higher score
    return average_precision_binary(labels, scores)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--corruptions", nargs="+",
                    default=["gaussian_noise","air_absorption","band_pass","band_stop",
                             "high_pass","high_shelf","low_pass","low_shelf",
                             "peaking_filter","tanh_distortion","time_mask","time_stretch"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1,2,3,4,5])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSCNN_Speech(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    featurizer = _Mel49x10()
    # ---------- Pass 1: ID only -> μ_ID, σ_ID of ASW ----------
    id_stream = _make_id_stream(args, device, featurizer)  # list of (x,y)
    y_true, y_pred, conf = _predict_stream(model, device, id_stream)
    acc = _accuracy_series(y_true, y_pred)
    asw_id = sliding_mean(acc, args.window)
    mu_id, sig_id = float(np.mean(asw_id)), float(np.std(asw_id) + 1e-8)

    print(f"[ID stats] window={args.window}  mu_ID={mu_id:.4f}  sigma_ID={sig_id:.4f}")

    # ---------- Pass 2: per corruption/severity ----------
    table = []
    for corr in args.corruptions:
        row = []
        for sev in args.severities:
            cid_stream = _make_cid_stream(args, device, featurizer, corr, sev)
            # concatenate ID + CID
            full = id_stream + cid_stream
            y_t, y_p, conf = _predict_stream(model, device, full)
            acc = _accuracy_series(y_t, y_p)
            asw = sliding_mean(acc, args.window)
            csw = sliding_mean(conf, args.window)
            ap = _auprc_from_series(asw, csw, mu_id, sig_id)
            row.append(ap)
            print(f"{corr:18s} sev={sev}  AUPRC={ap:.3f}")
        table.append((corr, row))

    # summary (mean over severities for each corruption; then mean over corruptions)
    per_corr = [float(np.mean(r)) for _, r in table]
    overall = float(np.mean(per_corr))
    print("\n=== AUPRC (mean over severities) per corruption ===")
    for (corr, row), meanv in zip(table, per_corr):
        print(f"{corr:18s}: {meanv:.3f}   // severities: {', '.join(f'{v:.3f}' for v in row)}")
    print(f"\nOverall mean AUPRC across corruptions: {overall:.3f}")

if __name__ == "__main__":
    main()
