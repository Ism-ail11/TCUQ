TCUQ — Temporal-Consistency Uncertainty for Streaming TinyML

TCUQ is a single-pass, label-free uncertainty monitor for TinyML. It converts short-horizon temporal consistency—captured as lightweight signals on posteriors and features—into a calibrated risk score maintained online with an 
𝑂
(
𝑊
)
O(W) ring buffer and 
𝑂
(
1
)
O(1) updates. A streaming conformal layer turns this score into a budgeted accept/abstain decision for on-device monitoring without extra forward passes.

This repository contains code to:

Train compact backbones on MNIST, CIFAR-10, SpeechCommands, and TinyImageNet.

Fit a tiny logistic TCUQ head from a small dev split.

Run streaming evaluation (failure detection, corrupted-in-distribution accuracy-drop).

Compute calibration metrics (F1, Brier, NLL, ECE).

Profile footprint/latency/energy on desktop as a proxy for MCUs.
