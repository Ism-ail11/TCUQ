# Minimal tiny MobileNetV2-like backbone for TinyImageNet (scaffold)
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2Tiny(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, groups=16, bias=False),
            nn.Conv2d(16, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24), nn.ReLU6(inplace=True),
            nn.Conv2d(24, 24, 3, 2, 1, groups=24, bias=False),
            nn.Conv2d(24, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x, return_feat=False):
        x = self.features(x)
        f = self.gap(x).flatten(1)
        out = self.fc(f)
        return (out, f) if return_feat else out
