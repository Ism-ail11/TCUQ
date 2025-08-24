import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: [B, 1, 49, 10]  (mel x time)
# Compact DSCNN (depthwise-separable) for 10-way KWS
class DSCNN_Speech(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # pointwise 1x1 to expand a bit
        self.pw1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        # DW blocks
        self.dw1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=(2,1), padding=1, groups=64, bias=False),
            nn.Conv2d(64, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True)
        )
        self.dw3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=(2,2), padding=1, groups=96, bias=False),
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        x = self.pw1(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        f = self.gap(x).flatten(1)
        out = self.fc(f)
        return (out, f) if return_feat else out
