import torch.nn as nn
from .common import ConvBNAct, flatten

class CNN4_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.f1 = ConvBNAct(1, 32, 3, 1, 1)
        self.f2 = ConvBNAct(32,64, 3, 2, 1)
        self.f3 = ConvBNAct(64,64, 3, 2, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x, return_feat=False):
        x = self.f1(x); x = self.f2(x); x = self.f3(x)
        f = self.gap(x).flatten(1)
        out = self.fc(f)
        if return_feat:
            return out, f
        return out
