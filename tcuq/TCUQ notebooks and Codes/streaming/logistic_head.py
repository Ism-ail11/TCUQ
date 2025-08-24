import torch, torch.nn as nn

class LogisticHead(nn.Module):
    def __init__(self, in_dim=4, init="zero"):
        super().__init__()
        self.w = nn.Linear(in_dim, 1, bias=True)
        if init == "zero":
            nn.init.zeros_(self.w.weight)
            nn.init.zeros_(self.w.bias)

    def forward(self, s_t):
        # s_t: [B,4] -> U_t in [0,1]
        return torch.sigmoid(self.w(s_t)).squeeze(-1)
