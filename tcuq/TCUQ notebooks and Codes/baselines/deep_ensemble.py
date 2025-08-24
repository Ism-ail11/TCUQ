import torch

class DeepEnsemble:
    def __init__(self, members):
        self.members = members
        for m in self.members: m.eval()
    @torch.no_grad()
    def predict(self, x):
        ps = [torch.softmax(m(x), dim=-1) for m in self.members]
        return torch.stack(ps, dim=0).mean(dim=0)
