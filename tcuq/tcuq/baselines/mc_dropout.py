
import torch, torch.nn as nn

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()

@torch.no_grad()
def mc_dropout_predict(model, x, passes:int=10):
    model.eval(); enable_dropout(model)
    out=None
    for _ in range(passes):
        p=torch.softmax(model(x), dim=-1)
        out = p if out is None else out+p
    return out/float(passes)
