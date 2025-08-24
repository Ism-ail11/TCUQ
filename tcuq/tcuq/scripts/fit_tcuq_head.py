import yaml, os, torch
from torch.utils.data import DataLoader
from tcuq.utils.io import load_ckpt, save_ckpt, ensure_dir
from tcuq.utils.logging import get_logger
from tcuq.models.resnet8_tiny import ResNet8Tiny
from tcuq.models.cnn4_mnist import CNN4_MNIST
from tcuq.data.cifar10 import get_cifar10_loaders
from tcuq.data.mnist import get_mnist_loaders
from tcuq.streaming.logistic_head import LogisticHead
from tcuq.streaming.ring_buffer import RingBuffer
from tcuq.streaming.signals import build_signal_vector
from tcuq.utils.seed import set_seed

def main():
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    set_seed(cfg.get("seed",1337))
    logger = get_logger("fit_head")

    # small dev mix: use part of CIFAR-10 test + on-the-fly corruption
    if cfg["data_mix"]["dataset"] == "cifar10":
        _, loader = get_cifar10_loaders("data", batch_size=128)
        num = cfg["data_mix"]["num_samples"]
    else:
        _, loader = get_mnist_loaders("data", batch_size=256)
        num = cfg["data_mix"]["num_samples"]

    ckpt = load_ckpt(cfg["ckpt_in"]["backbone"])
    model_name = ckpt["cfg"]["model"]["name"]
    num_classes = ckpt["cfg"]["model"]["num_classes"]
    if model_name == "resnet8_tiny": model = ResNet8Tiny(num_classes)
    else: model = CNN4_MNIST(num_classes)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)

    head = LogisticHead(in_dim=4, init=cfg["head"]["init"])
    head.train(); head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=cfg["head"]["l2"])

    buf = RingBuffer(cfg["temporal"]["W"])
    lags, w_lags = cfg["temporal"]["lags"], torch.tensor(cfg["temporal"]["w_lags"], dtype=torch.float)

    X, y = [], []
    with torch.no_grad():
        cnt = 0
        for x, yy in loader:
            x, yy = x.to(device), yy.to(device)
            logits, feats = model(x, return_feat=True)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            svec = build_signal_vector(probs, feats, preds, buf, lags, w_lags)
            # pseudo-target: 1 if misclassified else 0
            target = (preds != yy).float().unsqueeze(-1)
            X.append(svec); y.append(target)
            for i in range(x.size(0)):
                buf.push(feats[i].cpu(), probs[i].cpu(), preds[i].cpu())
            cnt += x.size(0)
            if cnt >= num: break

    X = torch.cat(X, dim=0); y = torch.cat(y, dim=0).to(device)
    for it in range(cfg["head"]["max_iter"]):
        opt.zero_grad()
        out = head(X.to(device)).unsqueeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward(); opt.step()

    ensure_dir(os.path.dirname(cfg["ckpt_out"]["head"]))
    save_ckpt({"state_dict": head.state_dict(), "cfg": cfg}, cfg["ckpt_out"]["head"])
    logger.info(f"Saved TCUQ head to {cfg['ckpt_out']['head']}")

if __name__ == "__main__":
    main()
