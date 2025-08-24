import yaml, os, torch, torch.nn as nn, torch.optim as optim
from tcuq.utils.seed import set_seed
from tcuq.utils.io import ensure_dir, save_ckpt
from tcuq.utils.logging import get_logger
from tcuq.data.mnist import get_mnist_loaders
from tcuq.data.cifar10 import get_cifar10_loaders
from tcuq.models.cnn4_mnist import CNN4_MNIST
from tcuq.models.resnet8_tiny import ResNet8Tiny

def get_data(cfg):
    if cfg["dataset"] == "mnist":
        return get_mnist_loaders(cfg["data_root"], cfg["train"]["batch_size"])
    if cfg["dataset"] == "cifar10":
        return get_cifar10_loaders(cfg["data_root"], cfg["train"]["batch_size"])
    raise ValueError("Dataset not supported in minimal repro.")

def get_model(cfg):
    name = cfg["model"]["name"]
    nc = cfg["model"]["num_classes"]
    if name == "cnn4_mnist": return CNN4_MNIST(nc)
    if name == "resnet8_tiny": return ResNet8Tiny(nc)
    raise ValueError("Model not supported in minimal repro.")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    logger = get_logger("train")
    set_seed(cfg.get("seed", 1337))
    train_loader, test_loader = get_data(cfg)
    model = get_model(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay",0.0))
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=cfg["train"].get("lr_decay", 1.0))
    loss = nn.CrossEntropyLoss()

    for ep in range(cfg["train"]["epochs"]):
        model.train()
        total = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            l = loss(logits, y)
            l.backward()
            opt.step()
            pred = logits.argmax(dim=-1)
            correct += (pred==y).sum().item(); total += y.numel()
        sched.step()
        logger.info(f"Epoch {ep+1}/{cfg['train']['epochs']}: train_acc={correct/total:.3f}")

    ensure_dir(os.path.dirname(cfg["ckpt"]["path"]))
    save_ckpt({"state_dict": model.state_dict(), "cfg": cfg}, cfg["ckpt"]["path"])
    logger.info(f"Saved checkpoint to {cfg['ckpt']['path']}")

if __name__ == "__main__":
    main()
