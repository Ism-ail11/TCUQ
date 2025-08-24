import yaml, torch, os
from tcuq.utils.io import load_ckpt, ensure_dir, append_jsonl
from tcuq.utils.logging import get_logger
from tcuq.data.cifar10 import get_cifar10_loaders
from tcuq.models.resnet8_tiny import ResNet8Tiny
from tcuq.streaming.state import TCUQMonitor
from tcuq.streaming.logistic_head import LogisticHead

def main():
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    logger = get_logger("stream_eval")
    ensure_dir(cfg["output_dir"])

    # backbone
    bck = load_ckpt(cfg["ckpt"]["backbone"])
    model = ResNet8Tiny(bck["cfg"]["model"]["num_classes"])
    model.load_state_dict(bck["state_dict"]); model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)

    # head
    h = load_ckpt(cfg["ckpt"]["head"])
    head = LogisticHead(4); head.load_state_dict(h["state_dict"]); head.eval().to(device)

    # data (use CIFAR-10 test as base stream)
    _, loader = get_cifar10_loaders(cfg["data_root"], batch_size=128)

    mon = TCUQMonitor(head,
                      W=cfg["temporal"]["W"],
                      lags=tuple(cfg["temporal"]["lags"]),
                      w_lags=tuple(cfg["temporal"]["w_lags"]),
                      alpha=cfg["temporal"]["alpha"],
                      q_alpha=cfg["temporal"]["q_alpha"],
                      warmup=cfg["temporal"]["warmup"],
                      budget=0.1,
                      device=device)

    from tcuq.eval.accdrop_detection import eval_accuracy_drop_auprc
    auprc = eval_accuracy_drop_auprc(model, loader, device=device)
    append_jsonl(os.path.join(cfg["output_dir"], "metrics.jsonl"), {"AUPRC": auprc})
    logger.info(f"AUPRC={auprc:.3f}")

if __name__ == "__main__":
    main()
