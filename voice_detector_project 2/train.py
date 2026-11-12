"""Train the CNN: python train.py --root /path/to/LA --model resnet"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ASVspoofLA
from evaluate import compute_eer
from model import build_model, count_parameters


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n, all_scores, all_labels = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            all_scores += torch.sigmoid(logits).detach().cpu().tolist()
            all_labels += y.cpu().tolist()
    eer, _ = compute_eer(np.array(all_labels), np.array(all_scores))
    return total / n, eer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="path to the LA folder of ASVspoof 2019")
    p.add_argument("--model", default="resnet", choices=["resnet", "baseline"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)   # repeatable runs
    # CHANGED: uses the Apple GPU ("mps") when there is no NVIDIA GPU
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    train_ds = ASVspoofLA(args.root, "train", augment=True)
    dev_ds = ASVspoofLA(args.root, "dev")
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    dev_dl = DataLoader(dev_ds, args.batch, num_workers=args.workers)

    n_real = sum(1 for _, l, _ in train_ds.items if l == 0)
    n_fake = len(train_ds) - n_real
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_fake).to(device))   # class weights

    model = build_model(args.model).to(device)
    print(f"model={args.model} parameters={count_parameters(model):,} device={device} real={n_real} fake={n_fake}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best, bad, path = float("inf"), 0, os.path.join(args.out, f"{args.model}_best.pt")
    with open(os.path.join(args.out, f"{args.model}_history.csv"), "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_eer", "dev_loss", "dev_eer"])
        for epoch in range(1, args.epochs + 1):
            tl, te = run_epoch(model, train_dl, loss_fn, device, optimizer)
            dl, de = run_epoch(model, dev_dl, loss_fn, device)
            log.writerow([epoch, tl, te, dl, de]); f.flush()
            print(f"epoch {epoch:2d}  train loss {tl:.4f}  dev loss {dl:.4f}  dev EER {de * 100:.2f}%")
            if dl < best:
                best, bad = dl, 0
                torch.save({"model_name": args.model, "state_dict": model.state_dict(), "epoch": epoch,
                            "dev_loss": dl, "dev_eer": de, "args": vars(args)}, path)
            else:
                bad += 1
                if bad >= args.patience:          # early stopping
                    print("early stopping")
                    break


if __name__ == "__main__":
    main()
