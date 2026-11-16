"""How learning rate shapes the training curve of a small CNN.

Trains the same small CNN on a Fashion-MNIST subset with plain SGD at four
learning rates. Everything else (data, initial weights, batch order, epochs)
is identical across runs. Writes results.json and figures/*.png.

Run:  python lr_experiment.py
"""
import copy
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

# ---- setup (the only thing that changes between runs is LR) ----
LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
N_TRAIN, N_VAL = 10_000, 5_000   # subset of the 60k training images, to keep CPU runs short
MEAN, STD = 0.2860, 0.3530       # Fashion-MNIST pixel mean/std

OUT = Path(__file__).parent
(OUT / "figures").mkdir(exist_ok=True)


def to_tensors(ds):
    x = (ds.data.float() / 255 - MEAN) / STD
    return x.unsqueeze(1), ds.targets


def load_data():
    train_full = FashionMNIST(OUT / "data", train=True, download=True)
    test = FashionMNIST(OUT / "data", train=False, download=True)
    x, y = to_tensors(train_full)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(SEED))
    tr, va = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_VAL]
    assert len(set(tr.tolist()) & set(va.tolist())) == 0, "train/val overlap"
    return (x[tr], y[tr]), (x[va], y[va]), to_tensors(test)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = torch.cat([model(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train_one(lr, train, val):
    torch.manual_seed(SEED)                       # same initial weights for every LR
    model = SmallCNN()
    opt = torch.optim.SGD(model.parameters(), lr=lr)   # plain SGD: w <- w - lr * grad
    order = torch.Generator().manual_seed(SEED)   # same batch order for every LR
    x, y = train
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_state, hist["best_epoch"] = None, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x), generator=order)
        tot_loss = tot_correct = 0.0
        for i in range(0, len(x), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            logits = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(1) == y[idx]).sum().item()
        vl, va = evaluate(model, *val)
        hist["train_loss"].append(tot_loss / len(x))
        hist["train_acc"].append(tot_correct / len(x))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        if va > max(hist["val_acc"][:-1], default=-1):   # keep the best-validation checkpoint
            best_state, hist["best_epoch"] = copy.deepcopy(model.state_dict()), epoch + 1
        print(f"lr={lr:<6} epoch {epoch + 1:2d}  train_loss={hist['train_loss'][-1]:.4f}  "
              f"val_loss={vl:.4f}  val_acc={va:.4f}")
    model.load_state_dict(best_state)
    return model, hist


def nan_to_none(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def plot(results):
    epochs = range(1, EPOCHS + 1)
    colors = dict(zip(LEARNING_RATES, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))

    # Fig 1: training loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in LEARNING_RATES:
        ax.plot(epochs, results[lr]["train_loss"], marker="o", ms=3, color=colors[lr], label=f"lr = {lr}")
    ax.axhline(math.log(10), color="gray", ls=":", lw=1, label="chance (ln 10)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (cross-entropy)")
    ax.set_title("Training loss per epoch for four learning rates")
    ax.legend(title="Learning rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig1_train_loss.png", dpi=150)

    # Fig 2: validation accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in LEARNING_RATES:
        ax.plot(epochs, [100 * a for a in results[lr]["val_acc"]], marker="o", ms=3, color=colors[lr], label=f"lr = {lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy per epoch for four learning rates")
    ax.legend(title="Learning rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig2_val_accuracy.png", dpi=150)

    # Fig 3: train vs validation loss, one panel per LR
    fig, axes = plt.subplots(1, len(LEARNING_RATES), figsize=(14, 3.6), sharex=True)
    for ax, lr in zip(axes, LEARNING_RATES):
        ax.plot(epochs, results[lr]["train_loss"], color=colors[lr], label="training")
        ax.plot(epochs, results[lr]["val_loss"], color=colors[lr], ls="--", label="validation")
        ax.set_title(f"lr = {lr}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Loss (cross-entropy)")
    fig.suptitle("Training vs. validation loss for each learning rate")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig3_train_vs_val.png", dpi=150)


def main():
    train, val, test = load_data()
    results, models = {}, {}
    for lr in LEARNING_RATES:
        t0 = time.time()
        models[lr], results[lr] = train_one(lr, train, val)
        results[lr]["seconds"] = round(time.time() - t0, 1)

    # Pick the LR by best validation accuracy (its best-epoch checkpoint);
    # touch the test set only once, for that checkpoint.
    best = max(LEARNING_RATES, key=lambda lr: max(results[lr]["val_acc"]))
    test_loss, test_acc = evaluate(models[best], *test)
    print(f"\nChosen by validation: lr={best} (epoch {results[best]['best_epoch']})  test_acc={test_acc:.4f}")

    plot(results)
    summary = {
        "setup": dict(seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, n_train=N_TRAIN, n_val=N_VAL,
                      optimizer="SGD (no momentum, no weight decay)", model_params=sum(p.numel() for p in SmallCNN().parameters()),
                      torch=torch.__version__),
        "runs": {str(lr): {k: [nan_to_none(v) for v in vals] if isinstance(vals, list) else vals
                           for k, vals in r.items()} for lr, r in results.items()},
        "chosen_lr": best, "test_acc_of_chosen": test_acc, "test_loss_of_chosen": test_loss,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
