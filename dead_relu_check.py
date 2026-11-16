"""Why does lr=1.0 get stuck at chance? Count ReLU units that never fire.

Trains the same model as lr_experiment.py for 1 epoch at lr=0.1 and lr=1.0,
then counts channels/units whose ReLU output is 0 for every validation image.
A dead unit outputs 0 and gets 0 gradient, so it cannot recover.

Run:  python dead_relu_check.py
"""
import torch
import torch.nn.functional as F

import lr_experiment as E

E.EPOCHS = 1  # train_one reads this module-level setting


def dead_count(h):
    """'dead/total (pct)' for channels/units whose max activation over all images is 0."""
    dead = (h.transpose(0, 1).flatten(1).amax(1) == 0)
    return f"{int(dead.sum())}/{dead.numel()} ({dead.float().mean().item():.0%})"


@torch.no_grad()
def inspect(model, x):
    model.eval()
    h1 = F.relu(model.conv1(x))
    h2 = F.relu(model.conv2(F.max_pool2d(h1, 2)))
    f1 = F.relu(model.fc1(F.max_pool2d(h2, 2).flatten(1)))
    logits = model.fc2(f1)
    return dead_count(h1), dead_count(h2), dead_count(f1), logits.std(0).mean().item()


def main():
    train, val, _ = E.load_data()
    for lr in [0.1, 1.0]:
        model, _ = E.train_one(lr, train, val)
        c1, c2, fc, spread = inspect(model, val[0])
        print(f"lr={lr}: dead conv1 {c1}, conv2 {c2}, fc1 {fc} | logit spread across images {spread:.4f}")


if __name__ == "__main__":
    main()
