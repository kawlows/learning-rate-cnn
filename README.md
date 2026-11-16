# Learning rate experiment: small CNN on Fashion-MNIST

Code behind the blog post *How Learning Rate Shapes the Training Curve of a Small CNN*.

Trains one small CNN with plain SGD at four learning rates (0.001, 0.01, 0.1, 1.0).
Data, initial weights, batch order, batch size (64), epochs (20) and seed (42) are the
same for every run. Uses 10,000 training and 5,000 validation images from the
Fashion-MNIST training set; the test set is used once, for the learning rate picked on validation.

## Run

```bash
pip install torch torchvision matplotlib
python lr_experiment.py
```

Runs on a laptop CPU in a few minutes. Writes `results.json` and `figures/*.png`.
Fashion-MNIST downloads to `data/` on first run.

## Why lr = 1.0 gets stuck at chance

```bash
python dead_relu_check.py
```

Trains 1 epoch at lr = 0.1 and lr = 1.0 and counts ReLU units that output 0 for every
validation image (dead units get no gradient and cannot recover). Expected output:

```
lr=0.1: dead conv1 0/16 (0%), conv2 0/32 (0%), fc1 0/64 (0%) | logit spread across images 2.6069
lr=1.0: dead conv1 0/16 (0%), conv2 27/32 (84%), fc1 62/64 (97%) | logit spread across images 0.0000
```
