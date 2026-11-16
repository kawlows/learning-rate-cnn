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
