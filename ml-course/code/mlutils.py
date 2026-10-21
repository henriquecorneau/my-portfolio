"""Small helpers shared by every lesson script.

Each lesson is a plain Python script:

    python 04_logistic_regression.py            # opens the figures in a window
    python 04_logistic_regression.py --save out # writes the figures to out/ instead

The helpers only deal with seeds, downloading data and drawing figures, so the
machine learning code in the lessons stays readable.
"""
import argparse
import random
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"            # small files shipped with the course
CACHE = DATA / "cache"          # bigger datasets, downloaded on first use

BLUE, RED, GREEN, ORANGE, GRAY = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#6b7280"
CLASS_COLORS = [BLUE, RED, GREEN, ORANGE, "#9333ea", "#0891b2", "#ca8a04", "#db2777", GRAY, "#4d7c0f"]

_save_dir = None
_prefix = ""


# --8<-- [start:seed]
def seed_everything(seed=0):
    """Same seed for Python, NumPy and PyTorch, so every run gives the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
# --8<-- [end:seed]


def setup(lesson, seed=0):
    """Parse the command line and fix the seeds. Call once at the top of a lesson script."""
    global _save_dir, _prefix
    parser = argparse.ArgumentParser(description=f"Lesson {lesson}")
    parser.add_argument("--save", metavar="DIR", help="save figures to DIR instead of showing them")
    args = parser.parse_args()
    _prefix = lesson
    if args.save:
        matplotlib.use("Agg")
        _save_dir = Path(args.save)
        _save_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10})


def path(name):
    """Absolute path of a file inside the data/ folder."""
    return DATA / name


def fetch(url, name):
    """Download url to data/cache/name once; afterwards just return the local path."""
    out = CACHE / name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
    return out


def load_mnist(fashion=False):
    """MNIST or Fashion-MNIST as NumPy arrays: images (N, 28, 28) uint8, labels (N,) int64."""
    from torchvision import datasets

    cls = datasets.FashionMNIST if fashion else datasets.MNIST
    out = []
    for train in (True, False):
        ds = cls(str(CACHE), train=train, download=True)
        out += [ds.data.numpy(), ds.targets.numpy()]
    return out                                     # x_train, y_train, x_test, y_test


FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def figure(cols=1, rows=1, size=(4.2, 3.2)):
    """Create a figure with a grid of axes (always 2-D: axes[row, col])."""
    import matplotlib.pyplot as plt

    return plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows), squeeze=False)


def finish(name, fig):
    """Save the figure (with --save) or show it."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if _save_dir is not None:
        out = _save_dir / f"{_prefix}-{name}.webp"
        fig.savefig(out, dpi=110, pil_kwargs={"quality": 82})
        print(f"saved {out}")
        plt.close(fig)
    else:
        plt.show()


# --8<-- [start:decision]
def plot_decision(ax, predict, X, y, title=None, resolution=300, proba=False, margin=0.3, lims=None):
    """Colour the plane by the class `predict` gives to each point, then draw the samples.

    predict: function (n, 2) array -> (n,) labels, or probabilities of class 1 if proba=True.
    lims: optional (x0, x1, y0, y1); by default the data range plus a margin.
    """
    from matplotlib.colors import ListedColormap

    x0, x1 = X[:, 0].min() - margin, X[:, 0].max() + margin
    y0, y1 = X[:, 1].min() - margin, X[:, 1].max() + margin
    if lims is not None:
        x0, x1, y0, y1 = lims
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]                  # (resolution², 2): every pixel is a sample
    z = np.asarray(predict(grid)).reshape(xx.shape)
    classes = np.unique(y)
    colors = CLASS_COLORS[:len(classes)]
    if proba:
        ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
        ax.contour(xx, yy, z, levels=[0.5], colors="k", linewidths=1)
    else:
        ax.contourf(xx, yy, z, levels=np.arange(len(classes) + 1) - 0.5,
                    cmap=ListedColormap(colors), alpha=0.18)
        ax.contour(xx, yy, z, levels=np.arange(len(classes)) + 0.5, colors="k", linewidths=0.6)
    for c, color in zip(classes, colors):
        ax.scatter(*X[y == c].T, s=10, color=color, edgecolor="white", linewidth=0.3, label=f"class {c}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.grid(False)
    if title:
        ax.set_title(title)
# --8<-- [end:decision]


# --8<-- [start:train]
def train_classifier(model, train_loader, X_val, y_val, epochs, lr=1e-3, weight_decay=0.0, device="cpu",
                     cosine=True, verbose=True):
    """Lesson 12's training loop for a multi-class classifier, with AdamW and (optionally) a cosine
    learning-rate schedule. Returns per-epoch (train loss, train acc, val loss, val acc)."""
    import torch
    from torch import nn

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda t: 0.5 * (1 + np.cos(np.pi * t / steps))) if cosine else (lambda t: 1.0))
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        tot_loss, tot_ok, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)                                  # (B, n_classes) logits
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(xb)
            tot_ok += (out.argmax(1) == yb).sum().item()
            n += len(xb)
        vl, va = evaluate(model, X_val, y_val, device)
        history.append((tot_loss / n, tot_ok / n, vl, va))
        if verbose:
            print(f"  epoch {epoch + 1:2d}: train loss {tot_loss / n:.4f} acc {tot_ok / n:.4f} | "
                  f"val loss {vl:.4f} acc {va:.4f}")
    return np.array(history)


def evaluate(model, X, y, device="cpu", batch=1000):
    """Mean cross-entropy and accuracy of a classifier on tensors X (N, ...) and y (N,)."""
    import torch
    from torch import nn

    model.eval()
    loss, ok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out = model(X[i:i + batch].to(device))
            loss += nn.functional.cross_entropy(out, y[i:i + batch].to(device), reduction="sum").item()
            ok += (out.argmax(1).cpu() == y[i:i + batch]).sum().item()
    return loss / len(X), ok / len(X)
# --8<-- [end:train]
