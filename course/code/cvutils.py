"""Small helpers shared by every lesson script.

Each lesson is a plain Python script:

    python 05_filtering.py            # opens the figures in a window
    python 05_filtering.py --save out # writes the figures to out/ instead

The helpers only deal with loading data and drawing figures, so the computer
vision code in the lessons stays readable.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

DATA = Path(__file__).resolve().parent / "data"

_save_dir = None
_prefix = ""


def setup(lesson):
    """Parse the command line. Call once at the top of a lesson script."""
    global _save_dir, _prefix
    parser = argparse.ArgumentParser(description=f"Lesson {lesson}")
    parser.add_argument("--save", metavar="DIR", help="save figures to DIR instead of showing them")
    args = parser.parse_args()
    _prefix = lesson
    if args.save:
        matplotlib.use("Agg")
        _save_dir = Path(args.save)
        _save_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)


def path(name):
    """Absolute path of a file inside the data/ folder."""
    return DATA / name


def imread(name, flags=cv2.IMREAD_COLOR):
    """cv2.imread that fails loudly instead of silently returning None."""
    img = cv2.imread(str(path(name)), flags)
    if img is None:
        raise FileNotFoundError(f"could not read {path(name)}")
    return img


def _draw(ax, img, title, cmap):
    if img.ndim == 3:
        # OpenCV stores colour images as BGR, Matplotlib expects RGB.
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif img.dtype == np.uint8:
        # Never let Matplotlib stretch the contrast of a uint8 image.
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    elif img.dtype == bool:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(img, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")


# --8<-- [start:show]
def show(name, images, titles=None, cols=None, size=3.4, cmap="gray"):
    """Draw a grid of images. Colour images are assumed to be BGR."""
    import matplotlib.pyplot as plt

    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    h, w = images[0].shape[:2]
    cell_h = size * min(max(h / w, 0.3), 1.6)       # follow the aspect ratio of the images
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, (cell_h + 0.35) * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            _draw(ax, images[i], titles[i] if titles else None, cmap)
        else:
            ax.axis("off")
    finish(name, fig)
# --8<-- [end:show]


def figure(cols=1, rows=1, size=(4.2, 3.2)):
    """Create a figure for custom plots (histograms, curves, 3-D...)."""
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
