"""Lesson 04 - Histograms, point operations and look-up tables (LUTs)."""
import cv2
import numpy as np

import cvutils as U

U.setup("04")

img = U.imread("basics/Hawke_Bay.jpg", cv2.IMREAD_GRAYSCALE)


# --8<-- [start:histogram]
def histogram(img):
    return np.bincount(img.ravel(), minlength=256)


def cumulative_histogram(img):
    return np.cumsum(histogram(img))
# --8<-- [end:histogram]


def plot_pairs(name, images, titles, luts=None):
    """Image on top, histogram (and the LUT, if any) below."""
    n = len(images)
    fig, axes = U.figure(n, 2, size=(3.6, 2.7))
    for k, (im, t) in enumerate(zip(images, titles)):
        axes[0, k].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_title(t, fontsize=10)
        axes[0, k].axis("off")
        ax = axes[1, k]
        ax.bar(range(256), histogram(im), width=1.0, color="#94a3b8")
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        if luts is not None and luts[k] is not None:
            ax2 = ax.twinx()
            ax2.plot(luts[k], color="#dc2626", lw=1.6)
            ax2.set_ylim(0, 260)
            ax2.set_yticks([0, 255])
    U.finish(name, fig)


# --8<-- [start:display]
# plt.imshow(img, cmap="gray")                    # stretches min..max to black..white!
# plt.imshow(img, cmap="gray", vmin=0, vmax=255)  # shows the real values
# --8<-- [end:display]

# --8<-- [start:point_ops]
def negative(img):
    return 255 - img


def brightness(img, delta):
    return cv2.add(img, delta)                    # saturates instead of wrapping


def threshold(img, a):
    return np.where(img < a, 0, 255).astype(np.uint8)


def intensity_window(img, a, b):
    return np.where((img >= a) & (img <= b), img, 0).astype(np.uint8)


def stretch(img, a, b):
    """Dynamic expansion: map [a, b] to [0, 255] and clip the rest."""
    out = 255.0 * (img.astype(np.float32) - a) / (b - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def sigmoid_contrast(img, k=0.1, r0=128):
    """Smooth contrast stretch centred on r0 (from the UFRN lectures)."""
    return (255 / (1 + np.exp(-k * (img.astype(np.float32) - r0)))).astype(np.uint8)
# --8<-- [end:point_ops]

plot_pairs("point-ops", [img, negative(img), threshold(img, 150), stretch(img, 110, 200)],
           ["original", "negative", "threshold a=150", "stretch [110, 200]"])


# --8<-- [start:equalize]
def equalize(img):
    """t(f) = (L - 1) / N * sum_{i<=f} h(i)"""
    cdf = cumulative_histogram(img)
    lut = np.round(255 * cdf / img.size).astype(np.uint8)
    return lut[img]                               # apply the LUT with fancy indexing
# --8<-- [end:equalize]

eq_mine = equalize(img)
eq_cv = cv2.equalizeHist(img)
print("equalize: max |mine - OpenCV| =", np.abs(eq_mine.astype(int) - eq_cv).max())

# --8<-- [start:clahe]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq_local = clahe.apply(img)
# --8<-- [end:clahe]

plot_pairs("equalize", [img, eq_mine, eq_local],
           ["original", "global equalization", "CLAHE (local)"],
           luts=[None, np.round(255 * cumulative_histogram(img) / img.size), None])


# --8<-- [start:lut]
def lut_threshold(a):
    return np.where(np.arange(256) < a, 0, 255).astype(np.uint8)


def lut_stretch(a, b):
    x = np.arange(256, dtype=np.float32)
    return np.clip(255 * (x - a) / (b - a), 0, 255).astype(np.uint8)


def lut_contrast(a, b):
    """Piece-wise linear: slope b/a below a, then a line to (255, 255)."""
    x = np.arange(256, dtype=np.float32)
    low = x * b / a
    high = ((255 - b) * x + 255 * (b - a)) / (255 - a)
    return np.clip(np.where(x <= a, low, high), 0, 255).astype(np.uint8)


def lut_gamma(gamma):
    x = np.arange(256) / 255.0
    return np.round(255 * x ** gamma).astype(np.uint8)


def apply_lut(img, lut):
    return lut[img]   # or cv2.LUT(img, lut)


# Composing transforms = composing LUTs: still one pass over the image.
combined = lut_threshold(128)[lut_stretch(110, 200)]
# --8<-- [end:lut]

luts = [lut_contrast(130, 30), lut_contrast(60, 160), lut_gamma(0.5), lut_gamma(2.0)]
names = ["contrast a=130 b=30", "contrast a=60 b=160", "gamma 0.5", "gamma 2.0"]
plot_pairs("luts", [apply_lut(img, L) for L in luts], names, luts=luts)
assert np.array_equal(apply_lut(img, lut_stretch(110, 200)), stretch(img, 110, 200))
assert np.array_equal(apply_lut(img, lut_stretch(110, 200)), cv2.LUT(img, lut_stretch(110, 200)))


# --8<-- [start:matching]
def cdf(channel):
    c = np.cumsum(np.bincount(channel.ravel(), minlength=256)).astype(np.float64)
    return c / c[-1]


def match_lut(source, reference):
    """For each source level, the reference level with the same cumulative frequency."""
    return np.clip(np.round(np.interp(cdf(source), cdf(reference), np.arange(256))), 0, 255).astype(np.uint8)


def match_histograms(source, reference):
    out = np.empty_like(source)
    for c in range(3):                            # B, G and R independently
        out[..., c] = match_lut(source[..., c], reference[..., c])[source[..., c]]
    return out
# --8<-- [end:matching]

src = U.imread("basics/rouen.jpg")
ref = cv2.resize(U.imread("features/cathedrale.jpeg"), (800, 533))
U.show("matching", [src, ref, match_histograms(src, ref)],
       ["source", "reference (style)", "source with reference histogram"], size=4)
