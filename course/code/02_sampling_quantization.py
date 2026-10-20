"""Lesson 02 - Sampling, quantization and interpolation."""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import cvutils as U

U.setup("02")

gray = U.imread("basics/cameraman.tif", cv2.IMREAD_GRAYSCALE)


# --8<-- [start:quantize]
def quantize(img, n_levels):
    """Uniform scalar quantization of a uint8 image to n_levels gray levels."""
    step = 256 // n_levels
    # integer division puts every pixel in a bin, then we move it to the bin centre
    return ((img // step) * step + step // 2).astype(np.uint8)
# --8<-- [end:quantize]


levels = [128, 32, 8, 4, 2]
U.show("quantize", [gray] + [quantize(gray, n) for n in levels],
       ["256 levels"] + [f"{n} levels" for n in levels], cols=6, size=2.6)


# --8<-- [start:histogram]
def histogram(img):
    """Count how many pixels have each of the 256 possible values."""
    hist = np.zeros(256, dtype=np.int64)
    for value in img.ravel():
        hist[value] += 1
    return hist

# Same result without a Python loop:
# np.bincount(img.ravel(), minlength=256)
# --8<-- [end:histogram]

fig, axes = U.figure(3, 1, size=(3.6, 2.6))
for ax, n in zip(axes[0], [256, 16, 4]):
    img = gray if n == 256 else quantize(gray, n)
    ax.bar(range(256), histogram(img), width=1.0, color="#2563eb")
    ax.set_title(f"{n} levels", fontsize=10)
    ax.set_xlim(0, 255)
U.finish("quantize-hist", fig)


# --8<-- [start:kmeans]
def color_quantize(img_bgr, n_colors):
    """Keep the n_colors most representative colours (k-means in L*a*b*)."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    km = MiniBatchKMeans(n_clusters=n_colors, n_init=3, random_state=0)
    labels = km.fit_predict(lab)
    quant = km.cluster_centers_.astype(np.uint8)[labels].reshape(h, w, 3)
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
# --8<-- [end:kmeans]

color = U.imread("basics/rouen.jpg")
per_channel = np.dstack([quantize(color[:, :, c], 4) for c in range(3)])
U.show("kmeans", [color, per_channel, color_quantize(color, 4), color_quantize(color, 16)],
       ["original", "4 levels per channel (64 colours)", "k-means, 4 colours", "k-means, 16 colours"],
       cols=2, size=4.6)

# --8<-- [start:subsample]
mire = U.imread("basics/mire.png", cv2.IMREAD_GRAYSCALE)


def subsample_loop(img, factor):
    h, w = img.shape
    out = np.zeros(((h + factor - 1) // factor, (w + factor - 1) // factor), img.dtype)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = img[i * factor, j * factor]
    return out


def subsample(img, factor):
    return img[::factor, ::factor]  # same thing with slicing


blurred = cv2.GaussianBlur(mire, (0, 0), sigmaX=2)  # low-pass before decimating
# --8<-- [end:subsample]

assert np.array_equal(subsample_loop(mire, 4), subsample(mire, 4))


def up(img):  # enlarge for display only, without smoothing
    return cv2.resize(img, (mire.shape[1], mire.shape[0]), interpolation=cv2.INTER_NEAREST)


U.show("aliasing", [mire, up(subsample(mire, 2)), up(subsample(mire, 4)), up(subsample(blurred, 4))],
       ["original", "factor 2", "factor 4 (aliasing)", "blur, then factor 4"], cols=4, size=3)


# --8<-- [start:downscale]
def downscale_area(img, scale):
    """Shrink by `scale` < 1: each output pixel is the mean of the block it covers."""
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    out = np.zeros((new_h, new_w) + img.shape[2:], np.uint8)
    step_y, step_x = img.shape[0] / new_h, img.shape[1] / new_w
    for i in range(new_h):
        for j in range(new_w):
            y0, y1 = int(i * step_y), int((i + 1) * step_y)
            x0, x1 = int(j * step_x), int((j + 1) * step_x)
            out[i, j] = img[y0:y1, x0:x1].mean(axis=(0, 1))
    return out
# --8<-- [end:downscale]


# --8<-- [start:upscale]
def upscale_nearest(img, scale):
    """Order-0 interpolation: copy the closest source pixel."""
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    out = np.zeros((new_h, new_w) + img.shape[2:], img.dtype)
    for i in range(new_h):
        for j in range(new_w):
            y = min(round(i / scale), img.shape[0] - 1)
            x = min(round(j / scale), img.shape[1] - 1)
            out[i, j] = img[y, x]
    return out


def upscale_bilinear(img, scale):
    """Order-1 interpolation: weighted mean of the 4 surrounding pixels."""
    src = img.astype(np.float32)
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    out = np.zeros((new_h, new_w) + img.shape[2:], np.float32)
    for i in range(new_h):
        for j in range(new_w):
            y, x = i / scale, j / scale                 # position in the source image
            y0, x0 = int(y), int(x)
            y1, x1 = min(y0 + 1, img.shape[0] - 1), min(x0 + 1, img.shape[1] - 1)
            dy, dx = y - y0, x - x0                     # fractional parts, in [0, 1)
            out[i, j] = ((1 - dy) * (1 - dx) * src[y0, x0] + (1 - dy) * dx * src[y0, x1]
                         + dy * (1 - dx) * src[y1, x0] + dy * dx * src[y1, x1])
    return np.clip(out, 0, 255).astype(np.uint8)
# --8<-- [end:upscale]


crop = gray[60:100, 100:140]  # 40x40 crop: the camera and the hand
small = downscale_area(gray, 0.25)
assert np.abs(small.astype(int) - cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)).max() <= 1
U.show("interp", [crop, upscale_nearest(crop, 6), upscale_bilinear(crop, 6),
                  cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)],
       ["40x40 crop", "nearest (order 0)", "bilinear (order 1)", "cv2.INTER_CUBIC"], cols=4, size=3)
