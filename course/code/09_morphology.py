"""Lesson 09 - Mathematical morphology on binary images."""
import cv2
import numpy as np

import cvutils as U

U.setup("09")

# --8<-- [start:kernels]
k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))    # (width, height)
print(k_cross)
# --8<-- [end:kernels]

fig, axes = U.figure(4, 1, size=(2.0, 2.2))
for ax, k, t in zip(axes[0], [k_rect, k_ellipse, k_cross, k_line], ["rect 5x5", "ellipse 7x7", "cross 5x5", "rect 3x11"]):
    ax.imshow(k, cmap="gray", vmin=0, vmax=1)
    ax.set_title(t, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("kernels", fig)


# --8<-- [start:erode_dilate]
def erode(img, k):
    """White at x only if the kernel, centred at x, fits entirely inside the white set."""
    b = img > 0
    kh, kw = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(b, ((kh, kh), (kw, kw)), constant_values=True)
    out = np.ones_like(b)
    for dy, dx in zip(*np.nonzero(k)):              # AND of shifted copies, one per kernel cell
        out &= padded[dy:dy + b.shape[0], dx:dx + b.shape[1]]
    return out.astype(np.uint8) * 255


def dilate(img, k):
    """White at x if the (reflected) kernel centred at x touches the white set at all."""
    b = img > 0
    kh, kw = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(b, ((kh, kh), (kw, kw)), constant_values=False)
    out = np.zeros_like(b)
    k_ref = k[::-1, ::-1]
    for dy, dx in zip(*np.nonzero(k_ref)):          # OR of shifted copies
        out |= padded[dy:dy + b.shape[0], dx:dx + b.shape[1]]
    return out.astype(np.uint8) * 255


def opening(img, k):          # erosion then dilation: removes small white specks
    return dilate(erode(img, k), k)


def closing(img, k):          # dilation then erosion: fills small black holes
    return erode(dilate(img, k), k)


def morph_gradient(img, k):   # dilation - erosion: the outline of every shape
    return cv2.subtract(dilate(img, k), erode(img, k))
# --8<-- [end:erode_dilate]

noisy = U.imread("morphology/noisy_yin_yang.png", cv2.IMREAD_GRAYSCALE)
noisy = np.where(noisy > 127, 255, 0).astype(np.uint8)

for name, mine, cv in [("erode", erode(noisy, k_cross), cv2.erode(noisy, k_cross)),
                       ("dilate", dilate(noisy, k_cross), cv2.dilate(noisy, k_cross)),
                       ("open", opening(noisy, k_cross), cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k_cross)),
                       ("close", closing(noisy, k_cross), cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, k_cross))]:
    print(f"{name:<6} identical to OpenCV: {np.array_equal(mine, cv)}")

U.show("basic-ops", [noisy, erode(noisy, k_cross), dilate(noisy, k_cross),
                     opening(noisy, k_cross), closing(noisy, k_cross)],
       ["noisy input", "erosion", "dilation", "opening", "closing"], cols=5, size=2.8)

# --8<-- [start:clean]
truth = np.where(U.imread("morphology/yin_yang.png", cv2.IMREAD_GRAYSCALE) > 127, 255, 0).astype(np.uint8)


def error(img):
    return 100 * np.mean(img != truth)                     # % of wrong pixels


def ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# One big opening followed by one big closing: the opening erases the white half too,
# because it is riddled with black specks (13% of its pixels).
naive = cv2.morphologyEx(cv2.morphologyEx(noisy, cv2.MORPH_OPEN, ellipse(7)), cv2.MORPH_CLOSE, ellipse(7))

# Alternating sequential filter: close/open with *growing* kernels, so each step only
# has to remove specks slightly bigger than the ones already gone.
asf = noisy
for k in (3, 5, 7):
    asf = cv2.morphologyEx(asf, cv2.MORPH_CLOSE, ellipse(k))
    asf = cv2.morphologyEx(asf, cv2.MORPH_OPEN, ellipse(k))

outline = cv2.morphologyEx(asf, cv2.MORPH_GRADIENT, k_cross)
print(f"error: noisy {error(noisy):.1f}%  open7+close7 {error(naive):.1f}%  ASF {error(asf):.2f}%")
# --8<-- [end:clean]
U.show("clean", [noisy, naive, asf, outline],
       [f"noisy ({error(noisy):.0f}% wrong)", f"open 7 + close 7 ({error(naive):.0f}% wrong)",
        f"alternating filter ({error(asf):.1f}% wrong)", "morphological gradient"], cols=4, size=3)

# --8<-- [start:qr]
qr = U.imread("morphology/QR_code.png", cv2.IMREAD_GRAYSCALE)
detector = cv2.QRCodeDetector()


def decode(img):
    # QR readers need a white "quiet zone" around the code
    padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    text, points, _ = detector.detectAndDecode(padded)
    return text or None


k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
qr_bin = cv2.threshold(qr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
qr_open = cv2.morphologyEx(qr_bin, cv2.MORPH_OPEN, k3)       # white noise inside black modules
qr_clean = cv2.morphologyEx(qr_open, cv2.MORPH_CLOSE, k3)    # black noise inside white modules
for name, im in [("input", qr), ("cleaned", qr_clean)]:
    print(f"QR {name:<8} decoded:", decode(im) is not None)
# --8<-- [end:qr]


def status(im):
    return "decoded ✓" if decode(im) else "not decoded ✗"


U.show("qr", [qr, qr_open, qr_clean],
       [f"input: {status(qr)}", "opening (cross 3x3)", f"+ closing: {status(qr_clean)}"], size=3.4)


# --8<-- [start:hitmiss]
def hit_or_miss(img, kernel):
    """kernel: 1 = must be white, -1 = must be black, 0 = don't care."""
    fg = img > 0
    hits = erode(fg.astype(np.uint8), (kernel == 1).astype(np.uint8)) > 0
    misses = erode((~fg).astype(np.uint8), (kernel == -1).astype(np.uint8)) > 0
    return (hits & misses).astype(np.uint8) * 255


corner = np.array([[0, -1, -1],
                   [1, 1, -1],
                   [0, 1, 0]])                     # top-right outer corner of a white shape
shapes = np.zeros((60, 80), np.uint8)
cv2.rectangle(shapes, (10, 10), (30, 40), 255, -1)
cv2.rectangle(shapes, (45, 20), (70, 50), 255, -1)
found_mine = hit_or_miss(shapes, corner)
found_cv = cv2.morphologyEx(shapes, cv2.MORPH_HITMISS, corner)
# --8<-- [end:hitmiss]
print("hit-or-miss identical to OpenCV:", np.array_equal(found_mine, found_cv),
      "| corners at (y, x):", list(zip(*np.nonzero(found_mine))))
vis = cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR)
vis[found_mine > 0] = (0, 0, 255)
U.show("hitmiss", [shapes, cv2.resize(vis, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)],
       ["input", "top-right corners (red)"], size=3.4)

# --8<-- [start:components]
blobs = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k_cross)
n, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs)
big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 500]     # label 0 is the background
print(f"{n - 1} white components, {len(big)} larger than 500 px")
# --8<-- [end:components]
