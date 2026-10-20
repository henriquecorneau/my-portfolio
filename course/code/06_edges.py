"""Lesson 06 - Edge detection with first and second derivatives."""
import cv2
import numpy as np

import cvutils as U

U.setup("06")

# --8<-- [start:signals]
step = np.r_[np.zeros(50), np.full(50, 255)]                  # sharp edge
ramp = np.r_[np.zeros(48), [25, 75, 175, 225], np.full(48, 250)]  # smooth edge
line = np.r_[np.zeros(47), [25, 75, 175, 250, 175, 75, 25], np.zeros(46)]  # thin line

d1 = np.diff(ramp)         # first derivative: a peak on the edge
d2 = np.diff(ramp, n=2)    # second derivative: + then -, crossing zero on the edge
# --8<-- [end:signals]

fig, axes = U.figure(3, 3, size=(3.4, 1.9))
for col, (sig, name) in enumerate([(step, "step"), (ramp, "ramp"), (line, "line")]):
    for row, (data, label) in enumerate([(sig, name), (np.diff(sig), "1st derivative"),
                                         (np.diff(sig, n=2), "2nd derivative")]):
        ax = axes[row, col]
        ax.plot(data, color=["#18181b", "#2563eb", "#dc2626"][row], lw=1.4)
        ax.axhline(0, color="#999", lw=.6)
        ax.set_xlim(35, 65)
        ax.set_title(f"{label}" if row else name, fontsize=9)
        ax.set_xticks([])
U.finish("signals", fig)

img = U.imread("basics/cameraman.tif", cv2.IMREAD_GRAYSCALE)

# --8<-- [start:gradient]
f = img.astype(np.float32)          # derivatives are signed: never compute them in uint8

dx = np.zeros_like(f)
dy = np.zeros_like(f)
dx[:, :-1] = np.diff(f, axis=1)     # I(x+1, y) - I(x, y)
dy[:-1, :] = np.diff(f, axis=0)     # I(x, y+1) - I(x, y)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)

gx = cv2.filter2D(f, cv2.CV_32F, sobel_x)      # ddepth=CV_32F keeps negative values
gy = cv2.filter2D(f, cv2.CV_32F, sobel_x.T)
magnitude = np.hypot(gx, gy)                   # sqrt(gx² + gy²); |gx| + |gy| is a cheaper variant
direction = np.arctan2(gy, gx)                 # radians, perpendicular to the edge

# OpenCV shortcut, same numbers:
gx_cv = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
# --8<-- [end:gradient]
assert np.allclose(gx, gx_cv)
px = cv2.filter2D(f, cv2.CV_32F, prewitt_x)
py = cv2.filter2D(f, cv2.CV_32F, prewitt_x.T)

wrong = cv2.filter2D(img, -1, sobel_x)   # uint8 output: negative responses clipped to 0


def to_u8(x):
    return cv2.normalize(np.abs(x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


U.show("gradient", [to_u8(gx), to_u8(gy), to_u8(magnitude), wrong],
       ["|Gx| (Sobel)", "|Gy| (Sobel)", "magnitude", "filter2D(uint8): half the edges lost"], cols=4, size=3)

hsv = np.zeros(img.shape + (3,), np.uint8)
hsv[..., 0] = ((direction + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
hsv[..., 1] = 255
hsv[..., 2] = to_u8(magnitude)
U.show("direction", [cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), to_u8(np.hypot(px, py)),
                     to_u8(np.abs(dx) + np.abs(dy))],
       ["direction as hue, magnitude as brightness", "Prewitt magnitude", "plain differences"], size=3.6)

# --8<-- [start:second_order]
lap4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
lap8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
laplacian = cv2.filter2D(f, cv2.CV_32F, lap4)


def gaussian_kernel(size, sigma):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


# Difference of Gaussians: a band-pass filter that approximates the Laplacian of Gaussian
dog = (cv2.filter2D(f, cv2.CV_32F, gaussian_kernel(9, 1.0))
       - cv2.filter2D(f, cv2.CV_32F, gaussian_kernel(9, 1.6)))
# --8<-- [end:second_order]


# --8<-- [start:zero_crossing]
def zero_crossings(resp, min_jump):
    """Edge where the sign changes between 4-neighbours AND the jump is large enough."""
    edges = np.zeros(resp.shape, bool)
    c = resp[1:-1, 1:-1]
    for n in (resp[:-2, 1:-1], resp[2:, 1:-1], resp[1:-1, :-2], resp[1:-1, 2:]):
        edges[1:-1, 1:-1] |= (np.sign(c) != np.sign(n)) & (np.abs(c - n) > min_jump)
    return edges
# --8<-- [end:zero_crossing]

U.show("second-order", [to_u8(laplacian), zero_crossings(laplacian, 0),
                        zero_crossings(laplacian, 60), zero_crossings(dog, 6)],
       ["|Laplacian|", "zero crossings, no threshold", "zero crossings, jump > 60", "DoG zero crossings, jump > 6"],
       cols=4, size=3)

# --8<-- [start:threshold]
def threshold(x, t):
    return (x >= t).astype(np.uint8) * 255


mag_u8 = to_u8(magnitude)
edges_sobel = threshold(mag_u8, 60)
_, edges_otsu = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# --8<-- [end:threshold]
U.show("threshold", [edges_sobel, edges_otsu], ["Sobel magnitude > 60", "Sobel magnitude, Otsu"], size=3.6)

# --8<-- [start:pipeline]
brain = U.imread("noise/noisy_brain.png", cv2.IMREAD_GRAYSCALE)
den = cv2.medianBlur(brain, 5)                                   # 1. pre-processing: S&P noise
gx = cv2.Sobel(den, cv2.CV_32F, 1, 0)                            # 2. processing: gradient
gy = cv2.Sobel(den, cv2.CV_32F, 0, 1)
mag = to_u8(np.hypot(gx, gy))
_, edges = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
overlay = cv2.cvtColor(brain, cv2.COLOR_GRAY2BGR)                # 3. display: edges in red
overlay[edges > 0] = (0, 0, 255)
# --8<-- [end:pipeline]
raw_mag = to_u8(np.hypot(cv2.Sobel(brain, cv2.CV_32F, 1, 0), cv2.Sobel(brain, cv2.CV_32F, 0, 1)))
U.show("pipeline", [brain, raw_mag, mag, overlay],
       ["noisy input", "Sobel without denoising", "Sobel after median 5x5", "edges in red"], cols=4, size=3)
