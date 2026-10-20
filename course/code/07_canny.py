"""Lesson 07 - The Canny edge detector, step by step."""
from collections import deque

import cv2
import numpy as np

import cvutils as U

U.setup("07")

img = U.imread("basics/cameraman.tif", cv2.IMREAD_GRAYSCALE)

# --8<-- [start:step1]
blurred = cv2.GaussianBlur(img.astype(np.float32), (5, 5), sigmaX=1.4)
# --8<-- [end:step1]

# --8<-- [start:step2]
gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
angle = np.rad2deg(np.arctan2(gy, gx)) % 180     # direction, folded into [0, 180)
# --8<-- [end:step2]


# --8<-- [start:step3]
def non_max_suppression(mag, ang):
    """Keep a pixel only if it is a local maximum along the gradient direction."""
    h, w = mag.shape
    out = np.zeros_like(mag)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = ang[i, j]
            if a < 22.5 or a >= 157.5:     # gradient ~horizontal -> compare left/right
                n1, n2 = mag[i, j - 1], mag[i, j + 1]
            elif a < 67.5:                 # ~45°: image y grows downwards, so this diagonal
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]
            elif a < 112.5:                # ~vertical -> compare up/down
                n1, n2 = mag[i - 1, j], mag[i + 1, j]
            else:                          # ~135°
                n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            if mag[i, j] >= n1 and mag[i, j] >= n2:
                out[i, j] = mag[i, j]
    return out
# --8<-- [end:step3]

thin = non_max_suppression(magnitude, angle)

# --8<-- [start:step4]
STRONG, WEAK = 255, 128


def double_threshold(nms, low, high):
    out = np.zeros(nms.shape, np.uint8)
    out[nms >= high] = STRONG
    out[(nms >= low) & (nms < high)] = WEAK
    return out
# --8<-- [end:step4]

labels = double_threshold(thin, low=40, high=100)


# --8<-- [start:step5]
def hysteresis(labels):
    """Weak pixels survive only if they are connected (8-neighbourhood) to a strong one."""
    out = np.where(labels == STRONG, 255, 0).astype(np.uint8)
    queue = deque(zip(*np.nonzero(labels == STRONG)))
    h, w = labels.shape
    while queue:                                   # flood fill starting from every strong pixel
        i, j = queue.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                y, x = i + di, j + dj
                if 0 <= y < h and 0 <= x < w and labels[y, x] == WEAK and out[y, x] == 0:
                    out[y, x] = 255
                    queue.append((y, x))
    return out
# --8<-- [end:step5]

edges = hysteresis(labels)


def to_u8(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


U.show("steps", [img, to_u8(magnitude), to_u8(thin), labels, edges],
       ["input", "1-2. blur + gradient", "3. non-max suppression", "4. strong (white) / weak (gray)",
        "5. hysteresis"], cols=5, size=2.8)


# --8<-- [start:canny]
def canny(img, low, high, sigma=1.4):
    f = cv2.GaussianBlur(img.astype(np.float32), (5, 5), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180
    return hysteresis(double_threshold(non_max_suppression(np.hypot(gx, gy), ang), low, high))


# OpenCV: blur yourself, then give the two thresholds (L2gradient=True uses the true magnitude)
edges_cv = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 1.4), 40, 100, L2gradient=True)
# --8<-- [end:canny]

agree = np.mean((edges > 0) == (edges_cv > 0))
print(f"pixels where ours and cv2.Canny agree: {100 * agree:.1f}%")

single_pass = labels.copy()   # naive version: weak pixels next to a strong one, one pass only
strong = labels == STRONG
grow = cv2.dilate(strong.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
single_pass = np.where(strong | ((labels == WEAK) & grow), 255, 0).astype(np.uint8)
print("edge pixels — single pass:", int((single_pass > 0).sum()), " flood fill:", int((edges > 0).sum()))

U.show("compare", [canny(img, 20, 50), edges, canny(img, 80, 160), edges_cv],
       ["ours (20, 50)", "ours (40, 100)", "ours (80, 160)", "cv2.Canny (40, 100)"], cols=4, size=3)

circle = U.imread("features/circulo.png", cv2.IMREAD_GRAYSCALE)
cf = circle.astype(np.float32)
cgx, cgy = cv2.Sobel(cf, cv2.CV_32F, 1, 0), cv2.Sobel(cf, cv2.CV_32F, 0, 1)
cmag = np.hypot(cgx, cgy)
cthin = non_max_suppression(cmag, np.rad2deg(np.arctan2(cgy, cgx)) % 180)
U.show("nms-circle", [circle, to_u8(cmag), to_u8(cthin)], ["circle", "Sobel magnitude (thick)", "after NMS (1 px)"],
       size=3)
