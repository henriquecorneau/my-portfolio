"""Lesson 10 - Colour spaces (CMYK, YUV, HSV) and colour segmentation."""
import cv2
import numpy as np

import cvutils as U

U.setup("10")

bgr = U.imread("color/crayons.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --8<-- [start:cmyk]
def rgb_to_cmyk(rgb):
    x = rgb.astype(np.float32) / 255
    k = 1 - x.max(axis=2)
    denom = np.where(k < 1, 1 - k, 1)                 # avoid 0/0 on pure black pixels
    c, m, y = [(1 - x[..., i] - k) / denom for i in range(3)]
    return np.dstack([c, m, y, k])                    # floats in [0, 1]


def cmyk_to_rgb(cmyk):
    c, m, y, k = np.moveaxis(cmyk, 2, 0)
    rgb = np.dstack([(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)])
    return np.round(rgb * 255).astype(np.uint8)
# --8<-- [end:cmyk]
# Note: many references (and the TP statement) write c = 1 - r - k. That is the
# "under-colour removal" before normalising; dividing by (1 - k) gives the usual CMYK.

cmyk = rgb_to_cmyk(rgb)
print("CMYK round trip max error:", np.abs(cmyk_to_rgb(cmyk).astype(int) - rgb).max())


def ink(channel, color):
    """Show one ink on white paper: subtract it from a white sheet."""
    paper = np.ones(channel.shape + (3,), np.float32)
    paper -= channel[..., None] * (1 - np.array(color, np.float32))
    return cv2.cvtColor((paper * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


U.show("cmyk", [ink(cmyk[..., 0], (0, 1, 1)), ink(cmyk[..., 1], (1, 0, 1)),
                ink(cmyk[..., 2], (1, 1, 0)), ink(cmyk[..., 3], (0, 0, 0))],
       ["Cyan ink", "Magenta ink", "Yellow ink", "blacK ink"], cols=4, size=3)


# --8<-- [start:yuv]
def rgb_to_yuv(rgb):
    x = rgb.astype(np.float32) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b     # luma: what a black-and-white TV shows
    u = 0.492 * (b - y)                       # blue-difference chroma, in [-0.436, 0.436]
    v = 0.877 * (r - y)                       # red-difference chroma, in [-0.615, 0.615]
    return np.dstack([y, u, v])


def yuv_to_rgb(yuv):
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    return np.clip(np.round(np.dstack([r, g, b]) * 255), 0, 255).astype(np.uint8)
# --8<-- [end:yuv]

yuv = rgb_to_yuv(rgb)
yuv_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)     # uint8, U and V offset by 128
print("Y: max |ours*255 - OpenCV| =", np.abs(yuv[..., 0] * 255 - yuv_cv[..., 0]).max().round(2))
print("YUV round trip max error:", np.abs(yuv_to_rgb(yuv).astype(int) - rgb).max())

fig, axes = U.figure(3, 1, size=(3.2, 3.2))
for ax, ch, t, lim in zip(axes[0], range(3), ["Y  [0, 1]", "U  [-0.44, 0.44]", "V  [-0.62, 0.62]"],
                          [(0, 1), (-.44, .44), (-.62, .62)]):
    ax.imshow(yuv[..., ch], cmap="gray" if ch == 0 else "coolwarm", vmin=lim[0], vmax=lim[1])
    ax.set_title(t, fontsize=10)
    ax.axis("off")
U.finish("yuv", fig)


# --8<-- [start:hsv]
def rgb_to_hsv(rgb):
    """H in [0, 180) (degrees / 2, fits in uint8), S and V in [0, 255], like OpenCV."""
    x = rgb.astype(np.float32) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=2), x.min(axis=2)
    delta = mx - mn
    safe = np.where(delta == 0, 1, delta)
    h = np.zeros_like(mx)
    h = np.where(mx == r, (60 * (g - b) / safe) % 360, h)
    h = np.where(mx == g, 60 * (b - r) / safe + 120, h)
    h = np.where(mx == b, 60 * (r - g) / safe + 240, h)
    h = np.where(delta == 0, 0, h)                         # gray: hue undefined, use 0
    s = np.where(mx == 0, 0, delta / np.where(mx == 0, 1, mx))
    return np.dstack([h / 2, s * 255, mx * 255]).round().astype(np.uint8)
# --8<-- [end:hsv]

hsv = rgb_to_hsv(rgb)
hsv_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
diff = np.abs(hsv.astype(int) - hsv_cv)
diff[..., 0] = np.minimum(diff[..., 0], 180 - diff[..., 0])   # hue is circular
print("HSV vs OpenCV, max abs diff per channel:", diff.reshape(-1, 3).max(axis=0))

U.show("hsv", [hsv[..., 0], hsv[..., 1], hsv[..., 2]], ["H (0-179)", "S (0-255)", "V (0-255)"], size=3.2)

fig, axes = U.figure(1, 1, size=(4.2, 4.2))
ax = axes[0, 0]
sub = hsv_cv[::3, ::3].reshape(-1, 3).astype(np.float32)
keep = sub[:, 2] > 100
angle = np.deg2rad(sub[keep, 0] * 2)
radius = sub[keep, 1] / 255 * sub[keep, 2] / 255
ax.scatter(radius * np.cos(angle), radius * np.sin(angle), s=2,
           c=rgb[::3, ::3].reshape(-1, 3)[keep] / 255)
ax.add_patch(__import__("matplotlib").patches.Circle((0, 0), 1, fill=False))
ax.set_aspect("equal")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_title("pixels in the HSV disc (angle = H, radius = S·V)", fontsize=9)
ax.axis("off")
U.finish("hsv-disc", fig)

# --8<-- [start:recolor]
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# Yellow sits in a narrow hue band (~35-65 degrees, i.e. 17-33 in OpenCV units),
# whatever its brightness. Saturation and value thresholds drop the white paper.
yellow = cv2.inRange(hsv, (18, 120, 100), (34, 255, 255))

recolored = hsv.copy()
recolored[..., 0][yellow > 0] = 60           # 120 degrees / 2: green. S and V untouched,
recolored = cv2.cvtColor(recolored, cv2.COLOR_HSV2BGR)  # so the shading still looks natural

rotated = hsv.copy()
rotated[..., 0] = (rotated[..., 0].astype(int) + 60) % 180   # rotate *every* hue by 120 degrees
rotated = cv2.cvtColor(rotated, cv2.COLOR_HSV2BGR)

# Red wraps around 0 degrees, so it needs two ranges:
red = cv2.inRange(hsv, (0, 120, 70), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))
# --8<-- [end:recolor]
U.show("recolor", [bgr, yellow, recolored, rotated],
       ["input", "yellow mask: H in [18, 34]", "yellow hue → green", "all hues + 120°"], cols=4, size=3)

# --8<-- [start:track]
car = U.imread("color/toy_car.jpg")
car_hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
# The pale car (S ~ 40) is hard to threshold, the saturated red background (S ~ 135) is easy:
# segment the background and invert.
background = cv2.inRange(car_hsv, (0, 80, 80), (12, 255, 255)) | \
             cv2.inRange(car_hsv, (168, 80, 80), (179, 255, 255))
mask = cv2.bitwise_not(background)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

# keep the largest blob and fill its holes (windows, reflections)
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
mask = np.where(labels == largest, 255, 0).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

m = cv2.moments(mask, binaryImage=True)
cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]                   # centre of mass
x, y, w, h = cv2.boundingRect(mask)
# --8<-- [end:track]
print(f"car centroid: ({cx:.0f}, {cy:.0f}), box {w}x{h}")
vis = car.copy()
cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.drawMarker(vis, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
U.show("track", [background, mask, vis], ["red background (two hue ranges)", "NOT background, cleaned",
                                          "centroid + bounding box"], size=3.6)
