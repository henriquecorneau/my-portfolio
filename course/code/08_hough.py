"""Lesson 08 - The Hough transform for lines (and circles)."""
import cv2
import numpy as np

import cvutils as U

U.setup("08")

# --8<-- [start:duality]
# Each image point (x, y) is a whole *line* b = -x a + y in the (a, b) parameter space.
# Points that are collinear in the image give lines that meet at one (a, b).
pts = [(1, 4), (3, 2), (4, 1)]           # all on y = -x + 5
a = np.linspace(-3, 1, 100)
# --8<-- [end:duality]

fig, axes = U.figure(3, 1, size=(3.4, 3.0))
ax = axes[0, 0]
ax.plot(*zip(*pts), "o", color="#2563eb")
ax.plot([0, 5], [5, 0], color="#94a3b8", lw=1)
ax.set_title("image space (x, y)", fontsize=10)
ax.grid(alpha=.3)
ax = axes[0, 1]
for x, y in pts:
    ax.plot(a, -x * a + y, label=f"b = -{x}a + {y}")
ax.plot(-1, 5, "o", color="k")
ax.set_title("parameter space (a, b)", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=.3)
ax = axes[0, 2]
th = np.linspace(-np.pi / 2, np.pi / 2, 200)
for x, y in pts:
    ax.plot(np.rad2deg(th), x * np.cos(th) + y * np.sin(th))
ax.set_title("(θ, ρ) space: sinusoids", fontsize=10)
ax.set_xlabel("θ (deg)")
ax.grid(alpha=.3)
U.finish("duality", fig)


# --8<-- [start:accumulator]
def hough_lines_acc(edges, rho_res=1, theta_res_deg=1):
    """Vote in (rho, theta): rho = x cos(theta) + y sin(theta)."""
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag, diag + 1, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res_deg))
    acc = np.zeros((len(rhos), len(thetas)), np.int32)
    ys, xs = np.nonzero(edges)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for x, y in zip(xs, ys):                      # one sinusoid per edge pixel
        r = np.round((x * cos_t + y * sin_t) / rho_res).astype(int) + diag // rho_res
        acc[r, np.arange(len(thetas))] += 1
    return acc, rhos, thetas
# --8<-- [end:accumulator]


# --8<-- [start:peaks]
def hough_peaks(acc, n, min_votes=1, nms_size=21):
    """The n highest cells, each one clearing its neighbourhood (non-maximum suppression)."""
    acc = acc.copy()
    peaks = []
    half = nms_size // 2
    for _ in range(n):
        r, t = np.unravel_index(np.argmax(acc), acc.shape)
        if acc[r, t] < min_votes:
            break
        peaks.append((r, t))
        acc[max(0, r - half):r + half + 1, max(0, t - half):t + half + 1] = 0
    return peaks


def draw_lines(img, lines, color=(0, 0, 255), thickness=2):
    """lines: iterable of (rho, theta)."""
    out = img.copy()
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho                 # closest point of the line to the origin
        p1 = (int(x0 - 2000 * b), int(y0 + 2000 * a))
        p2 = (int(x0 + 2000 * b), int(y0 - 2000 * a))
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
    return out
# --8<-- [end:peaks]

road = U.imread("features/estrada.jpg")
gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)

# --8<-- [start:lanes]
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

# keep only a trapezoid in front of the car: the region of interest
roi = np.zeros_like(edges)
h, w = edges.shape
poly = np.array([[(0, h), (w, h), (w // 2 + 30, int(h * 0.55)), (w // 2 - 30, int(h * 0.55))]], np.int32)
cv2.fillPoly(roi, poly, 255)
edges_roi = cv2.bitwise_and(edges, roi)

acc, rhos, thetas = hough_lines_acc(edges_roi)
peaks = hough_peaks(acc, n=4, min_votes=40)
lines = [(rhos[r], thetas[t]) for r, t in peaks]
# --8<-- [end:lanes]
print("our peaks (rho, theta deg):", [(int(r), int(np.rad2deg(t))) for r, t in lines])

# --8<-- [start:opencv]
cv_lines = cv2.HoughLines(edges_roi, rho=1, theta=np.pi / 180, threshold=60)          # (rho, theta)
segments = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=30,
                           minLineLength=30, maxLineGap=20)                          # (x1, y1, x2, y2)
# --8<-- [end:opencv]

seg_img = road.copy()
for x1, y1, x2, y2 in segments.reshape(-1, 4):   # (N, 1, 4) in OpenCV 4, (N, 4) in 5
    cv2.line(seg_img, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
layer = draw_lines(np.zeros_like(road), lines, thickness=3)
layer[roi == 0] = 0                                  # draw the infinite lines only inside the ROI
lanes = road.copy()
lanes[layer.any(axis=2)] = layer[layer.any(axis=2)]
U.show("lanes", [edges_roi, lanes, seg_img],
       ["Canny inside the ROI", "our accumulator: 4 peaks", "cv2.HoughLinesP segments"], size=3.8)

fig, axes = U.figure(1, 1, size=(5.5, 3.6))
ax = axes[0, 0]
ax.imshow(np.sqrt(acc), cmap="magma", aspect="auto",
          extent=[-90, 90, rhos[-1], rhos[0]])
for r, t in peaks:
    ax.plot(np.rad2deg(thetas[t]), rhos[r], "o", mfc="none", mec="#22d3ee", ms=12, mew=2)
ax.set_xlabel("θ (degrees)")
ax.set_ylabel("ρ (pixels)")
ax.set_title("accumulator (√votes) and detected peaks", fontsize=10)
U.finish("accumulator", fig)

# --8<-- [start:circles]
canvas = np.full((300, 400), 40, np.uint8)
for (cx, cy, r) in [(90, 100, 50), (250, 160, 70), (330, 60, 30)]:
    cv2.circle(canvas, (cx, cy), r, 200, -1)
canvas = cv2.add(canvas, np.random.randint(0, 40, canvas.shape, dtype=np.uint8))
circles = cv2.HoughCircles(cv2.medianBlur(canvas, 5), cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                           param1=100, param2=20, minRadius=10, maxRadius=100)
# --8<-- [end:circles]
out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
for cx, cy, r in np.round(circles.reshape(-1, 3)).astype(int):
    cv2.circle(out, (cx, cy), r, (0, 0, 255), 2)
    cv2.circle(out, (cx, cy), 2, (0, 255, 0), 3)
print("circles (x, y, r):", np.round(circles.reshape(-1, 3)).astype(int).tolist())
U.show("circles", [canvas, out], ["synthetic disks + noise", "cv2.HoughCircles"], size=3.6)
