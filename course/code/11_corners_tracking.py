"""Lesson 11 - Corner detection (Harris, Shi-Tomasi) and tracking (NCC, Lucas-Kanade)."""
import cv2
import numpy as np

import cvutils as U

U.setup("11")

img = U.imread("features/blox.jpg", cv2.IMREAD_GRAYSCALE)

# --8<-- [start:structure]
def structure_tensor(img, block=5, sigma=1.0):
    """Sxx, Sxy, Syy: Gaussian-weighted sums of gradient products around each pixel."""
    f = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)      # tame the noise first
    ix = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    sxx = cv2.GaussianBlur(ix * ix, (block, block), sigma)
    sxy = cv2.GaussianBlur(ix * iy, (block, block), sigma)
    syy = cv2.GaussianBlur(iy * iy, (block, block), sigma)
    return sxx, sxy, syy


def harris_response(img, k=0.05, block=5):
    sxx, sxy, syy = structure_tensor(img, block)
    det = sxx * syy - sxy ** 2          # lambda1 * lambda2
    trace = sxx + syy                   # lambda1 + lambda2
    return det - k * trace ** 2


def shi_tomasi_response(img, block=5):
    sxx, sxy, syy = structure_tensor(img, block)
    det = sxx * syy - sxy ** 2
    trace = sxx + syy
    return (trace - np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))) / 2   # min(lambda1, lambda2)
# --8<-- [end:structure]


# --8<-- [start:nms]
def greedy_nms(response, rel_threshold=0.01, min_distance=10, max_corners=None):
    """Best candidate first; every accepted corner forbids a square around it."""
    ys, xs = np.nonzero(response > rel_threshold * response.max())
    order = np.argsort(response[ys, xs])[::-1]               # strongest first
    allowed = np.ones(response.shape, bool)
    corners = []
    for y, x in zip(ys[order], xs[order]):
        if allowed[y, x]:
            corners.append((x, y))
            allowed[max(0, y - min_distance):y + min_distance + 1,
                    max(0, x - min_distance):x + min_distance + 1] = False
            if max_corners and len(corners) == max_corners:
                break
    return np.array(corners, np.float32)
# --8<-- [end:nms]


def draw(gray, pts, color=(0, 0, 255)):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in np.asarray(pts).reshape(-1, 2):
        cv2.circle(out, (int(round(x)), int(round(y))), 4, color, 1, cv2.LINE_AA)
    return out


R = harris_response(img)
candidates = np.argwhere(R > 0.01 * R.max())[:, ::-1]
harris = greedy_nms(R, 0.01, 8)
shi = greedy_nms(shi_tomasi_response(img), 0.05, 8)
print(f"Harris: {len(candidates)} pixels above threshold -> {len(harris)} corners after NMS")


def show_r(r):
    return cv2.normalize(np.sign(r) * np.abs(r) ** 0.25, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


U.show("harris", [show_r(R), draw(img, candidates), draw(img, harris), draw(img, shi, (0, 160, 0))],
       ["R (white corner, dark edge)", "R > 1% of max", "Harris + NMS", "Shi-Tomasi + NMS"], cols=4, size=3)

# --8<-- [start:opencv]
R_cv = cv2.cornerHarris(img.astype(np.float32), blockSize=5, ksize=3, k=0.05)   # the map only
pts = cv2.goodFeaturesToTrack(img, maxCorners=50, qualityLevel=0.05, minDistance=8,
                              blockSize=5, useHarrisDetector=False)          # Shi-Tomasi + NMS
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
refined = cv2.cornerSubPix(img, pts.copy(), winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
# --8<-- [end:opencv]
print("mean sub-pixel shift: %.2f px" % np.linalg.norm(refined - pts, axis=2).mean())

zoom_box = (slice(170, 230), slice(60, 130))
z = cv2.resize(img[zoom_box], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
zv = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
for p, c in [(pts, (0, 0, 255)), (refined, (0, 200, 0))]:
    for x, y in p.reshape(-1, 2):
        if 60 <= x < 130 and 170 <= y < 230:
            cv2.circle(zv, (int((x - 60 + .5) * 6), int((y - 170 + .5) * 6)), 5, c, 2)
U.show("subpix", [draw(img, pts), zv], ["goodFeaturesToTrack", "zoom: integer (red) vs cornerSubPix (green)"], size=4)

frame0 = U.imread("kitti/000000.png", cv2.IMREAD_GRAYSCALE)
frame1 = U.imread("kitti/000002.png", cv2.IMREAD_GRAYSCALE)


# --8<-- [start:ncc]
def ncc(a, b):
    """Zero-mean normalized cross-correlation of two equal-size patches, in [-1, 1]."""
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def track_ncc(img0, img1, points, half=7, search=20):
    """For each point, find the patch of img1 (within +-search px) most similar to its patch in img0."""
    out = []
    for x, y in points.astype(int):
        tpl = img0[y - half:y + half + 1, x - half:x + half + 1]
        best, best_xy = -1.0, (x, y)
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                yy, xx = y + dy, x + dx
                win = img1[yy - half:yy + half + 1, xx - half:xx + half + 1]
                if win.shape != tpl.shape:
                    continue
                score = ncc(tpl, win)
                if score > best:
                    best, best_xy = score, (xx, yy)
        out.append((*best_xy, best))
    return np.array(out, np.float32)
# --8<-- [end:ncc]

margin = 30
p0 = cv2.goodFeaturesToTrack(frame0, 60, 0.02, 15).reshape(-1, 2)
p0 = p0[(p0[:, 0] > margin) & (p0[:, 0] < frame0.shape[1] - margin) &
        (p0[:, 1] > margin) & (p0[:, 1] < frame0.shape[0] - margin)][:25]
tracked = track_ncc(frame0, frame1, p0)

# --8<-- [start:matchtemplate]
x, y = p0[0].astype(int)
tpl = frame0[y - 7:y + 8, x - 7:x + 8]
area = frame1[y - 27:y + 28, x - 27:x + 28]                       # +-20 px search window
scores = cv2.matchTemplate(area, tpl, cv2.TM_CCOEFF_NORMED)       # same NCC, vectorised
_, best_score, _, best_loc = cv2.minMaxLoc(scores)
# --8<-- [end:matchtemplate]
print("point 0: ours", tracked[0, :2], "score %.3f" % tracked[0, 2],
      "| matchTemplate", (x - 20 + best_loc[0], y - 20 + best_loc[1]), "score %.3f" % best_score)

# --8<-- [start:lk]
p0_lk = p0.reshape(-1, 1, 2)
p1_lk, status, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0_lk, None,
                                              winSize=(21, 21), maxLevel=3)
# forward-backward check: track back to frame 0 and demand that we land where we started
p0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(frame1, frame0, p1_lk, None,
                                                   winSize=(21, 21), maxLevel=3)
fb_error = np.linalg.norm(p0_back - p0_lk, axis=2).ravel()
good = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < 1.0)
# --8<-- [end:lk]
print(f"LK kept {good.sum()}/{len(p0)} points after the forward-backward check; median |NCC - LK| = "
      f"{np.median(np.linalg.norm(tracked[good, :2] - p1_lk.reshape(-1, 2)[good], axis=1)):.2f} px")

vis = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
for (xa, ya), (xb, yb, s) in zip(p0, tracked):
    cv2.arrowedLine(vis, (int(xa), int(ya)), (int(xb), int(yb)), (0, 0, 255), 2, tipLength=0.3)
vis2 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
for (xa, ya), (xb, yb), ok in zip(p0, p1_lk.reshape(-1, 2), good):
    if ok:
        cv2.arrowedLine(vis2, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 0), 2, tipLength=0.3)
U.show("tracking", [vis, vis2], ["NCC template matching (ours)", "pyramidal Lucas-Kanade + forward-backward check"],
       cols=1, size=9)
