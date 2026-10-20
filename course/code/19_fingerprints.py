"""Lesson 19 - Case study: fingerprint minutiae extraction and matching."""
import glob
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import thin

import cvutils as U

U.setup("19")

# 8-neighbours in circular order (dy, dx), starting at the top
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
ENDING, BIFURCATION = 0, 1


# --8<-- [start:skeleton]
def skeleton(binary):
    """binary: uint8, ridges black (0) on white (255). Returns a bool 1-pixel-wide skeleton."""
    return thin(binary == 0)
# --8<-- [end:skeleton]


# --8<-- [start:crossing]
def neighbours(sk):
    """8 shifted copies of the skeleton, in RING order (zero padding at the border)."""
    p = np.pad(sk, 1).astype(np.int8)
    h, w = sk.shape
    return [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]


def crossing_number(sk):
    """CN = 1/2 * sum |P_i - P_(i+1)| around the pixel: 1 = ridge ending, 3 = bifurcation."""
    n = neighbours(sk)
    cn = sum(np.abs(n[i] - n[(i + 1) % 8]) for i in range(8)) // 2
    return np.where(sk, cn, 0)
# --8<-- [end:crossing]


# --8<-- [start:hitmiss]
def endings_hit_or_miss(sk):
    """Ridge endings as 8 hit-or-miss patterns: centre on, exactly one neighbour on."""
    img = sk.astype(np.uint8) * 255
    found = np.zeros(sk.shape, bool)
    for dy, dx in RING:
        k = -np.ones((3, 3), np.int32)
        k[1, 1] = 1
        k[1 + dy, 1 + dx] = 1
        found |= cv2.morphologyEx(img, cv2.MORPH_HITMISS, k) > 0
    return found
# --8<-- [end:hitmiss]


# --8<-- [start:orientation]
def trace(sk, y, x, first, steps=10):
    """Walk along the skeleton from (y, x) through `first`; return the last pixel reached."""
    prev, cur = (y, x), first
    for _ in range(steps):
        nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in RING
               if 0 <= cur[0] + dy < sk.shape[0] and 0 <= cur[1] + dx < sk.shape[1]
               and sk[cur[0] + dy, cur[1] + dx] and (cur[0] + dy, cur[1] + dx) not in (prev, (y, x))]
        if len(nxt) != 1:                  # another minutia or the end of the ridge
            break
        prev, cur = cur, nxt[0]
    return cur


def minutiae(sk):
    """N x 4 array (x, y, angle, type). Angle of an ending: direction from the ridge to its tip.
    Angle of a bifurcation: direction of the branch opposite to the two closest ones."""
    cn = crossing_number(sk)
    out = []
    for y, x in zip(*np.nonzero((cn == 1) | (cn == 3))):
        branches = [(y + dy, x + dx) for dy, dx in RING
                    if 0 <= y + dy < sk.shape[0] and 0 <= x + dx < sk.shape[1] and sk[y + dy, x + dx]]
        ends = [trace(sk, y, x, b) for b in branches]
        angles = [np.arctan2(y - ey, x - ex) for ey, ex in ends]     # pointing towards the minutia
        if cn[y, x] == 1 and len(angles) == 1:
            out.append((x, y, angles[0], ENDING))
        elif cn[y, x] == 3 and len(angles) == 3:
            # the "stem" is the branch whose direction differs most from the other two
            d = [sum(abs(np.angle(np.exp(1j * (angles[i] - angles[j])))) for j in range(3)) for i in range(3)]
            out.append((x, y, angles[int(np.argmax(d))], BIFURCATION))
    return np.array(out, np.float64).reshape(-1, 4)
# --8<-- [end:orientation]


# --8<-- [start:filter]
def print_mask(binary, erode_px=15):
    """Convex hull of the ridges, eroded: minutiae near the border of the capture are not reliable."""
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours))
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [hull], -1, 255, cv2.FILLED)
    # erode as if outside the image were background (the default border mode replicates the edge)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px)),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def clean_minutiae(m, mask, min_dist=8):
    """Drop minutiae outside the mask and pairs that are too close (spurs, broken ridges)."""
    m = m[mask[m[:, 1].astype(int), m[:, 0].astype(int)] > 0]
    d = np.linalg.norm(m[:, None, :2] - m[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)
    return m[d.min(axis=1) >= min_dist]


def extract(binary):
    sk = skeleton(binary)
    return clean_minutiae(minutiae(sk), print_mask(binary)), sk
# --8<-- [end:filter]


def draw_minutiae(binary, m, sk=None):
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if sk is not None:
        vis[sk] = (200, 120, 0)
    for x, y, a, t in m:
        c = (0, 0, 255) if t == ENDING else (0, 170, 0)
        p = (int(x), int(y))
        cv2.circle(vis, p, 5, c, 2)
        cv2.line(vis, p, (int(x + 12 * np.cos(a)), int(y + 12 * np.sin(a))), c, 2)
    return vis


# ------------------------------------------------------------------ the FBI database
db = {Path(p).stem: U.imread(f"fingerprints/db/{Path(p).name}", cv2.IMREAD_GRAYSCALE)
      for p in sorted(glob.glob(str(U.path("fingerprints/db/*.tif"))))}
example = db["Ted"]
sk = skeleton(example)
cn = crossing_number(sk)
hm = endings_hit_or_miss(sk)
print(f"crossing number: {np.sum(cn == 1)} endings, {np.sum(cn == 3)} bifurcations; "
      f"hit-or-miss endings: {hm.sum()} (identical: {np.array_equal(hm, cn == 1)})")
raw = minutiae(sk)
clean, _ = extract(example)
print(f"Ted: {len(raw)} raw minutiae -> {len(clean)} after border and distance filtering")

crop = (slice(130, 260), slice(120, 250))
zoom = cv2.cvtColor(example, cv2.COLOR_GRAY2BGR)
zoom[sk] = (0, 0, 255)
U.show("skeleton", [example, cv2.resize(zoom[crop], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                    draw_minutiae(example, raw), draw_minutiae(example, clean)],
       ["binary print", "zoom: thinned ridges", f"all CN minutiae ({len(raw)})", f"filtered ({len(clean)})"],
       cols=4, size=3)

db_minutiae = {name: extract(img)[0] for name, img in db.items()}


# ------------------------------------------------------------------ crime-scene prints
# --8<-- [start:preprocess]
def preprocess(img):
    f = img.astype(np.float32)
    f -= np.median(f, axis=1, keepdims=True)              # 1. horizontal banding: remove row offsets
    f -= cv2.GaussianBlur(f, (0, 0), 25)                  # 2. smooth illumination gradient
    lo, hi = np.percentile(f, [1, 99])
    if abs(hi) > abs(lo):                                 # 3. polarity: ridges must be the dark side
        f = -f
    f = cv2.GaussianBlur(f, (0, 0), 1.2)                  # 4. light denoising

    # 5. where is the print? ridges give a high local standard deviation
    mean, sq = cv2.blur(f, (21, 21)), cv2.blur(f * f, (21, 21))
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    region = (std > 0.2 * std.max()).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region)
    region = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])   # the biggest blob: drops stains

    # 6. binarise: a ridge pixel is darker than its neighbourhood (f was high-passed)
    out = np.where((f < 0) & region, 0, 255).astype(np.uint8)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
# --8<-- [end:preprocess]


suspects = {Path(p).stem: U.imread(f"fingerprints/suspects/{Path(p).name}", cv2.IMREAD_GRAYSCALE)
            for p in sorted(glob.glob(str(U.path("fingerprints/suspects/*.tif"))))}
prepared = {k: preprocess(v) for k, v in suspects.items()}
U.show("preprocess", [cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX) for v in suspects.values()]
       + list(prepared.values()),
       [f"{k} (stretched)" for k in suspects] + ["pre-processed"] * len(prepared), cols=5, size=2.8)


# --8<-- [start:match]
def match_score(A, B, radius=12.0, angle_tol=np.deg2rad(30)):
    """Alignment-based matching. Every pair of ridge endings (a, b) proposes a rigid motion
    (rotation + translation) taking a onto b; we keep the motion that makes the most minutiae
    of A land on a minutia of B of the same type and similar direction."""
    best, best_pairs = 0, []
    ea, eb = A[A[:, 3] == ENDING], B[B[:, 3] == ENDING]
    for a in ea:
        dth = eb[:, 2] - a[2]                                     # one rotation per candidate b
        c, s = np.cos(dth), np.sin(dth)
        ax, ay = A[:, 0] - a[0], A[:, 1] - a[1]                   # A centred on a
        x = c[:, None] * ax - s[:, None] * ay + eb[:, 0:1]        # (len(eb), len(A))
        y = s[:, None] * ax + c[:, None] * ay + eb[:, 1:2]
        th = A[None, :, 2] + dth[:, None]
        dist = np.hypot(x[..., None] - B[:, 0], y[..., None] - B[:, 1])     # (eb, A, B)
        dang = np.abs(np.angle(np.exp(1j * (th[..., None] - B[:, 2]))))
        ok = (dist < radius) & (dang < angle_tol) & (A[None, :, None, 3] == B[:, 3])
        counts = np.minimum(ok.any(axis=2).sum(axis=1), ok.any(axis=1).sum(axis=1))
        k = int(np.argmax(counts))
        if counts[k] > best:
            best = int(counts[k])
            best_pairs = [(i, int(np.argmax(ok[k, i]))) for i in range(len(A)) if ok[k, i].any()]
    return best ** 2 / (len(A) * len(B)), best_pairs        # fraction of both sets explained
# --8<-- [end:match]

# --8<-- [start:identify]
results = {}
for name, binary in prepared.items():
    m, _ = extract(binary)
    scores = {person: match_score(m, ref)[0] for person, ref in db_minutiae.items()}
    ranking = sorted(scores, key=scores.get, reverse=True)
    results[name] = (m, ranking, scores)
    print(f"{name}: {len(m):3d} minutiae -> best {ranking[0]} ({scores[ranking[0]]:.3f}), "
          f"then {ranking[1]} ({scores[ranking[1]]:.3f}), {ranking[2]} ({scores[ranking[2]]:.3f})")
# --8<-- [end:identify]

fig, axes = U.figure(1, 1, size=(9, 3.2))
ax = axes[0, 0]
names = list(db_minutiae)
mat = np.array([[results[s][2][n] for n in names] for s in results])
im = ax.imshow(mat, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=60, fontsize=8)
ax.set_yticks(range(len(results)))
ax.set_yticklabels(list(results), fontsize=8)
fig.colorbar(im, ax=ax, label="match score")
U.finish("scores", fig)

panels, titles = [], []
for name in list(results)[:5]:
    m, ranking, scores = results[name]
    best = ranking[0]
    _, pairs = match_score(m, db_minutiae[best])
    left = draw_minutiae(prepared[name], m)
    right = draw_minutiae(db[best], db_minutiae[best])
    pair = np.hstack([left, right])
    for i, j in pairs:
        p = (int(m[i, 0]), int(m[i, 1]))
        q = (int(db_minutiae[best][j, 0]) + left.shape[1], int(db_minutiae[best][j, 1]))
        cv2.line(pair, p, q, (255, 160, 0), 1, cv2.LINE_AA)
    panels.append(pair)
    titles.append(f"{name} ↔ {best}: {len(pairs)} matched minutiae")
U.show("matches", panels, titles, cols=1, size=7)
