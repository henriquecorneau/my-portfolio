"""Lesson 12 - Keypoints and descriptors: FAST, oriented FAST, BRIEF, ORB and SIFT."""
import cv2
import numpy as np

import cvutils as U

U.setup("12")

# --8<-- [start:circle]
# Bresenham circle of radius 3 around the candidate pixel: 16 (dx, dy) offsets, in order.
CIRCLE = [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
          (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
# --8<-- [end:circle]


# --8<-- [start:fast]
def fast_score_map(img, t=20, n=9):
    """FAST-n segment test for every pixel at once.

    A pixel p is a corner if n *contiguous* circle pixels are all brighter than p + t
    or all darker than p - t. Returns a score map (0 = not a corner).
    """
    f = img.astype(np.int16)
    h, w = f.shape
    pad = np.pad(f, 3, mode="edge")
    ring = np.stack([pad[3 + dy:3 + dy + h, 3 + dx:3 + dx + w] for dx, dy in CIRCLE])  # 16 x h x w
    brighter = ring > f + t
    darker = ring < f - t

    def has_arc(flags):
        # circular: does any run of n consecutive True exist among the 16?
        ext = np.concatenate([flags, flags[:n - 1]])               # wrap around
        found = np.zeros((h, w), bool)
        for start in range(16):
            found |= np.all(ext[start:start + n], axis=0)
        return found

    corner = has_arc(brighter) | has_arc(darker)
    score = np.abs(ring - f).sum(axis=0)          # how strongly the ring differs from p
    score[~corner] = 0
    score[:3], score[-3:], score[:, :3], score[:, -3:] = 0, 0, 0, 0
    return score


def nms(score, radius=3):
    """Keep pixels that are the maximum of their (2r+1)^2 neighbourhood."""
    local_max = cv2.dilate(score.astype(np.float32), np.ones((2 * radius + 1,) * 2))
    ys, xs = np.nonzero((score > 0) & (score >= local_max))
    return np.stack([xs, ys], axis=1)
# --8<-- [end:fast]

rubiks = U.imread("features/rubiks_cube.png", cv2.IMREAD_GRAYSCALE)
kp_mine = nms(fast_score_map(rubiks, t=25, n=9))
fast_cv = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True,
                                         type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
kp_cv = fast_cv.detect(rubiks)
print(f"FAST: ours {len(kp_mine)} keypoints, OpenCV {len(kp_cv)}")


def draw_pts(gray, pts, color=(0, 255, 0), r=4):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), r, color, 1, cv2.LINE_AA)
    return out


U.show("fast", [draw_pts(rubiks, kp_mine), draw_pts(rubiks, [k.pt for k in kp_cv], (0, 0, 255))],
       ["our FAST-9 (t=25) + NMS", "cv2.FastFeatureDetector"], size=4.4)


# --8<-- [start:orientation]
def orientation(img, x, y, radius=15):
    """Intensity centroid (oFAST): the angle from the keypoint to the patch's centre of mass."""
    patch = img[y - radius:y + radius + 1, x - radius:x + radius + 1].astype(np.float32)
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = dx ** 2 + dy ** 2 <= radius ** 2           # circular patch: rotation invariant
    m10 = np.sum(dx * patch * disk)
    m01 = np.sum(dy * patch * disk)
    return np.arctan2(m01, m10)
# --8<-- [end:orientation]


# --8<-- [start:brief]
rng = np.random.default_rng(42)
PAIRS = np.clip(rng.normal(0, 31 / 5, size=(256, 4)), -13, 13).round()   # (x1, y1, x2, y2) per bit


def brief(img, x, y, angle=None):
    """256-bit binary descriptor: bit i = I(p1_i) < I(p2_i), on a smoothed image.
    With an angle, the sampling pattern is rotated first ("steered" BRIEF, as in ORB)."""
    pts = PAIRS.reshape(-1, 2)
    if angle is not None:
        c, s = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c, s], [-s, c]])        # rotate every sample point
    pts = np.round(pts).astype(int)
    vals = img[y + pts[:, 1], x + pts[:, 0]].reshape(-1, 2)
    bits = (vals[:, 0] < vals[:, 1]).astype(np.uint8)
    return np.packbits(bits)                           # 32 bytes


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())             # number of different bits
# --8<-- [end:brief]


def mini_orb(gray, n_best=300, steer=True):
    smooth = cv2.GaussianBlur(gray, (5, 5), 2)
    score = fast_score_map(gray, t=20, n=9)
    pts = nms(score, 3)
    pts = pts[(pts[:, 0] >= 25) & (pts[:, 0] < gray.shape[1] - 25) &
              (pts[:, 1] >= 25) & (pts[:, 1] < gray.shape[0] - 25)]
    pts = pts[np.argsort(score[pts[:, 1], pts[:, 0]])[::-1][:n_best]]
    angles = [orientation(smooth, x, y) for x, y in pts]
    des = np.array([brief(smooth, x, y, a if steer else None) for (x, y), a in zip(pts, angles)])
    return pts, des


def match_hamming(d1, d2, max_dist=40):
    """Mutual nearest neighbours ("cross check") under the Hamming distance."""
    dist = np.unpackbits(d1[:, None, :] ^ d2[None, :, :], axis=2).sum(axis=2)
    best12, best21 = dist.argmin(axis=1), dist.argmin(axis=0)
    return [(i, j) for i, j in enumerate(best12) if best21[j] == i and dist[i, j] <= max_dist]


# --8<-- [start:evaluate]
a = U.imread("features/bat.jpg", cv2.IMREAD_GRAYSCALE)
h, w = a.shape
M = cv2.getRotationMatrix2D((w / 2, h / 2), 50, 1.0)    # known ground truth: 50 degree rotation
b = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def precision(pa, pb, matches, tol=3.0):
    """Fraction of matches that land where the known rotation says they should."""
    if not matches:
        return 0.0
    src = np.array([pa[i] for i, _ in matches], np.float32)
    dst = np.array([pb[j] for _, j in matches], np.float32)
    predicted = src @ M[:, :2].T + M[:, 2]
    return float(np.mean(np.linalg.norm(predicted - dst, axis=1) < tol))
# --8<-- [end:evaluate]


panels, titles = [], []
for steer in (False, True):
    pa, da = mini_orb(a, steer=steer)
    pb, db = mini_orb(b, steer=steer)
    m = match_hamming(da, db)
    prec = precision(pa, pb, m)
    correct = [(i, j) for i, j in m if precision(pa, pb, [(i, j)]) == 1.0]
    print(f"mini-ORB steer={steer}: {len(m)} mutual matches, {100 * prec:.0f}% correct")
    kp_a = [cv2.KeyPoint(float(x), float(y), 7) for x, y in pa]
    kp_b = [cv2.KeyPoint(float(x), float(y), 7) for x, y in pb]
    good_dm = [cv2.DMatch(i, j, 0) for i, j in correct]
    bad_dm = [cv2.DMatch(i, j, 0) for i, j in m if (i, j) not in correct]
    vis = cv2.drawMatches(a, kp_a, b, kp_b, bad_dm, None, matchColor=(0, 0, 255),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis = cv2.drawMatches(a, kp_a, b, kp_b, good_dm, vis, matchColor=(0, 200, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    panels.append(vis)
    titles.append(f"{'steered' if steer else 'plain'} BRIEF: {100 * prec:.0f}% of {len(m)} matches correct")
U.show("steered", panels, titles, size=4.6)

# --8<-- [start:orb]
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   # Hamming: XOR + popcount


def orb_matches(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)  # best first
    return kp1, kp2, matches
# --8<-- [end:orb]

cathedral = U.imread("features/cathedraleNoisy.jpeg", cv2.IMREAD_GRAYSCALE)
variants = ["cathedraleZoom.jpeg", "cathedraleBlur.jpeg", "cathedraleLowContrast.jpeg"]
cath_small = cv2.resize(U.imread("features/cathedrale.jpeg", cv2.IMREAD_GRAYSCALE), (450, 300))
tower_tests = [("cathedraleLowContrast.jpeg", "low contrast"), ("cathedraleBlur.jpeg", "blurred"),
               ("cathedraleZoom.jpeg", "zoomed")]
panels, titles = [], []
for name, label in tower_tests:
    other = U.imread(f"features/{name}", cv2.IMREAD_GRAYSCALE)
    kp1, kp2, m = orb_matches(cathedral, other)
    good = [x for x in m if x.distance < 40]
    print(f"ORB noisy tower vs {label:<12}: {len(m)} cross-checked matches, {len(good)} with distance < 40")
    panels.append(cv2.drawMatches(cathedral, kp1, other, kp2, m[:30], None, matchColor=(0, 200, 0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
    titles.append(f"vs {label}: 30 best of {len(m)}")
U.show("orb-robustness", panels, titles, cols=3, size=4)

# --8<-- [start:sift]
sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))   # KD-tree


def sift_matches(img1, img2, ratio=0.75):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    knn = flann.knnMatch(des1, des2, k=2)          # two nearest neighbours per descriptor
    # Lowe's ratio test: keep a match only if it is clearly better than the runner-up
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return kp1, kp2, good
# --8<-- [end:sift]

scene = U.imread("features/match1.png", cv2.IMREAD_GRAYSCALE)
box = U.imread("features/match2.png", cv2.IMREAD_GRAYSCALE)
kp1, kp2, good = sift_matches(box, scene)
_, _, all_pairs = sift_matches(box, scene, ratio=1.01)
print(f"SIFT box vs scene: {len(all_pairs)} nearest neighbours, {len(good)} survive the ratio test")
kpo1, kpo2, orb_m = orb_matches(box, scene)
U.show("sift", [cv2.drawMatches(box, kp1, scene, kp2, good, None, matchColor=(0, 200, 0),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS),
                cv2.drawMatches(box, kpo1, scene, kpo2, orb_m[:len(good)], None, matchColor=(0, 0, 255),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)],
       [f"SIFT + ratio test ({len(good)} matches)", f"ORB, the {len(good)} best cross-checked matches"],
       cols=1, size=8)

other_building = cv2.resize(U.imread("features/mecca_clock_tower.jpg", cv2.IMREAD_GRAYSCALE), (400, 600))
kpa, kpb, wrong = orb_matches(cathedral, other_building)
print(f"ORB cathedral vs a different building: still {len(wrong)} cross-checked 'matches'")
U.show("false-matches", [cv2.drawMatches(cathedral, kpa, other_building, kpb, wrong[:30], None,
                                         matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)],
       ["two different towers: brute force still returns 'best' matches"], size=7)
