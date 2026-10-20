"""Lesson 13 - Homography estimation (DLT) and RANSAC, from scratch."""
import cv2
import numpy as np

import cvutils as U

U.setup("13")

box = U.imread("features/match2.png", cv2.IMREAD_GRAYSCALE)     # the object, seen from the front
scene = U.imread("features/match1.png", cv2.IMREAD_GRAYSCALE)   # the object somewhere in a cluttered scene

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(box, None)
kp2, des2 = sift.detectAndCompute(scene, None)
knn = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)).knnMatch(des1, des2, k=2)
matches = [m for m, n in knn if m.distance < 0.8 * n.distance]
src = np.float32([kp1[m.queryIdx].pt for m in matches])   # N x 2, in the box image
dst = np.float32([kp2[m.trainIdx].pt for m in matches])   # N x 2, in the scene
print(f"{len(matches)} tentative SIFT matches")


# --8<-- [start:dlt]
def normalize(pts):
    """Hartley normalisation: centroid at 0, mean distance sqrt(2). Returns (pts', T)."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph[:, :2], T


def dlt_homography(src, dst):
    """H such that dst ~ H @ src, from >= 4 correspondences, by solving A h = 0 with the SVD."""
    s, Ts = normalize(src)
    d, Td = normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)                 # right singular vector of the smallest singular value
    H = np.linalg.inv(Td) @ Hn @ Ts           # undo the normalisation
    return H / H[2, 2]


def project(H, pts):
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]             # back from homogeneous coordinates
# --8<-- [end:dlt]


# --8<-- [start:ransac]
def ransac_homography(src, dst, thresh=3.0, confidence=0.99, max_iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(src)
    best_inliers = np.zeros(n, bool)
    iters, i = max_iters, 0
    while i < iters:
        sample = rng.choice(n, 4, replace=False)             # minimal sample
        i += 1
        H = dlt_homography(src[sample], dst[sample])
        if abs(np.linalg.det(H)) < 1e-8:                     # degenerate sample (repeated or collinear points)
            continue
        # symmetric transfer error: forward and backward reprojection
        err = (np.linalg.norm(project(H, src) - dst, axis=1) ** 2
               + np.linalg.norm(project(np.linalg.inv(H), dst) - src, axis=1) ** 2)
        inliers = err < thresh ** 2
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            w = inliers.mean()                               # current inlier ratio estimate
            # iterations needed so that P(at least one all-inlier sample) >= confidence
            iters = min(max_iters, int(np.ceil(np.log(1 - confidence) / np.log(1 - w ** 4 + 1e-12))))
    H = dlt_homography(src[best_inliers], dst[best_inliers])  # refit on all the inliers
    return H, best_inliers, i
# --8<-- [end:ransac]

H_all = dlt_homography(src, dst)                 # least squares on everything: outliers ruin it
H, inliers, n_iter = ransac_homography(src, dst)
print(f"RANSAC: {inliers.sum()}/{len(src)} inliers after {n_iter} iterations")

# --8<-- [start:opencv]
H_cv, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
# --8<-- [end:opencv]
print(f"OpenCV: {int(mask.sum())} inliers")
corners = np.float32([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]])
print("corner disagreement ours vs OpenCV (px):", np.linalg.norm(project(H, corners) - project(H_cv, corners), axis=1).round(1))


def outline(img, H, color):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    poly = project(H, corners).astype(np.int32)
    cv2.polylines(out, [poly], True, color, 4, cv2.LINE_AA)
    return out


vis = cv2.drawMatches(box, kp1, scene, kp2, [m for m, ok in zip(matches, inliers) if not ok], None,
                      matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
vis = cv2.drawMatches(box, kp1, scene, kp2, [m for m, ok in zip(matches, inliers) if ok], vis,
                      matchColor=(0, 200, 0),
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
U.show("inliers", [vis], [f"RANSAC inliers (green, {inliers.sum()}) and outliers (red, {(~inliers).sum()})"], size=8)

# --8<-- [start:warp]
warped = cv2.warpPerspective(box, H, (scene.shape[1], scene.shape[0]))   # the box, as the camera sees it
mask_w = cv2.warpPerspective(np.full_like(box, 255), H, (scene.shape[1], scene.shape[0]))
overlay = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
overlay[mask_w > 0] = (0.5 * overlay[mask_w > 0] + 0.5 * np.array([0, 180, 255])).astype(np.uint8)

rectified = cv2.warpPerspective(scene, np.linalg.inv(H), (box.shape[1], box.shape[0]))  # and back
# --8<-- [end:warp]
scene_all = outline(scene, H_all, (0, 0, 255))
U.show("result", [scene_all, outline(scene, H, (0, 200, 0)), overlay, rectified],
       ["least squares on all matches", "RANSAC (ours)", "box warped into the scene", "scene rectified with H⁻¹"],
       cols=4, size=3.4)
