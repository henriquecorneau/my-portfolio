"""Lesson 16 - Structure from Motion: essential matrix, triangulation and PnP on KITTI."""
import glob

import cv2
import numpy as np

import cvutils as U

U.setup("16")

# --8<-- [start:data]
fs = cv2.FileStorage(str(U.path("kitti/calib.yml")), cv2.FILE_STORAGE_READ)
K = fs.getNode("K").mat()                      # fx = fy = 718.856, cx = 607.19, cy = 185.22
fs.release()
frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in sorted(glob.glob(str(U.path("kitti/*.png"))))]

sift = cv2.SIFT_create(nfeatures=3000)
features = [sift.detectAndCompute(f, None) for f in frames]   # (keypoints, descriptors) per frame
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def match(i, j, ratio=0.7):
    """Indices (qi, tj) of ratio-test matches between frames i and j."""
    knn = flann.knnMatch(features[i][1], features[j][1], k=2)
    return [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < ratio * n.distance]


def pts(i, idx):
    return np.float32([features[i][0][k].pt for k in idx])
# --8<-- [end:data]

# --8<-- [start:init]
pairs = match(0, 1)
p0, p1 = pts(0, [a for a, _ in pairs]), pts(1, [b for _, b in pairs])

E, inl = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
# E has 4 possible (R, t) decompositions; recoverPose keeps the one that puts the
# triangulated points *in front of both cameras* (the cheirality check).
n_good, R, t, inl = cv2.recoverPose(E, p0, p1, K, mask=inl)
# --8<-- [end:init]
print(f"init: {len(pairs)} matches, {n_good} pass RANSAC + cheirality")
print("R ~ I (small rotation):", np.round(cv2.Rodrigues(R)[0].ravel(), 4), "| t direction:", t.ravel().round(3))

R1s, R2s, t_dec = cv2.decomposeEssentialMat(E)
print("the 4 candidates: (R1, t), (R1, -t), (R2, t), (R2, -t); R1 angle %.1f deg, R2 angle %.1f deg"
      % (np.degrees(np.linalg.norm(cv2.Rodrigues(R1s)[0])), np.degrees(np.linalg.norm(cv2.Rodrigues(R2s)[0]))))


# --8<-- [start:triangulate]
def triangulate(Ra, ta, Rb, tb, xa, xb):
    """Linear triangulation; returns N x 3 world points and a validity mask."""
    Pa = K @ np.hstack([Ra, ta])                  # 3 x 4 projection matrices
    Pb = K @ np.hstack([Rb, tb])
    X = cv2.triangulatePoints(Pa, Pb, xa.T, xb.T)  # 4 x N, homogeneous
    X = (X[:3] / X[3]).T
    # keep points in front of both cameras, not absurdly far, with small reprojection error
    za = (X @ Ra.T + ta.T)[:, 2]
    zb = (X @ Rb.T + tb.T)[:, 2]
    proj = cv2.projectPoints(X, cv2.Rodrigues(Rb)[0], tb, K, None)[0].reshape(-1, 2)
    ok = (za > 0) & (zb > 0) & (za < 80) & (np.linalg.norm(proj - xb, axis=1) < 2.0)
    return X, ok
# --8<-- [end:triangulate]

# --8<-- [start:map]
poses = [(np.eye(3), np.zeros((3, 1))), (R, t)]   # world = camera 0; |t| = 1 sets the scale
points, colors = [], []
track = [dict(), dict()]                           # per frame: keypoint index -> 3-D point index

inl = inl.ravel() > 0
good_pairs = [p for p, ok in zip(pairs, inl) if ok]
X, ok = triangulate(*poses[0], *poses[1], pts(0, [a for a, _ in good_pairs]), pts(1, [b for _, b in good_pairs]))
for (a, b), x, valid in zip(good_pairs, X, ok):
    if valid:
        track[0][a] = track[1][b] = len(points)
        points.append(x)
# --8<-- [end:map]
print(f"initial map: {len(points)} points")

# --8<-- [start:incremental]
for k in range(2, len(frames)):
    pairs = match(k - 1, k)
    track.append(dict())
    # 2-D/3-D correspondences: keypoints of frame k whose partner in frame k-1 is already a map point
    known = [(track[k - 1][a], b) for a, b in pairs if a in track[k - 1]]
    obj = np.float32([points[i] for i, _ in known])
    img = pts(k, [b for _, b in known])
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=2.0,
                                                 iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
    Rk = cv2.Rodrigues(rvec)[0]
    poses.append((Rk, tvec))
    for idx in inliers.ravel():                  # the map point is seen again in frame k
        track[k][known[idx][1]] = known[idx][0]

    # triangulate the matches that are not in the map yet
    new = [(a, b) for a, b in pairs if a not in track[k - 1]]
    X, ok = triangulate(*poses[k - 1], Rk, tvec, pts(k - 1, [a for a, _ in new]), pts(k, [b for _, b in new]))
    for (a, b), x, valid in zip(new, X, ok):
        if valid:
            track[k - 1][a] = track[k][b] = len(points)
            points.append(x)
    print(f"frame {k:2d}: PnP on {len(known):4d} points ({len(inliers):4d} inliers), "
          f"map size {len(points)}")
# --8<-- [end:incremental]

# --8<-- [start:centres]
centres = np.array([(-Rk.T @ tk).ravel() for Rk, tk in poses])   # camera centre C = -R^T t
# --8<-- [end:centres]
steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
print("camera centres (x, z):", centres[:, [0, 2]].round(2).tolist())
print("step length relative to the first one: mean %.2f, std %.2f" % (steps.mean(), steps.std()))

P = np.array(points)
fig, axes = U.figure(2, 1, size=(4.4, 4.4))
ax = axes[0, 0]
near = (P[:, 2] > 0) & (P[:, 2] < 60) & (np.abs(P[:, 0]) < 30)
ax.scatter(P[near, 0], P[near, 2], s=0.6, c="#16a34a", alpha=0.5, label="map points")
ax.plot(centres[:, 0], centres[:, 2], "r^-", ms=5, label="camera centres")
ax.set_xlabel("X (units of the first baseline)")
ax.set_ylabel("Z, forward")
ax.set_aspect("equal")
ax.legend(fontsize=8)
ax.set_title("top view of the reconstruction", fontsize=10)

ax = axes[0, 1]
ax.remove()
ax = fig.add_subplot(1, 2, 2, projection="3d")
sel = near & (np.random.default_rng(0).random(len(P)) < 0.5)
ax.scatter(P[sel, 0], P[sel, 2], -P[sel, 1], s=0.5, c=-P[sel, 1], cmap="viridis")
ax.plot(centres[:, 0], centres[:, 2], -centres[:, 1], "r^-", ms=4)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("-Y (up)")
ax.view_init(elev=25, azim=-70)
ax.set_title("3-D view", fontsize=10)
U.finish("reconstruction", fig)

last = len(frames) - 1
vis = cv2.cvtColor(frames[last], cv2.COLOR_GRAY2BGR)
idx = list(track[last].values())
proj = cv2.projectPoints(P[idx], cv2.Rodrigues(poses[last][0])[0], poses[last][1], K, None)[0].reshape(-1, 2)
obs = pts(last, list(track[last].keys()))
for o, p in zip(obs, proj):
    cv2.circle(vis, tuple(np.int32(o)), 3, (0, 255, 0), -1)
    cv2.line(vis, tuple(np.int32(o)), tuple(np.int32(p)), (0, 0, 255), 1)
err = np.linalg.norm(obs - proj, axis=1)
print(f"frame {last}: {len(idx)} map points observed, median reprojection error {np.median(err):.2f} px")
U.show("reprojection", [vis], [f"frame {last}: observed keypoints (green) and reprojected map points"], size=9)
