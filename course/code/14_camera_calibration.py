"""Lesson 14 - The pinhole camera model and calibration with a chessboard."""
import glob

import cv2
import numpy as np

import cvutils as U

U.setup("14")


# --8<-- [start:pinhole]
def project_points(X, K, R, t, dist=(0, 0, 0, 0, 0)):
    """World points (N x 3) -> pixels (N x 2), with the Brown-Conrady distortion model."""
    k1, k2, p1, p2, k3 = dist
    Xc = X @ R.T + t                          # 1. world -> camera frame (extrinsics)
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]   # 2. perspective division (normalised plane)
    r2 = x ** 2 + y ** 2                      # 3. lens distortion
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    yd = y * radial + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]  # 4. intrinsics: metric -> pixels
    v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], axis=1)
# --8<-- [end:pinhole]


K_demo = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
dist_demo = np.array([-0.30, 0.10, 0, 0, 0])
rvec_demo, tvec_demo = np.array([0.1, -0.2, 0.05]), np.array([0.0, 0.0, 6.0])
R_demo, _ = cv2.Rodrigues(rvec_demo)
grid = np.array([[x, y, 0] for y in np.linspace(-3, 3, 13) for x in np.linspace(-4, 4, 17)], float)
ours = project_points(grid, K_demo, R_demo, tvec_demo, dist_demo)
cv_pts, _ = cv2.projectPoints(grid, rvec_demo, tvec_demo, K_demo, dist_demo)
print("projectPoints vs ours, max diff (px):", np.abs(ours - cv_pts.reshape(-1, 2)).max().round(6))

fig, axes = U.figure(1, 1, size=(5.0, 3.2))
ax = axes[0, 0]
ideal = project_points(grid, K_demo, R_demo, tvec_demo)
ax.scatter(*ideal.T, s=6, c="#94a3b8", label="pinhole (no distortion)")
ax.scatter(*ours.T, s=6, c="#dc2626", label="k1 = -0.30 (barrel)")
ax.set_xlim(0, 1280)
ax.set_ylim(720, 0)
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("a flat grid seen through a distorting lens", fontsize=10)
U.finish("distortion-model", fig)

# --8<-- [start:corners]
PATTERN = (9, 6)            # inner corners per row and per column (not squares!)
SQUARE = 1.0                # square side; use real units (e.g. 0.025 m) to get metric extrinsics

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE   # z = 0 plane

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
object_points, image_points, used = [], [], []
for fname in sorted(glob.glob(str(U.path("calibration/*.jpg")))):
    gray = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN)
    if not found:
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)   # sub-pixel accuracy
    object_points.append(objp)
    image_points.append(corners)
    used.append(fname)
image_size = gray.shape[::-1]               # (width, height)
# --8<-- [end:corners]
print(f"chessboard found in {len(used)} images of size {image_size}")

# --8<-- [start:calibrate]
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
print("RMS reprojection error: %.3f px" % rms)
print("K =\n", K.round(2))
print("dist (k1, k2, p1, p2, k3) =", dist.ravel().round(4))
# --8<-- [end:calibrate]

# --8<-- [start:per_view]
per_view = []
for objp_i, imgp_i, r, t in zip(object_points, image_points, rvecs, tvecs):
    projected, _ = cv2.projectPoints(objp_i, r, t, K, dist)
    per_view.append(np.sqrt(np.mean(np.sum((projected - imgp_i) ** 2, axis=2))))
# --8<-- [end:per_view]

fig, axes = U.figure(1, 1, size=(5.0, 2.6))
ax = axes[0, 0]
ax.bar(range(len(per_view)), per_view, color="#2563eb")
ax.axhline(rms, color="#dc2626", lw=1, ls="--", label=f"overall RMS {rms:.2f} px")
ax.set_xlabel("image")
ax.set_ylabel("RMS error (px)")
ax.legend(fontsize=8)
U.finish("per-view", fig)

# --8<-- [start:undistort]
img = cv2.imread(str(U.path("calibration/frame3.jpg")))
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, alpha=0)   # alpha=0: crop black borders
undistorted = cv2.undistort(img, K, dist, None, new_K)

# for video, build the maps once and remap every frame (much faster)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, image_size, cv2.CV_16SC2)
undistorted_fast = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
# --8<-- [end:undistort]

drawn = cv2.drawChessboardCorners(img.copy(), PATTERN, image_points[used.index(str(U.path("calibration/frame3.jpg")))], True)
U.show("undistort", [drawn, undistorted], ["detected corners", "undistorted (alpha = 0)"], size=4.6)

# --8<-- [start:pose]
view = used.index(str(U.path("calibration/frame4.jpg")))
_, rvec, tvec = cv2.solvePnP(objp, image_points[view], K, dist)    # board pose in this image
s = 3 * SQUARE
cube = np.float32([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
                   [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]])  # -z points towards the camera
pix, _ = cv2.projectPoints(cube, rvec, tvec, K, dist)
pix = pix.reshape(-1, 2).astype(int)
ar = cv2.imread(used[view])
cv2.drawContours(ar, [pix[:4]], -1, (0, 200, 0), -1)
for i, j in zip(range(4), range(4, 8)):
    cv2.line(ar, tuple(pix[i]), tuple(pix[j]), (255, 0, 0), 3)
cv2.drawContours(ar, [pix[4:]], -1, (0, 0, 255), 3)
cv2.drawFrameAxes(ar, K, dist, rvec, tvec, 2 * SQUARE, 3)
# --8<-- [end:pose]
U.show("pose", [ar], ["solvePnP + projectPoints: a virtual cube on the board"], size=6)

# --8<-- [start:save]
fs = cv2.FileStorage("camera_calibration.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.write("image_size", np.array(image_size))
fs.release()

fs = cv2.FileStorage("camera_calibration.yaml", cv2.FILE_STORAGE_READ)
K_loaded = fs.getNode("camera_matrix").mat()
fs.release()
# --8<-- [end:save]
assert np.allclose(K, K_loaded)

# --8<-- [start:synthetic]
# Bonus: a camera we *know*, so we can check that calibration recovers it.
rng = np.random.default_rng(1)
K_true = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]], float)
dist_true = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
obj_s, img_s = [], []
for _ in range(15):
    rvec = rng.uniform(-0.5, 0.5, 3)
    tvec = np.array([rng.uniform(-3, -1), rng.uniform(-2, 0), rng.uniform(9, 14)])
    pts, _ = cv2.projectPoints(objp.astype(float), rvec, tvec, K_true, dist_true)
    pts += rng.normal(0, 0.2, pts.shape)                  # 0.2 px of detection noise
    obj_s.append(objp)
    img_s.append(pts.astype(np.float32))
rms_s, K_est, dist_est, _, _ = cv2.calibrateCamera(obj_s, img_s, (1280, 720), None, None)
# --8<-- [end:synthetic]
print("synthetic: fx true 900 -> %.1f, k1 true -0.25 -> %.3f, k2 true 0.08 -> %.3f (RMS %.2f px)"
      % (K_est[0, 0], dist_est.ravel()[0], dist_est.ravel()[1], rms_s))
