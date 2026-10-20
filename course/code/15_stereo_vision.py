"""Lesson 15 - Stereo vision: calibration, rectification, disparity, depth and georeferencing."""
import glob

import cv2
import numpy as np
from pyproj import Transformer

import cvutils as U

U.setup("15")

# ---------------------------------------------------------------- part 1: calibrate a stereo pair
# --8<-- [start:stereo_calib]
PATTERN = (9, 6)
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)

lefts = sorted(glob.glob(str(U.path("stereo/left/*.jpg"))))
rights = sorted(glob.glob(str(U.path("stereo/right/*.jpg"))))
obj_pts, left_pts, right_pts = [], [], []
for fl, fr in zip(lefts, rights):
    gl = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(fr, cv2.IMREAD_GRAYSCALE)
    ok_l, cl = cv2.findChessboardCorners(gl, PATTERN)
    ok_r, cr = cv2.findChessboardCorners(gr, PATTERN)
    if ok_l and ok_r:                              # the board must be seen by *both* cameras
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        left_pts.append(cv2.cornerSubPix(gl, cl, (11, 11), (-1, -1), crit))
        right_pts.append(cv2.cornerSubPix(gr, cr, (11, 11), (-1, -1), crit))
        obj_pts.append(objp)
size = gl.shape[::-1]

# 1. intrinsics of each camera alone
_, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, left_pts, size, None, None)
_, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, right_pts, size, None, None)
# 2. extrinsics between them, keeping the intrinsics fixed
rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_pts, left_pts, right_pts, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC)
# --8<-- [end:stereo_calib]
print(f"{len(obj_pts)} stereo pairs, RMS {rms:.3f} px")
print("R =\n", R.round(4))
print("T =", T.ravel().round(3), "-> baseline %.3f squares" % np.linalg.norm(T))

# --8<-- [start:epilines]
img_l = cv2.imread(lefts[0], cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread(rights[0], cv2.IMREAD_GRAYSCALE)
pts_l = left_pts[0][::9].reshape(-1, 2)            # a few corners in the left image
lines_r = cv2.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 1, F).reshape(-1, 3)  # a x + b y + c = 0
# --8<-- [end:epilines]


def draw_epilines(img, lines, pts):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (200, 0, 200), (0, 160, 200), (200, 160, 0)]
    for (a, b, c), p, col in zip(lines, pts, colors):
        x0, x1 = 0, out.shape[1]
        cv2.line(out, (x0, int(-c / b)), (x1, int(-(c + a * x1) / b)), col, 2)
        cv2.circle(out, tuple(np.int32(p)), 6, col, -1)
    return out


left_vis = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
for p, col in zip(pts_l, [(255, 0, 0), (0, 160, 0), (0, 0, 255), (200, 0, 200), (0, 160, 200), (200, 160, 0)]):
    cv2.circle(left_vis, tuple(np.int32(p)), 6, col, -1)
U.show("epilines", [left_vis, draw_epilines(img_r, lines_r, right_pts[0][::9].reshape(-1, 2))],
       ["points in the left image", "their epipolar lines in the right image"], size=4.2)

# --8<-- [start:rectify]
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
map_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
rect_l = cv2.remap(img_l, *map_l, cv2.INTER_LINEAR)
rect_r = cv2.remap(img_r, *map_r, cv2.INTER_LINEAR)
# --8<-- [end:rectify]


def with_rows(a, b, step=40):
    pair = cv2.cvtColor(np.hstack([a, b]), cv2.COLOR_GRAY2BGR)
    for y in range(step // 2, pair.shape[0], step):
        cv2.line(pair, (0, y), (pair.shape[1], y), (0, 200, 0), 1)
    return pair


U.show("rectified", [with_rows(img_l, img_r), with_rows(rect_l, rect_r)],
       ["before: the same corner sits on different rows", "after rectification: epipolar lines are rows"],
       cols=1, size=8)

# ---------------------------------------------------------------- part 2: depth on the boat dataset
# --8<-- [start:check]
left = U.imread("modd/00013824L.jpg")
right = U.imread("modd/00013824R.jpg")
gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)


def row_offsets(img_l, img_r):
    """Vertical offset of RANSAC-filtered ORB matches: ~0 everywhere if the pair is rectified."""
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(img_l, None)
    k2, d2 = orb.detectAndCompute(img_r, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    _, inl = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0)
    inl = inl.ravel() > 0
    return p1[inl, 1] - p2[inl, 1], p1[inl, 0] - p2[inl, 0]


dy, dx = row_offsets(gray_l, gray_r)
print("dy percentiles (5, 50, 95):", np.percentile(dy, [5, 50, 95]).round(2))   # [-1, 0, 0]
# --8<-- [end:check]
print("dx (disparity) percentiles (5, 50, 95):", np.percentile(dx, [5, 50, 95]).round(1))

# --8<-- [start:load]
# The MODD frames are *already rectified*: rectifying them again with the raw calibration
# would break them. We only need the focal length (px) and the baseline (m).
fs = cv2.FileStorage(str(U.path("modd/calibration.yaml")), cv2.FILE_STORAGE_READ)
K = fs.getNode("M1").mat()
T = fs.getNode("T").mat()                    # millimetres
fs.release()
f_px = K[0, 0]                               # ~958 px
cx, cy = K[0, 2], K[1, 2]
baseline_m = np.linalg.norm(T) / 1000        # 0.356 m

# reprojection matrix for a rectified pair: [X Y Z W]^T = Q [u v d 1]^T
Q = np.float64([[1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, f_px],
                [0, 0, 1 / baseline_m, 0]])
# --8<-- [end:load]
print(f"focal length {f_px:.1f} px, baseline {baseline_m:.3f} m")
rect_l = left

# --8<-- [start:disparity]
fs = cv2.FileStorage(str(U.path("modd/stereo-method.yaml")), cv2.FILE_STORAGE_READ)
param = {k: int(fs.getNode(k).real()) for k in
         ["PreFilterSize", "PreFilterCap", "SADWindowSize", "MinDisparity", "NumDisparities",
          "TextureThreshold", "UniquenessRatio", "SpeckleWindowSize", "SpeckleRange", "Disp12MaxDiff"]}
fs.release()

bm = cv2.StereoBM_create(numDisparities=param["NumDisparities"], blockSize=param["SADWindowSize"])
bm.setPreFilterSize(param["PreFilterSize"])
bm.setPreFilterCap(param["PreFilterCap"])
bm.setTextureThreshold(param["TextureThreshold"])
bm.setUniquenessRatio(param["UniquenessRatio"])
bm.setSpeckleWindowSize(param["SpeckleWindowSize"])
bm.setSpeckleRange(param["SpeckleRange"])
bm.setDisp12MaxDiff(param["Disp12MaxDiff"])

block = 5
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=block,
                             P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,   # smoothness penalties
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                             disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# compute() returns fixed-point disparities (x16); invalid pixels are negative
disp_bm = bm.compute(gray_l, gray_r).astype(np.float32) / 16
disp = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16
valid = disp > 0
# --8<-- [end:disparity]
print(f"valid disparities: BM {100 * np.mean(disp_bm > 0):.0f}%, SGBM {100 * valid.mean():.0f}%")


def colorize(d):
    v = np.zeros(d.shape, np.uint8)
    ok = d > 0
    v[ok] = cv2.normalize(d[ok], None, 0, 255, cv2.NORM_MINMAX).ravel()
    out = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
    out[~ok] = 0
    return out


# --8<-- [start:roi]
x, y, w, h = 252, 461, 412, 184                      # the dinghy (e.g. a YOLO bounding box)
roi_disp = disp[y:y + h, x:x + w]
d_obj = np.median(roi_disp[roi_disp > 0])            # median: robust to holes and to background pixels
Z = f_px * baseline_m / d_obj                        # depth = f B / d
X = (x + w / 2 - cx) * Z / f_px                      # lateral offset of the box centre
# # --8<-- [end:roi]
print(f"dinghy: median disparity {d_obj:.1f} px -> depth {Z:.2f} m, {X:+.2f} m to the side")

vis_l = rect_l.copy()
cv2.rectangle(vis_l, (x, y), (x + w, y + h), (0, 0, 255), 3)
vis_d = colorize(disp)
cv2.rectangle(vis_d, (x, y), (x + w, y + h), (255, 255, 255), 3)
cv2.putText(vis_d, f"{Z:.1f} m", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
U.show("disparity", [vis_l, colorize(disp_bm), vis_d],
       ["left image (already rectified)", "StereoBM (MODD parameters)", "StereoSGBM + distance to the dinghy"], size=4.4)

# --8<-- [start:pointcloud]
points = cv2.reprojectImageTo3D(disp, Q)             # metres, because the baseline is in metres
colors = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
mask = valid & (points[..., 2] > 0) & (points[..., 2] < 40)
xyz, rgb = points[mask], colors[mask]


def write_ply(path, xyz, rgb):
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, np.hstack([xyz, rgb]), fmt="%.4f %.4f %.4f %d %d %d")

# write_ply("boat.ply", xyz, rgb)   # open it in MeshLab or CloudCompare
# --8<-- [end:pointcloud]
print(f"point cloud: {len(xyz)} points")

fig, axes = U.figure(1, 1, size=(5.6, 4.2))
ax = axes[0, 0]
sel = np.random.default_rng(0).choice(len(xyz), min(40000, len(xyz)), replace=False)
ax.scatter(xyz[sel, 0], xyz[sel, 2], s=0.4, c=rgb[sel] / 255)
in_roi = np.zeros(disp.shape, bool)
in_roi[y:y + h, x:x + w] = True
roi_xyz = points[mask & in_roi]
ax.scatter(roi_xyz[:, 0], roi_xyz[:, 2], s=1, c="#dc2626", label="dinghy ROI")
ax.plot(0, 0, "k^", ms=10, label="camera")
ax.set_xlabel("X, to the right (m)")
ax.set_ylabel("Z, forward (m)")
ax.set_ylim(0, 40)
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("point cloud seen from above", fontsize=10)
U.finish("pointcloud", fig)

# --8<-- [start:georef]
utm_e, utm_n = np.loadtxt(U.path("modd/gps.txt"))              # boat position, UTM zone 33 N (metres)
mag_x, mag_y, _ = np.loadtxt(U.path("modd/imu.txt"))           # magnetometer reading
declination = 4.41                                             # degrees, Gulf of Koper

heading = np.degrees(-np.arctan2(mag_y, mag_x)) + declination  # where the camera points (deg from north)
bearing = heading + np.degrees(np.arctan2(X, Z))               # add the object's angle in the image
distance = np.hypot(X, Z)                                      # X, Z of the dinghy, from the ROI step

obj_e = utm_e + distance * np.sin(np.radians(bearing))         # metres east
obj_n = utm_n + distance * np.cos(np.radians(bearing))         # metres north
to_wgs84 = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
boat_lon, boat_lat = to_wgs84.transform(utm_e, utm_n)
obj_lon, obj_lat = to_wgs84.transform(obj_e, obj_n)
# --8<-- [end:georef]
print(f"boat   {boat_lat:.6f}, {boat_lon:.6f}")
print(f"object {obj_lat:.6f}, {obj_lon:.6f}  ({distance:.1f} m at bearing {bearing % 360:.1f} deg)")
