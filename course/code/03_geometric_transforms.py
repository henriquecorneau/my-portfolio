"""Lesson 03 - Geometric transforms: rotation, homogeneous coordinates, affine and perspective warps."""
import cv2
import numpy as np

import cvutils as U

U.setup("03")

# --8<-- [start:rotate_point]
def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s, c]])


p = np.array([2, 4])
p_rot = rotation(np.pi / 2) @ p
print(p, "->", p_rot.round(3))   # [2 4] -> [-4. 2.]
# --8<-- [end:rotate_point]

fig, axes = U.figure(1, 1, size=(3.6, 3.6))
ax = axes[0, 0]
for q, c, label in [(p, "#2563eb", "p"), (p_rot, "#dc2626", "R(90°) p")]:
    ax.plot([0, q[0]], [0, q[1]], color=c)
    ax.plot(*q, "o", color=c, label=label)
ax.axhline(0, color="k", lw=.8)
ax.axvline(0, color="k", lw=.8)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.grid(alpha=.3)
ax.legend()
U.finish("rotate-point", fig)


# --8<-- [start:homogeneous]
def T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def S(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)


M = T(1.5, 2.3) @ R(np.pi / 2) @ S(1, 1)      # scale, then rotate, then translate
ph = np.array([2, 4, 1])                      # the point (2, 4) in homogeneous form
q = M @ ph
back = np.linalg.inv(M) @ q                   # the inverse undoes the whole chain
print(q[:2], back[:2])                        # [-2.5  4.3] [2. 4.]

# 3-D works the same way with 4x4 matrices: [[R, t], [0 0 0 1]]
# --8<-- [end:homogeneous]

img = U.imread("basics/rouen.jpg")
h, w = img.shape[:2]


# --8<-- [start:forward]
def warp_forward(img, A):
    """Push every source pixel to its destination: leaves holes."""
    out = np.zeros_like(img)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    src = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])  # 3 x N, (x, y, 1)
    dst = A @ src
    xd, yd = np.round(dst[0]).astype(int), np.round(dst[1]).astype(int)
    ok = (xd >= 0) & (xd < img.shape[1]) & (yd >= 0) & (yd < img.shape[0])
    out[yd[ok], xd[ok]] = img[ys.ravel()[ok], xs.ravel()[ok]]
    return out
# --8<-- [end:forward]


# --8<-- [start:inverse]
def warp_inverse(img, A, out_shape=None):
    """For every *destination* pixel, look up where it comes from (no holes)."""
    oh, ow = out_shape or img.shape[:2]
    A_inv = np.linalg.inv(A)
    ys, xs = np.mgrid[0:oh, 0:ow]
    dst = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    src = A_inv @ dst
    src = src[:2] / src[2]                      # divide by w (needed for perspective)
    xs_src, ys_src = np.round(src).astype(int)  # nearest neighbour lookup
    ok = (xs_src >= 0) & (xs_src < img.shape[1]) & (ys_src >= 0) & (ys_src < img.shape[0])
    out = np.zeros((oh, ow) + img.shape[2:], img.dtype)
    out.reshape(oh * ow, -1)[ok] = img[ys_src[ok], xs_src[ok]].reshape(ok.sum(), -1)
    return out
# --8<-- [end:inverse]


# rotate 30 degrees and scale 1.4x around the image centre
A = T(w / 2, h / 2) @ R(np.deg2rad(30)) @ S(1.4, 1.4) @ T(-w / 2, -h / 2)
U.show("forward-inverse", [warp_forward(img, A), warp_inverse(img, A)],
       ["forward mapping (holes)", "inverse mapping"], size=4.6)

# --8<-- [start:opencv_affine]
M_rot = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=45, scale=1.0)  # 2x3 matrix
rotated = cv2.warpAffine(img, M_rot, (w, h))

shear = np.float32([[1, 0.4, 0], [0, 1, 0]])
sheared = cv2.warpAffine(img, shear, (int(w * 1.4), h))

# an affine map is fully defined by 3 point correspondences
src_tri = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
dst_tri = np.float32([[0, h * 0.33], [w * 0.85, h * 0.25], [w * 0.15, h * 0.7]])
M_aff = cv2.getAffineTransform(src_tri, dst_tri)
affine = cv2.warpAffine(img, M_aff, (w, h))
# --8<-- [end:opencv_affine]
U.show("affine", [rotated, sheared, affine], ["getRotationMatrix2D(45°)", "shear", "getAffineTransform"],
       size=4)

# --8<-- [start:perspective]
road = U.imread("features/estrada.jpg")       # 500 x 281 road picture
# four points on the road surface (a trapezoid in the image)...
src = np.float32([[205, 160], [295, 160], [480, 280], [20, 280]])
# ...that are a rectangle when seen from above
dst = np.float32([[150, 0], [350, 0], [350, 500], [150, 500]])
H = cv2.getPerspectiveTransform(src, dst)     # 3x3 homography
top_view = cv2.warpPerspective(road, H, (500, 500))
top_view_mine = warp_inverse(road, H, (500, 500))
# --8<-- [end:perspective]
print("max |mine - OpenCV|:", np.abs(top_view.astype(int) - top_view_mine).mean().round(2), "(mean)")

annot = road.copy()
cv2.polylines(annot, [src.astype(np.int32)], True, (0, 0, 255), 2)
U.show("perspective", [annot, top_view, top_view_mine],
       ["4 points on the road", "warpPerspective", "warp_inverse (ours)"], size=4)
