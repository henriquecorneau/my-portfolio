---
title: Camera model and calibration
summary: Write down how a 3-D point becomes a pixel — extrinsics, perspective, lens distortion, intrinsics — then recover the camera's parameters from chessboard photos, undistort images and place a virtual cube on the board.
topics: pinhole model · intrinsics K · distortion · calibrateCamera · reprojection error · solvePnP
script: 14_camera_calibration.py
sources: UFRN activity 02
---

## From the world to a pixel

Every measurement we want to make in 3-D — a distance, a pose, a map — depends on knowing exactly how the camera turns rays into pixels. The **pinhole model** does it in four steps:

1. **Extrinsics** — express the world point in the camera frame: $\mathbf{X}_c = R\,\mathbf{X}_w + \mathbf{t}$.
2. **Perspective division** — project on the normalised image plane: $x = X_c / Z_c$, $y = Y_c / Z_c$.
3. **Lens distortion** — real lenses bend straight lines. The Brown–Conrady model, with $r^2 = x^2 + y^2$:
   $$x_d = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2)$$
   (and symmetrically for $y_d$). $k_i$ are radial terms (barrel when $k_1 < 0$), $p_i$ tangential ones.
4. **Intrinsics** — from metric to pixels with the matrix
   $$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \qquad \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} x_d \\ y_d \\ 1 \end{bmatrix}$$
   $f_x, f_y$ are the focal length in pixels, $(c_x, c_y)$ the principal point, $s$ the skew (≈ 0).

```python
--8<-- "14_camera_calibration.py:pinhole"
```

This 15-line function gives exactly the same pixels as `cv2.projectPoints` (max difference 0.0 px).

!fig 14-distortion-model | A flat grid through an ideal pinhole (gray) and through a lens with $k_1 = -0.3$ (red): barrel distortion pulls the edges inwards.

## Calibration = inverting the model

**Calibrating** means finding $K$ and the distortion coefficients (plus a pose per image) from images of an object whose geometry we know exactly: a chessboard. Its inner corners are, in the board's own frame, at $(i \cdot s, j \cdot s, 0)$. Zhang's method (2000) gets a first estimate from the homographies between the board and each image, then refines everything by minimising the **reprojection error** — the distance between the detected corners and the corners projected by the model.

### 1. Detect the corners

```python
--8<-- "14_camera_calibration.py:corners"
```

!!! warning "Inner corners, not squares"
    A board with 10 × 7 squares has **9 × 6 inner corners**: `PATTERN = (9, 6)`. Getting this wrong makes `findChessboardCorners` fail silently on every image.

`cornerSubPix` (lesson 11) is not optional: calibration is only as accurate as the corners.

### 2. Calibrate

```python
--8<-- "14_camera_calibration.py:calibrate"
```

```text
chessboard found in 21 images of size (1280, 720)
RMS reprojection error: 0.438 px
K =
 [[780.61   0.   725.46]
 [  0.   787.5  365.94]
 [  0.     0.     1.  ]]
dist (k1, k2, p1, p2, k3) = [-0.4282  0.2407  0.0013 -0.0013 -0.0992]
```

These are the parameters of my webcam: a wide-angle lens with strong barrel distortion ($k_1 = -0.43$). An RMS error below 0.5 px is a good calibration.

### 3. Check every view

```python
--8<-- "14_camera_calibration.py:per_view"
```

!fig 14-per-view | Per-image reprojection error. A view far above the others usually has a blurred or misdetected board: remove it and recalibrate.

!!! tip "Taking good calibration pictures"
    15–25 images; the board **flat** and sharp; cover the **whole field of view**, especially the corners where distortion is strongest; tilt the board up to ±45° in different directions; vary the distance. If you only show the board in the centre, $k_1, k_2, k_3$ are poorly constrained.

## Undistorting

```python
--8<-- "14_camera_calibration.py:undistort"
```

`alpha` chooses between keeping only valid pixels (`0`, cropped) and keeping every source pixel (`1`, black borders).

!fig 14-undistort | Detected corners and the undistorted image: the edges of the board and the frame of the whiteboard become straight lines.

## Pose estimation: solvePnP

Once $K$ is known, the pose of the board in any new image follows from its 2-D/3-D correspondences: this is the **Perspective-n-Point** problem.

```python
--8<-- "14_camera_calibration.py:pose"
```

!fig 14-pose | A 3 × 3 × 3-square cube drawn on the board with `solvePnP` + `projectPoints`: the core of marker-based augmented reality.

## Saving the result

```python
--8<-- "14_camera_calibration.py:save"
```

The YAML written by `cv2.FileStorage` is the format of the calibration files in lessons 15–17 and in ORB-SLAM3.

## Bonus: does calibration find the truth?

To know whether calibration recovers the real parameters, we need a camera whose parameters we *know*: simulate one. We project the chessboard with a chosen $K$ and distortion from 15 random poses, add 0.2 px of noise, and calibrate:

```python
--8<-- "14_camera_calibration.py:synthetic"
```

```text
synthetic: fx true 900 -> 901.3, k1 true -0.25 -> -0.252, k2 true 0.08 -> 0.083 (RMS 0.28 px)
```

The focal length is recovered within 0.15% and the distortion within a few thousandths — and the RMS error equals the noise we injected ($0.2\sqrt{2} \approx 0.28$ px), the sign of a model that fits perfectly.

## Exercises

1. Calibrate with only the 5 views where the board is closest to the centre. How do $k_1, k_2, k_3$ change?
2. Calibrate your own phone or webcam (print a chessboard, use `cv2.VideoCapture(0)` and press a key to save frames).
3. In the synthetic bonus, increase the noise to 1 px and the number of views to 50. How does the error on $f_x$ scale?
