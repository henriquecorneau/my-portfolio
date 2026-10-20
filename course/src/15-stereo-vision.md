---
title: Stereo vision
summary: Two calibrated cameras see depth. Calibrate and rectify a stereo pair, compute disparity maps with block matching and SGBM, convert them to metres and point clouds, and georeference an object seen from a boat.
topics: epipolar geometry · stereoCalibrate · rectification · StereoBM / SGBM · depth = fB/d · point cloud · GPS
script: 15_stereo_vision.py
sources: UFRN activities 08 and 09, my PIBIC boat project
---

## Depth from two eyes

Two cameras side by side, separated by a **baseline** $B$, see the same point at slightly different horizontal positions. The difference is the **disparity** $d = u_\text{left} - u_\text{right}$, and similar triangles give the depth directly:

$$Z = \frac{f\,B}{d}$$

with $f$ in pixels and $B$ in metres. Near objects have large disparity, far ones small. Since $d$ is measured with ~1 px error, the depth error grows like $Z^2$: stereo is precise up close and poor far away.

The hard part is finding, for each left pixel, *the same point* in the right image. Epipolar geometry makes it a 1-D search.

## Epipolar geometry

The ray of a left pixel, seen from the right camera, is a line: the **epipolar line**. Its match can only be on that line. Algebraically, corresponding points satisfy

$$\mathbf{x}_r^\top F\, \mathbf{x}_l = 0$$

where $F$ is the $3 \times 3$ **fundamental matrix** (in normalised coordinates it becomes the essential matrix $E = [\mathbf{t}]_\times R$, which we use in lesson 16).

## Part 1 — calibrating a stereo pair

Calibrate each camera alone (lesson 14), then estimate the rotation $R$ and translation $T$ **between** them with images where both see the board, keeping the intrinsics fixed:

```python
--8<-- "15_stereo_vision.py:stereo_calib"
```

```text
13 stereo pairs, RMS 0.448 px
T = [-3.344  0.042  0.053] -> baseline 3.345 squares
```

$R$ is almost the identity (the cameras are parallel) and $T$ is mostly along $x$: the right camera is 3.3 chessboard squares to the right.

!!! bug "What I changed from my first version"
    My first version called `stereoCalibrate` with no initial intrinsics and `flags=0`, so it estimated 30+ parameters at once from 13 image pairs. Calibrating each camera first and then using `CALIB_FIX_INTRINSIC` is more stable.

```python
--8<-- "15_stereo_vision.py:epilines"
```

!fig 15-epilines | Points in the left image and their epipolar lines in the right image: each matching corner lies on its line.

## Rectification

If we warp both images so that epipolar lines become **the same image rows**, matching is a search along a row. That is **rectification**: a homography per image computed by `stereoRectify`, applied (together with undistortion) with `remap`.

```python
--8<-- "15_stereo_vision.py:rectify"
```

!fig 15-rectified | Before: the same corner sits on different rows. After: every corner is on the same row in both images.

`stereoRectify` also returns $P_1, P_2$ (the new projection matrices, $f$ is `P1[0, 0]`) and $Q$, the matrix that turns a disparity into a 3-D point.

## Part 2 — depth on the water

The second dataset is a frame of the **MODD** maritime dataset, filmed from a small boat — the setting of my research project on autonomous boats.

### Check before you rectify

```python
--8<-- "15_stereo_vision.py:check"
```

```text
dy percentiles (5, 50, 95): [-1.  0.  0.]
```

The matched features are already on the same rows: **the MODD frames are distributed rectified**. Rectifying them again with the raw calibration would break them (I tried: no feature stayed within 3 rows of its match). Always check the row offset of a few matches before computing disparities — it takes five lines and saves hours.

```python
--8<-- "15_stereo_vision.py:load"
```

### Disparity: block matching and SGBM

**StereoBM** compares small blocks along each row (sum of absolute differences) and keeps the best. Fast, but it fails on weak texture. **StereoSGBM** (semi-global matching) adds a smoothness cost along several directions: `P1` penalises disparity changes of 1 between neighbours, `P2` larger jumps. It is slower and much denser.

```python
--8<-- "15_stereo_vision.py:disparity"
```

!!! warning "Disparities are fixed-point"
    `compute()` returns `int16` values **multiplied by 16**. Divide by 16 before using them, and treat values ≤ 0 as invalid.

### The distance to the dinghy

Given a bounding box (from a detector like YOLO, in my project), take the **median** disparity inside it — the box also contains water and sky, and the median ignores them — and apply $Z = fB/d$:

```python
--8<-- "15_stereo_vision.py:roi"
```

```text
dinghy: median disparity 47.3 px -> depth 7.21 m, -1.49 m to the side
```

The dinghy's disparity (47 px) matches the median horizontal offset of the ORB matches (46 px) from the rectification check — two independent measurements agreeing.

!fig 15-disparity | Water has little texture: StereoBM (with the dataset's parameters) finds only 3% of valid disparities, SGBM 22%, including the whole dinghy. The black band on the left is where no right-image pixel can match (`numDisparities`).

## Point clouds

$Q$ turns every $(u, v, d)$ into $(X, Y, Z)$; `reprojectImageTo3D` applies it to the whole map. For a rectified pair it is simply

$$Q = \begin{bmatrix} 1 & 0 & 0 & -c_x \\ 0 & 1 & 0 & -c_y \\ 0 & 0 & 0 & f \\ 0 & 0 & 1/B & 0 \end{bmatrix}$$

```python
--8<-- "15_stereo_vision.py:pointcloud"
```

!fig 15-pointcloud | The point cloud from above. The dinghy (red) is at ~7 m; the red points far behind are background pixels inside its bounding box — the reason we used the median. The `.ply` file opens in MeshLab or CloudCompare.

!!! bug "What I changed from my first version"
    My first point cloud reprojected the *colour-mapped, normalised* disparity image (converted back to gray) instead of the real disparity, so the depths were meaningless. `reprojectImageTo3D` needs the float disparity in pixels.

## Georeferencing

The boat knows where it is (GPS, in UTM coordinates) and where it points (magnetometer + local magnetic declination). The object's **bearing** is the boat's heading plus the object's angle in the image, $\operatorname{atan2}(X, Z)$; moving `distance` metres along it gives the object's position. Doing the offset in UTM (metres) and converting to latitude/longitude once with `pyproj` avoids the "1° ≈ 111 km" approximations:

```python
--8<-- "15_stereo_vision.py:georef"
```

```text
boat   45.547620, 13.705387
object 45.547674, 13.705441  (7.4 m at bearing 35.7 deg)
```

This is the pipeline of my [boat perception project](../projects/boat-vision.html), where it runs on every frame with YOLO detections.

## Exercises

1. Plot depth error vs depth for a 1 px disparity error with this camera ($f = 958$ px, $B = 0.356$ m). At what distance is the error 1 m?
2. Tune SGBM (`blockSize`, `uniquenessRatio`, `P1`, `P2`) to get more valid pixels on the water without noise.
3. Compute the dinghy's depth from the ORB matches inside its box only, without any disparity map.
