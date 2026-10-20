---
title: Structure from Motion
summary: Recover the camera's motion and a 3-D map of the scene from a single moving camera — essential matrix, cheirality, triangulation and PnP — on 15 frames of the KITTI driving dataset.
topics: essential matrix · recoverPose · triangulation · solvePnPRansac · scale ambiguity · visual odometry
script: 16_structure_from_motion.py
sources: UFRN activity 07
---

## One camera, two positions

Stereo needs two cameras at a known distance. A single *moving* camera also sees the scene from two positions — we just don't know the motion between them. **Structure from Motion** (SfM) estimates both at once: the camera poses (*motion*) and the 3-D points (*structure*). Run frame after frame, it is **visual odometry**, the core of visual SLAM (lesson 17).

Our data: the first 15 frames of KITTI sequence 00, a car driving down a street, with the camera's calibration.

!!! bug "The calibration has to be the right one"
    In my original activity I loaded `intrinsics.yml`, which was in fact the calibration of OpenCV's 640×480 stereo samples ($f = 535$), not of KITTI ($f = 718.9$, images 1241×376). The reconstruction "worked" but its geometry was distorted. Everything downstream depends on $K$.

```python
--8<-- "16_structure_from_motion.py:data"
```

## The essential matrix

For two views of the same calibrated camera, with relative rotation $R$ and translation $\mathbf{t}$, corresponding points in normalised coordinates $\hat{\mathbf{x}} = K^{-1}\mathbf{x}$ satisfy the **epipolar constraint**

$$\hat{\mathbf{x}}_2^\top E\, \hat{\mathbf{x}}_1 = 0, \qquad E = [\mathbf{t}]_\times R$$

$E$ (5 degrees of freedom) can be estimated from 5 matches; `findEssentialMat` runs Nistér's 5-point algorithm inside RANSAC.

Decomposing $E$ gives **four** possible $(R, \mathbf{t})$: two rotations × two signs of $\mathbf{t}$. Only one places the scene **in front of both cameras** (the *cheirality* condition). `recoverPose` triangulates the points for each candidate and keeps the right one:

```python
--8<-- "16_structure_from_motion.py:init"
```

```text
init: 1209 matches, 912 pass RANSAC + cheirality
R ~ I (small rotation): [-0.0023  0.0041 -0.0019] | t direction: [-0.021 -0.016 -1.   ]
the 4 candidates: (R1, t), (R1, -t), (R2, t), (R2, -t); R1 angle 0.3 deg, R2 angle 179.9 deg
```

!!! bug "decomposeEssentialMat is not enough"
    My first version took `R, _, t = cv2.decomposeEssentialMat(E)`: the first rotation and the translation, without checking which of the four combinations is physical. Here one of the rotations is a 180° flip — picking it puts the whole scene behind the camera.

Note that $\|\mathbf{t}\| = 1$: from images alone, **the scale is unknown**. A street seen from 1 m of motion looks exactly like a twice larger street seen from 2 m. Monocular reconstructions are only defined up to scale; we will measure everything in units of the first baseline. (Stereo, an IMU, or a known object size fixes the scale.)

## Triangulation

Each camera $k$ has a projection matrix $P_k = K\,[R_k \mid \mathbf{t}_k]$. A point seen at $\mathbf{x}_a$ and $\mathbf{x}_b$ lies at the intersection of two rays; `triangulatePoints` solves the linear system $\mathbf{x} \times P\mathbf{X} = \mathbf{0}$ for both views with the SVD (a DLT, like lesson 13). Rays never meet exactly, so we keep only points in front of both cameras with a small reprojection error:

```python
--8<-- "16_structure_from_motion.py:triangulate"
```

```python
--8<-- "16_structure_from_motion.py:map"
```

## Adding frames with PnP

For a new frame we should **not** estimate a new essential matrix: its translation would again have length 1, and the scale would drift at every frame. Instead we use the map: the new frame sees points we have already triangulated, so its pose is a **Perspective-n-Point** problem (lesson 14), solved robustly with RANSAC, and its scale is automatically consistent with the map.

The key is the bookkeeping: for each frame, a dictionary *keypoint index → map point index*. A match between frame $k-1$ and frame $k$ whose keypoint in $k-1$ is in the map gives a 2-D/3-D correspondence for frame $k$.

```python
--8<-- "16_structure_from_motion.py:incremental"
```

!!! bug "And the bug that made my first version wrong"
    My first `align_points` used `points_3D[match.queryIdx]`, assuming the $i$-th 3-D point belonged to the $i$-th keypoint of the new match list. It does not: the indices of different match lists are unrelated. Without explicit tracks, PnP receives random 2-D/3-D pairs.

```text
frame  2: PnP on  559 points ( 546 inliers), map size 1462
...
frame 14: PnP on  627 points ( 573 inliers), map size 6981
```

The camera centre is $C = -R^\top \mathbf{t}$ (the point that $[R \mid \mathbf{t}]$ sends to the origin):

```python
--8<-- "16_structure_from_motion.py:centres"
```

```text
camera centres (x, z): [[0.0, 0.0], [0.02, 1.0], [0.01, 2.02], ... [-0.45, 14.54], [-0.52, 15.76]]
step length relative to the first one: mean 1.13, std 0.07
```

The trajectory is a straight line forward along $Z$, as it should be for a car driving down a street, with steps of consistent length (the car accelerating slightly).

!fig 16-reconstruction | Top: the map points (building façades, parked cars, trees) and the 15 camera positions. Right: the same in 3-D.

!fig 16-reprojection | Validation: the map points re-projected into the last frame (red lines) land on the observed keypoints (green). Median reprojection error: 0.23 px.

## What is missing for a real system

This is a working visual odometry, but a real one adds:

- **Bundle adjustment** — jointly refine all poses and points to minimise every reprojection error (Ceres, g2o, GTSAM).
- **Keyframes** — skip frames with too little motion (small baseline = bad triangulation).
- **Loop closure** — recognise a place seen before and correct the accumulated drift.
- **Relocalisation** — recover when tracking is lost.

That is what ORB-SLAM does, in the next lesson.

## Exercises

1. Initialise with frames 0 and 2 instead of 0 and 1 (twice the baseline). Do you get more map points? A different scale?
2. Replace SIFT by ORB (`cv2.ORB_create(3000)` + Hamming matcher with the ratio test). How do the reprojection errors compare?
3. Download the KITTI ground-truth poses for sequence 00, align the scale with the first step, and plot the error over the 15 frames.
