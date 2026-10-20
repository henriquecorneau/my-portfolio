---
title: Corners and tracking
summary: Corners are points you can find again in another image. Detect them with Harris and Shi-Tomasi from the structure tensor, refine them to sub-pixel accuracy, and follow them between video frames with template matching and Lucas-Kanade.
topics: structure tensor · Harris · Shi-Tomasi · NMS · cornerSubPix · NCC · optical flow
script: 11_corners_tracking.py
sources: INSA TIM lab 7, UFRN activity 05
---

## Why corners?

To track motion, match two photos or calibrate a camera we need points that can be **found again**. A pixel in a flat region looks like all its neighbours; a pixel on an edge can slide along the edge without changing; only at a **corner** does any small move change the local appearance.

Make that precise: shift a window by $(u, v)$ and measure the change

$$E(u, v) = \sum_{x, y} w(x, y)\, [I(x + u, y + v) - I(x, y)]^2$$

A first-order Taylor expansion, $I(x+u, y+v) \approx I + I_x u + I_y v$, turns it into a quadratic form:

$$E(u, v) \approx \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}, \qquad
M = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix}$$

$M$ is the **structure tensor**. Its two eigenvalues $\lambda_1, \lambda_2$ say how much $E$ grows in the two principal directions: both small → flat; one large → edge; both large → **corner**.

## Harris and Shi-Tomasi

Computing eigenvalues at every pixel used to be expensive, so Harris and Stephens (1988) used a score built from the determinant and the trace, which are cheap:

$$R = \det M - k\,(\operatorname{tr} M)^2 = \lambda_1 \lambda_2 - k\,(\lambda_1 + \lambda_2)^2, \qquad k \approx 0.04\text{–}0.06$$

$R$ is large and positive at corners, negative on edges, near zero on flat areas. Shi and Tomasi (1994) used the smallest eigenvalue directly, $\min(\lambda_1, \lambda_2) = \tfrac{1}{2}\big(\operatorname{tr} M - \sqrt{(\operatorname{tr} M)^2 - 4 \det M}\big)$, which tracks better.

```python
--8<-- "11_corners_tracking.py:structure"
```

The window $w$ is a Gaussian, so the sums become Gaussian blurs of the gradient products.

## Non-maximum suppression

Thresholding $R$ gives *blobs* of candidates around each corner. Greedy NMS keeps the strongest candidate, forbids a square around it, and moves on:

```python
--8<-- "11_corners_tracking.py:nms"
```

```text
Harris: 554 pixels above threshold -> 41 corners after NMS
```

!fig 11-harris | $R$ (white = corner, dark = edge); the 554 candidates; 41 corners after NMS; Shi-Tomasi with the same NMS.

## The OpenCV way, and sub-pixel accuracy

```python
--8<-- "11_corners_tracking.py:opencv"
```

`goodFeaturesToTrack` is Shi-Tomasi (or Harris) + NMS + "keep the N best". Corners found on the pixel grid are only accurate to ±0.5 px, and the blur moves them further; `cornerSubPix` refines each one by finding the point where the image gradients around it are orthogonal to the vector towards it. Here corners moved by 1.9 px on average — huge for calibration (lesson 14), where we need ~0.1 px.

!fig 11-subpix | Integer corners (red) and refined ones (green) on a zoom of the blocks.

## Tracking by template matching

To follow a corner into the next frame, take a small patch around it and look for the most similar patch nearby. The similarity that does not care about brightness or contrast changes is the **zero-mean normalised cross-correlation**:

$$\mathrm{NCC}(T, W) = \frac{\sum (T - \bar T)(W - \bar W)}{\sqrt{\sum (T - \bar T)^2 \, \sum (W - \bar W)^2}} \in [-1, 1]$$

```python
--8<-- "11_corners_tracking.py:ncc"
```

`cv2.matchTemplate` computes the NCC at every position of the search window at once:

```python
--8<-- "11_corners_tracking.py:matchtemplate"
```

Both find the same position with the same score (0.849).

## Lucas-Kanade optical flow

Template search costs $O(\text{window}^2)$ per point. **Lucas-Kanade** instead assumes brightness constancy, $I_x u + I_y v + I_t = 0$, writes one such equation per pixel of a small window and solves the least-squares system $M\,[u, v]^\top = -[\sum I_x I_t, \sum I_y I_t]^\top$ — the same structure tensor $M$. That is why it needs corners: on an edge $M$ is singular (the aperture problem). An image **pyramid** handles large motions by solving coarse-to-fine.

```python
--8<-- "11_corners_tracking.py:lk"
```

The **forward-backward check** tracks every point back to the first frame and rejects those that do not return home (here 2 of 25). It is the cheapest outlier filter there is.

!fig 11-tracking | KITTI frames 0 → 2. Points move away from the centre of the image as the car drives forward. NCC and Lucas-Kanade agree to 0.6 px (median) on the points LK kept.

## Exercises

1. Plot $\lambda_1$ vs $\lambda_2$ (computed with `np.linalg.eigvalsh` on $M$) for every pixel of `blox.jpg` and colour the points by Harris' $R$.
2. Track 200 corners through all 15 KITTI frames with LK, re-detecting new corners when fewer than 100 remain, and draw the tracks.
3. Make `track_ncc` vectorised with `cv2.matchTemplate` and compare speed.
