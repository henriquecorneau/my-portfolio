---
title: Homography and RANSAC
summary: Estimate the homography between two views of a plane from point matches with the DLT and the SVD, and make it robust to wrong matches with RANSAC — written from scratch, then with cv2.findHomography.
topics: DLT · Hartley normalisation · SVD · RANSAC · inliers · warping
script: 13_homography_ransac.py
sources: UFRN activity 06
---

## The problem

From lesson 12 we have ~150 SIFT matches between a cookie box and a cluttered scene. Many are wrong. Since the box face is a **plane**, the correct matches are all related by one homography $H$ (lesson 03): $\mathbf{x}' \sim H \mathbf{x}$. We want $H$ — and to know which matches agree with it.

## The Direct Linear Transform

Write $H$ with rows $\mathbf{h}_1^\top, \mathbf{h}_2^\top, \mathbf{h}_3^\top$. For one correspondence $(x, y) \to (u, v)$:

$$u = \frac{\mathbf{h}_1^\top \mathbf{x}}{\mathbf{h}_3^\top \mathbf{x}}, \quad v = \frac{\mathbf{h}_2^\top \mathbf{x}}{\mathbf{h}_3^\top \mathbf{x}}
\;\Longrightarrow\;
\begin{bmatrix} -x & -y & -1 & 0 & 0 & 0 & ux & uy & u \\ 0 & 0 & 0 & -x & -y & -1 & vx & vy & v \end{bmatrix} \mathbf{h} = \mathbf{0}$$

Each correspondence gives two linear equations in the 9 entries of $\mathbf{h}$. Stack $n \ge 4$ of them into $A\mathbf{h} = \mathbf{0}$. Since $H$ is defined up to scale, we look for the unit vector that minimises $\|A\mathbf{h}\|$: the **right singular vector of $A$ with the smallest singular value**, i.e. the last row of $V^\top$ in $A = U \Sigma V^\top$.

One more ingredient makes it numerically sound: **Hartley normalisation**. Pixel coordinates are in the hundreds, so the columns of $A$ differ by orders of magnitude ($x$ vs $ux$). Translate and scale the points so their centroid is at the origin and their mean distance is $\sqrt 2$, solve, then undo: $H = T'^{-1} \tilde H T$.

```python
--8<-- "13_homography_ransac.py:dlt"
```

## RANSAC

Least squares on *all* matches is ruined by the outliers. **RANSAC** (Fischler & Bolles, 1981) turns the problem around:

1. pick a **minimal sample** at random (4 matches for a homography) and fit $H$;
2. count the **inliers**: matches whose reprojection error is below a threshold $t$;
3. repeat, keep the $H$ with the most inliers, and refit it on all of them.

How many iterations? If $w$ is the inlier fraction, a sample of 4 is all-inlier with probability $w^4$. To get at least one good sample with confidence $p$:

$$N = \frac{\log(1 - p)}{\log(1 - w^4)}$$

$w$ is unknown, so we update $N$ with the best inlier fraction seen so far — RANSAC stops early when the data is good.

```python
--8<-- "13_homography_ransac.py:ransac"
```

We use the **symmetric transfer error** — forward and backward reprojection — and skip degenerate samples (repeated or collinear points give a singular $H$).

```text
157 tentative SIFT matches
RANSAC: 71/157 inliers after 135 iterations
OpenCV: 90 inliers
corner disagreement ours vs OpenCV (px): [1.  0.2 0.4 0.4]
```

With 45% inliers, theory says $N \approx 110$ iterations for 99% confidence, and we stopped after 135. OpenCV counts more inliers because its error is one-way and not squared-and-summed, but both homographies put the box corners within 1 px of each other.

!fig 13-inliers | RANSAC inliers in green, outliers in red. Every green line is consistent with one plane-to-plane mapping.

## The OpenCV way

```python
--8<-- "13_homography_ransac.py:opencv"
```

`mask` marks the inliers. Other robust methods are available: `cv2.LMEDS` (no threshold, needs > 50% inliers) and `cv2.USAC_MAGSAC` (newer, often better).

## Using the homography

```python
--8<-- "13_homography_ransac.py:warp"
```

!fig 13-result | Least squares on all matches (red) is nonsense. RANSAC finds the box (green); warping the box into the scene overlays it exactly; $H^{-1}$ "un-perspectives" the scene into a front view of the box.

The same three steps — features, matches, robust model fitting — are the backbone of panorama stitching, augmented reality and, with a different model (the essential matrix), of 3-D reconstruction (lesson 16).

## Exercises

1. Remove the Hartley normalisation and compare the corner positions. On which images does it matter most?
2. Plot the number of RANSAC iterations as a function of the ratio-test threshold (0.6 to 0.9). Explain the curve.
3. Stitch two overlapping photos of a building: estimate $H$, warp one image into the other's frame on a larger canvas, and blend the overlap.
