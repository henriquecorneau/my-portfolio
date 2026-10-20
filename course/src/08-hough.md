---
title: The Hough transform
summary: Turn edge pixels into lines by letting every pixel vote for all the lines that could pass through it. Detect road lanes, and circles too.
topics: parameter space · accumulator · peak detection · HoughLinesP · HoughCircles
script: 08_hough.py
sources: UFRN lecture 10 and activity 04
---

## From pixels to lines: voting

Edge detection gives us *pixels*. Many tasks need *shapes*: the lanes of a road, the edges of a building, a circular marker. The Hough transform finds them by voting.

A line $y = ax + b$ through a point $(x_0, y_0)$ satisfies $b = -x_0 a + y_0$: in the **parameter space** $(a, b)$, all the lines through one image point form a line. Points that are collinear in the image produce lines that all cross at one $(a, b)$ — the parameters of the line they share.

```python
--8<-- "08_hough.py:duality"
```

!fig 08-duality | Three collinear points (left) become three lines meeting at $(a, b) = (-1, 5)$ (middle). In $(\theta, \rho)$ they become sinusoids, again meeting at one point (right).

The slope $a$ is infinite for vertical lines, so in practice we use the **normal form**: $\rho$ is the distance from the origin to the line and $\theta$ the angle of its normal,

$$\rho = x \cos\theta + y \sin\theta, \qquad \theta \in [-90°, 90°), \quad |\rho| \le \sqrt{W^2 + H^2}$$

## The accumulator

Discretise $(\rho, \theta)$ into a grid of counters. Every edge pixel adds one vote to every cell $(\rho(\theta), \theta)$ along its sinusoid. Cells where many sinusoids cross collect many votes.

```python
--8<-- "08_hough.py:accumulator"
```

## Finding the peaks

The $n$ largest cells are not $n$ different lines: a real line produces a *blob* of high cells around its true parameters. After taking the maximum, clear its neighbourhood before looking for the next one — non-maximum suppression again.

```python
--8<-- "08_hough.py:peaks"
```

## Lane detection

A minimal lane detector: Canny edges, a trapezoidal **region of interest** in front of the car (so trees and the horizon don't vote), Hough, draw.

```python
--8<-- "08_hough.py:lanes"
```

!fig 08-accumulator | The accumulator: each bright curve is one edge pixel's sinusoid. The circled peaks are the detected lines; the two edges of a painted line are two close peaks.

```python
--8<-- "08_hough.py:opencv"
```

`HoughLines` returns $(\rho, \theta)$ pairs like ours. `HoughLinesP` — the *probabilistic* Hough transform — samples a subset of the edge pixels and returns finite **segments** $(x_1, y_1, x_2, y_2)$, which is usually what you want.

!!! warning "Output shapes changed in OpenCV 5"
    OpenCV 4 returns arrays of shape `(N, 1, 4)` (or `(N, 1, 2)`); OpenCV 5 returns `(N, 4)`. Writing `lines.reshape(-1, 4)` works with both. Also check for `None` when nothing is found.

!fig 08-lanes | Edges inside the ROI, the four strongest lines from our accumulator, and the segments from `HoughLinesP`.

## Circles

A circle has three parameters $(x_c, y_c, r)$, so a naive accumulator is 3-D and slow. `HoughCircles` uses the gradient direction to vote only along the normal of each edge pixel.

```python
--8<-- "08_hough.py:circles"
```

!fig 08-circles | Three noisy disks, all found with their radius within 2 pixels. `param1` is Canny's high threshold, `param2` the minimum number of votes — lower it to find more (and more false) circles.

## Exercises

1. Weight each vote by the gradient magnitude of the pixel instead of 1. Do the peaks become sharper?
2. Use the gradient direction to vote for a single $\theta$ per pixel (± 5°) instead of all 180. How much faster is it?
3. Run the lane detector on every frame of a road video, and average the line parameters over the last 5 frames for stability.
