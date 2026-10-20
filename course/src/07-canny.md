---
title: The Canny edge detector
summary: Build the most widely used edge detector step by step — smoothing, gradient, non-maximum suppression, double threshold and hysteresis — and check it against cv2.Canny.
topics: Gaussian blur · non-maximum suppression · double threshold · hysteresis
script: 07_canny.py
sources: UFRN lecture 09 and activity 04
---

## What makes a good edge detector

John Canny (1986) wrote down three criteria: **detect** real edges and nothing else, **localise** them where they really are, and give **one response per edge**. Sobel + threshold fails the last two: its edges are several pixels thick and broken. Canny's algorithm fixes this in five steps.

## 1–2. Smooth, then take the gradient

```python
--8<-- "07_canny.py:step1"
```

```python
--8<-- "07_canny.py:step2"
```

The blur ($\sigma \approx 1.4$) controls the scale: more blur, fewer and smoother edges. We keep the direction folded into $[0°, 180°)$, because an edge and its opposite are the same line.

## 3. Non-maximum suppression

A thick ridge of gradient magnitude should become a 1-pixel line at its crest. For each pixel, look at its two neighbours **along the gradient direction** (across the edge); if it is not at least as large as both, it is not the crest — set it to zero. The direction is quantised to the 4 neighbour axes:

| Gradient angle | Edge runs | Compare with |
|---|---|---|
| ≈ 0° | vertically | left and right |
| ≈ 45° | diagonally | top-left and bottom-right |
| ≈ 90° | horizontally | above and below |
| ≈ 135° | diagonally | top-right and bottom-left |

```python
--8<-- "07_canny.py:step3"
```

!!! bug "Two bugs I made the first time"
    I wrote `np.arctan2(gx, gy)` instead of `np.arctan2(gy, gx)`, which mirrors every angle around 45° and compares the wrong neighbours. And in image coordinates *y grows downwards*: a gradient at +45° points to the bottom-right, so its neighbours are $(i-1, j-1)$ and $(i+1, j+1)$. Test on a circle (below): every direction appears once.

!fig 07-nms-circle | On a circle every gradient direction is present. NMS turns the thick Sobel ring into a 1-pixel contour — the best check that the four cases are right.

## 4. Double threshold

One threshold cannot be both strict enough to reject noise and permissive enough to keep the faint part of a real edge. Use two: pixels above `high` are **strong** edges, pixels between `low` and `high` are **weak** (maybe), the rest are dropped.

```python
--8<-- "07_canny.py:step4"
```

## 5. Hysteresis

A weak pixel is kept only if it is **connected to a strong one through a chain of weak pixels**. This is a flood fill: start from every strong pixel and grow into weak neighbours.

```python
--8<-- "07_canny.py:step5"
```

Checking only the 8 immediate neighbours of each weak pixel once is not enough — a weak segment ten pixels long would only keep its first pixel. On this image the single-pass version keeps 3021 edge pixels, the flood fill 3464.

!fig 07-steps | The five steps. Thick gradient ridges become thin crests (3), which are split into strong and weak (4); hysteresis keeps weak pixels that continue a strong edge (5).

## Putting it together, and OpenCV

```python
--8<-- "07_canny.py:canny"
```

`cv2.Canny` does steps 2–5 (do the blur yourself), and expects `low` and `high` on the same scale as the Sobel magnitude. With `L2gradient=True` it uses the true magnitude like we do. Our result and OpenCV's agree on **99.8 %** of the pixels.

!fig 07-compare | Lower thresholds keep texture (grass); higher ones keep only the main outlines. A common rule of thumb: `high` ≈ 2–3 × `low`.

## Exercises

1. Make the thresholds automatic: `high = 0.33 * percentile(magnitude, 90)`, `low = high / 2.5`. Does it work on `rue_rouen.jpg`?
2. Vectorise non-maximum suppression with four boolean masks (one per direction) and `np.roll`.
3. Implement hysteresis with `cv2.connectedComponents` on the weak-or-strong mask: keep every component that contains a strong pixel. Compare with the flood fill.
