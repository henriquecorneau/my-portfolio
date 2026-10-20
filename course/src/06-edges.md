---
title: Edge detection
summary: Edges are where the intensity changes fast. Find them with first derivatives (Sobel, Prewitt) or second derivatives (Laplacian, difference of Gaussians), and turn the result into a clean edge map.
topics: gradient · Sobel · Prewitt · Laplacian · DoG · zero crossings
script: 06_edges.py
sources: INSA TIM lab 4, UFRN lecture 08
---

## Edges in one dimension

Take one row of an image. An edge is a jump in intensity. Its derivative — for discrete signals, the difference between neighbours, `np.diff` — is:

- a **peak** at the edge, for the first derivative;
- a **positive/negative pair that crosses zero** at the edge, for the second derivative.

```python
--8<-- "06_edges.py:signals"
```

!fig 06-signals | A sharp step, a smooth ramp and a thin line, with their first and second derivatives. Note that a thin *line* gives two first-derivative peaks of opposite sign.

So there are two families of edge detectors: look for **maxima of the first derivative**, or for **zero crossings of the second**.

## The gradient

In 2-D, the first derivative becomes the **gradient**, a vector of two partial derivatives:

$$\nabla I = \left(\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right), \qquad
|\nabla I| = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \operatorname{atan2}(G_y, G_x)$$

The magnitude says *how strong* the edge is, the direction is perpendicular to the edge. Plain differences $I(x+1) - I(x)$ are very sensitive to noise, so practical operators combine a derivative in one direction with a smoothing in the other. **Prewitt** smooths with $[1, 1, 1]$, **Sobel** with $[1, 2, 1]$:

$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad S_y = S_x^\top$$

```python
--8<-- "06_edges.py:gradient"
```

!!! danger "The classic bug"
    `cv2.filter2D(img, -1, kernel)` on a `uint8` image returns `uint8`: every negative response is clipped to 0. Half of the edges (those going from bright to dark) disappear. Always ask for a float output (`cv2.CV_32F`, or `cv2.CV_64F` with `cv2.Sobel`).

!fig 06-gradient | $|G_x|$ responds to vertical edges, $|G_y|$ to horizontal ones, the magnitude to both. The last panel is the clipped `uint8` version: only one side of each edge survives.

!fig 06-direction | Left: gradient direction as colour, magnitude as brightness — the two sides of the coat have opposite colours (opposite directions). Prewitt is visually almost identical to Sobel; plain differences are noisier.

## Second derivatives: the Laplacian

The Laplacian sums the two second derivatives, $\nabla^2 I = \partial^2 I/\partial x^2 + \partial^2 I/\partial y^2$, and is approximated by

$$L_4 = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix} \quad\text{or}\quad L_8 = \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$

(sign conventions vary; only the zero crossings matter). It is isotropic — one kernel, no direction — but differentiating twice amplifies noise twice. The fix is to smooth first: the **Laplacian of Gaussian** (LoG), well approximated by the **Difference of Gaussians** (DoG), the same band-pass filter SIFT uses to find blobs in lesson 12.

```python
--8<-- "06_edges.py:second_order"
```

## Finding zero crossings

A pixel is on an edge if one of its 4 neighbours has the opposite sign **and** the jump between them is large enough — without that second condition, every tiny ripple of noise around zero becomes an "edge".

```python
--8<-- "06_edges.py:zero_crossing"
```

!fig 06-second-order | Without a threshold on the jump, the zero crossings of the Laplacian are everywhere. With it, the outlines appear, closed and one pixel thin. The DoG (smoothed first) gives cleaner contours.

## First-order edges: threshold the magnitude

For first-order operators, edges are bright ridges of the magnitude, so a threshold is enough; Otsu's method picks it automatically by splitting the histogram into two classes with minimal variance.

```python
--8<-- "06_edges.py:threshold"
```

!fig 06-threshold | Thresholded Sobel magnitude. The edges are thick (several pixels across) — lesson 07 thins them to exactly one pixel.

| | First order (Sobel, Prewitt) | Second order (Laplacian, DoG) |
|---|---|---|
| Edge = | maximum of $|\nabla I|$ | zero crossing of $\nabla^2 I$ |
| Gives | magnitude **and** direction | position only, no direction |
| Edges are | thick, need thinning | thin and closed |
| Noise | sensitive | very sensitive (smooth first) |

## A complete pipeline: noisy CT scan

A real processing chain has three parts: **pre-processing** (fix the image), **processing** (extract the information), **display**. The CT slice below has salt-and-pepper noise, so lesson 05 tells us to start with a median filter:

```python
--8<-- "06_edges.py:pipeline"
```

!fig 06-pipeline | Without denoising, every impulse becomes a small ring of edges. After a 5×5 median, the Sobel magnitude shows the lungs, the heart and the vessels, and the thresholded edges are drawn in red on the input.

## Exercises

1. Compare $|G_x| + |G_y|$ and $\sqrt{G_x^2 + G_y^2}$: where do they differ most? (Hint: diagonal edges.)
2. Replace the DoG with `cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), 1.5), cv2.CV_32F)` and tune the zero-crossing jump.
3. Draw the edges of `noisy_brain.png` in a different colour depending on the gradient direction (4 bins).
