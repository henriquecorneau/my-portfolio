---
title: Sampling, quantization and interpolation
summary: A digital image is a continuous scene sampled on a grid and quantized to a few levels. See what happens when we use too few samples or too few levels, and how to resize an image properly.
topics: quantization · histograms · k-means · aliasing · nearest / bilinear
script: 02_sampling_quantization.py
sources: INSA TIM lab 1, UFRN activity 01
---

## Two discretisations

A camera turns a continuous light field into numbers in two steps:

- **Sampling** — the scene is measured only on a grid of pixels. Fewer pixels, less detail.
- **Quantization** — each measurement is rounded to one of $L$ levels. With 8 bits, $L = 256$.

Both lose information, and both create *artefacts* — structures in the image that were not in the scene.

## Uniform quantization

To go from 256 levels to $N$, split $[0, 255]$ into $N$ bins of width $\Delta = 256 / N$ and replace every value by the centre of its bin:

$$q(f) = \left\lfloor \frac{f}{\Delta} \right\rfloor \Delta + \frac{\Delta}{2}$$

```python
--8<-- "02_sampling_quantization.py:quantize"
```

!fig 02-quantize | Down to 32 levels the eye barely notices. From 8 levels on, **false contours** appear: the smooth sky becomes visible bands.

## Histograms

The histogram $h(k)$ counts the pixels with value $k$. On the x-axis go *all possible values* (0–255), even absent ones; the bars add up to the number of pixels. It is the single most useful diagnostic in image processing — lesson 04 is built on it.

```python
--8<-- "02_sampling_quantization.py:histogram"
```

!fig 02-quantize-hist | After quantization the histogram collapses into $N$ spikes: the image literally has $N$ different values left.

## Quantizing colour

Quantizing each channel to 4 levels gives $4^3 = 64$ fixed colours, chosen without looking at the image. A much better palette comes from the image itself: run **k-means** on the pixel colours and replace each pixel by its cluster centre. Clustering in L\*a\*b\* (a perceptually uniform space, see lesson 10) makes "close" mean "looks similar".

```python
--8<-- "02_sampling_quantization.py:kmeans"
```

!fig 02-kmeans | With only 4 colours chosen by k-means, the picture is still readable; 64 fixed colours look worse than 16 adaptive ones.

## Subsampling and aliasing

Keeping one pixel out of $k$ is a one-liner with slicing:

```python
--8<-- "02_sampling_quantization.py:subsample"
```

On a test chart with increasingly fine circles, the result is not just "less sharp" — new, fake patterns appear. This is **aliasing**: by the Nyquist–Shannon theorem, a pattern can be represented only if we take at least two samples per period. Finer patterns fold back as false, coarser ones. The cure is to remove those frequencies *before* subsampling with a low-pass filter (a blur), which is exactly what `cv2.pyrDown` and `cv2.resize(..., interpolation=cv2.INTER_AREA)` do.

!fig 02-aliasing | Factor 4 creates whole new rings (aliasing). Blurring first leaves a uniform gray where detail is too fine to be represented — the honest answer.

## Shrinking: average the block

When shrinking by a factor $s < 1$, each output pixel covers a block of $1/s \times 1/s$ input pixels. Averaging that block is a box low-pass filter plus subsampling in one go:

```python
--8<-- "02_sampling_quantization.py:downscale"
```

Our function matches `cv2.resize(…, interpolation=cv2.INTER_AREA)` to within one gray level.

## Enlarging: interpolation

Enlarging needs values *between* the known pixels. For an output pixel $(i, j)$ we compute where it falls in the source, $(y, x) = (i/s, j/s)$, generally between four pixels.

**Order 0 — nearest neighbour.** Take the closest one. Fast, but blocky.

**Order 1 — bilinear.** Interpolate linearly in $x$, then in $y$. With $d_x, d_y \in [0, 1)$ the fractional parts:

$$I(y, x) = (1-d_y)(1-d_x)\,I_{00} + (1-d_y)\,d_x\,I_{01} + d_y(1-d_x)\,I_{10} + d_y d_x\,I_{11}$$

```python
--8<-- "02_sampling_quantization.py:upscale"
```

!fig 02-interp | Nearest neighbour keeps the pixel grid visible; bilinear is smooth but soft; bicubic (16 neighbours) is a little sharper.

!!! bug "A bug I made the first time"
    My first bilinear version computed `dx = x1 - x0`, which is always 1 — so every pixel was a copy of a corner. The weights must use the **fractional part** of the source position, `dx = x - int(x)`.

## In practice

```python
small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)    # shrink
big = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)            # enlarge
```

Use `INTER_AREA` to shrink, `INTER_LINEAR` or `INTER_CUBIC` to enlarge, and `INTER_NEAREST` for label images and masks, where inventing in-between values makes no sense.

## Exercises

1. Plot the number of distinct values of `quantize(gray, N)` for $N = 2 \dots 128$. Is it always $N$?
2. Implement `pyr_down(img)` = Gaussian blur + keep one pixel out of two, apply it twice on `mire.png` and compare with `subsample(mire, 4)`.
3. Vectorise `upscale_bilinear` with `np.meshgrid` (no loops) and check that the output is identical.
