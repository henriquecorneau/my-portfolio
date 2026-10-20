---
title: Histograms, point operations and LUTs
summary: Change the brightness, contrast and dynamic range of an image by remapping its gray levels. Equalise histograms, transfer the "look" of one photo to another, and do all of it with look-up tables.
topics: thresholding · contrast stretch · equalisation · CLAHE · gamma · histogram matching
script: 04_histograms_lut.py
sources: INSA TIM lab 2, UFRN lecture 06
---

## Look before you process

The test image of this lesson is a hazy landscape whose pixels all sit between about 120 and 200. Before anything else, show it **without** letting Matplotlib cheat:

```python
--8<-- "04_histograms_lut.py:display"
```

By default `imshow` maps the image's own min and max to black and white — a free, invisible contrast stretch. The course helper always passes `vmin=0, vmax=255`, so what you see is what is in the array.

```python
--8<-- "04_histograms_lut.py:histogram"
```

## Point operations

A **point operation** computes each output pixel from the input pixel *at the same place only*: $g(x, y) = t(f(x, y))$. The function $t$ is a curve from $[0, 255]$ to $[0, 255]$, drawn in red below.

- **Negative**: $t(f) = 255 - f$.
- **Threshold** (binarisation): $t(f) = 0$ if $f < a$, else $255$.
- **Intensity window**: keep $f$ if $a \le f \le b$, else $0$.
- **Dynamic expansion** (contrast stretch): map $[a, b]$ linearly onto $[0, 255]$ and clip the rest,
  $t(f) = 255 \frac{f - a}{b - a}$.
- **Sigmoid contrast**: $t(f) = 255 / (1 + e^{-k(f - r_0)})$, a smooth version of the stretch centred on $r_0$.

```python
--8<-- "04_histograms_lut.py:point_ops"
```

!fig 04-point-ops | Top: images. Bottom: their histograms. Stretching $[110, 200]$ to the full range reveals the fields and trees hidden in the haze, and spreads the histogram out.

## Histogram equalisation

Choosing $a$ and $b$ by hand is tedious. **Equalisation** picks the curve automatically so that the output histogram is as flat as possible — every gray level used about equally, which maximises the information (entropy) of the image. The curve is the normalised **cumulative histogram**:

$$t(f) = \frac{L - 1}{N} \sum_{i=0}^{f} h(i)$$

where $N$ is the number of pixels and $L = 256$. Intuition: a level that many pixels share gets a big step, so crowded ranges are pulled apart.

```python
--8<-- "04_histograms_lut.py:equalize"
```

Our version is identical to `cv2.equalizeHist` (maximum difference: 0).

Global equalisation can over-amplify noise and wash out regions that were already fine. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) equalises small tiles separately and limits how steep the curve may get:

```python
--8<-- "04_histograms_lut.py:clahe"
```

!fig 04-equalize | Global equalisation (middle) dramatically increases contrast; its curve (red) is the cumulative histogram. CLAHE (right) is gentler and local.

## Look-up tables

All these transforms share a property: every pixel with value $f$ becomes $t(f)$, wherever it is. So instead of computing $t$ millions of times, compute it **256 times** into a table, then index:

```python
--8<-- "04_histograms_lut.py:lut"
```

`lut[img]` is NumPy fancy indexing: for every pixel, read the table at that pixel's value. `cv2.LUT(img, lut)` does the same, faster. Composing two transforms is composing two tables — `lut2[lut1]` — and the image is still traversed only once.

A piece-wise linear **contrast** curve with parameters $(a, b)$ is: slope $b/a$ on $[0, a]$, then a straight line to $(255, 255)$. With $a > b$ the dark tones are compressed and the bright ones expanded; with $a < b$ the reverse. A **gamma** curve $t(f) = 255\,(f/255)^\gamma$ brightens ($\gamma < 1$) or darkens ($\gamma > 1$) the mid-tones while keeping black and white fixed.

!fig 04-luts | Four LUTs (red curves) and their effect. The histograms show how each curve stretches or compresses ranges of gray levels.

## Histogram matching: transfer a "look"

Equalisation makes the histogram flat. **Matching** makes it look like the histogram of another image. For each source level $f$, find the reference level $g$ with the same cumulative frequency, $C_\text{ref}(g) = C_\text{src}(f)$ — that is, $t = C_\text{ref}^{-1} \circ C_\text{src}$. It is still a LUT, one per colour channel.

```python
--8<-- "04_histograms_lut.py:matching"
```

!fig 04-matching | The colourful street (left) takes on the cooler, contrasted tones of the cathedral photo (middle). Nothing but three 256-entry tables.

## Exercises

1. Build the LUT "equalise, then threshold at 128" in one table and check it against applying both operations in sequence.
2. Plot `cumulative_histogram` of the equalised image. Why is it (almost) a straight line?
3. Implement `stretch_auto(img, p)` that sets $a$ and $b$ to the $p$-th and $(100-p)$-th percentiles of the image. Why is it more robust than using the min and max?
