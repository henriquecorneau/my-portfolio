---
title: Noise and spatial filtering
summary: Identify the kind of noise in an image, remove it with mean, median and Gaussian filters, understand convolution as the one operation behind most of them, and clean up a badly lit barcode.
topics: Gaussian & salt-and-pepper noise · PSNR · convolution · median · filter2D · background removal
script: 05_noise_filtering.py
sources: INSA TIM lab 3, UFRN lectures 02, 07 and activity 03
---

## Know your noise

Two noise models cover most cases:

- **Gaussian noise**: every pixel gets a small random offset $n \sim \mathcal{N}(0, \sigma^2)$. Typical of sensor electronics in low light.
- **Salt-and-pepper (impulse) noise**: a fraction $p$ of the pixels is replaced by 0 or 255. Typical of transmission errors and dead pixels.

```python
--8<-- "05_noise_filtering.py:noise"
```

To find out which noise an image has, subtract the clean version — in a **signed** type, otherwise negative differences wrap around (lesson 01):

```python
--8<-- "05_noise_filtering.py:diff"
```

```text
Gaussian noise: mean -0.45 std 9.87
S&P: 3.0% of the pixels changed
```

!fig 05-noise | The Gaussian difference is a bell curve centred on 0 ($\sigma \approx 10$). The salt-and-pepper difference is zero almost everywhere with a few huge values (log scale).

To compare filters with a number instead of by eye, we use the **PSNR**, $10 \log_{10}(255^2 / \text{MSE})$ in decibels: higher is closer to the clean image.

## The best filter: more measurements

If you can take several pictures of a static scene, averaging $n$ of them divides the noise standard deviation by $\sqrt{n}$ without blurring anything. The running mean $\mu_k = (1 - \tfrac{1}{k})\,\mu_{k-1} + \tfrac{1}{k}\,x_k$ needs only the current average in memory:

```python
--8<-- "05_noise_filtering.py:averaging"
```

!fig 05-averaging | Astrophotographers "stack" exposures for exactly this reason. With one image only, we have to trade noise for detail — that is what spatial filters do.

## Mean and median filters

A **mean** (box) filter replaces each pixel by the average of its $n \times n$ neighbourhood. A **median** filter replaces it by the *median*: sort the $n^2$ values and take the middle one.

```python
--8<-- "05_noise_filtering.py:mean_median"
```

!!! bug "Off by one"
    The window around pixel $i$ is `img[i-k : i+k+1]`. In my first version I wrote `img[i-k : i+k]`: Python slices exclude the end, so the window was $(n-1) \times (n-1)$ and shifted by half a pixel.

The median ignores outliers: a single 255 among eight values near 100 does not move it at all, while the mean is dragged up. That makes it *the* filter for salt-and-pepper noise.

## Convolution: one operation, many filters

The mean filter is a weighted sum of the neighbours with all weights equal to $1/n^2$. Allow arbitrary weights — a **kernel** $h$ — and you get the 2-D convolution:

$$g(i, j) = \sum_{k} \sum_{l} h(k, l)\, f(i - k,\, j - l)$$

(What we implement below, without flipping the kernel, is strictly a *correlation*; for symmetric kernels the two are identical, and OpenCV's `filter2D` also computes the correlation.)

```python
--8<-- "05_noise_filtering.py:convolution"
```

The **Gaussian** kernel $h(x, y) \propto e^{-(x^2 + y^2) / 2\sigma^2}$ gives more weight to the centre than a box, so it blurs more gracefully; the 3×3 kernel $\frac{1}{16}[1, 2, 1; 2, 4, 2; 1, 2, 1]$ is its small integer approximation. The **exponential** kernel $h \propto e^{-\gamma(|x| + |y|)}$ is a sharper-peaked alternative. All weights sum to 1 so that a flat region keeps its brightness. The median, by contrast, is *not* a convolution: it is not a weighted sum.

!fig 05-filters-g | Gaussian noise. All filters reduce the grain but blur the edges.

!fig 05-filters-sp | Salt-and-pepper noise. Averaging filters smear the impulses into gray blobs; the 3×3 median removes nearly all of them.

The numbers tell an important story:

| Filter | PSNR, Gaussian noise | PSNR, salt & pepper |
|---|---|---|
| none (noisy input) | 28.2 dB | 20.3 dB |
| mean 5×5 | 22.9 dB | 22.6 dB |
| median 3×3 | 26.6 dB | **27.0 dB** |
| Gaussian 5×5, σ = 1 | 25.9 dB | 24.7 dB |
| exponential 5×5 | 25.9 dB | 24.8 dB |

For salt-and-pepper the median wins by far. For the mild Gaussian noise ($\sigma \approx 10$) of this image, **every filter lowers the PSNR**: the detail destroyed by the blur costs more than the noise removed. Denoising is always a trade-off between noise and detail — the stronger the noise, the more blur is worth paying.

!fig 05-median-size | Bigger medians remove bigger clusters of impulses, but erase thin structures (the tripod legs) and flatten textures.

## The OpenCV way

```python
--8<-- "05_noise_filtering.py:opencv"
```

`filter2D` matches our convolution up to rounding inside the image, but handles the border differently: we left a black frame of $k$ pixels, OpenCV reflects the image (`BORDER_REFLECT_101`) so the output has no frame. And it is roughly **1000× faster** (≈ 0.2 ms vs ≈ 170 ms for our loop on a 256×256 image). The **bilateral** filter is a bonus: a Gaussian whose weights also drop when the neighbour's *value* is different, so it smooths flat regions while keeping edges.

## Case study: a badly lit barcode

The barcode photo below has a strong lighting gradient: a single global threshold (even Otsu's automatic one) cuts along the gradient, not along the bars. The fix is to **estimate the background** and remove it before thresholding.

My notes from the lab said "a very large mean filter keeps only the background". That is almost right: it does remove the gradient, but the wide black bars are wide enough to darken the mean around them, so they come out hollow. A morphological **closing** (a max filter followed by a min filter, lesson 09) with a disc larger than the widest bar erases the dark bars completely and leaves only the lighting:

```python
--8<-- "05_noise_filtering.py:barcode"
```

!fig 05-barcode | Top: input, failed global Otsu, mean background, and the hollow bars it produces. Bottom: background from a closing, the flat result, and the final binarisation.

Subtracting a background estimated with a closing has a name, the **black-hat** transform (`cv2.MORPH_BLACKHAT`); for bright objects on a dark background the mirror operation is the **top-hat**.

## Exercises

1. Add Gaussian noise with $\sigma = 30$ to `cameraman.tif`. Does filtering now increase the PSNR? Which kernel size is best?
2. Implement the median filter with `np.lib.stride_tricks.sliding_window_view` (no Python loop) and time it against yours and `cv2.medianBlur`.
3. Replace step 2b of the barcode pipeline with `cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, big)` and check that you get the same image.
