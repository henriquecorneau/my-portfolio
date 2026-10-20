---
title: Images are arrays
summary: A digital image is a NumPy matrix. Learn how OpenCV stores it, why data types cause half of all image-processing bugs, and how masks replace loops.
topics: NumPy · dtype · BGR · channels · ROI · masks
script: 01_images_as_arrays.py
sources: UFRN lecture 01, INSA TIM lab 1
---

## A picture is a matrix

For a computer, a grayscale image of height $H$ and width $W$ is an $H \times W$ matrix: element $(i, j)$ — **row $i$, column $j$** — is the brightness of that pixel, usually an integer between 0 (black) and 255 (white). A colour image adds a third axis with one plane per channel, giving an $H \times W \times 3$ array.

That is the whole secret: every algorithm in this course is "just" arithmetic on these arrays. So before touching an image, we need to be fluent with NumPy.

## Data types: the bug factory

Images are almost always stored as `uint8`: unsigned 8-bit integers, from 0 to 255. Arithmetic on `uint8` **wraps around** silently:

```python
--8<-- "01_images_as_arrays.py:dtype"
```

```text
uint8  255 + 1 -> 0
uint16 255 + 1 -> 256
cv2.add saturates -> 255
safe NumPy sum -> 256
```

`255 + 1` gives `0`: a white pixel becomes black. This is why so many first attempts at "increase the brightness" produce strange black spots. Three ways out:

| Situation | What to do |
|---|---|
| You want to clip at 0 and 255 | Use OpenCV's saturating ops: `cv2.add`, `cv2.subtract`, `cv2.addWeighted` |
| You need negative or large intermediate values (derivatives, sums) | Convert first: `img.astype(np.float32)` (or `np.int16`), compute, then clip and convert back |
| You divide or average | Floats, always — integer division loses information |

!!! warning "Rule of thumb"
    Read in `uint8`, **compute in `float32`**, `np.clip(…, 0, 255).astype(np.uint8)` at the very end. We will meet this rule in nearly every lesson.

## Masks: the end of nested loops

A *mask* is a boolean array with the same shape as the image. Indexing with it selects (or writes) only the pixels where it is `True`:

```python
--8<-- "01_images_as_arrays.py:masks"
```

`A[mask]` returns a 1-D array of the selected values, and `B[A < 5] = 0` writes zero exactly where the condition holds. Combine conditions with `&`, `|`, `~` or `np.logical_and/or/xor`. Masks are how we will express thresholding, segmentation and filtering without writing `for i in range(h): for j in range(w):` — which is about **1000× slower** in Python.

## Reading an image with OpenCV

```python
--8<-- "01_images_as_arrays.py:read"
```

The second argument of `cv2.imread` selects the mode: `cv2.IMREAD_GRAYSCALE` (0), `cv2.IMREAD_COLOR` (1, the default) or `cv2.IMREAD_UNCHANGED` (-1, keeps 16-bit depth and the alpha channel). Two traps:

- `cv2.imread` returns **`None`** instead of raising an error when the path is wrong. The course helper `U.imread` raises a clear `FileNotFoundError` instead.
- `img.shape` is `(height, width, channels)`, but most OpenCV functions that take a *size* want `(width, height)`. And pixel access is `img[row, col]` — that is `img[y, x]` — while drawing functions take points as `(x, y)`.

## BGR, not RGB

For historical reasons OpenCV loads colour images with the channels in **Blue, Green, Red** order. Matplotlib (and almost everything else) expects RGB. Forget the conversion and your sky turns orange:

```python
--8<-- "01_images_as_arrays.py:bgr"
```

!fig 01-bgr | Left: converted with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. Right: the same array shown without conversion — red and blue are swapped.

## Channels

```python
--8<-- "01_images_as_arrays.py:channels"
```

!fig 01-channels | Each channel is itself a grayscale image. The sky is bright in all three (white), the red awnings only in R. Zeroing the B and G planes keeps only the red light.

## From colour to gray

The naive gray level is the mean of the three channels, but our eyes are much more sensitive to green than to blue. The standard (ITU-R BT.601) **luma** weights them:

$$Y = 0.299\,R + 0.587\,G + 0.114\,B$$

```python
--8<-- "01_images_as_arrays.py:gray"
```

`cvtColor(…, COLOR_BGR2GRAY)` uses exactly these weights — our manual version differs by at most 0.5, which is just rounding. Note the result types: the manual one is `float32`, OpenCV's is `uint8`.

!fig 01-gray | The plain mean (left) makes the blue sky too bright and the red/green areas too dark compared with the perceptual luma (right).

## Regions of interest and drawing

Slicing returns a **view**: no data is copied, and writing into the slice writes into the original image. Use `.copy()` when you want an independent crop.

```python
--8<-- "01_images_as_arrays.py:roi"
```

!fig 01-roi | A slice of the image (left) and drawing primitives on a copy (right). Colours are given in BGR: `(0, 0, 255)` is red.

## Synthetic images, blending and simple transforms

```python
--8<-- "01_images_as_arrays.py:synthetic"
```

```python
--8<-- "01_images_as_arrays.py:blend"
```

!fig 01-synthetic | A gradient built with `np.tile` (each pixel equals its column index) and a 50/50 blend of two images with `cv2.addWeighted`.

```python
--8<-- "01_images_as_arrays.py:transpose"
```

!fig 01-transpose | Transposing swaps rows and columns — a mirror around the diagonal. `np.rot90` is a transpose followed by a flip.

## Exercises

1. Write `brightness(img, delta)` with plain NumPy that does **not** wrap around. Compare with `cv2.add`.
2. Using only masks, set to white every pixel of `rouen.jpg` whose red channel is larger than both the green and the blue channels. What did you select?
3. Time `gray[i, j]` in a double loop against `gray_cv` for the whole image (`time.perf_counter`). How many times faster is the vectorised version?
