---
title: Mathematical morphology
summary: Erode, dilate, open and close binary images with a structuring element. Clean a noisy shape, make a damaged QR code readable again, and find patterns with hit-or-miss.
topics: structuring elements · erosion · dilation · opening · closing · gradient · hit-or-miss · connected components
script: 09_morphology.py
sources: INSA TIM labs 5 and 9
---

## Shapes, not intensities

Filters of lesson 05 compute weighted *sums*. Morphology asks *set* questions: does a small shape — the **structuring element** (SE, or kernel) — fit inside the white region at this position? Does it touch it? That makes it ideal for cleaning up binary images after a threshold, and for measuring shapes.

```python
--8<-- "09_morphology.py:kernels"
```

!fig 09-kernels | Structuring elements built with `cv2.getStructuringElement`. Note that the size is given as **(width, height)**.

## Erosion and dilation

With $I$ the set of white pixels and $K_x$ the kernel centred at $x$:

$$\varepsilon_K(I) = \{\, x \mid K_x \subseteq I \,\} \qquad \delta_K(I) = \{\, x \mid \check K_x \cap I \neq \emptyset \,\}$$

**Erosion** keeps a pixel white only if the whole kernel fits in the white region: white objects shrink, white specks vanish. **Dilation** makes a pixel white if the (reflected) kernel touches any white pixel: objects grow, black holes fill.

A neat way to compute them: for every "on" cell $(d_y, d_x)$ of the kernel, shift the image by that offset; erosion is the AND of all the shifted copies, dilation is the OR.

```python
--8<-- "09_morphology.py:erode_dilate"
```

All four operations are **pixel-identical** to `cv2.erode`, `cv2.dilate` and `cv2.morphologyEx`, which the script checks.

!!! bug "What went wrong in my first erosion"
    I wrote `if not k.all() and window.all()`. That ignores the kernel's shape (a cross or a disc has zeros in it) and only compares whole windows. The definition says: *the window must be white wherever the kernel is 1* — `window[k == 1].all()`.

!fig 09-basic-ops | Erosion removes white specks but enlarges black ones; dilation does the opposite. Opening (erode, then dilate) and closing (dilate, then erode) restore the size of the big shapes while removing the small ones.

| Operation | Definition | Effect on white objects |
|---|---|---|
| Opening | $\delta(\varepsilon(I))$ | removes white parts smaller than the kernel |
| Closing | $\varepsilon(\delta(I))$ | fills black holes and gaps smaller than the kernel |
| Gradient | $\delta(I) - \varepsilon(I)$ | outlines of every shape |
| Top-hat / black-hat | $I - \text{open}(I)$ / $\text{close}(I) - I$ | small bright / dark details (lesson 05's barcode) |

## Cleaning very noisy shapes

The yin-yang below has 14% of its pixels flipped. The obvious recipe — a big opening to remove the white specks, then a big closing for the black ones — **fails**: the white half is so riddled with black specks that the opening eats it. The solution is an **alternating sequential filter**: close and open with a *growing* kernel, so every step only has to deal with specks slightly larger than those already removed.

```python
--8<-- "09_morphology.py:clean"
```

```text
error: noisy 13.7%  open7+close7 36.0%  ASF 0.72%
```

!fig 09-clean | One big opening + closing triples the error. The alternating filter brings it from 13.7% to 0.7% wrong pixels, and its morphological gradient is a clean outline.

## Making a QR code readable

A QR code is just black and white squares, so noise inside a module can be removed with a kernel smaller than a module. To *measure* success, we ask OpenCV's QR decoder:

```python
--8<-- "09_morphology.py:qr"
```

!fig 09-qr | The noisy code cannot be decoded. After an opening and a closing with a 3×3 cross, `cv2.QRCodeDetector` reads it.

Two practical lessons: the kernel must be **smaller than one module** (a 5×5 kernel starts eating the thin black lines), and decoders need a white **quiet zone** around the code — I had to pad the image with a white border before it worked.

## Hit-or-miss: finding patterns

Hit-or-miss looks for an exact local configuration. The kernel has three values: $1$ = must be white, $-1$ = must be black, $0$ = don't care. It is two erosions: the foreground eroded by the "1" cells AND the background eroded by the "−1" cells.

```python
--8<-- "09_morphology.py:hitmiss"
```

!fig 09-hitmiss | The kernel describes a top-right outer corner; exactly the two top-right corners are found. Lesson 19 uses the same tool to find the endings of fingerprint ridges.

## Counting objects

After cleaning, `connectedComponentsWithStats` labels every white blob and returns its area, bounding box and centroid — the bridge between pixels and objects:

```python
--8<-- "09_morphology.py:components"
```

## Exercises

1. Find the smallest square kernel for which the QR code stops being readable. Relate it to the module size in pixels.
2. Write `hit_or_miss` kernels for the four outer corners (use `np.rot90`) and count the corners of every rectangle in an image.
3. Use the morphological gradient of the cleaned yin-yang and `cv2.findContours` to measure the perimeter of each half.
