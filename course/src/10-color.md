---
title: Colour spaces and colour segmentation
summary: RGB is how screens work, not how to reason about colour. Convert to CMYK, YUV and HSV by hand, then use hue to recolour objects and to segment and locate a toy car.
topics: CMYK · YUV · HSV · inRange · hue shift · moments · centroid
script: 10_color.py
sources: INSA TIM lab 6, UFRN lecture 06
---

## Why other colour spaces?

In RGB, all three channels mix *what colour* a pixel is with *how bright* it is. A dark blue and a light blue pencil are far apart in RGB, so "select the blue pixels" is not a simple box. Other colour spaces separate those two pieces of information.

## CMYK: subtractive colour for printing

Screens *add* light (RGB, starting from black); printers *subtract* it with inks (starting from white paper). Cyan absorbs red, magenta absorbs green, yellow absorbs blue. Since mixing the three inks to make black is expensive and muddy, a fourth black ink (K) takes over the common part. With $r, g, b \in [0, 1]$:

$$k = 1 - \max(r, g, b), \qquad c = \frac{1 - r - k}{1 - k}, \quad m = \frac{1 - g - k}{1 - k}, \quad y = \frac{1 - b - k}{1 - k}$$

and back: $r = (1 - c)(1 - k)$, and so on.

```python
--8<-- "10_color.py:cmyk"
```

The round trip RGB → CMYK → RGB is exact (max error 0). OpenCV has no CMYK conversion — it is a printing concern.

!fig 10-cmyk | Each ink printed alone on white paper. Most of the dark shading goes to the K plate.

## YUV: luma plus two colour differences

YUV was designed in the 1950s so that colour TV could stay compatible with black-and-white sets: **Y** is exactly the gray image of lesson 01, and **U**, **V** carry the colour as differences from it.

$$Y = 0.299 R + 0.587 G + 0.114 B, \qquad U = 0.492\,(B - Y), \qquad V = 0.877\,(R - Y)$$

```python
--8<-- "10_color.py:yuv"
```

$U$ and $V$ are signed ($U \in [-0.44, 0.44]$, $V \in [-0.62, 0.62]$). OpenCV's `COLOR_RGB2YUV` stores them in `uint8` with an offset of 128, so you cannot compare the two numbers directly — only after converting. Since the eye is less sensitive to colour detail than to brightness, video codecs store U and V at half resolution (4:2:0) — and you can barely tell.

!fig 10-yuv | Y is the gray image. U is high on blue pencils, V on red/orange ones.

## HSV: hue, saturation, value

HSV is built for humans: **H** (hue) is the colour as an angle on the colour wheel, **S** (saturation) how pure it is, **V** (value) how bright. With $M = \max(r, g, b)$, $m = \min(r, g, b)$, $\Delta = M - m$:

$$H = \begin{cases} 60° \cdot \frac{g - b}{\Delta} \bmod 360° & M = r \\ 60° \cdot \frac{b - r}{\Delta} + 120° & M = g \\ 60° \cdot \frac{r - g}{\Delta} + 240° & M = b \end{cases} \qquad S = \frac{\Delta}{M}, \qquad V = M$$

```python
--8<-- "10_color.py:hsv"
```

In OpenCV's 8-bit HSV, **H is divided by 2** (0–179) to fit in a byte, while S and V go to 255. Our version matches `cv2.COLOR_BGR2HSV` within 1 unit.

!fig 10-hsv | The H channel is (almost) constant along a pencil, despite shading — that is the point. Note how it jumps between ~0 and ~179 around red: hue is circular.

!fig 10-hsv-disc | Each pixel placed at angle H and radius S·V. Pencils of the same colour form tight angular wedges, whatever their brightness.

## Recolouring by hue

Select a hue range with `cv2.inRange`, change H, keep S and V: the shading stays, so the result looks natural.

```python
--8<-- "10_color.py:recolor"
```

!fig 10-recolor | The yellow pencil becomes green. Rotating every hue by 120° recolours the whole image consistently.

!!! warning "Red wraps around"
    Red sits at both ends of the hue axis (0° and 360°, i.e. 0 and 179 in OpenCV). A red mask therefore needs **two** ranges combined with `|`. Also always constrain S and V: a white or black pixel has an undefined, meaningless hue.

## Segment and locate an object

This is the robotics version (my UFRN lecture tracked a toy car this way): segment the object by colour, then compute its **centre of mass** from the image moments $m_{pq} = \sum_{x,y} x^p y^q\, I(x, y)$:

$$\bar x = \frac{m_{10}}{m_{00}}, \qquad \bar y = \frac{m_{01}}{m_{00}}$$

The car is pale (S ≈ 40) — hard to threshold. The background is a saturated red (S ≈ 135) — easy. So we segment the **background** and invert, keep the largest blob and fill its holes:

```python
--8<-- "10_color.py:track"
```

!fig 10-track | Segment what is easy, then invert. The cross is the centroid, the box comes from `cv2.boundingRect`. Running this on every frame of a video gives a simple colour tracker.

## Exercises

1. Plot the H histogram of `crayons.jpg` (only pixels with S > 100). Can you count the pencil colours from its peaks?
2. Recolour all *blue* pencils to orange (careful with the target hue in OpenCV units).
3. Convert to L\*a\*b\* (`cv2.COLOR_BGR2LAB`) and segment the car with a threshold on the *a\** channel only. Compare with the HSV approach.
