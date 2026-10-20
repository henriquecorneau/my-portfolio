---
title: Geometric transforms
summary: Rotate, scale, shear and change the perspective of an image with matrices. Homogeneous coordinates turn all of them into one matrix product, and inverse mapping turns that into an image without holes.
topics: rotation · homogeneous coordinates · affine · homography · warping
script: 03_geometric_transforms.py
sources: UFRN lectures 03, 04, 06 and 07, INSA TIM lab 1
---

## Rotating a point

Rotating the point $\mathbf{p} = (x, y)$ by an angle $\theta$ around the origin is a matrix product:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

```python
--8<-- "03_geometric_transforms.py:rotate_point"
```

!fig 03-rotate-point | $(2, 4)$ rotated by 90° becomes $(-4, 2)$.

## Homogeneous coordinates

Scaling and rotation are matrix products, but translation is an *addition* — which breaks the nice "one matrix per transform" story. The trick is to add a coordinate: the 2-D point $(x, y)$ becomes $(x, y, 1)$, and every transform becomes a $3 \times 3$ matrix:

$$T = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix},\quad
R = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix},\quad
S = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Chaining transforms is now multiplying matrices — **read from right to left**: $M = T\,R\,S$ scales first, then rotates, then translates. Undoing everything is `np.linalg.inv(M)`.

```python
--8<-- "03_geometric_transforms.py:homogeneous"
```

To rotate an image around its *centre* instead of the top-left corner, sandwich the rotation: move the centre to the origin, rotate, move it back: $T(c)\,R\,T(-c)$. In 3-D the same idea gives the $4 \times 4$ matrices $\begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$ that we will use for camera poses in Part V.

## Warping an image: forward or inverse?

To transform an image we could take every source pixel and *push* it to $M\mathbf{p}$ (forward mapping). But when the image is enlarged or rotated, several destination pixels receive nothing — the result is full of holes.

The right way is **inverse mapping**: loop over the *destination* pixels and *pull* the value from $M^{-1}\mathbf{p}'$ in the source. Every output pixel gets exactly one value.

```python
--8<-- "03_geometric_transforms.py:forward"
```

```python
--8<-- "03_geometric_transforms.py:inverse"
```

!fig 03-forward-inverse | The same rotation + 1.4× zoom. Forward mapping (left) leaves a grid of black holes; inverse mapping (right) is complete.

The division `src[:2] / src[2]` does nothing for affine matrices (the last row is $[0, 0, 1]$, so $w = 1$) but is essential for perspective, below. Here we take the nearest source pixel; using bilinear interpolation (lesson 02) instead is what `cv2.warpAffine` does by default.

## Affine transforms with OpenCV

An **affine** transform keeps parallel lines parallel. It has 6 parameters — the top two rows of the $3 \times 3$ matrix — and OpenCV represents it as a $2 \times 3$ array.

```python
--8<-- "03_geometric_transforms.py:opencv_affine"
```

!fig 03-affine | Rotation around the centre, a horizontal shear, and a general affine map defined by where three corners must go.

Since each point gives 2 equations, **3 point pairs** define an affine map (`getAffineTransform`).

## Perspective: the homography

A camera looking at a plane at an angle does not preserve parallelism — the edges of a road meet at the horizon. The general projective map between two planes is a **homography**, a full $3 \times 3$ matrix $H$ defined up to scale (8 degrees of freedom):

$$\begin{bmatrix} x' w \\ y' w \\ w \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \qquad (x', y') = \left(\frac{x'w}{w}, \frac{y'w}{w}\right)$$

The division by $w$ is what makes far things small. Four point correspondences (no three collinear) determine $H$.

A classic use in mobile robotics is the **bird's-eye view** (inverse perspective mapping): pick four points that form a trapezoid on the road and declare that, seen from above, they form a rectangle.

```python
--8<-- "03_geometric_transforms.py:perspective"
```

!fig 03-perspective | Four points on the road (left) mapped to a rectangle give a top view where lane lines are parallel. Our `warp_inverse` (right) produces the same geometry as OpenCV, only blockier because it uses nearest-neighbour lookup.

Lesson 13 shows how to *estimate* a homography from hundreds of noisy, partly wrong correspondences, instead of four hand-picked ones.

!!! tip "Which function?"
    `cv2.warpAffine(img, M_2x3, (w, h))` for rotations, scalings and shears; `cv2.warpPerspective(img, H_3x3, (w, h))` for homographies. Both use inverse mapping internally and take the output size as **(width, height)**.

## Exercises

1. Write `rotate_around(img, angle, center)` with `warp_inverse` and compare with `cv2.getRotationMatrix2D` + `cv2.warpAffine`.
2. Rotate an image by 45° *without cropping the corners*: compute the size of the bounding box of the rotated image and add the needed translation to $M$.
3. Take a photo of a sheet of paper on a table, click its 4 corners and rectify it into an A4-shaped image ("document scanner").
