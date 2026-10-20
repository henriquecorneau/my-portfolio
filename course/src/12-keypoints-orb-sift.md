---
title: Keypoints and descriptors — FAST, ORB and SIFT
summary: Detect keypoints with FAST, give them an orientation, describe them with binary BRIEF strings and match them with the Hamming distance — then see why ORB steers its descriptors and how SIFT and the ratio test compare.
topics: FAST · intensity centroid · BRIEF · Hamming distance · ORB · SIFT · FLANN · ratio test
script: 12_keypoints_orb_sift.py
sources: INSA ORB project, UFRN activity 06
---

## Detector + descriptor

Matching two images of the same scene from different viewpoints needs two things:

1. a **detector** that finds the same physical points in both images (repeatability), and
2. a **descriptor**, a vector that summarises the patch around each point so that the *same* point gives *similar* vectors, even if the image is rotated, scaled, blurred or darker.

ORB (Rublee et al., 2011) — "Oriented FAST and Rotated BRIEF" — is the fast, free, binary answer, and the feature behind ORB-SLAM (lesson 17). We build a small version of it.

## FAST

FAST looks at the 16 pixels on a circle of radius 3 around a candidate $p$. It is a corner if **$n$ contiguous** circle pixels are all brighter than $I_p + t$ or all darker than $I_p - t$ ($n = 9$ for FAST-9, 12 in the original paper).

```python
--8<-- "12_keypoints_orb_sift.py:circle"
```

The original version of my implementation tested each pixel in a Python loop, which takes seconds per image. Vectorised, the 16 circle pixels become 16 shifted copies of the image, and "$n$ contiguous" becomes an AND over $n$ consecutive copies, for each of the 16 starting positions (wrapping around):

```python
--8<-- "12_keypoints_orb_sift.py:fast"
```

!fig 12-fast | Our FAST-9 (153 keypoints) and OpenCV's (182) on the same threshold. The difference comes from the score used for NMS: OpenCV uses the largest $t$ for which the point is still a corner.

## Orientation: the intensity centroid

FAST says *where* but not *which way*. ORB computes the centroid of the patch intensities with image moments; the vector from the keypoint to the centroid defines its angle:

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y), \qquad \theta = \operatorname{atan2}(m_{01}, m_{10})$$

```python
--8<-- "12_keypoints_orb_sift.py:orientation"
```

The patch is a disc, so rotating the image rotates the centroid by the same angle.

## BRIEF: a descriptor made of comparisons

BRIEF picks 256 random pairs of points $(\mathbf{p}_i, \mathbf{q}_i)$ inside the (smoothed) patch, fixed once and for all, and sets bit $i$ to $[I(\mathbf{p}_i) < I(\mathbf{q}_i)]$. The descriptor is 32 bytes, and two descriptors are compared with the **Hamming distance** — the number of different bits, a XOR and a popcount. That is why ORB matching is so fast.

"Steered" BRIEF rotates the sampling pattern by the keypoint's angle before sampling:

```python
--8<-- "12_keypoints_orb_sift.py:brief"
```

## Does the orientation matter? Measure it

We rotate an image by a **known** 50° so that we can check every match against the ground truth:

```python
--8<-- "12_keypoints_orb_sift.py:evaluate"
```

```text
mini-ORB steer=False: 4 mutual matches, 0% correct
mini-ORB steer=True: 98 mutual matches, 91% correct
```

!fig 12-steered | Correct matches in green, wrong ones in red. Without orientation, the bits compare different pixels after rotation and nothing matches. Steered, 91% of the matches are right.

## ORB in OpenCV

```python
--8<-- "12_keypoints_orb_sift.py:orb"
```

`crossCheck=True` keeps a match only if A's best match is B *and* B's best match is A. Sorting by distance puts the most reliable first.

!fig 12-orb-robustness | ORB between a noisy tower photo and degraded versions. Contrast changes barely matter (binary tests only compare pixels). Blur and zoom are harder: the number of matches with distance < 40 drops from 236 (low contrast) to 31 (blur) and 13 (zoom).

## SIFT and the ratio test

SIFT (Lowe, 2004) detects blobs as extrema of a **difference of Gaussians** across scales (lesson 06) — so it knows each keypoint's scale — and describes them with 128 floats: histograms of gradient orientations in a 4×4 grid. It is slower than ORB but more distinctive and robust to scale.

With float descriptors we use the Euclidean distance and a FLANN k-d tree. Instead of a fixed threshold, **Lowe's ratio test** compares the best match with the second best: a correct match is much closer than any other candidate, while an ambiguous one (a repeated window, a texture) is not.

```python
--8<-- "12_keypoints_orb_sift.py:sift"
```

```text
SIFT box vs scene: 1888 nearest neighbours, 114 survive the ratio test
```

!fig 12-sift | Finding a cookie box in a cluttered scene. SIFT + ratio test (top) gives mostly consistent matches. ORB's best cross-checked matches (bottom) struggle: the box is much bigger in its reference image than in the scene.

## Matches are not proof

!fig 12-false-matches | Two different towers. Brute force always returns a "best" match for every descriptor: 283 cross-checked matches between unrelated buildings.

Descriptor distance alone cannot tell you whether two images show the same thing. The missing ingredient is **geometry**: correct matches must all agree on one transformation between the images. That is the next lesson.

## Exercises

1. Plot the fraction of correct matches of steered vs plain BRIEF for rotations from 0° to 180°.
2. Add the ratio test to the ORB matcher (`bf = cv2.BFMatcher(cv2.NORM_HAMMING)` + `knnMatch(k=2)`). How many of the 283 false tower matches survive?
3. Replace the random BRIEF pairs by pairs sampled from a uniform distribution. Does accuracy change?
