---
title: "Case study: fingerprint identification"
summary: Five fingerprints were found at a crime scene, damaged in five different ways. Thin the ridges, detect minutiae with the crossing number and hit-or-miss, clean them up, restore the crime-scene prints and match them against a database of 20 people.
topics: thinning · crossing number · hit-or-miss · convex hull · pre-processing · alignment-based matching
script: 19_fingerprints.py
sources: INSA TIM lab 9
---

## The case

The lab's scenario: someone stole a blue French horn from a restaurant in Manhattan, and five fingerprints were collected at the scene. We have a "database" of 20 binarised prints and must say whose they are. The crime-scene prints were rotated, cropped — and degraded on purpose, each in a different way.

Fingerprints are compared not by pixels but by **minutiae**: the points where a ridge **ends** or **splits in two** (a bifurcation). Their positions and directions are stable across impressions and are what forensic systems store.

## 1. Thinning

Minutiae are defined on ridge *lines*, so first reduce every ridge to a 1-pixel-wide skeleton that keeps its connectivity. Thinning is an iterated hit-or-miss (lesson 09) that peels pixels off the boundary as long as that does not break the line; `skimage.morphology.thin` implements it.

```python
--8<-- "19_fingerprints.py:skeleton"
```

## 2. Finding minutiae: the crossing number

Walk around the 8 neighbours $P_1 \dots P_8$ of a skeleton pixel and count the transitions between 0 and 1:

$$CN = \frac{1}{2} \sum_{i=1}^{8} |P_i - P_{i+1}|, \qquad P_9 = P_1$$

$CN = 1$: the ridge **ends** here. $CN = 2$: an ordinary ridge pixel. $CN = 3$: a **bifurcation**.

```python
--8<-- "19_fingerprints.py:crossing"
```

The lab also asked for the hit-or-miss version of ridge endings: eight $3 \times 3$ kernels, one per position of the single neighbour.

```python
--8<-- "19_fingerprints.py:hitmiss"
```

```text
crossing number: 48 endings, 22 bifurcations; hit-or-miss endings: 48 (identical: True)
```

Both methods find exactly the same 48 endings. Bifurcations are where the crossing number really pays off: with hit-or-miss they need many kernels (T and Y shapes in all orientations), with $CN$ it is one comparison.

## 3. Orientation

Matching needs a direction for each minutia. Instead of looking only at the next pixel (which gives a direction rounded to 45°), we **walk ten pixels along the ridge** and take the direction of that segment. A bifurcation gets the direction of its "stem": the branch that points away from the other two.

```python
--8<-- "19_fingerprints.py:orientation"
```

## 4. Throwing away false minutiae

Many detected minutiae are artefacts: ridges that end because the **capture area** ends, small spurs created by thinning, ridges broken by noise. Two filters remove most of them:

- build the **convex hull** of the print, erode it, and keep only minutiae inside;
- drop minutiae that are closer than 8 px to another one (spurs and breaks come in pairs).

```python
--8<-- "19_fingerprints.py:filter"
```

!fig 19-skeleton | A database print, a zoom on its skeleton, all 70 crossing-number minutiae (endings red, bifurcations green), and the 35 left after filtering.

!!! bug "Borders and erosion"
    One crime-scene print touches the top of the image. `cv2.erode` replicates the border by default, so the hull mask was *not* eroded along that edge and a row of fake endings survived. Passing `borderType=cv2.BORDER_CONSTANT, borderValue=0` treats outside the image as background.

## 5. Restoring the crime-scene prints

The five prints are damaged differently: a lighting gradient (1), almost no contrast — all values between 96 and 104 (2), horizontal line noise (3), **inverted** polarity and an ink stain (4), faded and blurred patches (5). One pipeline handles all five, each step aimed at one problem:

```python
--8<-- "19_fingerprints.py:preprocess"
```

- **Banding**: subtracting each row's median removes line-wise offsets.
- **Illumination**: subtracting a very wide Gaussian blur is a high-pass filter; ridges become positive or negative oscillations around 0, whatever the original contrast was.
- **Polarity**: ridges are the thin structures, so they produce the extreme values; if the bright extreme is the larger one, the ridges are bright and we flip the sign.
- **Region**: ridges make the *local standard deviation* high; the largest blob of high variance, after an opening that cuts thin bridges, is the print — the stain is dropped.
- **Binarisation**: after the high-pass, "darker than its neighbourhood" is simply $f < 0$, a local threshold.

!fig 19-preprocess | Top: the crime-scene prints (contrast stretched for display). Bottom: the same pipeline applied to all of them.

## 6. Matching

The prints are rotated and shifted with respect to the database, so we cannot compare coordinates directly. **Alignment-based matching**: every pair (ending $a$ in print A, ending $b$ in print B) proposes a rigid motion that takes $a$ onto $b$ (rotation = difference of their directions). Apply it to all of A and count the minutiae that land within 12 px of a minutia of B of the same type and with a direction within 30°. Keep the best motion. The score is $n_\text{matched}^2 / (|A|\,|B|)$, the fraction of both sets that is explained.

```python
--8<-- "19_fingerprints.py:match"
```

The whole computation for one motion candidate is vectorised (an `eb × A × B` distance array), so comparing one print with the whole database of 20 takes a few seconds.

```python
--8<-- "19_fingerprints.py:identify"
```

```text
suspect_1:  20 minutiae -> best Barney (0.492), then Marshall (0.032), Dwight (0.026)
suspect_2:  40 minutiae -> best Lily (0.402), then Rajesh (0.037), Ted (0.035)
suspect_3:  54 minutiae -> best Robin (0.478), then Michael (0.028), Lily (0.027)
suspect_4:  29 minutiae -> best Marshall (0.354), then Leonard (0.040), Barney (0.033)
suspect_5:  46 minutiae -> best Ted (0.089), then Sheldon (0.042), Penny (0.041)
```

!fig 19-scores | Match scores, crime-scene prints × database. Four identifications are unambiguous (10× the runner-up); print 5, the blurred and faded one, is the weakest, but still twice the next candidate.

!fig 19-matches | Each crime-scene print (left) with its best database match (right) and the matched minutiae. The parallel lines are the signature of a single rigid motion explaining many points.

**Verdict:** the five prints belong to Barney, Lily, Robin, Marshall and Ted — the five friends having dinner in the lieutenant's report. Fans of a certain sitcom will know which of them steals blue French horns.

## Exercises

1. Make print 5 more convincing: estimate the ridge period of each print and rescale it to a common period before extracting minutiae. Does the margin improve? (In my test: 0.075 vs 0.023, but the other prints got slightly worse.)
2. Use bifurcations as motion candidates too and compare the scores.
3. Replace the score by the number of matched minutiae only. Why does the normalisation by $|A|\,|B|$ matter when prints have very different numbers of minutiae?
