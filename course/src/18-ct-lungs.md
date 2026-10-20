---
title: "Case study: lungs and vessels in CT scans"
summary: Medical images are not photos. Read CT slices in Hounsfield units, choose display windows, segment the lungs with thresholds, connected components and contours, segment the vessels inside them, and screen 21 patients.
topics: NIfTI · Hounsfield units · windowing · connected components · contours · masks · areas in mm²
script: 18_ct_lungs.py
sources: INSA TIM lab 8
---

!!! warning "Not a medical tool"
    This is an image-processing exercise, not a diagnostic method. The "vessel to lung ratio" below is a toy criterion chosen for the lab.

## Computed tomography in two paragraphs

A CT scanner sends X-rays through the body from many angles and reconstructs, slice by slice, how much each small volume absorbs them. The result is not brightness in arbitrary units but a physical quantity on an absolute scale, the **Hounsfield unit** (HU): air is −1000 HU, water 0 HU, soft tissue about +20 to +80, bone +400 to +1000 and more. Lungs, full of air, sit around −900 to −500 HU; blood vessels inside them around +40.

Scans are stored in medical formats (DICOM, NIfTI) that also carry the **physical size of a pixel**, so areas can be measured in mm². The `nibabel` library reads NIfTI:

```python
--8<-- "18_ct_lungs.py:load"
```

```text
(512, 512) float64 HU range [-1024, 1554] pixel (mm): [0.76171875 0.76171875]
```

The array is `float64`, from −1024 to 1554: it cannot be shown "as is" on a 0–255 screen.

## Windowing

Radiologists display a range of HU chosen for the tissue they study: a **window** of width $W$ centred on a **level** $L$, mapped linearly to 0–255 and clipped outside. It is the dynamic expansion of lesson 04, in physical units:

$$\text{display} = 255 \cdot \frac{\operatorname{clip}(HU,\, L - W/2,\, L + W/2) - (L - W/2)}{W}$$

| Region | Level | Width |
|---|---|---|
| Lungs | −600 | 1500 |
| Soft tissue | 50 | 350 |
| Brain | 40 | 80 |
| Bone | 400 | 1800 |

```python
--8<-- "18_ct_lungs.py:window"
```

!fig 18-windows | The same slice with four windows. The full range wastes the screen's 256 levels; the lung window shows the airways and vessels, the soft-tissue window the heart and muscles.

## Segmenting the lungs

The plan, one idea per line:

1. **Air** is everything below about −320 HU — but that includes the air *around* the patient.
2. The outside air **touches the image border**; the lungs don't. Label the connected components (lesson 09) and drop those that touch the border, and the tiny ones (bowel gas, noise).
3. The vessels inside the lungs are *not* air, so they are holes in the mask. Find the outer contour of each lung and **fill** it.

```python
--8<-- "18_ct_lungs.py:lungs"
```

My lab version selected contours by length and then took "the 2nd to 4th largest" (to skip the body outline). It worked on most slices but depended on the order of the contours. Removing everything that touches the border expresses the intent directly and works on all 21 slices — including those where the scanner padded the image with −8192 HU outside the field of view.

## Segmenting the vessels

Inside the lungs, a vessel is a pixel above −500 HU. But the lung *wall* is also above −500 HU, and the filled mask includes it. Eroding the mask a few pixels first keeps only the inside:

```python
--8<-- "18_ct_lungs.py:vessels"
```

!fig 18-pipeline | Air below −320 HU; the lungs (no border contact, large, filled); dense pixels inside them — the wall is included; with the eroded mask, only vessels remain; final overlay.

```text
lungs 153.2 cm², vessels 6.1 cm², ratio 0.040
```

## Screening all the patients

```python
--8<-- "18_ct_lungs.py:all"
```

!!! bug "Sort your file lists"
    `os.listdir` and `glob` return files in an **arbitrary** order. In my lab report I numbered patients by their position in the list and then reloaded `slice{i+1}` — so the flagged "patients" were not the ones whose ratios I had computed. `sorted(glob.glob(...))` and keeping the name with the result avoid that.

!fig 18-ratios | Vessel-to-lung area ratio per slice. Four slices reach the (arbitrary) 5% threshold.

!fig 18-all-slices | Eight of the 21 segmentations. Slices 010, 012, 020 and 021 contain dense regions inside the lungs, which is what pushes their ratio up.

**Limitations.** The trachea and main bronchi are air too, and in a few slices they are counted in the lung area. A threshold of −500 HU also catches any dense tissue (scars, consolidations), not only vessels. Real pipelines work in 3-D (the airway is one connected tube across slices) and increasingly with neural networks — but they still start with Hounsfield windows and connected components.

## Exercises

1. Remove the trachea: it is the round air component closest to the image centre in the upper slices. Add a circularity test, $4\pi A / P^2 > 0.7$, with `cv2.contourArea` and `cv2.arcLength`.
2. Save the lung and vessel masks of every slice as PNG files next to the windowed image, for a radiologist to review.
3. Separate the left and right lung (the two largest components) and report the ratio for each one.
