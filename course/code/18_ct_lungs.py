"""Lesson 18 - Case study: lung and vessel segmentation in CT scans."""
import glob
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

import cvutils as U

U.setup("18")


# --8<-- [start:load]
def load_slice(path):
    scan = nib.load(str(path))
    hu = np.flip(scan.get_fdata(), axis=1).T          # Hounsfield units, radiological orientation
    pix_mm = scan.header["pixdim"][1:3]               # pixel size (mm) along the two image axes
    return hu, pix_mm


hu, pix_mm = load_slice(U.path("ct/slice017.nii.gz"))
print(hu.shape, hu.dtype, f"HU range [{hu.min():.0f}, {hu.max():.0f}]", "pixel (mm):", pix_mm)
# --8<-- [end:load]


# --8<-- [start:window]
def window(hu, level, width):
    """Map [level - width/2, level + width/2] HU to [0, 255]; clip everything else."""
    lo, hi = level - width / 2, level + width / 2
    return np.round(255 * (np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.uint8)


lung_view = window(hu, level=-600, width=1500)
soft_tissue = window(hu, level=50, width=350)
bone = window(hu, level=400, width=1800)
# --8<-- [end:window]
U.show("windows", [window(hu, hu.mean(), hu.max() - hu.min()), lung_view, soft_tissue, bone],
       ["full range", "lung window (L -600, W 1500)", "soft tissue (L 50, W 350)", "bone (L 400, W 1800)"],
       cols=4, size=3)


# --8<-- [start:lungs]
def segment_lungs(hu, air_hu=-320, min_area_px=2000):
    air = (hu < air_hu).astype(np.uint8)                  # air and lungs are both below ~-320 HU
    # the air *around* the patient touches the image border: remove it
    n, labels, stats, _ = cv2.connectedComponentsWithStats(air, connectivity=4)
    border = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = [i for i in range(1, n) if i not in border and stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    candidates = np.isin(labels, keep).astype(np.uint8) * 255
    # vessels are *not* air, so they are holes in the candidates: fill each outer contour
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(candidates)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask, contours
# --8<-- [end:lungs]

# --8<-- [start:vessels]
def segment_vessels(hu, lung_mask, vessel_hu=-500, erode_iter=3):
    # erode the lung mask so the dense lung wall is not counted as a vessel
    inner = cv2.erode(lung_mask, np.ones((3, 3), np.uint8), iterations=erode_iter)
    return ((hu > vessel_hu) & (inner > 0)).astype(np.uint8) * 255


def areas_mm2(mask, pix_mm):
    return np.count_nonzero(mask) * pix_mm[0] * pix_mm[1]
# --8<-- [end:vessels]

lung_mask, contours = segment_lungs(hu)
air = (hu < -320).astype(np.uint8) * 255
naive_vessels = ((hu > -500) & (lung_mask > 0)).astype(np.uint8) * 255
vessels = segment_vessels(hu, lung_mask)
lung_area, vessel_area = areas_mm2(lung_mask, pix_mm), areas_mm2(vessels, pix_mm)
print(f"lungs {lung_area / 100:.1f} cm², vessels {vessel_area / 100:.1f} cm², ratio {vessel_area / lung_area:.3f}")

overlay = cv2.cvtColor(lung_view, cv2.COLOR_GRAY2BGR)
cv2.drawContours(overlay, contours, -1, (0, 220, 0), 2)
overlay[vessels > 0] = (0, 0, 255)
U.show("pipeline", [air, lung_mask, naive_vessels, vessels, overlay],
       ["HU < -320", "lungs: no border, big, filled", "HU > -500 inside lungs", "… with eroded mask",
        "result"], cols=5, size=2.8)

# --8<-- [start:all]
results = []
for path in sorted(glob.glob(str(U.path("ct/*.nii.gz")))):
    hu, pix_mm = load_slice(path)
    lungs, _ = segment_lungs(hu)
    vessels = segment_vessels(hu, lungs)
    ratio = areas_mm2(vessels, pix_mm) / areas_mm2(lungs, pix_mm)
    results.append((Path(path).name.split(".")[0], ratio, hu, lungs, vessels))

for name, ratio, *_ in results:
    flag = "  <- suspicious (>= 5%)" if ratio >= 0.05 else ""
    print(f"{name}: vessel/lung = {100 * ratio:4.1f}%{flag}")
# --8<-- [end:all]

fig, axes = U.figure(1, 1, size=(7.5, 2.6))
ax = axes[0, 0]
names = [r[0].replace("slice0", "") for r in results]
vals = [100 * r[1] for r in results]
ax.bar(names, vals, color=["#dc2626" if v >= 5 else "#94a3b8" for v in vals])
ax.axhline(5, color="#18181b", lw=1, ls="--")
ax.set_ylabel("vessel / lung area (%)")
ax.set_xlabel("slice")
U.finish("ratios", fig)

tiles, titles = [], []
for name, ratio, hu, lungs, vessels in results[:: max(1, len(results) // 8)][:8]:
    t = cv2.cvtColor(window(hu, -600, 1500), cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(lungs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(t, cs, -1, (0, 220, 0), 2)
    t[vessels > 0] = (0, 0, 255)
    tiles.append(cv2.resize(t, (300, 300)))
    titles.append(f"{name}: {100 * ratio:.1f}%")
U.show("all-slices", tiles, titles, cols=4, size=2.8)
