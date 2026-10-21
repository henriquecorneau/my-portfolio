"""Lesson 15 - Object detection: IoU, non-maximum suppression, YOLOv8 inference and fine-tuning on a small dataset."""
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO

import mlutils as U

U.setup("15")


# --8<-- [start:iou]
def iou(a, b):
    """Intersection over union of boxes in (x1, y1, x2, y2) format. a: (N, 4), b: (M, 4) -> (N, M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


print("IoU of two 10x10 boxes shifted by 5 px:", iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))[0, 0].round(3))
# --8<-- [end:iou]


# --8<-- [start:nms]
def nms(boxes, scores, thr=0.5):
    """Greedy non-maximum suppression: keep the best box, drop every box overlapping it by more than thr."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        best, rest = order[0], order[1:]
        keep.append(best)
        order = rest[iou(boxes[best:best + 1], boxes[rest])[0] <= thr]
    return np.array(keep)


rng = np.random.default_rng(0)
centres = np.array([[30, 30], [70, 40], [50, 80]])
boxes = np.concatenate([np.c_[c - 12 + rng.normal(0, 2, (6, 2)), c + 12 + rng.normal(0, 2, (6, 2))]
                        for c in centres])                       # 3 objects x 6 overlapping candidates
scores = rng.uniform(0.3, 0.95, len(boxes))
kept = nms(boxes, scores)
ref = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32), 0.5)
print("NMS keeps boxes", sorted(kept.tolist()), "| torchvision:", sorted(ref.tolist()))
# --8<-- [end:nms]

fig, axes = U.figure(2, 1, size=(3.3, 3.3))
for ax, idx, title in [(axes[0, 0], range(len(boxes)), f"{len(boxes)} raw candidates"),
                       (axes[0, 1], kept, f"after NMS: {len(kept)} boxes")]:
    for i in idx:
        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(__import__("matplotlib").patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                                                  color=U.CLASS_COLORS[i // 6], lw=1 + 2 * scores[i]))
        ax.text(x1, y1 - 1, f"{scores[i]:.2f}", fontsize=6, color=U.CLASS_COLORS[i // 6])
    ax.set_xlim(0, 100)
    ax.set_ylim(105, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
U.finish("nms", fig)

# --8<-- [start:inference]
model = YOLO(str(U.CACHE / "yolov8n.pt"))           # pretrained on COCO (80 classes); downloaded once
img_path = str(U.path("rouen.jpg"))                 # 800x576 photo, people are ~25x60 pixels
for conf in (0.09, 0.25, 0.55):
    r = model.predict(img_path, conf=conf, verbose=False)[0]
    names = [r.names[int(c)] for c in r.boxes.cls]
    print(f"conf >= {conf}: {len(names):2d} detections", {k: int(v) for k, v in pd.Series(names).value_counts().items()})
r = model.predict(img_path, conf=0.25, verbose=False)[0]
print("boxes tensor", tuple(r.boxes.xyxy.shape), r.boxes.xyxy.dtype, "| first:",
      r.names[int(r.boxes.cls[0])], r.boxes.xyxy[0].round().tolist(), "conf %.2f" % r.boxes.conf[0])
# --8<-- [end:inference]

# --8<-- [start:resolution]
for size in (640, 1280):
    r = model.predict(img_path, conf=0.25, imgsz=size, verbose=False)[0]
    people = int((r.boxes.cls == 0).sum())
    print(f"imgsz {size}: {len(r.boxes):2d} detections, {people} people")
r_hi = r
# --8<-- [end:resolution]

# --8<-- [start:tasks]
seg = YOLO(str(U.CACHE / "yolov8n-seg.pt")).predict(img_path, conf=0.25, imgsz=1280, verbose=False)[0]
pose = YOLO(str(U.CACHE / "yolov8n-pose.pt")).predict(img_path, conf=0.25, imgsz=1280, verbose=False)[0]
print("segmentation masks", tuple(seg.masks.data.shape), "| pose keypoints", tuple(pose.keypoints.data.shape))
# --8<-- [end:tasks]

r_lo = model.predict(img_path, conf=0.25, verbose=False)[0]
fig, axes = U.figure(1, 4, size=(8.0, 2.45))
for ax, res, title in [(axes[0, 0], r_lo, "detection at 640 px (default)"), (axes[1, 0], r_hi, "detection at 1280 px"),
                       (axes[2, 0], seg, "instance segmentation (1280 px)"),
                       (axes[3, 0], pose, "pose estimation, 17 keypoints per person (1280 px)")]:
    ax.imshow(res.plot(line_width=1, font_size=7)[330:520, :, ::-1])     # crop: the street level
    ax.set_title(title)
    ax.axis("off")
U.finish("rouen", fig)

# --8<-- [start:finetune]
skate = U.path("skate")
yaml_path = U.CACHE / "skate.yaml"
yaml_path.write_text(f"path: {skate}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                     "names:\n  0: person\n  1: skate\n")
print("images: train %d, valid %d, test %d" % tuple(len(list((skate / s / "images").glob("*.jpg")))
                                                 for s in ("train", "valid", "test")))
t0 = time.time()
tuned = YOLO(str(U.CACHE / "yolov8n.pt"))           # start from the COCO weights: transfer learning
tuned.train(data=str(yaml_path), epochs=60, imgsz=640, batch=8, device="cpu", workers=2, seed=0,
            project=str(U.CACHE / "runs"), name="skate", exist_ok=True, plots=False, verbose=False)
print("fine-tuning took %.1f min" % ((time.time() - t0) / 60))
metrics = tuned.val(data=str(yaml_path), split="val", imgsz=640, device="cpu", verbose=False, plots=False)
print("validation: mAP50 %.3f, mAP50-95 %.3f | per class mAP50-95: %s"
      % (metrics.box.map50, metrics.box.map, {k: round(float(v), 3) for k, v in zip(["person", "skate"], metrics.box.maps)}))
# --8<-- [end:finetune]

# --8<-- [start:original]
orig = pd.read_csv(skate / "original_run_results.csv")
orig.columns = [c.strip() for c in orig.columns]
best = orig.loc[orig["metrics/mAP50-95(B)"].idxmax()]
print("my original run (100 epochs, 640 px, Colab T4): best mAP50-95 %.3f at epoch %d, final mAP50 %.3f"
      % (best["metrics/mAP50-95(B)"], best["epoch"], orig["metrics/mAP50(B)"].iloc[-1]))
# --8<-- [end:original]
mine = pd.read_csv(U.CACHE / "runs" / "skate" / "results.csv")
mine.columns = [c.strip() for c in mine.columns]

fig, axes = U.figure(2, 1, size=(4.2, 3.1))
for df, label, c in [(orig, "original run (100 epochs, 640 px)", U.GRAY), (mine, "this lesson (60 epochs, 640 px, CPU)", U.RED)]:
    axes[0, 0].plot(df["epoch"], df["train/box_loss"], color=c, label=label)
    axes[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"], color=c, ls="--", lw=1)
    axes[0, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"], color=c, label=label)
axes[0, 0].set_title("training box loss")
axes[0, 1].set_title("validation mAP50 (dashed) and mAP50-95")
for ax in axes[0]:
    ax.set_xlabel("epoch")
axes[0, 0].legend(fontsize=7)
U.finish("training", fig)

test_imgs = sorted((skate / "test" / "images").glob("*.jpg"))[:4]
fig, axes = U.figure(4, 1, size=(3.1, 2.5))
for ax, p in zip(axes[0], test_imgs):
    res = tuned.predict(str(p), conf=0.25, verbose=False)[0]
    ax.imshow(res.plot(line_width=2, font_size=10)[..., ::-1])
    ax.axis("off")
U.finish("skate-test", fig)
