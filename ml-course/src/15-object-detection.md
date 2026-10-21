---
title: Object detection with YOLO
summary: Find every object in an image and draw a box around it. Intersection over union and non-maximum suppression written from scratch, what a detector outputs and how confidence thresholds and input resolution change it, detection/segmentation/pose with pretrained YOLOv8, and fine-tuning on a 36-image skateboard dataset I labelled — on a CPU, compared with my original GPU run.
topics: bounding boxes · IoU · non-maximum suppression · confidence threshold · input resolution · mAP · YOLOv8 · fine-tuning · segmentation · pose
script: 15_object_detection.py
sources: UFRN Deep Learning class 12 (YOLOv8) and my YOLOv8 training assignment (Skate detection dataset)
---

## From "what" to "what and where"

A classifier answers *what is in this image*. A **detector** answers *what is where*: a list of boxes, each with a class and a confidence. That is what a robot needs — the boat must know *where* the other boat is, the drone where its target is.

**YOLO** ("You Only Look Once") is the family of real-time detectors most used in robotics. A CNN backbone (lesson 13–14) computes feature maps at three scales; at every cell of every map, a head predicts box coordinates and class scores. The network outputs thousands of candidate boxes per image in one forward pass, and two pieces of geometry turn them into a clean answer.

## Intersection over union

How much do two boxes agree? The **IoU** is the area of their intersection divided by the area of their union: 1 for identical boxes, 0 for disjoint ones.

```python
--8<-- "15_object_detection.py:iou"
```

```text
IoU of two 10x10 boxes shifted by 5 px: 0.333
```

Half-overlapping boxes have an IoU of only 1/3 — it is a strict measure. IoU is used three times in detection: to match predictions to ground truth during training, to remove duplicate predictions (below), and to decide whether a detection counts as correct during evaluation.

## Non-maximum suppression

Neighbouring cells all see the same object and all predict a box for it. **NMS** keeps the most confident box, removes every other box that overlaps it by more than a threshold, and repeats with what is left:

```python
--8<-- "15_object_detection.py:nms"
```

```text
NMS keeps boxes [5, 6, 12, 15] | torchvision: [5, 6, 12, 15]
```

!fig 15-nms | Eighteen noisy candidates around three objects (colour = object, thickness = confidence), and what NMS keeps.

Three objects, four boxes: one candidate of the green object was shifted far enough to overlap the best box by less than 0.5 and survived. The IoU threshold is a trade-off — too high and duplicates survive, too low and genuinely overlapping objects (people in a crowd) are merged. YOLO applies NMS per class inside `predict`; newer versions (YOLOv10, RT-DETR) are trained to avoid duplicates altogether.

## A pretrained detector

YOLOv8n ("nano", 3.2 M parameters) trained on COCO's 80 classes, on the photo of Rouen from lesson 09:

```python
--8<-- "15_object_detection.py:inference"
```

```text
conf >= 0.09: 22 detections {'person': 13, 'umbrella': 4, 'car': 3, 'motorcycle': 1, 'potted plant': 1}
conf >= 0.25:  8 detections {'person': 5, 'umbrella': 3}
conf >= 0.55:  3 detections {'person': 3}
boxes tensor (8, 4) torch.float32 | first: person [146.0, 384.0, 168.0, 442.0] conf 0.77
```

The **confidence threshold** is the precision/recall trade-off of lesson 06 again. My class notebook used `conf=0.09` to "see more objects": it does find more people, but also three "cars" and a "motorcycle" that are café furniture. There is no correct threshold in general — only one that matches the cost of a miss versus a false alarm for your application.

### Small objects need pixels

The people in this photo are about 25×60 pixels. YOLO resizes every image so its long side is `imgsz` (640 by default) before the network sees it — here a mild downscale — and the smallest feature map has a stride of 8 pixels:

```python
--8<-- "15_object_detection.py:resolution"
```

```text
imgsz 640:  8 detections, 5 people
imgsz 1280: 15 detections, 9 people
```

Doubling the input resolution finds the couple sitting by the fountain and the people at the right edge — at 4× the computation. For small, distant objects (boats on the horizon, a drone's target from 30 m) resolution matters more than model size; tiling the image into overlapping crops is the other common fix.

### Segmentation and pose

The same architecture with a different head gives a pixel mask per object, or 17 body keypoints per person:

```python
--8<-- "15_object_detection.py:tasks"
```

```text
segmentation masks (14, 928, 1280) | pose keypoints (1, 17, 3)
```

!fig 15-rouen | The street level of the photo. From top: detection at the default 640 px and at 1280 px, instance segmentation, and pose estimation — the nano pose model only finds the one person large enough to see their limbs.

## Training on your own data

COCO has *person* but not *skateboard-under-a-person-in-a-bowl*. For the UFRN assignment I labelled frames of a skateboarding video on Roboflow — 52 images, classes `person` and `skate` — and fine-tuned YOLOv8n on a free Colab GPU. The dataset is in the course data (CC BY 4.0), in YOLO format: one text file per image with `class x_center y_center width height` (normalised), or polygons as here, which Ultralytics converts to boxes.

```python
--8<-- "15_object_detection.py:finetune"
```

```text
images: train 36, valid 11, test 5
fine-tuning took 8.2 min
validation: mAP50 0.738, mAP50-95 0.512 | per class mAP50-95: {'person': 0.676, 'skate': 0.349}
```

!!! note "mAP, briefly"
    For each class, sort the detections by confidence and trace the precision–recall curve, counting a detection as correct if its IoU with an unmatched ground-truth box is ≥ 0.5; the area under that curve is the **average precision**. **mAP50** averages it over classes; **mAP50-95** averages it again over IoU thresholds 0.50, 0.55, … 0.95, so it also rewards *precise* boxes. It is the standard COCO metric.

Transfer learning (lesson 14) is what makes 36 images enough: the backbone and the *person* knowledge come from COCO, and the per-class numbers show it — `person` reaches 0.68 mAP50-95, the new `skate` class only 0.35. Skateboards are small, often half-hidden under feet, and appear in 36 images; that class is where more labels would pay off.

```python
--8<-- "15_object_detection.py:original"
```

```text
my original run (100 epochs, 640 px, Colab T4): best mAP50-95 0.534 at epoch 90, final mAP50 0.774
```

!fig 15-training | My original Colab run (grey, 100 epochs on a T4 GPU) and this lesson's run (red, 60 epochs on a laptop CPU). Same data, same model; the CPU run gets within 0.02 mAP50-95 in 8 minutes.

!fig 15-skate-test | Predictions of the fine-tuned model on the five held-out test frames (first four shown).

Validation numbers on 11 images jump by ±0.1 between epochs — with so few images, a single box more or less changes the metric. Don't read the third decimal; look at the trend, and keep a test set you never tune on (here 5 frames, which is too few for anything but a sanity check).

!!! warning "Keep API keys out of notebooks"
    My training notebook downloaded the dataset with `Roboflow(api_key="...")`, the key written in plain text in a notebook that is one `git push` or one shared Colab link away from being public. Anyone with the key can read and modify the workspace. Use an environment variable (`os.environ["ROBOFLOW_API_KEY"]`) or Colab's secrets, and revoke any key that was ever committed.

## Exercises

1. Run the pretrained model on a video with `model.track(source="video.mp4")` and look at how the tracker keeps identities between frames (the base of my drone-tracking project).
2. Fine-tune with `imgsz=960` and compare the `skate` mAP. Then try `yolov8s.pt`. Which helps the small class more, pixels or parameters?
3. Implement mAP50 for one class from scratch: match predictions to ground truth greedily by confidence with IoU ≥ 0.5, accumulate precision and recall, and integrate. Compare with `metrics.box.ap50`.
