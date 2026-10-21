---
title: Architectures and transfer learning
summary: A short tour of the architectures that shaped deep learning, the residual connection that made very deep networks trainable — measured, not just asserted — and the most useful trick in applied deep learning: start from a network pretrained on ImageNet. With 1,000 labelled satellite images, a fine-tuned ResNet-18 reaches 96%.
topics: AlexNet · VGG · ResNet · residual connections · EfficientNet · ViT · pretrained features · linear probe · fine-tuning · EuroSAT
script: 14_transfer_learning.py
sources: UFRN Deep Learning class 11 (architectures), my drone and boat projects
---

## A decade of architectures

Every image classifier since 2012 is a variation on lesson 13's recipe — convolutions, non-linearities, downsampling, a classifier at the end — with ideas added to go deeper, cheaper or more accurate. torchvision ships most of them with ImageNet weights, and their metadata gives a compact history:

```python
--8<-- "14_transfer_learning.py:zoo"
```

```text
AlexNet (2012)            61.1 M params    0.71 GFLOPs  ImageNet top-1 56.5%
VGG-16 (2014)            138.4 M params   15.47 GFLOPs  ImageNet top-1 71.6%
ResNet-18 (2015)          11.7 M params    1.81 GFLOPs  ImageNet top-1 69.8%
ResNet-50 (2015)          25.6 M params    4.09 GFLOPs  ImageNet top-1 80.9%
MobileNet-V3-S (2019)      2.5 M params    0.06 GFLOPs  ImageNet top-1 67.7%
EfficientNet-B0 (2019)     5.3 M params    0.39 GFLOPs  ImageNet top-1 77.7%
ConvNeXt-T (2022)         28.6 M params    4.46 GFLOPs  ImageNet top-1 82.5%
ViT-B/16 (2020)           86.6 M params   17.56 GFLOPs  ImageNet top-1 81.1%
```

- **AlexNet** (2012) — the network that won ImageNet by a landslide with GPUs, ReLUs and dropout, and started the deep learning era.
- **VGG** (2014) — "just stack 3×3 convolutions": simple and deep, but 138 M parameters, most of them in its dense layers.
- **ResNet** (2015) — residual connections (below) made 50, 101, even 1,000 layers trainable. ResNet-18 matches VGG-16 with 12× fewer parameters and 8× less compute; it is still the default backbone in robotics.
- **MobileNet, EfficientNet** — designed for phones and embedded boards (depthwise-separable convolutions, scaled width/depth/resolution). MobileNet-V3-Small runs in 60 MFLOPs: this is the family that fits on a drone.
- **Vision Transformers** (2020) — cut the image into 16×16 patches and process them with the attention layers of language models. Better than CNNs with huge pretraining data; ConvNeXt (2022) showed a modernised CNN can match them.

## The residual connection

Why couldn't networks simply be made deeper before 2015? Measure it: the gradient that reaches the first layer of a plain stack of convolutions, at initialisation, versus the same depth built from **residual blocks** $\mathbf{y} = \text{relu}(\mathbf{x} + F(\mathbf{x}))$:

```python
--8<-- "14_transfer_learning.py:residual"
```

```text
 4 conv layers: gradient norm at the first layer, plain 4.81e-04, residual 9.20e-02
16 conv layers: gradient norm at the first layer, plain 8.99e-10, residual 1.21e-01
48 conv layers: gradient norm at the first layer, plain 0.00e+00, residual 2.79e-01
```

In the plain network the signal shrinks a little at every layer, forward and backward (lesson 11's product of derivatives), and at 48 layers the gradient underflows to exactly zero: the first layers cannot learn at all. In the residual network the identity path $\mathbf{x}$ carries the signal and its gradient straight through every block, and each block only has to learn a *correction* $F$ to the identity — so an extra block can at worst learn $F = 0$ and do no harm. Batch normalisation and careful initialisation help plain networks too, but the skip connection is what scales: it is in every modern architecture, transformers included.

## Transfer learning

Training ResNet-18 from scratch takes 1.3 million labelled ImageNet images and days of GPU time. But its early layers learn edges and textures, and its later layers learn generic parts and shapes, useful far beyond ImageNet's 1,000 classes. **Transfer learning** reuses them: take the pretrained network, replace its last layer, and train on your small dataset.

The test: **EuroSAT**, 64×64 Sentinel-2 satellite tiles of 10 land-use classes — the kind of aerial imagery a drone or a mapping pipeline deals with, and quite unlike ImageNet's photos of dogs and cars. We pretend labels are expensive and keep only **100 images per class** for training:

```python
--8<-- "14_transfer_learning.py:data"
```

```text
classes: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
train (1000, 3, 64, 64) val (500, 3, 64, 64) test (2000, 3, 64, 64) torch.float32
```

!fig 14-eurosat | One training tile per class.

Satellite tiles have no "up", so flips and 90° rotations are free augmentations (8 versions of every tile):

```python
--8<-- "14_transfer_learning.py:augment"
```

### Baseline: a CNN from scratch

```python
--8<-- "14_transfer_learning.py:scratch"
```

```text
CNN from scratch: val acc 0.834, test acc 0.859 (109 s)
```

### Option 1: frozen features + a linear classifier

Remove ResNet-18's ImageNet head (`fc = Identity()`): the network now maps every tile to 512 numbers. Compute them once, with the network frozen, and train a logistic regression (lesson 04) on top — a **linear probe**. Two details are essential: the input must be preprocessed **exactly as during pretraining** (ImageNet mean and standard deviation), and tiny 64×64 images are upsampled, because ImageNet objects were seen at ~224 pixels.

```python
--8<-- "14_transfer_learning.py:features"
```

```text
frozen ResNet-18 + logistic regression: val acc 0.884, test acc 0.880 (16 s)
```

Better than the CNN trained for two minutes, in 16 seconds, with no deep learning training at all. And the advantage grows as labels get scarcer:

```python
--8<-- "14_transfer_learning.py:few-shot"
```

```text
  5 labelled images per class -> test acc 0.638
 10 labelled images per class -> test acc 0.740
 25 labelled images per class -> test acc 0.824
 50 labelled images per class -> test acc 0.849
100 labelled images per class -> test acc 0.880
```

With 25 images per class, pretrained features come within 3.5 points of the from-scratch CNN trained on four times more.

### Option 2: fine-tuning

Train the whole network — pretrained layers included — with a new 10-class head and a small learning rate (3e-4: we want to adjust the features, not destroy them):

```python
--8<-- "14_transfer_learning.py:finetune"
```

```text
fine-tuned ResNet-18: val acc 0.946, test acc 0.960 (121 s)
```

!fig 14-results | Left: validation accuracy per epoch. The fine-tuned network starts where the linear probe ends and passes 94% in five epochs; the CNN from scratch is still at 83% after 40. Right: confusion matrix of the fine-tuned model on 2,000 test tiles.

!fig 14-errors | One test error for each of six true classes (81 errors in total). Crop types, pasture and vegetation are hard to tell apart at 10 m per pixel even for a person.

| Approach | Labelled images | Training time (CPU) | Test accuracy |
|---|---|---|---|
| CNN from scratch | 1,000 | 2 min | 85.9% |
| Frozen ResNet-18 + logistic regression | 1,000 | 16 s | 88.0% |
| Fine-tuned ResNet-18 | 1,000 | 2 min | **96.0%** |

## A recipe for your own data

1. Start from a pretrained backbone, preprocessed exactly as it expects. Pick the size for your hardware: ResNet-18/50 on a laptop or a Jetson, MobileNet/EfficientNet on a microcontroller-class device.
2. Try a **linear probe** first: minutes, and a strong baseline. If it is good enough, stop.
3. Otherwise **fine-tune** with a learning rate 10× smaller than from scratch, augmentation that respects your labels, and early stopping. With very little data, freeze the early layers and train only the last block and the head.
4. Train from scratch only when your data is huge, or nothing like natural images (spectrograms, depth maps, medical volumes — and even then, pretrained models often still help).

This is also how detectors are trained: the YOLO model of the next lesson starts from COCO weights and adapts to a new dataset with a few dozen images.

## Exercises

1. Fine-tune only `layer4` and `fc` (set `requires_grad = False` on the rest). How do time and accuracy compare with full fine-tuning?
2. Replace ResNet-18 by `efficientnet_b0` or `mobilenet_v3_small` (look for the name of their classifier layer). Plot test accuracy against GFLOPs.
3. Run the linear probe with the backbone's **random** initialisation instead of ImageNet weights. What accuracy do random features give, and what does that tell you about where the 88% comes from?
