---
title: Convolutional neural networks
summary: Networks that know they are looking at images. Why a dense layer is the wrong tool for pixels, what a convolution layer computes and how many parameters it costs, an MLP against a CNN on Fashion-MNIST, the overfitting of the class model and how batch norm, dropout, augmentation and a learning-rate schedule fix it — and what the learnt filters actually detect.
topics: convolution · weight sharing · pooling · Fashion-MNIST · overfitting · batch normalisation · dropout · data augmentation · feature maps
script: 13_cnn.py
sources: UFRN Deep Learning classes 10–11 (CNNs, Fashion-MNIST) and exercise 3 (MNIST + Gradio)
---

## Images are not just long vectors

The MLP of lesson 12 started with `Flatten`: the 28×28 image became 784 unrelated numbers. Shuffle the pixels the same way in every image and the MLP learns exactly as well — it has no idea that pixel (10, 10) is next to pixel (10, 11). Two facts about images are thrown away:

- **Locality**: what matters are local patterns — an edge, a corner, a texture — made of neighbouring pixels.
- **Translation equivariance**: an edge is an edge wherever it is. A detector learnt in the top-left corner should work in the bottom-right.

A **convolution layer** builds both in. It slides a small kernel (3×3 here) over the image and computes the same weighted sum at every position — exactly the filtering of the [Computer Vision course](../course/05-noise-filtering.html), except that the kernel weights are *learnt*. A layer has many kernels (*channels*), each producing a **feature map**.

## The data

Fashion-MNIST (Zalando): 70,000 grey 28×28 images of 10 kinds of clothing, a harder drop-in replacement for MNIST — digits are too easy to tell methods apart.

```python
--8<-- "13_cnn.py:data"
```

```text
train (54000, 1, 28, 28) val (6000, 1, 28, 28) test (10000, 1, 28, 28)
```

PyTorch images are `(batch, channels, height, width)` — **channels first**, unlike NumPy/OpenCV images `(H, W, C)` and Keras `(N, H, W, C)`. Forgetting the `unsqueeze(1)` (the channel axis) is the first CNN bug everyone makes.

## What a convolution layer costs

```python
--8<-- "13_cnn.py:conv"
```

```text
conv weight (32, 1, 3, 3) -> parameters: 320
input (8, 1, 28, 28) -> conv (8, 32, 28, 28) -> max-pool (8, 32, 14, 14)
a dense layer with the same output size would need 19.7 M parameters
```

32 kernels of 3×3, plus 32 biases: **320 parameters** to produce 25,088 outputs. A dense layer with the same output would need 19.7 million. That is **weight sharing**: the same 9 weights are reused at all 784 positions. With `padding=1` the output keeps the input size (in general, $\text{out} = (\text{in} + 2p - k)/s + 1$ for kernel $k$, padding $p$, stride $s$).

**Max-pooling** keeps the largest value of each 2×2 block: it halves the resolution, makes the representation tolerant to small shifts, and lets the next layer's 3×3 kernels see a 2× larger region of the original image. Stacking conv + pool layers grows the **receptive field**: early layers see edges, later layers see parts made of edges.

## Three models

```python
--8<-- "13_cnn.py:models"
```

```text
MLP                101,770 parameters
small CNN          804,554 parameters
regularised CNN    431,018 parameters
```

The first two are the models of the UFRN class (Keras `model1` and `model2`). The small CNN has 8× more parameters than the MLP — but 99.9% of them are in the dense layer after `Flatten` (32 × 14 × 14 × 128), not in the convolution. The regularised CNN has more convolutions and *fewer* parameters, because it pools twice before the dense layer.

The third model adds three regularisers:

- **Batch normalisation** normalises each channel over the mini-batch (zero mean, unit variance, then a learnt scale and shift) — lesson 02's "normalise the inputs", applied inside the network at every layer. It makes training much less sensitive to the learning rate and initialisation.
- **Dropout** sets a random 40% of the activations to zero at each training step, so no unit can rely on a specific other unit; at test time all units are used (PyTorch rescales automatically — one reason `model.eval()` matters).
- **Data augmentation** shows a slightly different version of each image at every epoch:

```python
--8<-- "13_cnn.py:augment"
```

Augmentations must preserve the label: a mirrored shoe is still a shoe, but a mirrored "6" in MNIST is not a "6" any more, and upside-down clothes would not appear in the test set. Finally, weight decay and a **cosine learning-rate schedule** (large steps first, small at the end) — the training helper is lesson 12's loop moved to `mlutils.py`:

```python
--8<-- "mlutils.py:train"
```

## Training

```python
--8<-- "13_cnn.py:train"
```

```text
MLP
  final: train acc 0.9111, val acc 0.8935, test acc 0.8845 | lowest val loss 0.3018 at epoch 11, final val loss 0.3030
small CNN
  final: train acc 0.9741, val acc 0.9158, test acc 0.9140 | lowest val loss 0.2323 at epoch 7, final val loss 0.2832
regularised CNN + augmentation
  final: train acc 0.9039, val acc 0.9228, test acc 0.9201 | lowest val loss 0.2058 at epoch 15, final val loss 0.2058
```

!fig 13-curves | Loss and accuracy per epoch, training (dashed) and validation (solid). The small CNN's training loss keeps falling while its validation loss turns up after epoch 7: overfitting. The regularised CNN's validation loss keeps decreasing until the end.

Three things to read in this figure:

1. **The convolution helps**: +3 points over the MLP with the same classifier on top.
2. **The small CNN overfits**, as it did in my Colab run (Keras, 20 epochs: validation loss 0.257 at epoch 3, then up to 0.537 at epoch 20 while training accuracy kept climbing): training accuracy 97%, validation 92%, and the validation *loss* rising while the accuracy stays flat — the model becomes over-confident on the images it gets wrong. Early stopping (lesson 12) would have kept the epoch-7 weights.
3. **Regularisation removes the overfitting.** The regularised model's *training* accuracy is lower than its validation accuracy — normal, since it is measured with dropout on and on augmented images — and it is still improving after 15 epochs. The test gain is small here (92.0% vs 91.4%) because 15 epochs is short; regularised models need longer training, and they are the ones that keep improving.

## What it gets wrong

```python
--8<-- "13_cnn.py:errors"
```

```text
Shirt        accuracy 0.727, most often mistaken for T-shirt/top (125 times)
Pullover     accuracy 0.881, most often mistaken for Shirt (59 times)
Coat         accuracy 0.893, most often mistaken for Pullover (45 times)
```

!fig 13-confusion | Confusion matrix of the regularised CNN on the test set (colour on a log scale). Almost all errors are among the upper-body garments.

Trousers, bags, sandals and boots are near-perfect; shirt, T-shirt, pullover and coat are confused with each other. At 28×28 in greyscale, a shirt and a T-shirt can be literally the same blob of pixels — some of these errors are label noise and some are the limit of the resolution. A confusion matrix tells you where more data, more resolution or a better model would pay off.

## What the filters learn

```python
--8<-- "13_cnn.py:filters"
```

```text
first-layer filters (32, 1, 3, 3) | feature maps (32, 28, 28)
```

!fig 13-filters | Top: eight learnt 3×3 filters of the first layer (red positive, blue negative). Bottom: their feature maps on an ankle boot (after batch norm and ReLU). Several are edge detectors in different orientations — the network rediscovered Sobel-like kernels — and one responds to the background.

Nobody told the network to detect edges. Gradient descent found that horizontal, vertical and diagonal derivative filters are the most useful first step to classify clothes — the same conclusion image processing reached by hand decades ago. Deeper layers combine these maps into textures, parts and shapes.

!!! tip "Put a model behind an interface"
    The UFRN exercise wrapped an MNIST CNN in a [Gradio](https://www.gradio.app) app with a sketchpad: draw a digit, get the prediction. The lesson I took from it: the model reached 98.7% on the test set but failed on many *uploaded* photos of digits — they were dark-on-white, off-centre and not 28×28, unlike MNIST's white-on-black centred digits. A test set only measures performance on data like the training data; preprocessing new inputs the same way (invert, crop, centre, resize) matters as much as the network.

## Exercises

1. Train the small CNN with early stopping (`patience=3`) using lesson 12's `fit`. Which epoch is kept, and what is its test accuracy?
2. Remove one ingredient at a time from the regularised model (batch norm, dropout, augmentation, cosine schedule) and train for 15 epochs. Which one matters most here?
3. Replace max-pooling by convolutions with `stride=2`. Compare parameters, time per epoch and accuracy.
