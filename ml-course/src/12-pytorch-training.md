---
title: PyTorch and the training loop
summary: The framework that does lesson 11's calculus for you. Tensors with shapes, dtypes and devices, autograd checked against a hand derivation, the canonical training loop, SGD vs momentum vs Adam, the MNIST bug of forgetting to normalise the pixels, and early stopping with a checkpoint you can reload.
topics: tensors · dtype · device · autograd · nn.Module · DataLoader · optimisers · input normalisation · early stopping · checkpoints
script: 12_pytorch_training.py
sources: INSA IML deep learning lab (Keras), UFRN Deep Learning class 6
---

## Why a framework

Lesson 11 needed 20 lines of careful calculus for a two-layer network. A modern network has hundreds of layers of a dozen types. Frameworks give three things: **tensors** (NumPy-like arrays that can live on a GPU), **automatic differentiation** (the backward pass is derived for you), and building blocks (layers, losses, optimisers, data loaders). This course uses **PyTorch**; my original labs used Keras/TensorFlow, and every idea here translates one-to-one.

## Tensors: shape, dtype, device

```python
--8<-- "12_pytorch_training.py:tensors"
```

```text
torch.Size([3, 4]) torch.float32 cpu | a @ b = [6.0, 22.0, 38.0]
broadcasting: (3, 4) + (4,) -> (3, 4) | (3, 1) * (1, 4) -> (3, 4)
training on: cpu
```

Three attributes to keep in your head at every line:

- **shape** — most deep learning bugs are shape bugs. Write the expected shape as a comment next to every non-trivial tensor; I do it throughout the course.
- **dtype** — networks train in `float32` (or lower); NumPy defaults to `float64`. Class labels for `CrossEntropyLoss` must be `int64`.
- **device** — every tensor in an operation must be on the same device (`.to(device)`). The code runs unchanged on CPU or GPU.

### Is it faster?

The UFRN class compared matrix products in NumPy and the framework on CPU and GPU:

```python
--8<-- "12_pytorch_training.py:timing"
```

```text
  256 x 256: NumPy    3.87 ms | PyTorch CPU    1.56 ms |    22 GFLOP/s
 1024 x 1024: NumPy    7.09 ms | PyTorch CPU   17.09 ms |   126 GFLOP/s
 2048 x 2048: NumPy   45.53 ms | PyTorch CPU   65.47 ms |   262 GFLOP/s
```

On a CPU both call an optimised BLAS library, and neither is consistently faster (the numbers move with whatever else the machine is doing). The real gain is the GPU, typically 10–50× on large products. Two timing traps: the first call pays one-off initialisation costs (warm up first), and **GPU calls are asynchronous** — they return before the computation is done, so without `torch.cuda.synchronize()` you measure only the launch. (My version of this exercise had `np.arrange` and no loop over the sizes; it never ran.)

## Autograd

Mark parameters with `requires_grad=True`; every operation on them records itself in a graph; `loss.backward()` walks that graph backwards applying the chain rule — lesson 11's `backward`, generated automatically.

```python
--8<-- "12_pytorch_training.py:autograd"
```

```text
p = 0.1978 | dL/dW = [-0.802183985710144, -1.604367971420288] | by hand (p - y) x = [-0.8021838665008545, -1.604367733001709]
```

The "prediction minus label, times input" of lesson 04, to seven digits.

## Model, loss, optimiser, loop

A model is an `nn.Module`; for a plain stack of layers `nn.Sequential` is enough. Hyper-parameters go in **one config object**, not scattered through the code, so every experiment is described by a single value you can print, log or save next to the results:

```python
--8<-- "12_pytorch_training.py:model"
```

The training loop is always the same five steps — forward, loss, `zero_grad`, `backward`, `step` — inside two loops (epochs, mini-batches), followed by an evaluation in `eval()` mode under `torch.no_grad()`:

```python
--8<-- "12_pytorch_training.py:loop"
```

Two details matter. Gradients **accumulate** in `.grad` (useful for very large batches) so they must be zeroed at every step. And the losses take **logits** — the raw scores before the sigmoid/softmax: `BCEWithLogitsLoss` and `CrossEntropyLoss` apply the sigmoid/softmax internally in a numerically stable way (computing `log(softmax(z))` naively overflows for large $z$).

!!! bug "Softmax twice"
    In my UFRN MNIST + Gradio exercise the Keras model already ended with a `softmax` layer, and the prediction function applied `tf.nn.softmax` **again** to its output. The predicted class didn't change (softmax preserves the order), so the app "worked" — but the "probabilities" it displayed were all squashed towards 10%. Decide where the softmax lives (in PyTorch: in the loss during training, explicitly at inference) and apply it once.

## Optimisers

The INSA lab compared optimisers, batch sizes and epochs on the mixture data of lesson 11:

```python
--8<-- "12_pytorch_training.py:optimisers"
```

```text
SGD, batch 32  : test acc after 3 epochs 0.856, after 30 0.887
SGD, batch 256 : test acc after 3 epochs 0.835, after 30 0.856
SGD + momentum : test acc after 3 epochs 0.851, after 30 0.970
Adam           : test acc after 3 epochs 0.868, after 30 0.965
```

!fig 12-optimisers | Training loss and test accuracy per epoch for the four configurations.

- **Batch size.** With the same learning rate, batches of 256 make 8× fewer updates per epoch than batches of 32: smoother but slower progress per epoch.
- **Momentum** accumulates a velocity, $\mathbf{v} \leftarrow 0.9\,\mathbf{v} - \eta\nabla J$, $\theta \leftarrow \theta + \mathbf{v}$: consistent gradient directions add up, oscillating ones cancel — exactly the cure for the zig-zag in lesson 02's elongated valley.
- **Adam** keeps a running average of each parameter's gradient and of its square, and divides one by the square root of the other: every parameter gets its own step size. It is the robust default; well-tuned SGD + momentum often generalises as well or better on vision tasks.

!!! bug "The experiment that compared nothing"
    My Keras version looped over four configurations — but `model.compile(optimizer="sgd", ...)` was hard-coded inside the loop, so the "Adam" runs were SGD; and an `SGD(learning_rate=0.1, momentum=0.9)` object was created and never passed (the string `"sgd"` gives the default 0.01 without momentum). The written conclusions about Adam were explanations of noise. When an experiment varies a setting, print the setting *from the object that is actually used* (`print(opt)`).

## The bug that costs 50 points of accuracy

The lab's MLP on MNIST: 784 → 800 ReLU → 10, plain SGD. My Keras run reached a loss of 408 in the first epoch and **35%** test accuracy. Here is the same thing in PyTorch, with and without dividing the pixels by 255:

```python
--8<-- "12_pytorch_training.py:mnist"
```

```text
raw pixels 0-255: train loss per epoch [8.736, 1.698, 1.573], test acc 0.5128
pixels / 255    : train loss per epoch [0.799, 0.364, 0.311], test acc 0.9200
```

The lab text said "scale all inputs to [0, 1]"; my code converted to `float32` and forgot the division. With inputs up to 255, the first layer's pre-activations are ~255 times larger than the initialisation expects, the first gradient steps are enormous, many ReLUs die, and the network never recovers. Lesson 02 in its most expensive form: **normalise the inputs**. (My write-up blamed it on "multiple classes to predict".)

## Early stopping and checkpoints

Hold out a validation set, watch its loss after each epoch, and stop when it has not improved for a few epochs (the *patience*) — then roll back to the best weights:

```python
--8<-- "12_pytorch_training.py:early-stopping"
```

```text
early stopping after epoch 10: no improvement for 3 epochs
best validation loss 0.0738 at epoch 7 | reloaded model identical: True | test accuracy 0.9805
```

!fig 12-early-stopping | Training loss keeps falling; validation loss stops improving after epoch 7. Training continues for the patience window, then the epoch-7 weights are restored.

A checkpoint is the model's `state_dict()` — a dictionary of named tensors — saved with `torch.save`. To use it you rebuild the same architecture and `load_state_dict`. The equality test on 1,000 images is the check the lab asked for ("propose a way to verify the reloaded model is the same"): identical outputs, bit for bit. In a real project, save the optimiser's `state_dict` and the config too, so training can resume.

From 92% (3 epochs of SGD) to 98% (Adam, 7 epochs) with the same network. The next lesson gets to 99% with a network that knows it is looking at images.

## Exercises

1. Add a learning-rate schedule (`torch.optim.lr_scheduler.CosineAnnealingLR`) to the SGD runs on the mixture data. Does plain SGD catch up with momentum?
2. Move the model and data to the GPU if you have one (the code already does it through `device`) and time one epoch on each device.
3. Write `fit` so that it also saves the optimiser state, and resume an interrupted training run from the checkpoint. Check that the loss curve continues smoothly.
