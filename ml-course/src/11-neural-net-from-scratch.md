---
title: A neural network from scratch
summary: Open the black box before using a framework. Activation functions and their derivatives, why sigmoids make deep networks forget how to learn, a two-layer network written in NumPy with backpropagation derived line by line, a gradient check that proves it right, and what the hidden units actually draw.
topics: neuron · activation functions · vanishing gradients · forward pass · backpropagation · chain rule · gradient check · mini-batch SGD
script: 11_neural_net_from_scratch.py
sources: UFRN Deep Learning classes 6–7, INSA IML deep learning lab
---

## From logistic regression to a network

Logistic regression (lesson 04) is a single **neuron**: a weighted sum of the inputs followed by a non-linear function. Its boundary is a straight line. A **multilayer perceptron** (MLP) stacks neurons: a *hidden layer* of $H$ neurons each computes its own feature of the input, and an output neuron does logistic regression **on those features**:

$$\mathbf{h} = f(W_1^\top\mathbf{x} + \mathbf{b}_1), \qquad p = \sigma(W_2^\top\mathbf{h} + b_2)$$

The network learns the features *and* the classifier together. The function $f$ — the **activation** — is essential: without it, two linear layers collapse into one linear layer, and nothing is gained.

## Activation functions and their derivatives

Training uses the derivative of the activation (we'll see exactly where in a moment), so its shape matters as much as the function's:

```python
--8<-- "11_neural_net_from_scratch.py:activations"
```

```text
sigmoid    max derivative 0.25 | f'(z) at z = -5, 0, 5: [0.007 0.25  0.007]
tanh       max derivative 1.00 | f'(z) at z = -5, 0, 5: [0. 1. 0.]
ReLU       max derivative 1.00 | f'(z) at z = -5, 0, 5: [0. 1. 1.]
leaky ReLU max derivative 1.00 | f'(z) at z = -5, 0, 5: [0.01 1.   1.  ]
```

!fig 11-activations | Four activations (left) and their derivatives (right). Sigmoid and tanh saturate: for large |z| the derivative is almost zero. ReLU's derivative is exactly 1 for every positive input.

- **Sigmoid** — squashes to (0, 1). Its derivative is at most 0.25 and vanishes when the neuron saturates. Still the right choice for a binary *output*.
- **tanh** — a centred sigmoid with range (−1, 1); derivative up to 1, but it saturates too.
- **ReLU**, $\max(0, z)$ — cheap, does not saturate for positive inputs. A neuron whose input is always negative, though, gets zero gradient forever (a "dead" ReLU).
- **Leaky ReLU** — keeps a small slope (0.01) for negative inputs so no neuron dies completely. GELU and Swish are smooth variants used in modern architectures.

### Why depth needs non-saturating activations

The gradient that reaches the first layer is a **product** of one derivative per layer (the chain rule). With sigmoids each factor is at most 0.25:

```python
--8<-- "11_neural_net_from_scratch.py:vanishing"
```

```text
 2 sigmoid layers: gradient scaled by at most 0.25^2 = 6.2e-02
 5 sigmoid layers: gradient scaled by at most 0.25^5 = 9.8e-04
10 sigmoid layers: gradient scaled by at most 0.25^10 = 9.5e-07
20 sigmoid layers: gradient scaled by at most 0.25^20 = 9.1e-13
```

This is the **vanishing gradient** problem: the first layers of a deep sigmoid network barely learn. The opposite — factors above 1 compounding — makes gradients **explode**. ReLU (derivative exactly 1 where active), careful initialisation, normalisation layers and residual connections (lesson 15) are the cures that made deep learning possible.

## The data

The lab's 2-D *mixture* dataset: two classes from Gaussian mixtures that no straight line can separate.

```python
--8<-- "11_neural_net_from_scratch.py:data"
```

```text
train (1759, 2) test (5272, 2) | positives: 53.4%
```

## The network, forward and backward

**Forward pass**: compute the prediction, keeping the intermediate values. **Backward pass**: apply the chain rule from the loss back to every parameter, reusing those values. This is all **backpropagation** is — the chain rule, organised so that each intermediate gradient is computed once.

```python
--8<-- "11_neural_net_from_scratch.py:mlp"
```

Read `backward` from top to bottom, following the shapes:

1. With a sigmoid output and cross-entropy loss, $\partial L/\partial z_2 = p - y$ — the same "prediction minus label" as logistic regression (divided by $n$ because the loss is a mean).
2. $z_2 = \mathbf{h}W_2 + b_2$ is linear: $\partial L/\partial W_2 = \mathbf{h}^\top\,\partial L/\partial z_2$, and the gradient flowing back into the hidden layer is $\partial L/\partial z_2\, W_2^\top$.
3. Through ReLU the gradient passes unchanged where $z_1 > 0$ and is **blocked** where $z_1 \le 0$ — multiplication by $f'(z_1)$, here a 0/1 mask.
4. $z_1 = XW_1 + b_1$ is linear again: $\partial L/\partial W_1 = X^\top\,\partial L/\partial z_1$.

Every layer of every network follows this pattern: *gradient with respect to my output in, gradient with respect to my parameters and my input out*.

```text
parameters: 257 (2x64 + 64 + 64x1 + 1)
```

### Prove it: the gradient check

A backpropagation bug rarely crashes: the network just trains badly. Compare a few analytic gradients with finite differences, as in lesson 03:

```python
--8<-- "11_neural_net_from_scratch.py:gradcheck"
```

```text
gradient check: largest relative difference 1.1e-07
```

Relative differences around 1e-7 mean the derivation is right; a bug typically gives 1e-1 or more.

## Training with mini-batch SGD

Instead of the full gradient (lesson 02), take a step after each **mini-batch** of 32 samples, reshuffling every **epoch** (one pass over the data). Each step is noisier but 55 times cheaper, and the noise even helps to escape poor regions.

```python
--8<-- "11_neural_net_from_scratch.py:train"
```

```text
hidden units  1: train acc 0.848, test acc 0.842, test loss 0.362
hidden units  4: train acc 0.860, test acc 0.852, test loss 0.245
hidden units 64: train acc 0.949, test acc 0.943, test loss 0.160
```

!fig 11-mlp | Predicted probability of the red class with 1, 4 and 64 hidden ReLU units, and the loss curves of the 64-unit network.

The pictures show what hidden units *are*. With one unit the boundary is a line — logistic regression. Each ReLU unit contributes one "fold" along a line; with 4 units the boundary is a polygon of a few straight pieces; with 64 it can bend around the red tongue. A network with one hidden layer can approximate any continuous function given enough units (the *universal approximation theorem*); deeper networks do it with far fewer units.

The loss curve is jumpy, and still going down after 200 epochs: SGD with a constant learning rate keeps bouncing around the minimum. Learning-rate schedules and adaptive optimisers (lesson 12) smooth this out.

## Exercises

1. Add a second hidden layer (2 → 32 → 32 → 1). Write its `backward` and run the gradient check before training.
2. Replace ReLU by sigmoid in the hidden layer (and its derivative in `backward`). How many epochs does it need to reach the same test accuracy?
3. Initialise all weights to zero. What happens to the 64 hidden units, and why can they never become different?
