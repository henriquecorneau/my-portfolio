---
title: Linear regression and gradient descent
summary: Fit a line three ways — the closed-form formula, least squares, and gradient descent written by hand — and understand the learning rate, why feature scaling makes optimisation fast, and how a polynomial with too many degrees of freedom memorises noise.
topics: least squares · MSE · gradient descent · learning rate · conditioning · polynomial features · ridge
script: 02_linear_regression_gd.py
sources: UFRN Deep Learning class 9 and exercise 2
---

## The model

Linear regression predicts a number as a weighted sum of the inputs:

$$\hat{y} = w x + b \qquad\text{or, with } d \text{ features,}\qquad \hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

$w$ is the slope (weight) and $b$ the intercept (bias). "Learning" means choosing them so the predictions are close to the targets. Close in what sense? The usual answer is the **mean squared error**:

$$J(w, b) = \frac{1}{N}\sum_{i=1}^{N} \big(w x_i + b - y_i\big)^2$$

Why squares, and not absolute values? If the targets are the true line plus **Gaussian noise**, $y_i = w x_i + b + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, then the probability of the data is $\prod_i \exp\!\big(-(y_i - w x_i - b)^2 / 2\sigma^2\big)$, and maximising it is exactly minimising $J$. Least squares is the **maximum-likelihood** estimate under Gaussian noise — and it is also differentiable everywhere, which gradient descent will need.

```python
--8<-- "02_linear_regression_gd.py:data"
```

## Solution 1: the closed form

$J$ is a quadratic bowl in $(w, b)$; setting its two partial derivatives to zero gives the textbook formulas — the slope is the covariance of $x$ and $y$ over the variance of $x$:

$$w = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}, \qquad b = \bar{y} - w\,\bar{x}$$

In matrix form, put the inputs in a **design matrix** $A$ with a column of ones for the bias; the solution of $\min_\theta \|A\theta - \mathbf{y}\|^2$ is $\theta = (A^\top A)^{-1} A^\top \mathbf{y}$ (the *normal equations*). Never invert the matrix yourself: `np.linalg.lstsq` solves it stably with an SVD.

```python
--8<-- "02_linear_regression_gd.py:closed"
```

```text
closed form: w = 2.9842, b = 4.1111 | lstsq: [2.9842 4.1111]
```

Close to the true $w=3, b=4$; the difference is the noise.

```python
--8<-- "02_linear_regression_gd.py:rmse"
```

```text
RMSE = 0.4981, 'RMSE' with the misplaced parenthesis = 6.48e-15
```

!!! bug "One parenthesis, zero error"
    The class example computed `np.sqrt((np.sum(y - y_predict)**2) / n)`: the **square of the sum** of the residuals instead of the **sum of the squares**. For a least-squares fit with an intercept the residuals sum to exactly zero (that is the $\partial J / \partial b = 0$ condition), so this "error" is always ≈ 0, whatever the data. When a metric looks too good, test it on a case where you know the answer.

## Solution 2: gradient descent

The closed form needs all the data in memory and a matrix factorisation. Neural networks have millions of parameters and no closed form at all; what they use is **gradient descent**: start somewhere, compute the slope of the cost, take a small step downhill, repeat.

$$\theta \leftarrow \theta - \eta \, \nabla J(\theta)$$

$\eta$ is the **learning rate**. For our $J$ the gradient is

$$\frac{\partial J}{\partial w} = \frac{2}{N}\sum_i (\hat{y}_i - y_i)\, x_i, \qquad \frac{\partial J}{\partial b} = \frac{2}{N}\sum_i (\hat{y}_i - y_i)$$

```python
--8<-- "02_linear_regression_gd.py:gd"
```

```text
gradient descent: w = 2.9842, b = 4.1111, final cost 0.2481
```

Exactly the closed-form answer.

!!! bug "Implement it, don't import it"
    The exercise asked for this function. My submission called `sklearn.linear_model.SGDRegressor` inside it — which works, returns the right line, and teaches nothing about *how* the parameters move. The eight lines above are the core of every training loop in Part IV; it is worth writing them once by hand.

!fig 02-frames | The line after 1, 6, 31 and 500 epochs of gradient descent. The cost J drops quickly at first, then slowly.

### The learning rate

```python
--8<-- "02_linear_regression_gd.py:lr"
```

```text
Hessian eigenvalues [0.322 4.131] -> gradient descent diverges for lr > 2/4.13 = 0.484
lr = 0.01: cost after 100 epochs = 0.3619
lr = 0.1 : cost after 100 epochs = 0.2484
lr = 0.45: cost after 100 epochs = 0.2481
lr = 0.5 : cost after 100 epochs = 1.494e+07
```

Too small and training crawls; too large and every step overshoots the minimum by more than the previous one — the cost explodes. For a quadratic cost the limit is exact: gradient descent converges if and only if $\eta < 2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of the **Hessian** (the matrix of second derivatives, here $\frac{2}{N}A^\top A$). Neural networks are not quadratic, but the same picture holds locally — which is why a diverging loss is the first sign of a learning rate that is too high.

!fig 02-learning-rate | Left: cost per epoch for four learning rates (log scale). Right: the contours of J(w, b) and the path of gradient descent with lr = 0.1 — first straight down the steep direction, then slowly along the flat valley towards the optimum (star).

The right plot shows the other problem: the contours are **elongated ellipses**. The ratio of the Hessian's eigenvalues (here $4.13 / 0.32 \approx 13$, the **condition number**) says how much steeper one direction is than the other. The learning rate is limited by the steepest direction, so progress along the flattest one is slow.

## Why scaling makes optimisation fast

With several features of very different scales, the condition number explodes. California housing (20,640 districts, 8 features, target = median house value) has features with standard deviations from 0.5 to 1,100:

```python
--8<-- "02_linear_regression_gd.py:housing"
```

```text
optimal training MSE 0.5192 | GD raw features: 2.1176 | GD standardised: 0.5193
feature std range: 0.49 .. 1.1e+03
```

!fig 02-scaling | On raw features the largest stable learning rate is tiny (1e-7: the population feature dominates the Hessian) and after 300 epochs gradient descent is still far from the optimum. On standardised features it is almost there after a hundred epochs.

Same model, same data, same algorithm — standardising the inputs turns the long thin valley into a round bowl. In lesson 01 scaling changed the *answer* of k-NN; here it changes only the *speed* of finding the answer, but by orders of magnitude. Neural networks are trained by gradient descent, so they inherit this: normalise your inputs (lesson 12 forgets to, on purpose).

## Overfitting, again: polynomial regression

A linear model is linear in its **parameters**, not necessarily in $x$: with features $(1, x, x^2, \dots, x^p)$ it fits a polynomial, still by least squares. Fit 15 noisy samples of a sine wave with growing degree:

```python
--8<-- "02_linear_regression_gd.py:poly"
```

```text
degree  1, ridge alpha 0: train MSE 0.0930, validation MSE 0.3091
degree  3, ridge alpha 0: train MSE 0.0203, validation MSE 0.0755
degree 14, ridge alpha 0: train MSE 0.0022, validation MSE 263534.3869
degree 14, ridge alpha 0.001: train MSE 0.0338, validation MSE 0.0590
```

!fig 02-polynomial | Degree 1 underfits, degree 3 is about right, degree 14 passes through every point and oscillates wildly between them. Ridge regularisation tames the same degree-14 polynomial.

The degree-14 polynomial has 15 coefficients for 15 points: it interpolates the noise. **Ridge regression** adds a penalty $\alpha\|\mathbf{w}\|^2$ to the cost, which forbids huge coefficients; the model keeps its flexibility but has to pay for using it. This trade-off — **capacity versus data** — is the same as $k$ in k-NN, and it will reappear as $C$ in SVMs, depth in trees and weight decay in neural networks.

## Exercises

1. Implement **stochastic** gradient descent: at each step use one random sample (or a mini-batch of 10) instead of the whole set. Plot the cost: why is it noisy, and why does it still converge?
2. Make the animation the exercise asked for: save one PNG per epoch of the first figure and assemble them into a GIF with `imageio`.
3. On California housing, fit ridge regression with polynomial features of degree 2 and choose $\alpha$ on a validation set. How much does the test MSE improve over the plain linear model?
