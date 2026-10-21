---
title: Support vector machines
summary: The classifier with the widest margin. Write the SVM as the optimisation problem it is and solve it with CVXPY, see what C really does, derive the dual, and use it to draw curved boundaries with the kernel trick — an RBF SVM written from scratch that agrees with scikit-learn on every point.
topics: margin · hinge loss · soft margin · support vectors · dual · kernel trick · RBF · grid search
script: 05_svm.py
sources: INSA IML labs 3–4 (linear and kernel SVM)
---

## Many lines separate the data. Which one?

Logistic regression picks the line that maximises the likelihood. The **support vector machine** picks the line that stays **as far as possible from both classes** — the one with the widest *margin*. The intuition: a boundary that passes close to training points will misclassify slightly different test points; a wide street is robust.

With labels $y_i \in \{-1, +1\}$ and a hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$, scale $\mathbf{w}$ so that the closest points satisfy $y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$. The street between $\mathbf{w}^\top\mathbf{x} + b = \pm 1$ has width $2/\|\mathbf{w}\|$, so maximising the margin is minimising $\|\mathbf{w}\|$:

$$\min_{\mathbf{w}, b} \ \tfrac12\|\mathbf{w}\|^2 \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 \ \ \forall i$$

Real data overlap, so we let points violate the margin by $\xi_i \ge 0$ and pay $C$ per unit of violation — the **soft margin** (same idea as the soft ball of lesson 03):

$$\min_{\mathbf{w}, b, \xi} \ \tfrac12\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0$$

Eliminating $\xi$ shows that this is $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max(0, 1 - y_i f(\mathbf{x}_i))$: L2 regularisation plus the **hinge loss**, which is zero for points beyond the margin. That is the key difference with logistic regression — points that are correctly classified *with margin* have no influence at all.

## The primal, solved directly

It is a convex quadratic program, so CVXPY solves it as written:

```python
--8<-- "05_svm.py:primal"
```

```text
CVXPY primal : w = [0.3769 1.6538], b = 0.5263, margin width 2/|w| = 1.179
scikit-learn : w = [0.3768 1.6533], b = 0.5263
support vectors: 17 with CVXPY (margin <= 1), 17 in scikit-learn
```

The **support vectors** are the points on or inside the margin ($y_i f(\mathbf{x}_i) \le 1$). Move any other point (without crossing the margin) and the solution does not change.

## What C does

```python
--8<-- "05_svm.py:C"
```

```text
C = 0.001 : margin width 6.68, 160 support vectors, training error 0.025
C = 0.01  : margin width 3.35,  65 support vectors, training error 0.025
C = 1     : margin width 1.18,  17 support vectors, training error 0.020
C = 100   : margin width 0.89,  13 support vectors, training error 0.020
```

!fig 05-margin | Linear SVMs with C = 0.01, 1 and 100. Solid line: decision boundary; dashed: the margin; circles: support vectors. Small C buys a wide margin at the price of many violations.

!!! bug "Small C means a *wide* margin"
    My lab report says "when C decreases, the margin also decreases". It is the opposite, and the formula says why: $C$ is the price of a violation. Cheap violations (small $C$) let the optimiser shrink $\|\mathbf{w}\|$ freely, so the street gets wider and more points end up inside it. Large $C$ makes violations expensive: a narrow margin that bends to every training point — more variance, risk of overfitting. $C$ plays the role of $1/\lambda$ in ridge regression.

## The dual and the kernel trick

Apply lesson 03's recipe — Lagrangian, set the derivatives with respect to $\mathbf{w}$, $b$, $\xi$ to zero — and the primal becomes:

$$\max_\alpha \ \sum_i \alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\, \mathbf{x}_i^\top\mathbf{x}_j \quad\text{s.t.}\quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0$$

with $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$, so $f(\mathbf{x}) = \sum_i \alpha_i y_i\, \mathbf{x}_i^\top\mathbf{x} + b$. Points with $\alpha_i > 0$ are the support vectors.

Look at where the data appear: **only through dot products** $\mathbf{x}_i^\top\mathbf{x}_j$. Now imagine mapping each point to a richer feature space $\phi(\mathbf{x})$ — all monomials up to degree 3, say — where the classes become linearly separable. We never need $\phi$ itself, only $\phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$, and for many maps that dot product has a closed form, a **kernel** $k(\mathbf{x}_i, \mathbf{x}_j)$:

- polynomial: $k(\mathbf{a}, \mathbf{b}) = (1 + \mathbf{a}^\top\mathbf{b})^p$ — all monomials up to degree $p$;
- Gaussian (RBF): $k(\mathbf{a}, \mathbf{b}) = \exp(-\gamma\|\mathbf{a} - \mathbf{b}\|^2)$ — an *infinite-dimensional* feature space, yet one exponential to compute.

That is the **kernel trick**: replace every dot product by $k$, and a linear algorithm draws non-linear boundaries. Here is the dual, written for CVXPY, on the two-moons dataset:

```python
--8<-- "05_svm.py:dual"
```

```text
dual from scratch: 51 support vectors (scikit-learn 51), same predicted class on 100.0% of the plane
```

(The Cholesky factor $L$ is there only to write $\alpha^\top Q\alpha$ as a sum of squares, which CVXPY recognises as convex without checking a 200×200 matrix.) The bias comes from the support vectors strictly inside $(0, C)$, which lie exactly on the margin. scikit-learn's `SVC` uses a specialised solver (SMO, in libsvm), much faster than a generic QP, but it solves this same problem.

## The two hyper-parameters of the RBF kernel

$\gamma$ sets the **reach** of each training point: $k$ decays with distance at rate $\gamma$. Small $\gamma$ — every point influences far away, the boundary is smooth, almost linear. Large $\gamma$ — every support vector only affects its immediate neighbourhood, and the boundary wraps around individual points.

```python
--8<-- "05_svm.py:gamma"
```

```text
gamma = 0.01 : train error 0.137, validation error 0.153
gamma = 0.1  : train error 0.113, validation error 0.133
gamma = 1    : train error 0.067, validation error 0.063
gamma = 10   : train error 0.037, validation error 0.080
gamma = 100  : train error 0.007, validation error 0.117
```

!fig 05-gamma | RBF SVMs (C = 10) with increasing γ. From an almost linear boundary to islands around individual points.

$C$ and $\gamma$ interact, so we search them **together** on a grid and choose on the validation set:

```python
--8<-- "05_svm.py:grid"
```

```text
best on validation: C = 100, gamma = 1 -> validation error 0.060, test error 0.063
polynomial kernel: C = 10, degree 7 -> validation error 0.060, test error 0.083
```

!fig 05-grid | Validation error over the grid of hyper-parameters. Note the diagonal valley for the RBF kernel: a larger γ can be compensated by a smaller C.

Both kernels reach 6% on validation, but on the test set the RBF model holds up better. With validation errors this close (0.060 vs 0.060, on 300 samples), the choice of *kernel* is within noise; lesson 06 shows how to make such comparisons less noisy.

!!! bug "Never tune on the test set"
    In the spam part of my kernel-SVM lab I filled the grid with **test** errors, picked the best $(C, \gamma)$ and reported that same test error. The number is optimistic by construction: among 30 configurations, the winner is partly the luckiest on those particular test e-mails. In the same notebook, the "train error" printed for the polynomial SVM was a stale variable from the RBF model. The cure for both is lesson 06's discipline: a pipeline where selection only ever sees validation folds.

## SVMs in practice

- **Scale the features** — the RBF kernel uses distances, so lesson 01's warning applies in full.
- Training cost grows between $O(N^2)$ and $O(N^3)$: SVMs shine up to ~10⁴–10⁵ samples. Beyond that, linear SVMs (`LinearSVC`) or neural networks.
- **Multi-class**: `SVC` trains one SVM per *pair* of classes (one-vs-one) and votes; on MNIST digits 0/2/8 my linear SVM reached 97.5%.
- No probabilities by default (the output is a distance to the boundary); `probability=True` adds a calibration step.

## Exercises

1. Implement the linear SVM by **subgradient descent** on $\frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac1N\sum_i \max(0, 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b))$ (the Pegasos algorithm). Compare with the CVXPY solution.
2. Show numerically that the polynomial kernel with $p=2$ in 2-D equals the dot product of the explicit features $\phi(\mathbf{x}) = (1, \sqrt2 x_1, \sqrt2 x_2, x_1^2, x_2^2, \sqrt2 x_1x_2)$.
3. Train an RBF SVM on the spam data of lesson 04, choosing $(C, \gamma)$ by 5-fold cross-validation on the training set only (`GridSearchCV`). Compare with the logistic regression.
