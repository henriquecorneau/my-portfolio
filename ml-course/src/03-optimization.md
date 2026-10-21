---
title: Optimisation for machine learning
summary: Every model in this course is trained by minimising something. Compare fixed-step gradient descent, Armijo line search and Newton's method on the Rosenbrock function, check gradients numerically, then solve constrained problems with CVXPY — primal, dual, KKT multipliers — up to the minimum enclosing ball and its soft, outlier-tolerant version.
topics: gradient · Hessian · line search · Newton · Lagrangian · duality · CVXPY
script: 03_optimization.py
sources: INSA IML optimisation labs (unconstrained and constrained)
---

## Why a whole lesson on optimisation

Training a model = solving $\min_\theta J(\theta)$. Lesson 02 did it for a quadratic bowl. Real costs are curved valleys (neural networks), or come with **constraints** (SVMs in lesson 05, probabilities that must sum to one, physical limits). This lesson collects the tools; later lessons just use them.

## A hard test function

Rosenbrock's "banana" function has its minimum $J=0$ at $(1,1)$, at the bottom of a long, narrow, **curved** valley — the kind of geometry that makes simple methods struggle:

$$J(\theta) = (1-\theta_1)^2 + 100\,(\theta_2 - \theta_1^2)^2$$

```python
--8<-- "03_optimization.py:rosenbrock"
```

Hand-written derivatives are a classic source of silent bugs (a wrong sign still "trains", just badly). Always compare them once against **finite differences**, $\partial J/\partial\theta_i \approx \big(J(\theta + h e_i) - J(\theta - h e_i)\big)/2h$:

```python
--8<-- "03_optimization.py:check-grad"
```

```text
analytic gradient [-28.6 -18. ] | numerical [-28.6 -18. ]
analytic Hessian row 0 [430. 280.] | numerical [430. 280.]
```

Deep learning frameworks compute gradients automatically (lesson 12), and `torch.autograd.gradcheck` does exactly this test.

## Three ways down the valley

All descent methods iterate $\theta_{k+1} = \theta_k + \alpha_k\, \mathbf{d}_k$; they differ in the **direction** $\mathbf{d}_k$ and the **step** $\alpha_k$.

- **Gradient descent, fixed step.** $\mathbf{d} = -\nabla J$, $\alpha$ constant. The step must be small enough for the steepest part of the valley, so it is far too small everywhere else.
- **Gradient descent + Armijo backtracking.** Start with $\alpha = 1$ and halve it until the cost decreases "enough": $J(\theta + \alpha\mathbf{d}) \le J(\theta) + c\,\alpha\,\nabla J^\top \mathbf{d}$. The step adapts to the local landscape at the price of a few extra evaluations of $J$.
- **Newton's method.** Approximate $J$ by its second-order Taylor expansion and jump to the minimum of that quadratic: $\mathbf{d} = -H^{-1}\nabla J$, $\alpha = 1$. The Hessian $H$ rescales each direction by its curvature — the cure for the elongated valleys of lesson 02.

```python
--8<-- "03_optimization.py:descent"
```

```text
fixed : 14294 iterations -> theta = [0.9989 0.9978], J = 1.25e-06
armijo:  6029 iterations -> theta = [0.9992 0.9984], J = 6.22e-07
newton:     5 iterations -> theta = [1. 1.], J = 2.07e-17
```

!fig 03-descent | Left: the three paths from θ₀ = (−1, 0). Gradient descent zig-zags into the valley and then crawls along it; Newton takes one wild step and lands on the solution. Right: J against iteration (log-log). Newton's convergence is quadratic — the number of correct digits doubles at each step.

!!! bug "Print the right counter"
    In my lab report Newton's method "also took 2,499 iterations" — the maximum. It didn't: the function returned the path but not its iteration count, and I printed the variable `iter` left over from the gradient-descent loop. Newton needs 5 iterations here. When two methods give suspiciously identical numbers, check which variable you are printing.

If Newton is so fast, why does nobody train neural networks with it? Because $H$ has $p^2$ entries for $p$ parameters: for a network with 10⁷ weights that is 10¹⁴ numbers, and solving with it costs $O(p^3)$. Deep learning uses first-order methods (SGD, Adam — lesson 12) that *approximate* the curvature cheaply, e.g. by normalising each parameter's step by the running size of its gradients.

## Constraints: the Lagrangian and the dual

Now minimise under inequality constraints:

$$\min_\theta f(\theta) \quad\text{s.t.}\quad g_i(\theta) \le 0$$

The **Lagrangian** adds each constraint with a multiplier $\mu_i \ge 0$: $\mathcal{L}(\theta, \mu) = f(\theta) + \sum_i \mu_i g_i(\theta)$. At the optimum of a convex problem the **KKT conditions** hold: the gradient of $\mathcal{L}$ is zero, constraints are satisfied, and **complementary slackness** $\mu_i\, g_i(\theta^\star) = 0$ — a multiplier is non-zero only for constraints that are *active* (tight). Minimising $\mathcal{L}$ over $\theta$ first gives the **dual problem** in $\mu$ alone. For convex problems both have the same optimal value (zero *duality gap*), so we can solve whichever is easier.

### Example: projecting a point onto a polytope

Find the point of the diamond $|t_1| + |t_2| \le 1$ closest to $c = (3, 1)$. Four linear constraints $A\mathbf{t} \le \mathbf{b}$. [CVXPY](https://www.cvxpy.org) lets you write the problem as you would on paper and picks a solver:

```python
--8<-- "03_optimization.py:primal"
```

Setting $\nabla_t \mathcal{L} = (\mathbf{t} - \mathbf{c}) + A^\top\mu = 0$ gives $\mathbf{t} = \mathbf{c} - A^\top \mu$; substituting back yields a quadratic problem in $\mu \ge 0$:

```python
--8<-- "03_optimization.py:dual"
```

```text
primal: t* = [ 1. -0.] objective 2.5
dual: mu* = [1.5 0.5 0.  0. ] -> t = c - A^T mu = [1. 0.]
duality gap: 0.0 | multipliers from the primal solver: [1.5 0.5 0.  0. ]
```

Only the two constraints that meet at the vertex $(1, 0)$ have non-zero multipliers — complementary slackness in action. And CVXPY's primal solver hands us the same multipliers for free (`constraint.dual_value`). The dual is not a curiosity: in lesson 05 it is what turns a linear SVM into a kernel SVM.

## The minimum enclosing ball

Where do we put a fire station so that the farthest of 18 houses is as close as possible? Find the smallest ball (centre $\mathbf{z}$, radius $R$) containing all points: $\min R^2$ s.t. $\|\mathbf{x}_i - \mathbf{z}\|^2 \le R^2$. Its dual is

$$\max_{\mu \ge 0,\ \sum\mu_i = 1} \ \sum_i \mu_i \|\mathbf{x}_i\|^2 - \Big\|\sum_i \mu_i \mathbf{x}_i\Big\|^2, \qquad \mathbf{z} = \sum_i \mu_i \mathbf{x}_i$$

The centre is a weighted average of the **support points** — the few points on the boundary, the only ones with $\mu_i > 0$.

```python
--8<-- "03_optimization.py:meb"
```

```text
houses: centre [0. 0.], radius 1.4142, 4 support points
with 2 outliers: centre [ 1.65 -0.35], radius 2.9741, 3 support points
```

!!! bug "A factor ½ that only shows up with outliers"
    I copied the dual objective from the previous exercise, which had a $\frac12$ in front of the quadratic term: `0.5 * cp.sum_squares(X.T @ mu) - q @ mu`. That is a *different* problem. On the symmetric houses the centre is (0, 0) either way, so the mistake was invisible; with the two outliers it gave a centre at (2.80, −0.75) whose "ball" leaves the house at (−1, 1) 4.18 away, outside a radius of 2.81. Two lessons: derive the dual yourself, and **check the constraints of the answer** (`max distance ≤ R`?) — a one-line test would have caught it.

Two outliers move the centre and double the radius: the hard ball must contain *everything*. The fix is the same idea as the soft-margin SVM: allow points outside, but make them pay. With slack variables $\xi_i \ge 0$,

$$\min_{\mathbf{z}, R, \xi} \ R^2 + C\sum_i \xi_i \quad\text{s.t.}\quad \|\mathbf{x}_i - \mathbf{z}\|^2 \le R^2 + \xi_i$$

```python
--8<-- "03_optimization.py:soft"
```

```text
soft ball (C = 0.3): centre [ 1.2499 -0.    ], radius 2.4621, points left outside: [18 19]
```

!fig 03-constrained | Left: the projection of c onto the diamond lands on a vertex, where two constraints are active. Right: the minimum enclosing ball of the houses (blue), the ball dragged by two outliers (red), and the soft ball with C = 0.3 (green), which leaves the outliers outside.

The soft ball leaves the outliers out, but it is still bigger and off-centre: an outlier outside the ball keeps pulling on the centre, with a force proportional to $C$. Lower $C$ and the ball shrinks further — but starts excluding houses too. $C$ is a **hyper-parameter**: it sets the price of an error, and only validation data can tell you the right price. This model — a soft ball around "normal" data — is the **Support Vector Data Description**, used for anomaly detection.

## Exercises

1. Implement **momentum**: $\mathbf{v} \leftarrow \beta \mathbf{v} - \alpha\nabla J$, $\theta \leftarrow \theta + \mathbf{v}$ with $\beta = 0.9$. How many iterations on Rosenbrock with a fixed $\alpha$?
2. Newton fails when $H$ is not positive definite (try $\theta_0 = (0, 1)$ and print the eigenvalues of $H$). Add *damping*: solve $(H + \lambda I)\mathbf{d} = -\nabla J$ with $\lambda$ large enough.
3. Make the soft ball robust in two steps: fit it, drop the points with $\xi_i > 0$, refit the hard ball on what is left. Do you recover the houses-only ball?
