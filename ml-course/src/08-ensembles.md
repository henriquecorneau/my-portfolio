---
title: Ensembles — bagging, forests and boosting
summary: Many weak or unstable models, combined, beat one strong model. Build bagging and AdaBoost from scratch, tune a random forest's feature sampling, read feature importances, write gradient boosting as "fit the residuals", and benchmark everything on 100,000 forest-cover samples — including a gradient-boosting run that falls apart without regularisation.
topics: bootstrap · bagging · random forest · max_features · feature importance · AdaBoost · gradient boosting · learning rate
script: 08_ensembles.py
sources: INSA IML labs on random forests (TP2) and boosting (TP3)
---

## Two ways to combine models

Lesson 07 ended with a problem: deep trees have low bias and high variance. Ensembles attack the two sides of the bias–variance trade-off differently:

- **Bagging** and **random forests** average many *independent*, low-bias, high-variance models. Averaging keeps the bias and cuts the variance.
- **Boosting** adds many *simple*, high-bias models one after the other, each one correcting the mistakes of the ensemble so far. It drives the bias down.

## Bagging from scratch

If we had many independent training sets, we could train one tree on each and average: the noise of the individual trees would cancel. We have only one set, so we fake the others with the **bootstrap**: draw $N$ samples *with replacement* from the $N$ training samples. Each bootstrap sample contains about $1 - 1/e \approx 63\%$ of the distinct original samples, some of them several times.

```python
--8<-- "08_ensembles.py:bagging"
```

```text
one deep tree : held-out error 0.143
bagging,   1 trees: held-out error 0.143
bagging,  10 trees: held-out error 0.147
bagging,  50 trees: held-out error 0.120
bagging, 200 trees: held-out error 0.127
unique samples in a bootstrap sample: 63.2% (theory: 1 - 1/e = 63.2%)
```

!fig 08-bagging | Left: one fully grown tree on noisy moons. Middle: 200 bagged trees, coloured by the share of trees voting "red" — the boundary is smoother and uncertain regions appear. Right: held-out error against the number of trees.

The error drops and then plateaus: once there are enough trees, adding more doesn't hurt (no overfitting from the number of trees) but doesn't help either. The ~37% of samples a tree never saw (*out-of-bag*) give a free validation estimate (`oob_score=True` in scikit-learn).

## Random forests: decorrelate the trees

Averaging $B$ models of variance $\sigma^2$ with pairwise correlation $\rho$ gives variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. More trees kill the second term, but not the first: bagged trees are correlated, because they all start with the same strong features. A **random forest** adds a second source of randomness — at each split, only a random subset of `max_features` features is considered — so the trees differ more and the average is better.

```python
--8<-- "08_ensembles.py:max-features"
```

```text
max_features =  1: accuracy 0.9639 ± 0.0098
max_features =  2: accuracy 0.9661 ± 0.0094
max_features =  4: accuracy 0.9700 ± 0.0121
max_features =  8: accuracy 0.9733 ± 0.0084
max_features = 16: accuracy 0.9694 ± 0.0091
max_features = 32: accuracy 0.9644 ± 0.0116
max_features = 64: accuracy 0.9456 ± 0.0109
```

On the 8×8 digits, 64 features = plain bagging (94.6%). Considering only 8 per split — $\sqrt{64}$, scikit-learn's default for classification — gives 97.3%. Too few and each split is close to random; too many and the trees are all alike. Note the error bars: over 5 random splits, differences under ~1 point are noise.

### Feature importance

Each split reduces the impurity; summing those reductions per feature over all trees gives the **impurity-based importance**.

```python
--8<-- "08_ensembles.py:importance"
```

```text
top 6 features: [1, 0, 2, 4, 3, 6] | importance of the 5 informative: 0.78, of the 15 noise: 0.22
```

!fig 08-forest | Left: accuracy against max_features (mean ± std over 5 splits). Right: importances on a synthetic problem where only features 0–4 carry information (red), with the spread across individual trees.

The five informative features come out on top. But the 15 pure-noise features still collect 22% of the importance: deep trees split on noise too, and impurity importance favours features with many distinct values. **Permutation importance** (shuffle one feature on held-out data and measure the drop in accuracy, `sklearn.inspection.permutation_importance`) is the more reliable measure.

## Boosting

### AdaBoost

Train a **stump** (a depth-1 tree). Increase the weight of the samples it got wrong, train the next stump on the re-weighted data, and so on. Each stump gets a vote $\alpha_t = \frac12\log\frac{1 - \varepsilon_t}{\varepsilon_t}$ that grows as its weighted error $\varepsilon_t$ shrinks; the final classifier is the sign of the weighted sum.

```python
--8<-- "08_ensembles.py:adaboost"
```

```text
AdaBoost,   1 stumps: train error 0.190, held-out error 0.213
AdaBoost,  10 stumps: train error 0.077, held-out error 0.133
AdaBoost, 100 stumps: train error 0.050, held-out error 0.123
AdaBoost, 400 stumps: train error 0.000, held-out error 0.133
```

!fig 08-adaboost | AdaBoost with 1, 10 and 100 stumps. Each stump is a single horizontal or vertical cut; their weighted sum draws a staircase that follows the moons.

### Gradient boosting

AdaBoost turns out to be a special case of a more general idea. Build the model as a sum $F(x) = \sum_t \eta\, h_t(x)$, and at each step fit the new tree $h_t$ to the **negative gradient of the loss** with respect to the current predictions — gradient descent (lesson 02), but in the space of functions. For the squared loss the negative gradient is simply the **residual** $y - F(x)$:

```python
--8<-- "08_ensembles.py:gboost"
```

```text
  1 trees, lr 1.0: training MSE 0.133
 10 trees, lr 0.3: training MSE 0.077
100 trees, lr 0.1: training MSE 0.057
```

!fig 08-gboost | Gradient boosting with depth-2 regression trees on a noisy sine. Each tree corrects the residuals of the previous ones; the learning rate shrinks each correction.

The **learning rate** $\eta$ (*shrinkage*) and the number of trees trade off against each other: a small $\eta$ needs more trees but generalises better. With early stopping on a validation fraction, scikit-learn finds the number of trees automatically:

```python
--8<-- "08_ensembles.py:lr"
```

```text
lr = 0.1  :  485 trees, best validation R² 0.843
lr = 0.01 : 1000 trees, best validation R² 0.826
lr = 0.001: 1000 trees, best validation R² 0.588
```

!fig 08-learning-rate | Training (dashed) and validation (solid) R² on California housing. With lr = 0.001, 1,000 trees are nowhere near enough.

!!! bug "Judge the learning rate on validation data"
    My lab answer said lr = 0.1 "is too fast and overfits" and 0.01 is optimal. The first loop of that notebook scored each model **on its own training data** (`hgb.score(X, y)` after `hgb.fit(X, y)`), and the conclusion was read off curves without comparing their best validation values. Measured on validation data, 0.1 reaches the best R² (0.843) and early stopping prevents it from overfitting; 0.01 is slightly worse because 1,000 trees are not enough for it. With a larger budget, smaller learning rates usually win by a little — at a large cost in time.

## The benchmark: forest cover types

581,012 patches of forest in Colorado, 54 cartographic features (elevation, slope, soil type…), 7 cover types. We train on 100,000 and test on 50,000:

```python
--8<-- "08_ensembles.py:covtype"
```

```text
AdaBoost (100 trees, depth 5)      test accuracy 0.623  (train  40.4 s, predict  0.7 s)
random forest (100, depth 5)       test accuracy 0.676  (train   2.1 s, predict  0.1 s)
random forest (100, full depth)    test accuracy 0.914  (train   5.7 s, predict  0.5 s)
hist. GB (100, depth 5)            test accuracy 0.803  (train   6.4 s, predict  0.6 s)
hist. GB (300, 63 leaves, l2 = 0)  test accuracy 0.767  (train  32.3 s, predict  3.5 s)
hist. GB (300, 63 leaves, l2 = 1)  test accuracy 0.924  (train  84.6 s, predict  2.7 s)
```

Three lessons in one table:

1. **Random forests want deep trees.** The lab compared models with the same `max_depth=5` for everybody and concluded that gradient boosting was far ahead (80% vs 68%). That handicaps the forest: its whole design is to average *low-bias* trees. Fully grown, it reaches 91.4% in 6 seconds.
2. **Boosting wants shallow trees, many of them.** And it beats the forest when given enough capacity — 92.4%.
3. **Boosting can blow up.** The same 300 bigger trees *without* L2 regularisation drop to 76.7%, below the small model. Each leaf value in gradient boosting is a Newton step, $-\sum g_i / (\sum h_i + \lambda)$; in tiny leaves of rare classes the curvature $\sum h_i$ is almost zero, and the step is enormous. I checked: without regularisation the largest leaf value is 182,760 (9.2 with $\lambda = 1$), the test accuracy peaks at 86% after 50 trees and then *falls*, and even the training accuracy ends at 80% — this is not overfitting, the optimisation itself is breaking down. `l2_regularization` adds $\lambda$ to the denominator and fixes it. If a boosted model gets *worse* with more capacity, look for this before anything else.

For tabular data like this, gradient-boosted trees (scikit-learn's `HistGradientBoosting`, XGBoost, LightGBM, CatBoost) are still the method to beat — neural networks usually aren't.

## Exercises

1. Use the out-of-bag score of a `RandomForestClassifier` to choose `max_features` on the digits without any validation split. Does it agree with the curve above?
2. Compute the permutation importance of the 20 synthetic features on a held-out set. How much importance do the noise features get now?
3. Extend `gradient_boosting` to binary classification with the logistic loss: the trees fit $y - \sigma(F(x))$. Compare with `HistGradientBoostingClassifier` on the moons.
