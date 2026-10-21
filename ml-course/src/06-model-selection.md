---
title: Model selection and evaluation
summary: How to choose hyper-parameters without fooling yourself, and how to report a number someone can trust. Cross-validation from scratch, pipelines that make leakage impossible, grid search on a DNA splice-site problem, the confusion matrix, precision, recall, ROC, and learning curves to tell bias from variance.
topics: k-fold · stratification · Pipeline · GridSearchCV · confusion matrix · precision/recall · ROC AUC · learning curves
script: 06_model_selection.py
sources: INSA IML model-selection lab and lecture, A. Ng's "Advice for applying ML"
---

## The problem with a single validation split

In lessons 01 and 05 we chose hyper-parameters on one validation set. With 1,000 samples, a validation set of a few hundred gives accuracies with a standard error of 2–3 percentage points — as large as the differences we were trying to detect. And we threw away a third of the data for training.

**k-fold cross-validation** fixes both: cut the training data into $k$ folds, train $k$ times, each time validating on a different fold, and average. Every sample is used for validation exactly once and for training $k-1$ times.

The data for this lesson is the *splice* dataset (LIBSVM): 60 DNA nucleotides around a position, and whether that position is an exon/intron boundary — a real bioinformatics task.

```python
--8<-- "06_model_selection.py:data"
```

```text
train (1000, 60) test (2175, 60) | class balance: 51.7% / 52.0% positive
```

Balanced classes, so accuracy is a meaningful metric here (more on that below).

## Cross-validation from scratch

```python
--8<-- "06_model_selection.py:kfold"
```

```text
5-fold accuracy from scratch: [0.815 0.87  0.89  0.875 0.875] -> 0.865 ± 0.026
cross_val_score (stratified):  0.851 ± 0.021
```

Look at the spread across folds: from 81.5% to 89%, for the *same* model. A single validation split could have told you anything in that range. (The two means differ because the folds differ — by about one standard deviation, which is exactly the point.)

!!! tip "The pipeline is the model"
    `make_pipeline(StandardScaler(), SVC(...))` glues preprocessing and classifier into one estimator. When cross-validation calls `fit` on a training fold, the scaler is fitted on that fold only, and the validation fold is transformed with its statistics. Writing the scaling *outside* the loop — `X = scaler.fit_transform(X)` before cross-validating — leaks the validation folds' mean and variance into training (lesson 01's warning, in a form that is easy to miss). With a pipeline the mistake cannot be made.

## Grid search

`GridSearchCV` runs cross-validation for every combination of hyper-parameters and refits the best one on the whole training set:

```python
--8<-- "06_model_selection.py:grid"
```

```text
best: {'svc__C': 5.6234, 'svc__gamma': 0.01} cross-validated accuracy 0.879
test accuracy of the refitted best model: 0.900
linear SVM: C = 0.01, CV accuracy 0.801, test accuracy 0.847
```

!fig 06-grid | Cross-validated accuracy over the (γ, C) grid, both axes logarithmic. The red star is the selected model. To the right of γ ≈ 0.05 the RBF kernel becomes so narrow that the model memorises the training folds.

The RBF kernel beats the linear one by 5 points on the test set, and the cross-validation ranking predicted it.

!!! bug "Plotting a log grid with linear axes (and upside down)"
    My lab version drew the scores with `plt.imshow(scores.reshape(25, 25), extent=[min(gamma), max(gamma), min(C), max(C)])`. Three problems hide in that line: the grid is logarithmic but the extent is linear, so 24 of the 25 γ columns are squeezed near zero; `imshow` puts row 0 at the **top** while the extent labels the top with the **largest** C, so the C axis is flipped; and the reshape order (rows = C, columns = γ, as `ParameterGrid` iterates) only works because both grids had 25 values. `pcolormesh(gammas, Cs, grid)` with log axes uses the real coordinates and avoids all three.

## Beyond accuracy

Accuracy hides *which* errors a model makes. The **confusion matrix** counts them:

```python
--8<-- "06_model_selection.py:metrics"
```

```text
confusion matrix [[TN FP] [FN TP]] = [[958, 86], [131, 1000]]
accuracy 0.900 | precision 0.921 | recall 0.884 | F1 0.902
ROC AUC 0.963
```

- **Precision** $= TP/(TP+FP)$: when the model says "boundary", how often is it right?
- **Recall** (sensitivity) $= TP/(TP+FN)$: of all true boundaries, how many did it find?
- **F1**: their harmonic mean — high only if both are.

They matter when classes are imbalanced or errors have different costs. A cancer screening wants high recall (a missed tumour is worse than a false alarm); a spam filter wants high precision (a lost legitimate e-mail is worse than a spam in the inbox). On a dataset with 1% positives, "always negative" has 99% accuracy and 0% recall.

Most classifiers output a **score**, and the label is "score > threshold". Moving the threshold trades precision against recall. The **ROC curve** plots the true positive rate against the false positive rate for every threshold; the area under it (**AUC**) is the probability that a random positive gets a higher score than a random negative — a threshold-free summary.

!fig 06-metrics | Left: the confusion matrix of the selected model on the 2,175 test sequences. Right: ROC curves; the dot is the default threshold (score 0).

## Learning curves: bias or variance?

When a model is not good enough, should you get more data, or a more flexible model? Train on growing subsets and plot training and validation error (Andrew Ng's classic advice):

```python
--8<-- "06_model_selection.py:learning"
```

```text
linear SVM (high bias)           n = 800: train 0.842, validation 0.801
RBF SVM, γ = 1 (high variance)   n = 800: train 1.000, validation 0.558
```

!fig 06-learning-curves | Left: the linear SVM — both errors are high and close; more data will not help much, the model is too simple (high bias). Right: an RBF SVM with a much too large γ — zero training error and near-chance validation error, with a gap that closes only slowly (high variance).

| Symptom | Diagnosis | What helps |
|---|---|---|
| training error high, validation ≈ training | high bias (underfitting) | more flexible model, more/better features, less regularisation |
| training error low, validation much higher | high variance (overfitting) | more data, regularisation, simpler model, fewer features |

## The protocol, in one list

1. Split off a **test set** first and lock it away.
2. Put every preprocessing step in a **pipeline** with the model.
3. Choose hyper-parameters (and models) by **cross-validation** on the training set.
4. Refit the chosen pipeline on the whole training set.
5. Evaluate **once** on the test set. Report the metric that matches the cost of errors, with the confusion matrix.

If the dataset is too small for a test set, use **nested cross-validation** (an outer loop for evaluation around the inner grid search) — `cross_val_score(GridSearchCV(...), X, y)`.

## Exercises

1. Implement stratification in `kfold_indices`: split the indices of each class separately and merge the folds. Does the spread across folds shrink?
2. Run nested cross-validation on the 1,000 training sequences for the RBF pipeline. How does its estimate compare with the 0.879 of the grid search, and with the test accuracy?
3. On the spam data of lesson 04, choose the decision threshold that gives at most 1% of legitimate e-mails classified as spam. What recall do you get?
