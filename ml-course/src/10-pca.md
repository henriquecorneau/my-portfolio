---
title: Dimensionality reduction
summary: Find the few directions that carry most of the variance. Derive PCA as an eigenvalue problem and implement it in five lines, see the SVD give the same answer, learn why standardisation decides what PCA finds, compress handwritten digits into eigen-digits, and compare PCA with the non-linear t-SNE.
topics: covariance · eigenvectors · SVD · explained variance · standardisation · reconstruction · t-SNE
script: 10_pca.py
sources: INSA IML PCA lab
---

## Why reduce dimensions

The wine data of lesson 01 had 11 features but sugar and density said almost the same thing. An MNIST digit has 784 pixels, but most combinations of 784 values are noise, not digits — real digits live near a much smaller subset of that space. **Dimensionality reduction** finds a low-dimensional description that keeps what matters, to visualise data in 2-D, compress it, remove noise, or speed up and regularise a later model.

## PCA: the directions of largest variance

**Principal component analysis** looks for the unit vector $\mathbf{u}$ such that the projections $\mathbf{u}^\top\mathbf{x}_i$ of the centred data have the largest possible variance. That variance is $\mathbf{u}^\top \Sigma\, \mathbf{u}$, where $\Sigma$ is the covariance matrix. Maximising it under $\|\mathbf{u}\| = 1$ (a Lagrangian again, lesson 03) gives

$$\Sigma\,\mathbf{u} = \lambda\,\mathbf{u}$$

The best direction is the **eigenvector** with the largest eigenvalue, and the variance along it *is* that eigenvalue. The second principal component is the best direction orthogonal to the first, and so on: the principal axes are the eigenvectors of $\Sigma$ sorted by eigenvalue, and $\lambda_k / \sum_j \lambda_j$ is the fraction of the total variance each one **explains**.

```python
--8<-- "10_pca.py:pca"
```

Our data: 202 Australian athletes (the `ais` dataset), 11 blood tests and body measurements per athlete.

```python
--8<-- "10_pca.py:data"
```

```text
(202, 11) | std of each variable: {'rcc': 0.5, 'wcc': 1.8, 'hc': 3.7, 'hg': 1.4, 'ferr': 47.4, 'bmi': 2.9, 'ssf': 32.5, ...}
explained variance ratio (ours): [0.454 0.233 0.105 0.081]
explained variance ratio (sklearn): [0.454 0.233]
same axes up to sign: True
```

(An eigenvector is only defined up to its sign: $-\mathbf{u}$ is as good as $\mathbf{u}$. Two correct PCA implementations can disagree on signs — compare absolute values.)

In practice PCA is computed with the **singular value decomposition** of the centred data matrix, $X_c = U S V^\top$: the columns of $V$ are the principal axes and $\lambda_k = s_k^2 / (N - 1)$. It avoids forming $X^\top X$, which squares the condition number and loses precision.

```python
--8<-- "10_pca.py:svd"
```

```text
eigenvalues from the SVD: [5.016 2.57  1.163 0.894] | from the covariance: [5.016 2.57  1.163 0.894]
```

## Standardise — or PCA finds the units

PCA maximises *variance*, and variance depends on units. Ferritin is measured in ng/mL with a standard deviation of 47; red cell count in 10¹²/L with a standard deviation of 0.5.

```python
--8<-- "10_pca.py:scaling"
```

```text
raw data: PC1 explains 61% of the variance; its largest loadings: {'ferr': -0.98, 'ssf': 0.14}
standardised: PC1 explains 45%; largest loadings: {'lbm': -0.4, 'hg': -0.39, 'hc': -0.39, 'rcc': -0.37}
```

On raw data the "first principal component" is ferritin, alone: it has the biggest numbers. After standardisation (PCA on the **correlation** matrix) the first component combines lean body mass, haemoglobin, haematocrit and red cell count — a genuine pattern shared by several variables, which happens to separate male and female athletes.

!fig 10-ais | Left: explained variance of each component — 5 of 11 explain 94%. Middle: the athletes projected on the first two components of the standardised data, coloured by sex (which PCA never saw). Right: the same on raw data, dominated by ferritin.

Standardise when the variables have different units. Don't, when they share a meaningful common scale — the pixels of an image, below.

## Eigen-digits and reconstruction

Apply PCA to 5,000 MNIST digits (784 pixels each):

```python
--8<-- "10_pca.py:digits"
```

```text
components needed for 80/90/95/99% of the variance: [43, 84, 148, 322] of 784
```

Each principal axis is itself a 784-pixel image. The **reconstruction** from $k$ components is $\hat{\mathbf{x}} = \bar{\mathbf{x}} + \sum_{j \le k} (\mathbf{u}_j^\top(\mathbf{x} - \bar{\mathbf{x}}))\,\mathbf{u}_j$: the mean digit plus a weighted sum of $k$ "eigen-digits". PCA is the linear projection with the smallest reconstruction error for a given $k$.

!fig 10-eigendigits | The mean digit and the first seven principal axes (red adds ink, blue removes it). PC1 contrasts a "0"-like ring with a "1"-like stroke.

!fig 10-reconstruction | Six digits reconstructed from 5, 20, 50 and 150 coordinates instead of 784. With 5 the 4 becomes a 9; with 50 every digit is readable — a 16× compression.

## t-SNE: a non-linear map for visualisation

PCA is linear: it can only rotate and project. The digits of different classes sit on curved, tangled surfaces, and their 2-D projection overlaps heavily. **t-SNE** takes another approach: it places points in 2-D so that **neighbours stay neighbours** — it matches the probability that two points are close in the original space with the same probability in the map, and does not care about preserving large distances.

```python
--8<-- "10_pca.py:tsne"
```

!fig 10-tsne | The 1,797 scikit-learn 8×8 digits mapped to 2-D, coloured by their true label (which neither method used). PCA keeps the global geometry but mixes the classes; t-SNE separates nearly all of them into islands.

t-SNE is a **visualisation** tool, not a feature extractor: distances *between* islands and the sizes of the islands mean little, the result depends on the `perplexity` (roughly, the number of neighbours considered) and on the random seed, and there is no `transform` for new points. Use PCA to compress features for a model; use t-SNE (or UMAP) to look at your data — including, in lesson 17, at the embeddings learnt by a neural network.

## Exercises

1. Train a logistic regression on MNIST (lesson 04) on 20, 50 and 150 PCA components instead of 784 pixels. How do accuracy and training time change?
2. **Denoising**: add Gaussian noise to the digits, fit PCA on clean training digits and reconstruct the noisy ones from 50 components. Why does this remove much of the noise?
3. Run t-SNE with perplexity 5, 30 and 100 on the same digits. What changes, and what stays?
