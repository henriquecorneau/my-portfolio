"""Lesson 10 - Dimensionality reduction: PCA from scratch, scaling, reconstruction, t-SNE."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import mlutils as U

U.setup("10")

# --8<-- [start:data]
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/ais.csv"
ais = pd.read_csv(U.fetch(URL, "ais.csv"))
variables = ["rcc", "wcc", "hc", "hg", "ferr", "bmi", "ssf", "pcBfat", "lbm", "ht", "wt"]
X = ais[variables].to_numpy(float)                 # (202, 11) blood tests and body measurements
sex = (ais["sex"] == "f").to_numpy()               # used only to colour the plots
print(X.shape, "| std of each variable:", {v: round(float(s), 1) for v, s in zip(variables, X.std(0))})
# --8<-- [end:data]


# --8<-- [start:pca]
def pca(X, k):
    """Principal axes = eigenvectors of the covariance matrix, sorted by decreasing eigenvalue."""
    Xc = X - X.mean(0)                             # PCA is about variance: centre first
    cov = Xc.T @ Xc / (len(X) - 1)                 # (d, d)
    eigval, eigvec = np.linalg.eigh(cov)           # eigh: for symmetric matrices, ascending order
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    return Xc @ eigvec[:, :k], eigvec[:, :k], eigval


Z = (X - X.mean(0)) / X.std(0)                     # standardised variables
proj, axes_, eigval = pca(Z, 2)
ref = PCA(2).fit(Z)
print("explained variance ratio (ours):", np.round(eigval[:4] / eigval.sum(), 3))
print("explained variance ratio (sklearn):", np.round(ref.explained_variance_ratio_, 3))
print("same axes up to sign:", np.allclose(np.abs(axes_), np.abs(ref.components_.T), atol=1e-6))
# --8<-- [end:pca]

# --8<-- [start:svd]
U_, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
print("eigenvalues from the SVD: %s" % np.round(S[:4] ** 2 / (len(Z) - 1), 3),
      "| from the covariance: %s" % np.round(eigval[:4], 3))
# --8<-- [end:svd]

# --8<-- [start:scaling]
raw_proj, raw_axes, raw_eig = pca(X, 2)
print("raw data: PC1 explains %.0f%% of the variance; its largest loadings: %s"
      % (100 * raw_eig[0] / raw_eig.sum(),
         {variables[i]: round(float(raw_axes[i, 0]), 2) for i in np.argsort(-np.abs(raw_axes[:, 0]))[:2]}))
print("standardised: PC1 explains %.0f%%; largest loadings: %s"
      % (100 * eigval[0] / eigval.sum(),
         {variables[i]: round(float(axes_[i, 0]), 2) for i in np.argsort(-np.abs(axes_[:, 0]))[:4]}))
# --8<-- [end:scaling]

fig, axes = U.figure(3, 1, size=(3.5, 3.2))
ax = axes[0, 0]
ratio = eigval / eigval.sum()
ax.bar(range(1, 12), ratio, color=U.BLUE, alpha=0.7, label="per component")
ax.step(range(1, 12), np.cumsum(ratio), where="mid", color=U.RED, label="cumulative")
ax.set_xlabel("principal component")
ax.set_title("explained variance (standardised)")
ax.legend(fontsize=7)
for ax, (P, title) in zip(axes[0, 1:], [(proj, "PCA on standardised data"), (raw_proj, "PCA on raw data")]):
    ax.scatter(*P[~sex].T, s=10, color=U.BLUE, label="male")
    ax.scatter(*P[sex].T, s=10, color=U.RED, label="female")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
axes[0, 1].legend(fontsize=7)
U.finish("ais", fig)

# --8<-- [start:digits]
x_train, y_train, _, _ = U.load_mnist()
D = x_train[:5000].reshape(5000, -1) / 255.0       # (5000, 784) pixel intensities in [0, 1]
pca_d = PCA().fit(D)                               # no standardisation: pixels already share a scale
cum = np.cumsum(pca_d.explained_variance_ratio_)
print("components needed for 80/90/95/99% of the variance:",
      [int(np.searchsorted(cum, p) + 1) for p in (0.8, 0.9, 0.95, 0.99)], "of", D.shape[1])
recon = {}
for k in (5, 20, 50, 150):
    p = PCA(k).fit(D)
    recon[k] = p.inverse_transform(p.transform(D[:6]))   # project onto k axes, then back to 784 pixels
# --8<-- [end:digits]

fig, axes = U.figure(5, 6, size=(1.2, 1.25))
for row in range(6):
    for col, (title, img) in enumerate([("original", D[row])] + [(f"{k} PCs", recon[k][row]) for k in recon]):
        ax = axes[row, col]
        ax.imshow(img.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        if row == 0:
            ax.set_title(title, fontsize=8)
U.finish("reconstruction", fig)

fig, axes = U.figure(8, 1, size=(1.3, 1.45))
axes[0, 0].imshow(pca_d.mean_.reshape(28, 28), cmap="gray_r")
axes[0, 0].set_title("mean", fontsize=8)
for i, ax in enumerate(axes[0, 1:]):
    v = pca_d.components_[i].reshape(28, 28)
    ax.imshow(v, cmap="RdBu_r", vmin=-np.abs(v).max(), vmax=np.abs(v).max())
    ax.set_title(f"PC{i + 1}", fontsize=8)
for ax in axes[0]:
    ax.axis("off")
U.finish("eigendigits", fig)

# --8<-- [start:tsne]
Xs, ys = load_digits(return_X_y=True)              # (1797, 64): 8x8 digits, fast enough for t-SNE
p2 = PCA(2).fit_transform(Xs)
t2 = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Xs)
# --8<-- [end:tsne]

fig, axes = U.figure(2, 1, size=(4.2, 3.9))
for ax, (P, title) in zip(axes[0], [(p2, "PCA (linear)"), (t2, "t-SNE (non-linear)")]):
    for d in range(10):
        ax.scatter(*P[ys == d].T, s=4, color=U.CLASS_COLORS[d])
        cx, cy = np.median(P[ys == d], axis=0)
        ax.text(cx, cy, str(d), fontsize=12, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=U.CLASS_COLORS[d], alpha=0.85))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("tsne", fig)
