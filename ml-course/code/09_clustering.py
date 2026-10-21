"""Lesson 09 - Clustering: k-means and k-means++, choosing K, hierarchical clustering, colour quantisation."""
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score

import mlutils as U

U.setup("09")


# --8<-- [start:kmeans]
def kmeans_pp_init(X, k, rng):
    """k-means++: each new centre is drawn with probability proportional to D(x)^2."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centres)


def kmeans(X, k, rng, init="k-means++", iters=100):
    """Lloyd's algorithm: assign each point to the nearest centre, move each centre to its points' mean."""
    C = kmeans_pp_init(X, k, rng) if init == "k-means++" else X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)        # (n, k) squared distances
        labels = d2.argmin(1)
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    inertia = d2[np.arange(len(X)), labels].sum()            # sum of squared distances to the centre
    return labels, C, inertia
# --8<-- [end:kmeans]

# --8<-- [start:init]
X, y_true = make_blobs(n_samples=1500, centers=8, cluster_std=0.7, random_state=3)
rng = np.random.default_rng(0)
inert = {m: [kmeans(X, 8, rng, init=m)[2] for _ in range(50)] for m in ("random", "k-means++")}
best = KMeans(8, n_init=10, random_state=0).fit(X).inertia_
for m, v in inert.items():
    v = np.array(v)
    print(f"{m:10s}: inertia median {np.median(v):7.1f}, worst {v.max():7.1f}, "
          f"runs within 1% of the best known ({best:.1f}): {np.mean(v < 1.01 * best):.0%}")
# --8<-- [end:init]

fig, axes = U.figure(3, 1, size=(3.4, 3.2))
for ax, (title, m, seed) in zip(axes[0, :2], [("random init: a bad local minimum", "random", 0),
                                               ("k-means++ init", "k-means++", 0)]):
    r = np.random.default_rng(seed)
    runs = [kmeans(X, 8, r, init=m) for _ in range(50)]
    lab, C, inn = max(runs, key=lambda t: t[2]) if m == "random" else runs[0]
    ax.scatter(*X.T, c=[U.CLASS_COLORS[l] for l in lab], s=4)
    ax.scatter(*C.T, c="k", marker="x", s=60)
    ax.set_title(f"{title}\ninertia {inn:.0f}", fontsize=9)
axes[0, 2].hist([inert["random"], inert["k-means++"]], bins=20, color=[U.RED, U.BLUE], label=["random", "k-means++"])
axes[0, 2].set_xlabel("final inertia (50 runs each)")
axes[0, 2].legend()
U.finish("init", fig)

# --8<-- [start:choose-k]
Xk, _ = make_blobs(n_samples=600, centers=[[0, 0], [4, 0], [2, 3.5], [9, 1]], cluster_std=[0.6, 0.6, 0.6, 1.2],
                   random_state=1)
ks = range(2, 11)
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Xk)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Xk, km.labels_))
print("silhouette by K:", {k: round(s, 3) for k, s in zip(ks, silhouettes)})
# --8<-- [end:choose-k]

fig, axes = U.figure(3, 1, size=(3.4, 3.0))
axes[0, 0].plot(list(ks), inertias, "o-", color=U.BLUE)
axes[0, 0].set_title("inertia (elbow)")
axes[0, 0].set_xlabel("K")
axes[0, 1].plot(list(ks), silhouettes, "o-", color=U.RED)
axes[0, 1].set_title("mean silhouette")
axes[0, 1].set_xlabel("K")
kbest = list(ks)[int(np.argmax(silhouettes))]
lab = KMeans(kbest, n_init=10, random_state=0).fit_predict(Xk)
axes[0, 2].scatter(*Xk.T, c=[U.CLASS_COLORS[l] for l in lab], s=5)
axes[0, 2].set_title(f"K = {kbest}")
U.finish("choose-k", fig)

# --8<-- [start:hierarchical]
Xm, ym = make_moons(n_samples=300, noise=0.06, random_state=0)
Xg, yg = make_blobs(n_samples=300, centers=[[0, 0], [3, 3]], cluster_std=1.0, random_state=0)
for name, data in [("moons", Xm), ("two gaussians", Xg)]:
    line = []
    for link in ("ward", "complete", "average", "single"):
        lab = AgglomerativeClustering(n_clusters=2, linkage=link).fit_predict(data)
        sizes = np.bincount(lab)
        line.append(f"{link} {sizes.min()}/{sizes.max()}")
    print(f"{name:13s} cluster sizes:", ", ".join(line))
# --8<-- [end:hierarchical]

fig, axes = U.figure(4, 2, size=(2.7, 2.5))
for row, data in enumerate((Xm, Xg)):
    for col, link in enumerate(("ward", "complete", "average", "single")):
        lab = AgglomerativeClustering(n_clusters=2, linkage=link).fit_predict(data)
        ax = axes[row, col]
        ax.scatter(*data.T, c=[U.CLASS_COLORS[l] for l in lab], s=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f"{link} linkage")
U.finish("linkage", fig)

fig, axes = U.figure(2, 1, size=(4.2, 2.9))
for ax, link in zip(axes[0], ("ward", "single")):
    dendrogram(linkage(Xg[::6], method=link), ax=ax, no_labels=True, color_threshold=None,
               above_threshold_color=U.GRAY)
    ax.set_title(f"dendrogram, {link} linkage (50 points)")
    ax.set_ylabel("merge distance")
    ax.grid(False)
U.finish("dendrogram", fig)

# --8<-- [start:quantise]
img = np.asarray(Image.open(U.path("rouen.jpg")).convert("RGB"))   # (576, 800, 3) uint8
pixels = img.reshape(-1, 3).astype(np.float64)                      # one sample per pixel, 3 features
sample = pixels[np.random.default_rng(0).choice(len(pixels), 20000, replace=False)]
quantised = {}
for k in (2, 4, 8, 16):
    km = KMeans(k, n_init=4, random_state=0).fit(sample)            # learn the palette on a sample
    labels = km.predict(pixels)
    quantised[k] = km.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)
    bits = np.ceil(np.log2(k))
    ratio = 24 / bits
    print(f"K = {k:2d}: {bits:.0f} bits/pixel instead of 24 -> compression x{ratio:.0f}, "
          f"RMSE {np.sqrt(np.mean((quantised[k].astype(float) - img) ** 2)):.1f} grey levels")
# --8<-- [end:quantise]

fig, axes = U.figure(3, 1, size=(3.6, 2.8))
for ax, (title, im) in zip(axes[0], [("original (24 bits/pixel)", img), ("K = 4 colours", quantised[4]),
                                     ("K = 16 colours", quantised[16])]):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis("off")
U.finish("quantise", fig)
