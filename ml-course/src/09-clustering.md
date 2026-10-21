---
title: Clustering
summary: Learning without labels. Write k-means and its k-means++ initialisation from scratch, see how often a bad start traps it, choose the number of clusters with the elbow and the silhouette, compare hierarchical linkages on shapes k-means cannot handle, and compress a photo of Rouen to 16 colours.
topics: k-means · k-means++ · inertia · silhouette · hierarchical clustering · linkage · dendrogram · colour quantisation
script: 09_clustering.py
sources: INSA IML clustering lab
---

## Unsupervised learning

So far every sample came with a label. **Clustering** has only the $\mathbf{x}_i$, and asks: do they form groups? Customers with similar behaviour, pixels of similar colour, sensor readings from the same regime. There is no "right answer" to check against, which makes clustering both useful for exploration and slippery to evaluate.

## k-means

k-means looks for $K$ centres $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$ and an assignment of every point to one of them, minimising the **inertia** — the total squared distance of each point to its centre:

$$J = \sum_{i=1}^{N} \big\|\mathbf{x}_i - \boldsymbol{\mu}_{c(i)}\big\|^2$$

**Lloyd's algorithm** alternates two steps, each of which can only decrease $J$:

1. **Assign**: each point goes to its nearest centre.
2. **Update**: each centre moves to the mean of its points (the mean is what minimises the sum of squared distances).

It stops when nothing moves. Guaranteed to converge — but to a **local** minimum that depends on the starting centres.

```python
--8<-- "09_clustering.py:kmeans"
```

**k-means++** chooses the starting centres one at a time, each drawn with probability proportional to its squared distance $D(\mathbf{x})^2$ to the nearest centre already chosen. Far-away points are likely picks, so the initial centres tend to be spread over the different groups.

```python
--8<-- "09_clustering.py:init"
```

```text
random    : inertia median  2362.0, worst 20996.6, runs within 1% of the best known (1325.3): 14%
k-means++ : inertia median  1405.3, worst  4686.9, runs within 1% of the best known (1325.3): 18%
```

!fig 09-init | Left: the worst of 50 runs with random initialisation — three centres fight over one corner while one cluster spans three blobs. Middle: a k-means++ run. Right: the final inertia of 50 runs of each.

k-means++ makes disasters rarer and the typical result much better, but even it lands in a local minimum most of the time on this 8-blob problem. That is why scikit-learn runs k-means several times (`n_init`) and keeps the lowest inertia. Always do the same.

!!! note "On fixing the seed"
    My lab report concluded that k-means++ "with a random seed" is preferred because it is "more predictable and reproducible". Two different things are mixed there. A fixed `random_state` makes a run *reproducible* (same numbers every time), whatever the initialisation. k-means++ makes runs *better on average*. Neither guarantees the global optimum; several restarts do most of the work.

## How many clusters?

$K$ is an input of k-means, and the inertia always decreases when $K$ grows (with $K = N$ it is zero). Two common heuristics:

- **The elbow**: plot the inertia against $K$ and look for the point where adding a cluster stops paying off.
- **The silhouette**: for each point, $a$ = mean distance to its own cluster, $b$ = mean distance to the nearest other cluster, $s = (b - a)/\max(a, b)$. Close to 1: well inside its cluster; near 0: on a border; negative: probably in the wrong cluster. Average over all points and pick the $K$ with the highest score.

```python
--8<-- "09_clustering.py:choose-k"
```

```text
silhouette by K: {2: 0.589, 3: 0.539, 4: 0.681, 5: 0.611, 6: 0.61, 7: 0.603, 8: 0.525, 9: 0.419, 10: 0.333}
```

!fig 09-choose-k | Four groups of different spreads. The elbow and the silhouette both point to K = 4.

Here both agree. On real data they often don't, and K = 2 frequently wins the silhouette simply because two big blobs are "well separated" at a coarse scale. The honest answer is that the number of clusters depends on the **scale** you care about — and on what you will do with them.

## Hierarchical clustering

Agglomerative clustering starts with every point as its own cluster and repeatedly **merges the two closest clusters**. "Closest" between two groups needs a definition — the **linkage**:

- **single**: distance between their two closest points;
- **complete**: between their two farthest points;
- **average**: average over all pairs;
- **Ward**: the merge that increases the total within-cluster variance the least (k-means' criterion).

```python
--8<-- "09_clustering.py:hierarchical"
```

```text
moons         cluster sizes: ward 82/218, complete 129/171, average 98/202, single 150/150
two gaussians cluster sizes: ward 138/162, complete 128/172, average 143/157, single 1/299
```

!fig 09-linkage | Top: two interleaved moons. Only single linkage follows the chains of nearby points and recovers both moons. Bottom: two overlapping Gaussian clouds. Single linkage now fails completely: it chains everything into one cluster and leaves a single outlier alone.

No linkage is best: single linkage finds elongated shapes but is ruined by a few points bridging two groups; Ward finds compact, similarly-sized blobs, like k-means. The whole merge history is a tree, the **dendrogram**; cutting it at a given height gives the clusters at that scale, which is a nice way of postponing the choice of $K$.

!fig 09-dendrogram | Dendrograms of 50 points of the two Gaussian clouds. With Ward linkage the two groups appear as two branches joined very high up, far above every other merge. With single linkage all merge heights are similar, clusters grow by absorbing points one at a time (chaining), and the top split separates a few stragglers rather than the two clouds.

## Application: colour quantisation

A colour image is a set of pixels, each a point in 3-D RGB space. Run k-means on the pixels, then replace every pixel by its cluster centre: the image now uses only $K$ colours (a **palette**), and each pixel can be stored as a $\lceil\log_2 K\rceil$-bit index into it.

```python
--8<-- "09_clustering.py:quantise"
```

```text
K =  2: 1 bits/pixel instead of 24 -> compression x24, RMSE 33.0 grey levels
K =  4: 2 bits/pixel instead of 24 -> compression x12, RMSE 20.9 grey levels
K =  8: 3 bits/pixel instead of 24 -> compression x8, RMSE 14.8 grey levels
K = 16: 4 bits/pixel instead of 24 -> compression x6, RMSE 11.0 grey levels
```

!fig 09-quantise | The old market square of Rouen with 16.7 million possible colours, with 4 and with 16. The palette is learnt from a random sample of 20,000 pixels — fitting all 460,800 gives the same colours, much more slowly.

(The palette itself costs $K \times 24$ bits, negligible here.) This is how GIF images work, and the same "vector quantisation" idea is used to compress the embeddings of neural networks.

## Exercises

1. Quantise in the Lab colour space instead of RGB (`skimage.color.rgb2lab`). Distances in Lab are closer to perceived differences. Do the 4-colour results look better?
2. k-means assumes round clusters of similar size. Generate three elongated Gaussian clouds (apply a linear transform to blobs) and compare k-means with a Gaussian mixture model (`sklearn.mixture.GaussianMixture`).
3. Implement single-linkage clustering from scratch with a minimum spanning tree: build the MST of the points (Prim's algorithm) and remove its $K-1$ longest edges.
