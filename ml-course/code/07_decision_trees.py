"""Lesson 07 - Decision trees: CART from scratch, depth and overfitting, instability."""
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import mlutils as U

U.setup("07")


# --8<-- [start:gini]
def gini(y):
    """Probability that two samples drawn at random from y have different labels."""
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)


def best_split(X, y):
    """Try every feature and every threshold between consecutive values; keep the purest split."""
    best = (None, None, gini(y))                     # (feature, threshold, weighted impurity)
    n = len(y)
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for t in (values[:-1] + values[1:]) / 2:     # midpoints
            left = X[:, j] <= t
            imp = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / n
            if imp < best[2] - 1e-12:
                best = (j, t, imp)
    return best
# --8<-- [end:gini]


# --8<-- [start:tree]
def build_tree(X, y, depth=0, max_depth=None, min_samples=2):
    """Recursive CART. A node is either a leaf (a class) or (feature, threshold, left, right)."""
    majority = np.bincount(y).argmax()
    if len(np.unique(y)) == 1 or len(y) < min_samples or depth == max_depth:
        return majority
    j, t, _ = best_split(X, y)
    if j is None:                                    # no split reduces the impurity
        return majority
    left = X[:, j] <= t
    return (j, t, build_tree(X[left], y[left], depth + 1, max_depth, min_samples),
            build_tree(X[~left], y[~left], depth + 1, max_depth, min_samples))


def predict_one(node, x):
    while isinstance(node, tuple):
        j, t, left, right = node
        node = left if x[j] <= t else right
    return node


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])
# --8<-- [end:tree]


# --8<-- [start:iris]
iris = load_iris()
X, y = iris.data[:, 2:], iris.target               # petal length and width, 3 species
tree = build_tree(X, y, max_depth=3)
j, t, imp = best_split(X, y)
print("root split: %s <= %.2f, Gini impurity %.3f -> %.3f" % (iris.feature_names[2 + j], t, gini(y), imp))
ref = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
print(export_text(ref, feature_names=["petal length", "petal width"]))
print("depth-3 tree: training accuracy %.3f (ours) vs %.3f (scikit-learn), same predictions on %.0f%%"
      % (np.mean(predict(tree, X) == y), ref.score(X, y), 100 * np.mean(predict(tree, X) == ref.predict(X))))
# --8<-- [end:iris]

fig, axes = U.figure(3, 1, size=(3.4, 3.2))
for ax, d in zip(axes[0], (1, 3, None)):
    t = build_tree(X, y, max_depth=d)
    U.plot_decision(ax, lambda g: predict(t, g), X, y, title=f"max depth {d}", resolution=150)
    ax.set_xlabel("petal length (cm)")
axes[0, 0].set_ylabel("petal width (cm)")
U.finish("iris", fig)

# --8<-- [start:depth]
Xc, yc = make_classification(n_samples=2000, n_features=20, n_informative=20, n_redundant=0,
                             class_sep=0.9, random_state=11)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=0)
depths = range(1, 21)
tr_err, te_err = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    tr_err.append(1 - clf.score(Xtr, ytr))
    te_err.append(1 - clf.score(Xte, yte))
best = int(np.argmin(te_err))
print("depth with lowest held-out error: %d (%.3f); depth 20: train %.3f, held-out %.3f"
      % (depths[best], te_err[best], tr_err[-1], te_err[-1]))
# --8<-- [end:depth]

fig, axes = U.figure(1, 1, size=(4.8, 3.2))
ax = axes[0, 0]
ax.plot(depths, tr_err, "o-", ms=3, color=U.BLUE, label="training error")
ax.plot(depths, te_err, "o-", ms=3, color=U.RED, label="held-out error")
ax.set_xticks(range(1, 21, 2))
ax.set_xlabel("max depth")
ax.set_ylabel("error rate")
ax.legend()
ax.set_title("a single tree: depth vs overfitting")
U.finish("depth", fig)

# --8<-- [start:instability]
fig, axes = U.figure(4, 2, size=(2.6, 2.4))
for i, ax in enumerate(axes.flat):
    Xm, ym = make_moons(n_samples=100, noise=0.2, random_state=i)   # 8 samples of the same distribution
    t = DecisionTreeClassifier(random_state=0).fit(Xm, ym)            # grown until the leaves are pure
    U.plot_decision(ax, t.predict, Xm, ym, resolution=150, lims=(-1.6, 2.6, -1.2, 1.7))
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("instability", fig)
# --8<-- [end:instability]
