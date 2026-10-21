"""Lesson 08 - Ensembles: bagging, random forests, AdaBoost and gradient boosting."""
import time

import numpy as np
from sklearn.datasets import (fetch_california_housing, fetch_covtype, load_digits, make_classification,
                              make_moons)
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlutils as U

U.setup("08")

X, y = make_moons(n_samples=600, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)


# --8<-- [start:bagging]
def bagging(X, y, n_trees, rng):
    """Train each tree on a bootstrap sample: n draws WITH replacement from the training set."""
    trees = []
    for _ in range(n_trees):
        idx = rng.integers(0, len(y), len(y))
        trees.append(DecisionTreeClassifier(random_state=int(rng.integers(1 << 30))).fit(X[idx], y[idx]))
    return trees


def vote(trees, X):
    """Fraction of trees voting for class 1 (a 'soft' prediction)."""
    return np.mean([t.predict(X) for t in trees], axis=0)


rng = np.random.default_rng(0)
trees = bagging(Xtr, ytr, 200, rng)
single = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print("one deep tree : held-out error %.3f" % (1 - single.score(Xte, yte)))
for n in (1, 10, 50, 200):
    print(f"bagging, {n:3d} trees: held-out error {np.mean((vote(trees[:n], Xte) > 0.5) != yte):.3f}")
print("unique samples in a bootstrap sample: %.1f%% (theory: 1 - 1/e = 63.2%%)"
      % (100 * np.mean([len(np.unique(rng.integers(0, 1000, 1000))) / 1000 for _ in range(200)])))
# --8<-- [end:bagging]

fig, axes = U.figure(3, 1, size=(3.4, 3.2))
U.plot_decision(axes[0, 0], single.predict, Xtr, ytr, title="one deep tree", resolution=200)
U.plot_decision(axes[0, 1], lambda g: vote(trees, g), Xtr, ytr, proba=True, title="bagging, 200 trees (vote share)",
                resolution=200)
ns = np.arange(1, 201)
votes = np.cumsum([t.predict(Xte) for t in trees], axis=0) / ns[:, None]
axes[0, 2].plot(ns, np.mean((votes > 0.5) != yte, axis=1), color=U.BLUE)
axes[0, 2].axhline(1 - single.score(Xte, yte), color=U.RED, ls="--", lw=1, label="one deep tree")
axes[0, 2].set_xscale("log")
axes[0, 2].set_xlabel("number of trees")
axes[0, 2].set_ylabel("held-out error")
axes[0, 2].legend()
U.finish("bagging", fig)

# --8<-- [start:max-features]
Xd, yd = load_digits(return_X_y=True)              # (1797, 64): 8x8 images, one feature per pixel
results = {}
for m in (1, 2, 4, 8, 16, 32, 64):
    acc = []
    for rep in range(5):
        a, b, c, d = train_test_split(Xd, yd, test_size=0.2, random_state=rep)
        acc.append(RandomForestClassifier(n_estimators=100, max_features=m, random_state=rep,
                                          n_jobs=-1).fit(a, c).score(b, d))
    results[m] = (np.mean(acc), np.std(acc))
    print(f"max_features = {m:2d}: accuracy {results[m][0]:.4f} ± {results[m][1]:.4f}")
# --8<-- [end:max-features]

# --8<-- [start:importance]
Xi, yi = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0,
                             shuffle=False, random_state=0)   # features 0-4 informative, 5-19 noise
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Xi, yi)
imp = rf.feature_importances_
std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
order = np.argsort(-imp)
print("top 6 features:", order[:6].tolist(), "| importance of the 5 informative: %.2f, of the 15 noise: %.2f"
      % (imp[:5].sum(), imp[5:].sum()))
# --8<-- [end:importance]

fig, axes = U.figure(2, 1, size=(4.0, 3.1))
ms = list(results)
axes[0, 0].errorbar(ms, [results[m][0] for m in ms], yerr=[results[m][1] for m in ms], fmt="o-", color=U.BLUE,
                    capsize=3)
axes[0, 0].set_xscale("log", base=2)
axes[0, 0].set_xticks(ms, [str(m) for m in ms])
axes[0, 0].set_xlabel("max_features (of 64)")
axes[0, 0].set_ylabel("test accuracy")
axes[0, 0].set_title("random forest on digits")
axes[0, 1].bar(range(20), imp[order], yerr=std[order], color=[U.RED if i < 5 else U.GRAY for i in order],
               capsize=2)
axes[0, 1].set_xticks(range(20), order, fontsize=7)
axes[0, 1].set_xlabel("feature")
axes[0, 1].set_title("importances (red: truly informative)")
U.finish("forest", fig)


# --8<-- [start:adaboost]
def adaboost(X, y, rounds):
    """Discrete AdaBoost with decision stumps; labels must be -1/+1."""
    w = np.full(len(y), 1 / len(y))                    # sample weights
    stumps, alphas = [], []
    for _ in range(rounds):
        stump = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        err = np.sum(w[pred != y])                     # weighted error, < 0.5 for a useful stump
        alpha = 0.5 * np.log((1 - err) / max(err, 1e-12))
        w *= np.exp(-alpha * y * pred)                 # misclassified samples gain weight
        w /= w.sum()
        stumps.append(stump)
        alphas.append(alpha)
    return lambda Z: np.sign(sum(a * s.predict(Z) for a, s in zip(alphas, stumps)))


ys_tr, ys_te = 2 * ytr - 1, 2 * yte - 1
for rounds in (1, 10, 100, 400):
    f = adaboost(Xtr, ys_tr, rounds)
    print(f"AdaBoost, {rounds:3d} stumps: train error {np.mean(f(Xtr) != ys_tr):.3f}, "
          f"held-out error {np.mean(f(Xte) != ys_te):.3f}")
# --8<-- [end:adaboost]

fig, axes = U.figure(3, 1, size=(3.3, 3.1))
for ax, r in zip(axes[0], (1, 10, 100)):
    f = adaboost(Xtr, ys_tr, r)
    U.plot_decision(ax, lambda g: (f(g) > 0).astype(int), Xtr, ytr, title=f"AdaBoost, {r} stump(s)", resolution=200)
U.finish("adaboost", fig)

# --8<-- [start:gboost]
xr = np.sort(np.random.rand(200) * 6)
yr = np.sin(xr) + 0.3 * np.random.randn(200)


def gradient_boosting(x, y, n_trees, lr):
    """Squared loss: each new tree fits the residuals (= minus the gradient) of the current model."""
    pred = np.full_like(y, y.mean())
    trees = []
    for _ in range(n_trees):
        residual = y - pred
        t = DecisionTreeRegressor(max_depth=2).fit(x[:, None], residual)
        pred += lr * t.predict(x[:, None])
        trees.append(t)
    return lambda z: y.mean() + lr * sum(t.predict(z[:, None]) for t in trees)


for n_trees, lr in [(1, 1.0), (10, 0.3), (100, 0.1)]:
    f = gradient_boosting(xr, yr, n_trees, lr)
    print(f"{n_trees:3d} trees, lr {lr}: training MSE {np.mean((f(xr) - yr) ** 2):.3f}")
# --8<-- [end:gboost]

fig, axes = U.figure(3, 1, size=(3.3, 2.8))
xs = np.linspace(0, 6, 400)
for ax, (n_trees, lr) in zip(axes[0], [(1, 1.0), (10, 0.3), (100, 0.1)]):
    ax.scatter(xr, yr, s=6, color=U.GRAY)
    ax.plot(xs, np.sin(xs), "--", color="k", lw=1)
    ax.plot(xs, gradient_boosting(xr, yr, n_trees, lr)(xs), color=U.RED, lw=1.8)
    ax.set_title(f"{n_trees} tree(s), learning rate {lr}")
U.finish("gboost", fig)

# --8<-- [start:lr]
Xh, yh = fetch_california_housing(data_home=str(U.CACHE), return_X_y=True)
fig, axes = U.figure(1, 1, size=(5.0, 3.3))
for lr, c in zip((0.1, 0.01, 0.001), (U.RED, U.BLUE, U.GRAY)):
    hgb = HistGradientBoostingRegressor(max_iter=1000, learning_rate=lr, max_depth=5, early_stopping=True,
                                        validation_fraction=0.2, n_iter_no_change=50, scoring="r2",
                                        random_state=0).fit(Xh, yh)
    print(f"lr = {lr:<5}: {hgb.n_iter_:4d} trees, best validation R² {max(hgb.validation_score_):.3f}")
    axes[0, 0].plot(hgb.train_score_, color=c, lw=1, ls="--")
    axes[0, 0].plot(hgb.validation_score_, color=c, label=f"lr = {lr} (validation)")
# --8<-- [end:lr]
axes[0, 0].set_xlabel("number of trees")
axes[0, 0].set_ylabel("R² (dashed: training)")
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_title("gradient boosting on California housing")
axes[0, 0].legend()
U.finish("learning-rate", fig)

# --8<-- [start:covtype]
Xc, yc = fetch_covtype(data_home=str(U.CACHE), return_X_y=True)     # (581012, 54), 7 forest cover types
Xa, Xb, ya, yb = train_test_split(Xc, yc, train_size=100_000, test_size=50_000, stratify=yc, random_state=0)
models = {
    "AdaBoost (100 trees, depth 5)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100,
                                                        random_state=0),
    "random forest (100, depth 5)": RandomForestClassifier(100, max_depth=5, n_jobs=-1, random_state=0),
    "random forest (100, full depth)": RandomForestClassifier(100, n_jobs=-1, random_state=0),
    "hist. GB (100, depth 5)": HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=0),
    "hist. GB (300, 63 leaves, l2 = 0)": HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63,
                                                                         early_stopping=False, random_state=0),
    "hist. GB (300, 63 leaves, l2 = 1)": HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63,
                                                                         l2_regularization=1.0,
                                                                         early_stopping=False, random_state=0),
}
for name, m in models.items():
    t0 = time.time()
    m.fit(Xa, ya)
    t1 = time.time()
    acc = m.score(Xb, yb)
    print(f"{name:34s} test accuracy {acc:.3f}  (train {t1 - t0:5.1f} s, predict {time.time() - t1:4.1f} s)")
# --8<-- [end:covtype]
