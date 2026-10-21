"""Lesson 00 - A preview: five model families from the course on the same 2-D problem."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_moons
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import mlutils as U

U.setup("00")

# --8<-- [start:overview]
X, y = make_moons(n_samples=600, noise=0.28, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
models = {
    "logistic regression (04)": LogisticRegression(),
    "k-NN, k = 15 (01)": KNeighborsClassifier(15),
    "RBF SVM (05)": CalibratedClassifierCV(SVC(C=10, gamma=1), ensemble=False),   # scores -> probabilities
    "gradient boosting (08)": HistGradientBoostingClassifier(max_depth=3, random_state=0),
    "neural network (11)": MLPClassifier((64, 64), max_iter=2000, random_state=0),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(f"{name:26s} test accuracy {m.score(Xte, yte):.3f}")
# --8<-- [end:overview]

fig, axes = U.figure(5, 1, size=(2.9, 2.8))
for ax, (name, m) in zip(axes[0], models.items()):
    U.plot_decision(ax, lambda g: m.predict_proba(g)[:, 1], Xtr, ytr, proba=True, resolution=200,
                    title=name, lims=(-1.8, 2.8, -1.4, 1.9))
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("hero", fig)

# a 2x2 version of the same picture for the portfolio home page
fig, axes = U.figure(2, 2, size=(2.6, 1.75))
for ax, name in zip(axes.flat, ["k-NN, k = 15 (01)", "RBF SVM (05)", "gradient boosting (08)", "neural network (11)"]):
    m = models[name]
    U.plot_decision(ax, lambda g: m.predict_proba(g)[:, 1], Xtr, ytr, proba=True, resolution=200,
                    title=name, lims=(-1.8, 2.8, -1.4, 1.9))
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("banner", fig)
