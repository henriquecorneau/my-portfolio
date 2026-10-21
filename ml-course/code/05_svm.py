"""Lesson 05 - Support vector machines: margin, soft margin, the dual and the kernel trick."""
import cvxpy as cp
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mlutils as U

U.setup("05")


def two_gaussians(n):
    mu1, S1 = np.array([0, 2]), 1.5 * np.array([[1, 0.1], [0.1, 1]])
    mu2, S2 = np.array([-2, -2]), 3.5 * np.array([[1, -0.25], [-0.25, 0.5]])
    X = np.vstack([np.random.multivariate_normal(mu1, S1, n), np.random.multivariate_normal(mu2, S2, n)])
    return X, np.r_[np.ones(n), -np.ones(n)]        # SVM convention: labels in {-1, +1}


X, y = two_gaussians(100)                           # (200, 2), (200,)


# --8<-- [start:primal]
def svm_primal(X, y, C):
    """min 1/2 ||w||^2 + C sum(xi)  s.t.  y_i (w.x_i + b) >= 1 - xi_i,  xi_i >= 0."""
    w, b, xi = cp.Variable(X.shape[1]), cp.Variable(), cp.Variable(len(y))
    cons = [cp.multiply(y, X @ w + b) >= 1 - xi, xi >= 0]
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi)), cons).solve()
    return w.value, b.value


w, b = svm_primal(X, y, C=1.0)
ref = SVC(kernel="linear", C=1.0).fit(X, y)
margin = y * (X @ w + b)
print("CVXPY primal : w = %s, b = %.4f, margin width 2/|w| = %.3f" % (np.round(w, 4), b, 2 / np.linalg.norm(w)))
print("scikit-learn : w = %s, b = %.4f" % (np.round(ref.coef_[0], 4), ref.intercept_[0]))
print("support vectors: %d with CVXPY (margin <= 1), %d in scikit-learn"
      % (np.sum(margin <= 1 + 1e-4), len(ref.support_)))
# --8<-- [end:primal]

# --8<-- [start:C]
for C in (0.001, 0.01, 1, 100):
    clf = SVC(kernel="linear", C=C).fit(X, y)
    print(f"C = {C:<6}: margin width {2 / np.linalg.norm(clf.coef_):.2f}, "
          f"{len(clf.support_):3d} support vectors, training error {1 - clf.score(X, y):.3f}")
# --8<-- [end:C]

fig, axes = U.figure(3, 1, size=(3.5, 3.3))
for ax, C in zip(axes[0], (0.01, 1, 100)):
    clf = SVC(kernel="linear", C=C).fit(X, y)
    U.plot_decision(ax, clf.predict, X, (y > 0).astype(int), title=f"C = {C}: {len(clf.support_)} SVs")
    xx = np.linspace(*ax.get_xlim(), 2)
    wv, bv = clf.coef_[0], clf.intercept_[0]
    for off, ls in ((0, "-"), (1, "--"), (-1, "--")):
        ax.plot(xx, (off - bv - wv[0] * xx) / wv[1], "k", ls=ls, lw=1)
    ax.scatter(*clf.support_vectors_.T, s=40, facecolor="none", edgecolor="k", lw=0.8)
U.finish("margin", fig)

# --8<-- [start:dual]
Xm, ym = make_moons(n_samples=200, noise=0.2, random_state=0)
ym = 2 * ym - 1.0


def rbf(A, B, gamma):
    """Gaussian kernel matrix k(a, b) = exp(-gamma ||a - b||^2), shape (len(A), len(B))."""
    d2 = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None, :]
    return np.exp(-gamma * d2)


def svm_dual(X, y, C, gamma):
    """max sum(a) - 1/2 sum_ij a_i a_j y_i y_j k(x_i, x_j)  s.t.  0 <= a <= C,  sum a_i y_i = 0."""
    K = rbf(X, X, gamma)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))   # K = L L^T, so a^T Q a = ||L^T (a*y)||^2
    a = cp.Variable(len(y))
    obj = cp.Maximize(cp.sum(a) - 0.5 * cp.sum_squares(L.T @ cp.multiply(a, y)))
    cp.Problem(obj, [a >= 0, a <= C, a @ y == 0]).solve()
    a = a.value
    on_margin = (a > 1e-5) & (a < C - 1e-5)               # 0 < a_i < C: exactly on the margin
    b = np.mean(y[on_margin] - (a * y) @ K[:, on_margin])
    return lambda Z: (a * y) @ rbf(X, Z, gamma) + b, a


f, alpha = svm_dual(Xm, ym, C=1.0, gamma=1.0)
ref = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(Xm, ym)
grid = np.random.uniform(-1.5, 2.5, (2000, 2))
print("dual from scratch: %d support vectors (scikit-learn %d), same predicted class on %.1f%% of the plane"
      % (np.sum(alpha > 1e-5), len(ref.support_), 100 * np.mean(np.sign(f(grid)) == ref.predict(grid))))
# --8<-- [end:dual]

# --8<-- [start:gamma]
Xn, yn = make_moons(n_samples=900, noise=0.25, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=1 / 3, stratify=yn, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.5, stratify=ytr, random_state=0)
for g in (0.01, 0.1, 1, 10, 100):
    clf = SVC(kernel="rbf", C=10, gamma=g).fit(Xtr, ytr)
    print(f"gamma = {g:<5}: train error {1 - clf.score(Xtr, ytr):.3f}, validation error {1 - clf.score(Xva, yva):.3f}")
# --8<-- [end:gamma]

fig, axes = U.figure(4, 1, size=(3.0, 2.9))
for ax, g in zip(axes[0], (0.1, 1, 10, 100)):
    clf = SVC(kernel="rbf", C=10, gamma=g).fit(Xtr, ytr)
    U.plot_decision(ax, clf.predict, Xtr, ytr, title=f"RBF, γ = {g}", resolution=200)
U.finish("gamma", fig)

# --8<-- [start:grid]
Cs, gammas = np.logspace(-3, 2, 6), np.logspace(-2, 2, 5)
err = np.array([[1 - SVC(kernel="rbf", C=C, gamma=g).fit(Xtr, ytr).score(Xva, yva) for g in gammas] for C in Cs])
i, j = np.unravel_index(err.argmin(), err.shape)
best = SVC(kernel="rbf", C=Cs[i], gamma=gammas[j]).fit(Xtr, ytr)
print("best on validation: C = %g, gamma = %g -> validation error %.3f, test error %.3f"
      % (Cs[i], gammas[j], err[i, j], 1 - best.score(Xte, yte)))

degrees = range(1, 8)
err_p = np.array([[1 - SVC(kernel="poly", coef0=1, degree=d, C=C).fit(Xtr, ytr).score(Xva, yva)
                   for d in degrees] for C in Cs])
i2, j2 = np.unravel_index(err_p.argmin(), err_p.shape)
best_p = SVC(kernel="poly", coef0=1, degree=degrees[j2], C=Cs[i2]).fit(Xtr, ytr)
print("polynomial kernel: C = %g, degree %d -> validation error %.3f, test error %.3f"
      % (Cs[i2], degrees[j2], err_p[i2, j2], 1 - best_p.score(Xte, yte)))
# --8<-- [end:grid]

fig, axes = U.figure(2, 1, size=(3.9, 3.2))
for ax, e, xl, xt in [(axes[0, 0], err, "γ", [f"{g:g}" for g in gammas]),
                      (axes[0, 1], err_p, "degree", [str(d) for d in degrees])]:
    im = ax.imshow(e, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(Cs)), [f"{c:g}" for c in Cs])
    ax.set_xlabel(xl)
    ax.set_ylabel("C")
    ax.grid(False)
    for (r, c_), v in np.ndenumerate(e):
        ax.text(c_, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v > 0.17 else "k")
axes[0, 0].set_title("validation error, RBF kernel")
axes[0, 1].set_title("validation error, polynomial kernel")
U.finish("grid", fig)
