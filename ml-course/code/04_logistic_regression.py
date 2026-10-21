"""Lesson 04 - Logistic regression: from scratch, spam filtering, and multi-class softmax."""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlutils as U

U.setup("04")


# --8<-- [start:data]
def two_gaussians(n_per_class):
    """The lab's synthetic data: two 2-D Gaussian clouds with different covariances."""
    mu1, S1 = np.array([0, 2]), 1.5 * np.array([[1, 0.1], [0.1, 1]])
    mu2, S2 = np.array([-2, -2]), 3.5 * np.array([[1, -0.25], [-0.25, 0.5]])
    X = np.vstack([np.random.multivariate_normal(mu1, S1, n_per_class),
                   np.random.multivariate_normal(mu2, S2, n_per_class)])
    y = np.r_[np.ones(n_per_class), np.zeros(n_per_class)].astype(int)
    return X, y


X, y = two_gaussians(100)                          # X (200, 2), y (200,) in {0, 1}
# --8<-- [end:data]


# --8<-- [start:scratch]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_logistic(X, y, lr=0.1, epochs=5000):
    """Minimise the mean cross-entropy by gradient descent. Returns w (d,), b and the loss history."""
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    for _ in range(epochs):
        p = sigmoid(X @ w + b)                     # (n,) predicted P(y = 1 | x)
        eps = 1e-12
        losses.append(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        w -= lr * X.T @ (p - y) / len(y)           # the gradient is (prediction - label) * input
        b -= lr * np.mean(p - y)
    return w, b, np.array(losses)


w, b, losses = fit_logistic(X, y)
ref = LogisticRegression(C=np.inf).fit(X, y)     # C = inf: no regularisation, like ours
print("from scratch: w = %s, b = %.4f, loss %.4f" % (np.round(w, 4), b, losses[-1]))
print("scikit-learn: w = %s, b = %.4f" % (np.round(ref.coef_[0], 4), ref.intercept_[0]))
print("training accuracy %.3f" % np.mean((sigmoid(X @ w + b) > 0.5) == y))
# --8<-- [end:scratch]

fig, axes = U.figure(3, 1, size=(3.5, 3.2))
z = np.linspace(-8, 8, 200)
axes[0, 0].plot(z, sigmoid(z), color=U.BLUE)
axes[0, 0].axhline(0.5, color=U.GRAY, ls="--", lw=1)
axes[0, 0].set_title("sigmoid σ(z) = 1 / (1 + e⁻ᶻ)")
axes[0, 0].set_xlabel("z = wᵀx + b")
U.plot_decision(axes[0, 1], lambda g: sigmoid(g @ w + b), X, y, proba=True, title="P(class 1 | x)")
axes[0, 2].plot(losses, color=U.RED)
axes[0, 2].set_xscale("log")
axes[0, 2].set_xlabel("epoch")
axes[0, 2].set_title("cross-entropy loss")
U.finish("synthetic", fig)

# --8<-- [start:spam]
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/"
data = np.loadtxt(U.fetch(BASE + "spambase.data", "spambase.data"), delimiter=",")
names = [line.split(":")[0] for line in U.fetch(BASE + "spambase.names", "spambase.names").read_text()
         .splitlines() if line.endswith("continuous.")]                    # the 57 feature names
Xs, ys = data[:, :57], data[:, 57].astype(int)     # (4601, 57) word/char frequencies + capital-run lengths
print("%d e-mails: %d ham, %d spam; feature max values from %.2f to %.0f"
      % (len(ys), (ys == 0).sum(), ys.sum(), Xs.max(0).min(), Xs.max(0).max()))
Xa, Xt, ya, yt = train_test_split(Xs, ys, test_size=1 / 3, stratify=ys, random_state=0)
# --8<-- [end:spam]


# --8<-- [start:spam-models]
def evaluate(name, train, test, C=1.0):
    clf = LogisticRegression(C=C, max_iter=5000).fit(train, ya)
    print(f"{name:13s} train acc {clf.score(train, ya):.4f}  test acc {clf.score(test, yt):.4f}")
    return clf


evaluate("raw", Xa, Xt)
sc = StandardScaler().fit(Xa)
clf_std = evaluate("standardised", sc.transform(Xa), sc.transform(Xt))
eps = 0.01
sc_log = StandardScaler().fit(np.log(Xa + eps))
clf_log = evaluate("log + std", sc_log.transform(np.log(Xa + eps)), sc_log.transform(np.log(Xt + eps)))
# --8<-- [end:spam-models]

# --8<-- [start:coefs]
coef = clf_log.coef_[0]                             # one weight per standardised log-feature
order = np.argsort(-np.abs(coef))
for i in order[:10]:
    print(f"{names[i]:28s} {coef[i]:+.2f}   (odds x{np.exp(coef[i]):.2f} per std)")
# --8<-- [end:coefs]

fig, axes = U.figure(2, 1, size=(4.2, 3.3))
for k, (ax, title, xs) in enumerate([(axes[0, 0], "word_freq_remove, raw", Xs), (axes[0, 1], "log(x + 0.01)",
                                                                                np.log(Xs + eps))]):
    j = names.index("word_freq_remove")
    ax.hist(xs[ys == 0, j], bins=40, color=U.BLUE, alpha=0.6, label="ham")
    ax.hist(xs[ys == 1, j], bins=40, color=U.RED, alpha=0.6, label="spam")
    ax.set_yscale("log")
    ax.set_title(title)
axes[0, 0].legend()
U.finish("spam-hist", fig)

fig, axes = U.figure(1, 1, size=(6.5, 3.2))
top = order[:15]
axes[0, 0].barh(range(15), coef[top][::-1], color=[U.RED if c > 0 else U.BLUE for c in coef[top][::-1]])
axes[0, 0].set_yticks(range(15), [names[i] for i in top][::-1], fontsize=7)
axes[0, 0].set_xlabel("weight (standardised log-features): red pushes towards spam")
U.finish("spam-coefs", fig)

# --8<-- [start:softmax]
x_train, y_train, x_test, y_test = U.load_mnist()
n_train = 10000                                    # a subset keeps this under a minute on a CPU
Xm = x_train[:n_train].reshape(n_train, -1) / 255.0   # (10000, 784) float64 in [0, 1]
Xm_test = x_test.reshape(len(x_test), -1) / 255.0
t0 = time.time()
soft = LogisticRegression(C=0.05, max_iter=300).fit(Xm, y_train[:n_train])   # multinomial (softmax)
print("softmax regression on %d MNIST digits: train acc %.4f, test acc %.4f (%.0f s)"
      % (n_train, soft.score(Xm, y_train[:n_train]), soft.score(Xm_test, y_test), time.time() - t0))
print("weights:", soft.coef_.shape)                # (10 classes, 784 pixels)
# --8<-- [end:softmax]

fig, axes = U.figure(10, 1, size=(1.25, 1.45))
lim = np.abs(soft.coef_).max()
for d, ax in enumerate(axes[0]):
    ax.imshow(soft.coef_[d].reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(str(d))
    ax.axis("off")
U.finish("softmax-weights", fig)
