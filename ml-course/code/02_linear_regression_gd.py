"""Lesson 02 - Linear regression: closed form, gradient descent, scaling and overfitting."""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import mlutils as U

U.setup("02")

# --8<-- [start:data]
n = 100
x = 2 * np.random.rand(n)                         # (100,) inputs in [0, 2]
y = 4 + 3 * x + np.random.normal(0, 0.5, n)       # true line w=3, b=4, plus Gaussian noise
# --8<-- [end:data]

# --8<-- [start:closed]
# Simple regression: slope = cov(x, y) / var(x), intercept from the means
w_cf = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b_cf = y.mean() - w_cf * x.mean()

# Same thing in matrix form: add a column of ones and solve the least-squares problem
A = np.c_[x, np.ones(n)]                          # (100, 2) design matrix
theta, *_ = np.linalg.lstsq(A, y, rcond=None)
print("closed form: w = %.4f, b = %.4f | lstsq: %s" % (w_cf, b_cf, np.round(theta, 4)))
# --8<-- [end:closed]

# --8<-- [start:rmse]
y_hat = w_cf * x + b_cf
rmse = np.sqrt(np.mean((y - y_hat) ** 2))
rmse_wrong = np.sqrt(np.sum(y - y_hat) ** 2 / n)  # square of the sum, not sum of the squares
print("RMSE = %.4f, 'RMSE' with the misplaced parenthesis = %.2e" % (rmse, rmse_wrong))
# --8<-- [end:rmse]


# --8<-- [start:gd]
def gradient_descent_regression(x, y, lr, epochs):
    """Fit y = w x + b by minimising the mean squared error with batch gradient descent."""
    w, b = 0.0, 0.0
    history = []                                   # (w, b, cost) after each epoch
    for _ in range(epochs):
        err = (w * x + b) - y                      # (n,) residuals
        cost = np.mean(err ** 2)
        grad_w = 2 * np.mean(err * x)              # dJ/dw
        grad_b = 2 * np.mean(err)                  # dJ/db
        w -= lr * grad_w
        b -= lr * grad_b
        history.append((w, b, cost))
    return w, b, np.array(history)


w, b, hist = gradient_descent_regression(x, y, lr=0.1, epochs=500)
print("gradient descent: w = %.4f, b = %.4f, final cost %.4f" % (w, b, hist[-1, 2]))
# --8<-- [end:gd]

# the line at a few epochs, like frames of an animation
fig, axes = U.figure(4, 1, size=(2.9, 2.7))
for ax, ep in zip(axes[0], (0, 5, 30, 499)):
    wi, bi, ci = hist[ep]
    ax.scatter(x, y, s=8, color=U.GRAY)
    xs = np.array([0, 2])
    ax.plot(xs, wi * xs + bi, color=U.RED, lw=2)
    ax.set_title(f"epoch {ep + 1}: J = {ci:.2f}")
    ax.set_ylim(0, 12)
U.finish("frames", fig)

# --8<-- [start:lr]
H = 2 * A.T @ A / n                                # Hessian of J(w, b): constant for a quadratic
eig = np.linalg.eigvalsh(H)
print("Hessian eigenvalues %s -> gradient descent diverges for lr > 2/%.2f = %.3f"
      % (np.round(eig, 3), eig.max(), 2 / eig.max()))
for lr in (0.01, 0.1, 0.45, 0.5):
    _, _, h = gradient_descent_regression(x, y, lr=lr, epochs=100)
    print(f"lr = {lr:<4}: cost after 100 epochs = {h[-1, 2]:.4g}")
# --8<-- [end:lr]

fig, axes = U.figure(2, 1, size=(4.2, 3.2))
for lr, c in zip((0.01, 0.1, 0.45, 0.5), (U.GRAY, U.BLUE, U.GREEN, U.RED)):
    _, _, h = gradient_descent_regression(x, y, lr=lr, epochs=100)
    axes[0, 0].semilogy(h[:, 2], color=c, label=f"lr = {lr}")
axes[0, 0].set_ylim(0.1, 1e4)
axes[0, 0].set_xlabel("epoch")
axes[0, 0].set_ylabel("cost J (log)")
axes[0, 0].legend()
axes[0, 0].set_title("learning rate")
# cost surface and the path of gradient descent
ws, bs = np.meshgrid(np.linspace(0, 6, 200), np.linspace(0, 6, 200))
J = ((ws[..., None] * x + bs[..., None] - y) ** 2).mean(-1)
axes[0, 1].contour(ws, bs, J, levels=np.geomspace(0.2, 40, 14), cmap="viridis", linewidths=0.8)
_, _, h = gradient_descent_regression(x, y, lr=0.1, epochs=200)
axes[0, 1].plot(np.r_[0, h[:, 0]], np.r_[0, h[:, 1]], ".-", color=U.RED, ms=3, lw=1)
axes[0, 1].plot(w_cf, b_cf, "k*", ms=10)
axes[0, 1].set_xlabel("w")
axes[0, 1].set_ylabel("b")
axes[0, 1].set_title("path of gradient descent on J(w, b)")
U.finish("learning-rate", fig)

# --8<-- [start:housing]
housing = fetch_california_housing(data_home=str(U.CACHE))
Xh, yh = housing.data, housing.target              # (20640, 8), median house value in $100k
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.25, random_state=0)


def gd_linear(X, y, lr, epochs):
    """Multivariate version: theta = (w_1..w_d, b), gradient in matrix form."""
    A = np.c_[X, np.ones(len(X))]
    theta = np.zeros(A.shape[1])
    costs = []
    for _ in range(epochs):
        err = A @ theta - y
        costs.append(np.mean(err ** 2))
        theta -= lr * 2 * A.T @ err / len(y)
    return theta, np.array(costs)


ref = LinearRegression().fit(Xh_tr, yh_tr)
ref_mse = np.mean((ref.predict(Xh_tr) - yh_tr) ** 2)
sc = StandardScaler().fit(Xh_tr)
_, cost_raw = gd_linear(Xh_tr, yh_tr, lr=1e-7, epochs=300)   # larger lr diverges on raw data
_, cost_std = gd_linear(sc.transform(Xh_tr), yh_tr, lr=0.1, epochs=300)
print("optimal training MSE %.4f | GD raw features: %.4f | GD standardised: %.4f"
      % (ref_mse, cost_raw[-1], cost_std[-1]))
print("feature std range: %.2g .. %.2g" % (Xh_tr.std(0).min(), Xh_tr.std(0).max()))
# --8<-- [end:housing]

fig, axes = U.figure(1, 1, size=(4.6, 3.2))
ax = axes[0, 0]
ax.semilogy(cost_raw, color=U.RED, label="raw features (lr = 1e-7)")
ax.semilogy(cost_std, color=U.BLUE, label="standardised (lr = 0.1)")
ax.axhline(ref_mse, color="k", ls="--", lw=1, label="least-squares optimum")
ax.set_xlabel("epoch")
ax.set_ylabel("training MSE")
ax.set_title("California housing: GD with and without scaling")
ax.legend()
U.finish("scaling", fig)

# --8<-- [start:poly]
xs = np.sort(np.random.rand(15))                   # only 15 training points
f = lambda t: np.sin(2 * np.pi * t)
ys = f(xs) + np.random.normal(0, 0.2, 15)
xv = np.random.rand(200)
yv = f(xv) + np.random.normal(0, 0.2, 200)


def fit_poly(deg, alpha=0.0):
    model = Ridge(alpha=alpha) if alpha else LinearRegression()
    feats = PolynomialFeatures(deg)
    model.fit(feats.fit_transform(xs[:, None]), ys)
    return lambda t: model.predict(feats.transform(np.asarray(t)[:, None]))


for deg, alpha in [(1, 0), (3, 0), (14, 0), (14, 1e-3)]:
    p = fit_poly(deg, alpha)
    print(f"degree {deg:2d}, ridge alpha {alpha:g}: train MSE {np.mean((p(xs) - ys) ** 2):.4f}, "
          f"validation MSE {np.mean((p(xv) - yv) ** 2):.4f}")
# --8<-- [end:poly]

fig, axes = U.figure(4, 1, size=(2.9, 2.6))
t = np.linspace(0, 1, 300)
for ax, (deg, alpha) in zip(axes[0], [(1, 0), (3, 0), (14, 0), (14, 1e-3)]):
    p = fit_poly(deg, alpha)
    ax.plot(t, f(t), color=U.GRAY, ls="--", lw=1)
    ax.plot(t, p(t), color=U.RED, lw=1.8)
    ax.scatter(xs, ys, s=14, color=U.BLUE, zorder=3)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(f"degree {deg}" + (f", ridge {alpha:g}" if alpha else ""))
U.finish("polynomial", fig)
