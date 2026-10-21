"""Lesson 03 - Optimisation: gradient descent, Armijo line search, Newton, duality with CVXPY."""
import cvxpy as cp
import numpy as np

import mlutils as U

U.setup("03")


# --8<-- [start:rosenbrock]
def J(t):
    """Rosenbrock's banana function: minimum J = 0 at (1, 1), at the bottom of a curved valley."""
    return (1 - t[0]) ** 2 + 100 * (t[1] - t[0] ** 2) ** 2


def grad(t):
    return np.array([-2 * (1 - t[0]) - 400 * t[0] * (t[1] - t[0] ** 2),
                     200 * (t[1] - t[0] ** 2)])


def hessian(t):
    return np.array([[2 - 400 * t[1] + 1200 * t[0] ** 2, -400 * t[0]],
                     [-400 * t[0], 200]])
# --8<-- [end:rosenbrock]


# --8<-- [start:check-grad]
def numerical_grad(f, t, h=1e-6):
    """Central finite differences: slow, but a reliable check of a hand-written gradient."""
    e = np.eye(len(t))
    return np.array([(f(t + h * e[i]) - f(t - h * e[i])) / (2 * h) for i in range(len(t))])


t = np.array([-0.7, 0.4])
print("analytic gradient", grad(t), "| numerical", numerical_grad(J, t).round(6))
print("analytic Hessian row 0", hessian(t)[0], "| numerical",
      numerical_grad(lambda s: grad(s)[0], t).round(4))
# --8<-- [end:check-grad]


# --8<-- [start:descent]
def armijo(theta, d, g, alpha=1.0, rho=0.5, c=1e-4):
    """Backtracking: halve the step until J decreases at least c * alpha * (slope along d)."""
    while J(theta + alpha * d) > J(theta) + c * alpha * g @ d:
        alpha *= rho
    return alpha


def minimise(theta0, method, tol=1e-3, max_iter=20000):
    theta = theta0.astype(float).copy()
    path = [theta.copy()]
    for it in range(max_iter):
        g = grad(theta)
        if np.linalg.norm(g) <= tol:
            break
        if method == "fixed":
            theta = theta - 1e-3 * g                         # fixed step size
        elif method == "armijo":
            theta = theta + armijo(theta, -g, g) * -g        # steepest descent + line search
        elif method == "newton":
            theta = theta - np.linalg.solve(hessian(theta), g)  # jump to the minimum of the local quadratic
        path.append(theta.copy())
    return theta, np.array(path), it


theta0 = np.array([-1.0, 0.0])
results = {}
for method in ("fixed", "armijo", "newton"):
    theta, path, it = minimise(theta0, method)
    results[method] = path
    print(f"{method:6s}: {it:5d} iterations -> theta = {np.round(theta, 4)}, J = {J(theta):.2e}")
# --8<-- [end:descent]

fig, axes = U.figure(2, 1, size=(4.6, 3.5))
ax = axes[0, 0]
g1, g2 = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-3.2, 1.6, 300))
ax.contour(g1, g2, J([g1, g2]), levels=np.geomspace(0.05, 300, 16), cmap="viridis", linewidths=0.7)
for (m, path), c in zip(results.items(), (U.RED, U.ORANGE, U.BLUE)):
    step = max(1, len(path) // 60)
    ax.plot(path[::step, 0], path[::step, 1], ".-", color=c, ms=3, lw=1, label=f"{m} ({len(path) - 1} it.)")
ax.plot(1, 1, "k*", ms=11)
ax.set_title("paths on Rosenbrock's function")
ax.legend(fontsize=7)
ax = axes[0, 1]
for (m, path), c in zip(results.items(), (U.RED, U.ORANGE, U.BLUE)):
    ax.loglog(np.arange(1, len(path) + 1), [J(p) + 1e-16 for p in path], color=c, label=m)
ax.set_xlabel("iteration")
ax.set_ylabel("J(θ)")
ax.set_title("convergence")
ax.legend()
U.finish("descent", fig)

# --8<-- [start:primal]
# Project c onto the diamond |t1| + |t2| <= 1:  min 1/2 ||t - c||^2  s.t.  A t <= b
c = np.array([3.0, 1.0])
A = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
b = np.ones(4)

t = cp.Variable(2)
primal = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(t - c)), [A @ t <= b])
primal.solve()
print("primal: t* =", np.round(t.value, 4), "objective", round(primal.value, 4))
# --8<-- [end:primal]

# --8<-- [start:dual]
# Lagrangian L(t, mu) = 1/2||t - c||^2 + mu^T (A t - b), mu >= 0.
# dL/dt = 0  ->  t = c - A^T mu.  Plugging it back gives the dual problem in mu only:
mu = cp.Variable(4)
dual = cp.Problem(cp.Minimize(0.5 * cp.quad_form(mu, A @ A.T) - (A @ c - b) @ mu), [mu >= 0])
dual.solve()
t_from_dual = c - A.T @ mu.value
print("dual: mu* =", np.round(mu.value, 4), "-> t = c - A^T mu =", np.round(t_from_dual, 4))
print("duality gap:", round(primal.value + dual.value, 6),
      "| multipliers from the primal solver:", np.round(primal.constraints[0].dual_value, 4))
# --8<-- [end:dual]

# --8<-- [start:meb]
houses = np.loadtxt(U.path("houses.csv"), delimiter=",")     # (18, 2) coordinates


def enclosing_ball(X):
    """Minimum enclosing ball through its dual: max sum mu_i ||x_i||^2 - ||X^T mu||^2, mu in the simplex."""
    mu = cp.Variable(len(X))
    q = (X ** 2).sum(1)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(X.T @ mu) - q @ mu), [mu >= 0, cp.sum(mu) == 1])
    prob.solve()
    mu = mu.value
    z = X.T @ mu                                   # centre = weighted mean of the support points
    support = mu > 1e-4
    r = np.sqrt(((X[support] - z) ** 2).sum(1).max())
    return z, r, support


z, r, sup = enclosing_ball(houses)
print("houses: centre %s, radius %.4f, %d support points" % (np.round(z, 4), r, sup.sum()))
outliers = np.array([[3.0, -3.0], [2.5, 2.5]])
z2, r2, sup2 = enclosing_ball(np.vstack([houses, outliers]))
print("with 2 outliers: centre %s, radius %.4f, %d support points" % (np.round(z2, 4), r2, sup2.sum()))
# --8<-- [end:meb]

# --8<-- [start:soft]
def soft_ball(X, C):
    """Allow points outside the ball at a price: min R^2 + C sum(xi), ||x_i - z||^2 <= R^2 + xi."""
    z, R2, xi = cp.Variable(2), cp.Variable(), cp.Variable(len(X))
    cons = [cp.sum_squares(X[i] - z) <= R2 + xi[i] for i in range(len(X))] + [xi >= 0]
    cp.Problem(cp.Minimize(R2 + C * cp.sum(xi)), cons).solve()
    return z.value, np.sqrt(R2.value), xi.value > 1e-6


Xall = np.vstack([houses, outliers])
z3, r3, out3 = soft_ball(Xall, C=0.3)
print("soft ball (C = 0.3): centre %s, radius %.4f, points left outside: %s"
      % (np.round(z3, 4), r3, np.flatnonzero(out3)))
# --8<-- [end:soft]

fig, axes = U.figure(2, 1, size=(3.8, 3.6))
tt = np.linspace(0, 2 * np.pi, 200)
ax = axes[0, 0]
ax.fill([1, 0, -1, 0], [0, 1, 0, -1], color=U.BLUE, alpha=0.15)
ax.plot(*c, "o", color=U.RED)
ax.annotate("c", c, textcoords="offset points", xytext=(5, 5))
ax.plot(*t.value, "o", color=U.BLUE)
ax.annotate("t*", t.value, textcoords="offset points", xytext=(5, -12))
ax.plot([c[0], t.value[0]], [c[1], t.value[1]], "--", color=U.GRAY)
for r_ in np.linspace(0.5, 3, 6):
    ax.plot(c[0] + r_ * np.cos(tt), c[1] + r_ * np.sin(tt), color=U.GRAY, lw=0.5)
ax.set_aspect("equal")
ax.set_xlim(-1.5, 3.5)
ax.set_ylim(-1.5, 2.5)
ax.set_title("projection onto a polytope")
ax = axes[0, 1]
ax.plot(*houses.T, "s", color=U.BLUE, ms=5, label="houses")
ax.plot(*outliers.T, "x", color="k", ms=8, mew=2, label="outliers")
for (zc, rc), col, lab in [((z, r), U.BLUE, "ball (houses)"), ((z2, r2), U.RED, "ball (+ outliers)"),
                           ((z3, r3), U.GREEN, "soft ball, C = 0.3")]:
    ax.plot(zc[0] + rc * np.cos(tt), zc[1] + rc * np.sin(tt), color=col, label=lab)
    ax.plot(*zc, "+", color=col, ms=9)
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="lower left")
ax.set_title("minimum enclosing ball")
U.finish("constrained", fig)
