"""Lesson 11 - A neural network from scratch: activations, forward pass, backpropagation, gradient check."""
import numpy as np

import mlutils as U

U.setup("11")

# --8<-- [start:activations]
ACT = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (s := 1 / (1 + np.exp(-z))) * (1 - s)),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "ReLU": (lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float)),
    "leaky ReLU": (lambda z: np.where(z > 0, z, 0.01 * z), lambda z: np.where(z > 0, 1, 0.01)),
}
z = np.linspace(-5, 5, 11)
for name, (f, df) in ACT.items():
    print(f"{name:10s} max derivative {df(np.linspace(-6, 6, 1001)).max():.2f} | f'(z) at z = -5, 0, 5: "
          f"{df(np.array([-5.0, 0.01, 5.0])).round(3)}")
# --8<-- [end:activations]

fig, axes = U.figure(2, 1, size=(4.0, 3.0))
zz = np.linspace(-5, 5, 400)
for (name, (f, df)), c in zip(ACT.items(), (U.BLUE, U.GREEN, U.RED, U.ORANGE)):
    axes[0, 0].plot(zz, f(zz), color=c, label=name)
    axes[0, 1].plot(zz, df(zz), color=c, label=name)
axes[0, 0].set_ylim(-1.5, 3)
axes[0, 0].set_title("activation f(z)")
axes[0, 1].set_title("derivative f'(z)")
axes[0, 0].legend(fontsize=7)
U.finish("activations", fig)

# --8<-- [start:vanishing]
# Backpropagation multiplies one derivative per layer. With sigmoids each factor is at most 0.25.
for depth in (2, 5, 10, 20):
    print(f"{depth:2d} sigmoid layers: gradient scaled by at most 0.25^{depth} = {0.25 ** depth:.1e}")
# --8<-- [end:vanishing]

# --8<-- [start:data]
train = np.loadtxt(U.path("mixtures/mixtureexampleTrain.csv"))
test = np.loadtxt(U.path("mixtures/mixtureexampleTest.csv"))
X, y = train[:, :2], (train[:, 2] > 0).astype(float)     # (1759, 2) float64, labels -1/+1 -> 0/1
Xt, yt = test[:, :2], (test[:, 2] > 0).astype(float)     # (5272, 2)
print("train", X.shape, "test", Xt.shape, "| positives: %.1f%%" % (100 * y.mean()))
# --8<-- [end:data]


# --8<-- [start:mlp]
class MLP:
    """2 -> H (ReLU) -> 1 (sigmoid), trained with binary cross-entropy. Plain NumPy."""

    def __init__(self, n_in, n_hidden, rng):
        # He initialisation: keeps the variance of the activations stable through ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))   # (2, H)
        self.b1 = np.zeros(n_hidden)                                    # (H,)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, 1))   # (H, 1)
        self.b2 = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1             # (n, H) pre-activations
        self.h = np.maximum(self.z1, 0)             # (n, H) hidden activations
        z2 = self.h @ self.W2 + self.b2             # (n, 1)
        self.p = 1 / (1 + np.exp(-z2[:, 0]))        # (n,) P(y = 1)
        return self.p

    def loss(self, y):
        eps = 1e-12
        return -np.mean(y * np.log(self.p + eps) + (1 - y) * np.log(1 - self.p + eps))

    def backward(self, y):
        """Chain rule, from the output back to the input. Returns the gradients in params() order."""
        n = len(y)
        dz2 = (self.p - y)[:, None] / n             # (n, 1)  dL/dz2: sigmoid + cross-entropy simplify
        dW2 = self.h.T @ dz2                        # (H, 1)
        db2 = dz2.sum(0)
        dh = dz2 @ self.W2.T                        # (n, H)  back through the second layer
        dz1 = dh * (self.z1 > 0)                    # (n, H)  back through ReLU: gate the gradient
        dW1 = self.X.T @ dz1                        # (2, H)
        db1 = dz1.sum(0)
        return [dW1, db1, dW2, db2]
# --8<-- [end:mlp]


net = MLP(2, 64, np.random.default_rng(0))
print("parameters: %d (2x64 + 64 + 64x1 + 1)" % sum(p.size for p in net.params()))

# --8<-- [start:gradcheck]
Xs, ys = X[:20], y[:20]
net.forward(Xs)
analytic = net.backward(ys)
h = 1e-6
worst = 0
for P, G in zip(net.params(), analytic):
    for idx in [tuple(np.random.randint(s) for s in P.shape) for _ in range(5)]:
        old = P[idx]
        P[idx] = old + h
        net.forward(Xs)
        lp = net.loss(ys)
        P[idx] = old - h
        net.forward(Xs)
        lm = net.loss(ys)
        P[idx] = old
        num = (lp - lm) / (2 * h)
        worst = max(worst, abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx])))
print("gradient check: largest relative difference %.1e" % worst)
# --8<-- [end:gradcheck]


# --8<-- [start:train]
def train_mlp(n_hidden, epochs=200, lr=0.1, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    net = MLP(2, n_hidden, rng)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(y))             # reshuffle every epoch
        for i in range(0, len(y), batch):
            idx = order[i:i + batch]
            net.forward(X[idx])
            for P, G in zip(net.params(), net.backward(y[idx])):
                P -= lr * G                         # in-place SGD step
        net.forward(X)
        tr_loss, tr_acc = net.loss(y), np.mean((net.p > 0.5) == y)
        net.forward(Xt)
        hist.append((tr_loss, tr_acc, net.loss(yt), np.mean((net.p > 0.5) == yt)))
    return net, np.array(hist)


for H in (1, 4, 64):
    net, hist = train_mlp(H)
    print(f"hidden units {H:2d}: train acc {hist[-1, 1]:.3f}, test acc {hist[-1, 3]:.3f}, test loss {hist[-1, 2]:.3f}")
# --8<-- [end:train]

fig, axes = U.figure(4, 1, size=(3.1, 3.0))
for ax, H in zip(axes[0, :3], (1, 4, 64)):
    net, hist = train_mlp(H)
    U.plot_decision(ax, lambda g: net.forward(g), X, y.astype(int), proba=True, resolution=200,
                    title=f"{H} hidden unit{'s' if H > 1 else ''}: test acc {hist[-1, 3]:.3f}")
ax = axes[0, 3]
ax.plot(hist[:, 0], color=U.BLUE, label="train loss")
ax.plot(hist[:, 2], color=U.RED, label="test loss")
ax.set_xlabel("epoch")
ax.set_title("64 hidden units")
ax.legend()
U.finish("mlp", fig)
