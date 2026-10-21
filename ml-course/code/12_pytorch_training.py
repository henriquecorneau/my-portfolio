"""Lesson 12 - PyTorch: tensors, autograd, nn.Module, the training loop, optimisers, checkpoints."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import mlutils as U

U.setup("12")

# --8<-- [start:tensors]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.arange(12, dtype=torch.float32).reshape(3, 4)   # (3, 4) float32 on the CPU
b = torch.from_numpy(np.ones(4, dtype=np.float32))        # shares memory with the NumPy array
c = a @ b                                                  # (3,) matrix-vector product
print(a.shape, a.dtype, a.device, "| a @ b =", c.tolist())
print("broadcasting: (3, 4) + (4,) ->", tuple((a + b).shape), "| (3, 1) * (1, 4) ->",
      tuple((a[:, :1] * a[:1, :]).shape))
print("training on:", device)
# --8<-- [end:tensors]

# --8<-- [start:timing]
for n in (256, 1024, 2048):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    At, Bt = torch.from_numpy(A), torch.from_numpy(B)
    At @ Bt                                                # warm-up: the first call pays one-off costs
    t0 = time.perf_counter()
    for _ in range(5):
        A @ B
    t_np = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        At @ Bt
    if device.type == "cuda":
        torch.cuda.synchronize()                           # GPU calls are asynchronous: wait before timing
    t_pt = (time.perf_counter() - t0) / 5
    print(f"{n:5d} x {n}: NumPy {1e3 * t_np:7.2f} ms | PyTorch CPU {1e3 * t_pt:7.2f} ms "
          f"| {2 * n ** 3 / t_pt / 1e9:5.0f} GFLOP/s")
# --8<-- [end:timing]

# --8<-- [start:autograd]
x = torch.tensor([[1.0, 2.0]])                            # (1, 2) one sample
W = torch.tensor([[0.5], [-1.0]], requires_grad=True)     # (2, 1) a parameter: track its gradient
bias = torch.tensor([0.1], requires_grad=True)
p = torch.sigmoid(x @ W + bias)                           # forward pass builds a graph of operations
loss = nn.functional.binary_cross_entropy(p, torch.tensor([[1.0]]))
loss.backward()                                           # backward pass fills .grad of every leaf
print("p = %.4f | dL/dW = %s | by hand (p - y) x = %s"
      % (p.item(), W.grad.ravel().tolist(), ((p.item() - 1) * x).ravel().tolist()))
# --8<-- [end:autograd]

# --8<-- [start:data]
train = np.loadtxt(U.path("mixtures/mixtureexampleTrain.csv"), dtype=np.float32)
test = np.loadtxt(U.path("mixtures/mixtureexampleTest.csv"), dtype=np.float32)
X, y = torch.from_numpy(train[:, :2]), torch.from_numpy((train[:, 2] > 0).astype(np.float32))
Xt, yt = torch.from_numpy(test[:, :2]), torch.from_numpy((test[:, 2] > 0).astype(np.float32))
# --8<-- [end:data]


# --8<-- [start:model]
@dataclass
class Config:
    hidden: int = 64
    lr: float = 0.1
    batch_size: int = 32
    epochs: int = 30
    optimizer: str = "sgd"          # "sgd", "momentum" or "adam"
    seed: int = 0


def make_mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(name)
# --8<-- [end:model]


# --8<-- [start:loop]
def fit(cfg, X, y, Xv, yv, loss_fn, n_out=1, patience=None):
    """The canonical PyTorch training loop. Returns the model and per-epoch (train loss, val loss, val acc).

    With patience=k, stop when the validation loss has not improved for k epochs and
    return the model with the best weights seen (early stopping + checkpointing)."""
    torch.manual_seed(cfg.seed)
    model = make_mlp(X.shape[1], cfg.hidden, n_out).to(device)
    opt = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True)
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).squeeze(-1) if n_out == 1 else model(xb)   # (B,) logits or (B, 10)
            loss = loss_fn(out, yb)
            opt.zero_grad()                       # gradients accumulate by default: reset them
            loss.backward()                       # backpropagation
            opt.step()                            # parameter update
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():                     # no graph needed for evaluation
            out = model(Xv.to(device))
            out = out.squeeze(-1) if n_out == 1 else out
            vloss = loss_fn(out, yv.to(device)).item()
            pred = (out > 0).float() if n_out == 1 else out.argmax(1)
            vacc = (pred.cpu() == yv).float().mean().item()
        history.append((total / len(X), vloss, vacc))
        if patience is not None:
            if vloss < best_loss:
                best_loss, best_state = vloss, copy.deepcopy(model.state_dict())
            elif epoch - int(np.argmin([h[1] for h in history])) >= patience:
                print(f"early stopping after epoch {epoch + 1}: no improvement for {patience} epochs")
                break
    if best_state is not None:
        model.load_state_dict(best_state)            # roll back to the best checkpoint
    return model, np.array(history)
# --8<-- [end:loop]

# --8<-- [start:optimisers]
bce = nn.BCEWithLogitsLoss()        # sigmoid + binary cross-entropy in one numerically stable step
runs = {}
for name, cfg in {
    "SGD, batch 32": Config(optimizer="sgd", lr=0.1),
    "SGD, batch 256": Config(optimizer="sgd", lr=0.1, batch_size=256),
    "SGD + momentum": Config(optimizer="momentum", lr=0.1),
    "Adam": Config(optimizer="adam", lr=0.01),
}.items():
    _, runs[name] = fit(cfg, X, y, Xt, yt, bce)
    print(f"{name:15s}: test acc after 3 epochs {runs[name][2, 2]:.3f}, after 30 {runs[name][-1, 2]:.3f}")
# --8<-- [end:optimisers]

fig, axes = U.figure(2, 1, size=(4.2, 3.1))
for (name, h), c in zip(runs.items(), (U.BLUE, U.GRAY, U.GREEN, U.RED)):
    axes[0, 0].plot(range(1, 31), h[:, 0], color=c, label=name)
    axes[0, 1].plot(range(1, 31), h[:, 2], color=c, label=name)
axes[0, 0].set_title("training loss")
axes[0, 1].set_title("test accuracy")
for ax in axes[0]:
    ax.set_xlabel("epoch")
axes[0, 0].legend(fontsize=7)
U.finish("optimisers", fig)

# --8<-- [start:mnist]
x_train, y_train, x_test, y_test = U.load_mnist()
Xm = torch.from_numpy(x_train.reshape(-1, 784)).float()      # (60000, 784) float32, values 0..255
ym = torch.from_numpy(y_train)                                # (60000,) int64 class indices
Xm_test = torch.from_numpy(x_test.reshape(-1, 784)).float()
ym_test = torch.from_numpy(y_test)
ce = nn.CrossEntropyLoss()          # softmax + categorical cross-entropy, takes raw logits
cfg = Config(hidden=800, lr=0.01, batch_size=32, epochs=3)
for name, scale in (("raw pixels 0-255", 1.0), ("pixels / 255", 255.0)):
    _, h = fit(cfg, Xm / scale, ym, Xm_test / scale, ym_test, ce, n_out=10)
    print(f"{name:16s}: train loss per epoch {np.round(h[:, 0], 3).tolist()}, test acc {h[-1, 2]:.4f}")
# --8<-- [end:mnist]

# --8<-- [start:early-stopping]
perm = torch.randperm(60000)
tr, va = perm[:50000], perm[50000:]                 # hold out 10,000 training images for validation
Xs = Xm / 255.0
cfg = Config(hidden=800, optimizer="adam", lr=1e-3, batch_size=128, epochs=30)
model, curve = fit(cfg, Xs[tr], ym[tr], Xs[va], ym[va], ce, n_out=10, patience=3)

path = U.CACHE / "mlp_mnist_best.pt"
torch.save(model.state_dict(), path)                # a checkpoint is just the dict of tensors
reloaded = make_mlp(784, cfg.hidden, 10)
reloaded.load_state_dict(torch.load(path))
reloaded.eval()
with torch.no_grad():
    same = torch.equal(model(Xs[:1000]), reloaded(Xs[:1000]))
    acc = (reloaded(Xm_test / 255.0).argmax(1) == ym_test).float().mean().item()
print("best validation loss %.4f at epoch %d | reloaded model identical: %s | test accuracy %.4f"
      % (curve[:, 1].min(), curve[:, 1].argmin() + 1, same, acc))
# --8<-- [end:early-stopping]

fig, axes = U.figure(1, 1, size=(4.6, 3.1))
c = curve
axes[0, 0].plot(range(1, len(c) + 1), c[:, 0], "o-", ms=3, color=U.BLUE, label="training loss")
axes[0, 0].plot(range(1, len(c) + 1), c[:, 1], "o-", ms=3, color=U.RED, label="validation loss")
axes[0, 0].axvline(np.argmin(c[:, 1]) + 1, color=U.GRAY, ls="--", lw=1, label="best checkpoint")
axes[0, 0].set_xlabel("epoch")
axes[0, 0].set_title("MLP 784-800-10 on MNIST, Adam")
axes[0, 0].legend()
U.finish("early-stopping", fig)
