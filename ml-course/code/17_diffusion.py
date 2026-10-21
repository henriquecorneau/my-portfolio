"""Lesson 17 - Diffusion models from scratch: a DDPM that learns to generate a 2-D distribution."""
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn

import mlutils as U

U.setup("17")


# --8<-- [start:config]
@dataclass
class Config:
    T: int = 100                  # number of noise levels
    beta_start: float = 1e-4
    beta_end: float = 0.2
    steps: int = 6000             # training iterations
    batch: int = 256
    lr: float = 2e-3
    hidden: int = 128


cfg = Config()
# --8<-- [end:config]

# --8<-- [start:data]
data, _ = make_moons(n_samples=10000, noise=0.05, random_state=0)
x0_all = torch.tensor((data - data.mean(0)) / data.std(0), dtype=torch.float32)   # (10000, 2), zero mean, unit std
# --8<-- [end:data]

# --8<-- [start:schedule]
betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)     # (T,) noise added at each step
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, 0)                             # (T,) fraction of signal left after t steps


def q_sample(x0, t, noise):
    """Jump straight to step t: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) noise."""
    ab = alpha_bar[t][:, None]                                   # (B, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


print("signal left: t=0 %.3f, t=T/4 %.3f, t=T/2 %.3f, t=T %.4f"
      % tuple(alpha_bar[[0, cfg.T // 4, cfg.T // 2, cfg.T - 1]].sqrt().tolist()))
# --8<-- [end:schedule]


# --8<-- [start:model]
class Denoiser(nn.Module):
    """eps_theta(x_t, t): predicts the noise that was added. t enters as sinusoidal features."""

    def __init__(self, hidden):
        super().__init__()
        self.freqs = torch.exp(torch.linspace(0, np.log(100), 16))          # (16,)
        self.net = nn.Sequential(nn.Linear(2 + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        tt = (t.float() / cfg.T)[:, None] * self.freqs                      # (B, 16)
        return self.net(torch.cat([x, tt.sin(), tt.cos()], dim=1))          # (B, 2)
# --8<-- [end:model]


# --8<-- [start:train]
model = Denoiser(cfg.hidden)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
losses = []
for step in range(cfg.steps):
    x0 = x0_all[torch.randint(0, len(x0_all), (cfg.batch,))]                # (B, 2) clean samples
    t = torch.randint(0, cfg.T, (cfg.batch,))                               # a random noise level each
    noise = torch.randn_like(x0)
    loss = nn.functional.mse_loss(model(q_sample(x0, t, noise), t), noise)  # learn to predict the noise
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
print("training loss: first 100 steps %.3f, last 500 steps %.3f" % (np.mean(losses[:100]), np.mean(losses[-500:])))
# --8<-- [end:train]


# --8<-- [start:sample]
@torch.no_grad()
def sample(n, keep=(99, 75, 50, 25, 10, 0)):
    """Ancestral sampling: start from pure noise and remove a little of the predicted noise T times."""
    x = torch.randn(n, 2)
    snapshots = {}
    for t in reversed(range(cfg.T)):
        tt = torch.full((n,), t)
        eps = model(x, tt)
        mean = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
        if t in keep:
            snapshots[t] = x.clone()
    return x, snapshots


gen, snaps = sample(2000)
real = x0_all[:2000]


def mmd(a, b, s=0.5):
    """Maximum mean discrepancy with a Gaussian kernel: 0 when two sample sets have the same distribution."""
    k = lambda x, y: torch.exp(-torch.cdist(x, y) ** 2 / (2 * s ** 2)).mean()
    return (k(a, a) + k(b, b) - 2 * k(a, b)).item()


print("MMD generated vs real: %.4f | pure noise vs real: %.4f | two real halves: %.4f"
      % (mmd(gen, real), mmd(torch.randn(2000, 2), real), mmd(x0_all[2000:4000], real)))
# --8<-- [end:sample]

fig, axes = U.figure(5, 1, size=(2.6, 2.6))
for ax, t in zip(axes[0], (0, 10, 25, 50, 99)):
    xt = q_sample(real, torch.full((2000,), t), torch.randn_like(real))
    ax.scatter(*xt.T, s=1, color=U.BLUE)
    ax.set_title(f"forward: t = {t}")
for ax in axes[0]:
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("forward", fig)

fig, axes = U.figure(5, 1, size=(2.6, 2.6))
for ax, t in zip(axes[0], (99, 50, 25, 10, 0)):
    ax.scatter(*snaps[t].T, s=1, color=U.RED)
    ax.set_title(f"reverse: t = {t}")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
U.finish("reverse", fig)

# --8<-- [start:field]
# what the network learnt: at a given noise level, -eps points back towards the data
g = torch.stack(torch.meshgrid(torch.linspace(-3, 3, 21), torch.linspace(-3, 3, 21), indexing="xy"), -1).reshape(-1, 2)
with torch.no_grad():
    field = {t: -model(g, torch.full((len(g),), t)) for t in (5, 40)}
# --8<-- [end:field]

fig, axes = U.figure(3, 1, size=(3.2, 3.1))
for ax, t in zip(axes[0, :2], (40, 5)):
    ax.scatter(*real[:800].T, s=1, color=U.GRAY)
    v = field[t] / field[t].norm(dim=1).max()                     # same max arrow length in both panels
    ax.quiver(g[:, 0], g[:, 1], v[:, 0], v[:, 1], color=U.RED, scale=14, width=0.005, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"−ε prediction at t = {t}")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 2].plot(np.convolve(losses, np.ones(100) / 100, mode="valid"), color=U.BLUE)
axes[0, 2].set_title("training loss (moving avg.)")
axes[0, 2].set_xlabel("step")
U.finish("field", fig)
