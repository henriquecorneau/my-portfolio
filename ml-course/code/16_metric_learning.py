"""Lesson 16 - Metric learning for face recognition: embeddings, ArcFace, open-set identification, occlusion."""
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torchvision import models

import mlutils as U

U.setup("16")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --8<-- [start:config]
@dataclass
class Config:
    known_people: int = 120       # identities used for training; the others are never seen
    max_per_person: int = 40      # cap George W. Bush's 530 photos
    emb_dim: int = 128
    epochs: int = 10
    batch: int = 64
    lr: float = 3e-4
    arc_s: float = 30.0           # ArcFace scale
    arc_m: float = 0.4            # ArcFace angular margin (radians)
    p_occlude: float = 0.5        # probability of masking the lower face during training


cfg = Config()
# --8<-- [end:config]

# --8<-- [start:data]
lfw = fetch_lfw_people(data_home=str(U.CACHE), min_faces_per_person=10, color=True, resize=1.0)
rng = np.random.default_rng(0)
keep = np.concatenate([rng.permutation(np.flatnonzero(lfw.target == p))[:cfg.max_per_person]
                       for p in range(len(lfw.target_names))])
faces = torch.from_numpy(lfw.images[keep]).permute(0, 3, 1, 2)   # (N, 3, 125, 94) float32, already in [0, 1]
ids = lfw.target[keep]                                           # (N,) person index
people = rng.permutation(len(lfw.target_names))
known = people[:cfg.known_people]
train_mask = np.isin(ids, known)
print("%d images of %d people: %d images of %d people for training, %d images of %d unseen people for testing"
      % (len(ids), len(people), train_mask.sum(), len(known), (~train_mask).sum(), len(people) - len(known)))
# --8<-- [end:data]


# --8<-- [start:occlusion]
def occlude(x, start=0.55):
    """Black out the lower part of the face, like a mask (the RandomMaskOcclusion of my notebook)."""
    x = x.clone()
    x[..., int(x.shape[-2] * start):, :] = 0
    return x
# --8<-- [end:occlusion]


# --8<-- [start:model]
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Embedder(nn.Module):
    """ImageNet ResNet-18 trunk + a linear projection to a unit-length embedding."""

    def __init__(self, dim=None):
        super().__init__()
        self.trunk = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.trunk.fc = nn.Identity()                            # (B, 512) features
        self.proj = nn.Linear(512, dim) if dim else nn.Identity()  # dim=None: raw ImageNet features

    def forward(self, x):                                        # x: (B, 3, 125, 94) in [0, 1]
        return nn.functional.normalize(self.proj(self.trunk((x - MEAN) / STD)), dim=1)   # unit vectors
# --8<-- [end:model]


# --8<-- [start:arcface]
class ArcFace(nn.Module):
    """Additive angular margin loss: the true class must win by an angle m, on a sphere of radius s."""

    def __init__(self, dim, n_classes, s, m):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)   # one "centre" direction per class
        self.s, self.m = s, m

    def forward(self, emb, y):
        cos = emb @ nn.functional.normalize(self.W).T                # (B, C) cosine to each centre
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)                           # penalised true-class logit
        onehot = nn.functional.one_hot(y, cos.shape[1]).bool()
        logits = self.s * torch.where(onehot, target, cos)
        return nn.functional.cross_entropy(logits, y)
# --8<-- [end:arcface]


# --8<-- [start:train]
torch.manual_seed(0)
model = Embedder(cfg.emb_dim).to(device)
head = ArcFace(cfg.emb_dim, len(known), cfg.arc_s, cfg.arc_m).to(device)
remap = {p: i for i, p in enumerate(known)}
Xk = faces[train_mask]
yk = torch.tensor([remap[p] for p in ids[train_mask]])
opt = torch.optim.AdamW([{"params": model.parameters(), "lr": cfg.lr},
                         {"params": head.parameters(), "lr": 10 * cfg.lr}], weight_decay=1e-4)
t0 = time.time()
for epoch in range(cfg.epochs):
    model.train()
    total = 0.0
    for idx in torch.randperm(len(Xk)).split(cfg.batch):
        x = Xk[idx].clone()
        flip = torch.rand(len(x)) < 0.5
        x[flip] = x[flip].flip(-1)                                    # horizontal flip
        mask = torch.rand(len(x)) < cfg.p_occlude
        x[mask] = occlude(x[mask])                                    # simulated face mask
        loss = head(model(x.to(device)), yk[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    print(f"  epoch {epoch + 1:2d}: ArcFace loss {total / len(Xk):.3f}")
print("fine-tuning took %.1f min" % ((time.time() - t0) / 60))
# --8<-- [end:train]


# --8<-- [start:evaluate]
@torch.no_grad()
def embed(net, X):
    net.eval()
    return torch.cat([net(X[i:i + 256].to(device)).cpu() for i in range(0, len(X), 256)]).numpy()


test = ~train_mask
who = ids[test]
iu = np.triu_indices(test.sum(), 1)
same = (who[:, None] == who[None, :])[iu]                             # pairs of the same person


def evaluate(G, P):
    """Unseen people only. Rank-1: is the most similar OTHER image of the same person?
    Verification: ROC AUC of the cosine similarity over all pairs (same person vs different)."""
    S = P @ G.T                                                      # (n, n) cosine similarities
    sims = S[iu]
    np.fill_diagonal(S, -np.inf)                                     # never match an image with itself
    return np.mean(who[S.argmax(1)] == who), roc_auc_score(same, sims), sims


results = {}
for name, net in [("ImageNet features (frozen)", Embedder().to(device)), ("ArcFace fine-tuned", model)]:
    clean = embed(net, faces[test])
    masked = embed(net, occlude(faces[test]))
    results[name] = (evaluate(clean, clean), evaluate(clean, masked))  # masked probes vs clean gallery
    (r1, a1, _), (r2, a2, _) = results[name]
    print(f"{name:27s} clean: rank-1 {r1:.3f}, AUC {a1:.3f} | masked probes: rank-1 {r2:.3f}, AUC {a2:.3f}")
E = embed(model, faces)                                              # (N, 128) embeddings of every image
Em = embed(model, occlude(faces))
# --8<-- [end:evaluate]

fig, axes = U.figure(2, 1, size=(4.0, 3.3))
for (name, res), c in zip(results.items(), (U.GRAY, U.RED)):
    for ax, (r1, auc, sims) in zip(axes[0], res):
        fpr, tpr, _ = roc_curve(same, sims)
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC {auc:.3f})")
for ax, title in zip(axes[0], ("verification, unseen people", "masked probe vs clean photo")):
    ax.plot([0, 1], [0, 1], "--", color=U.GRAY, lw=1)
    ax.set_xlabel("false accept rate")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="lower right")
axes[0, 0].set_ylabel("true accept rate")
U.finish("roc", fig)

# --8<-- [start:open-set]
sims = results["ArcFace fine-tuned"][0][2]
fpr, tpr, thr = roc_curve(same, sims)
k = np.argmin(np.abs(fpr - 0.01))
t = thr[k]                                                           # accept a match only above this similarity
print("threshold for a 1%% false accept rate: cosine similarity > %.3f (true accept rate %.3f)" % (t, tpr[k]))

unseen = np.setdiff1d(people, known)
newcomer, stranger = unseen[0], unseen[1]
gallery = {lfw.target_names[p]: E[np.flatnonzero(ids == p)[0]] for p in unseen[2:22]}   # 20 enrolled people
photos = np.flatnonzero(ids == newcomer)
gallery[lfw.target_names[newcomer]] = E[photos[0]]                  # enrol the newcomer with ONE photo
names, G = list(gallery), np.stack(list(gallery.values()))
print("newcomer:", lfw.target_names[newcomer], "| stranger:", lfw.target_names[stranger])
for q, label, emb in [(photos[1], "newcomer, another photo", E), (photos[2], "newcomer, masked photo", Em),
                      (np.flatnonzero(ids == stranger)[0], "stranger (not enrolled)", E)]:
    s = G @ emb[q]
    b = int(s.argmax())
    print(f"{label:24s}: best match {names[b]:22s} similarity {s[b]:.3f} -> {names[b] if s[b] > t else 'unknown'}")
# --8<-- [end:open-set]

fig, axes = U.figure(2, 1, size=(4.0, 3.2))
for ax, name in zip(axes[0], results):
    s_ = results[name][0][2]
    ax.hist(s_[same], bins=40, density=True, alpha=0.6, color=U.BLUE, label="same person")
    ax.hist(s_[~same], bins=40, density=True, alpha=0.6, color=U.RED, label="different people")
    ax.set_xlabel("cosine similarity")
    ax.set_title(name)
axes[0, 1].axvline(t, color="k", ls="--", lw=1, label="1% false-accept threshold")
axes[0, 0].legend(fontsize=7)
axes[0, 1].legend(fontsize=7)
U.finish("similarities", fig)

fig, axes = U.figure(5, 2, size=(1.5, 1.95))
tidx = np.flatnonzero(test)
for row, p in enumerate(unseen[3:5]):
    qi = np.flatnonzero(ids == p)[1]
    s = E[tidx] @ Em[qi]
    s[tidx == qi] = -np.inf
    top = tidx[np.argsort(-s)[:4]]
    axes[row, 0].imshow(occlude(faces[qi]).permute(1, 2, 0))
    axes[row, 0].set_title("query (masked)", fontsize=7)
    for col, j in enumerate(top, 1):
        axes[row, col].imshow(faces[j].permute(1, 2, 0))
        ok = ids[j] == ids[qi]
        axes[row, col].set_title(("✓ " if ok else "✗ ") + f"{E[j] @ Em[qi]:.2f}", fontsize=7,
                                 color=U.GREEN if ok else U.RED)
for ax in axes.flat:
    ax.axis("off")
U.finish("retrieval", fig)
