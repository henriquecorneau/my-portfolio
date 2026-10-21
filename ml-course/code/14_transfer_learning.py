"""Lesson 14 - Architectures and transfer learning: residual blocks, pretrained ResNet-18 on 1,000 satellite images."""
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models
from torchvision.transforms import functional as TF

import mlutils as U

U.setup("14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --8<-- [start:zoo]
zoo = [("AlexNet (2012)", models.AlexNet_Weights.IMAGENET1K_V1), ("VGG-16 (2014)", models.VGG16_Weights.IMAGENET1K_V1),
       ("ResNet-18 (2015)", models.ResNet18_Weights.IMAGENET1K_V1), ("ResNet-50 (2015)", models.ResNet50_Weights.IMAGENET1K_V2),
       ("MobileNet-V3-S (2019)", models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
       ("EfficientNet-B0 (2019)", models.EfficientNet_B0_Weights.IMAGENET1K_V1),
       ("ConvNeXt-T (2022)", models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), ("ViT-B/16 (2020)", models.ViT_B_16_Weights.IMAGENET1K_V1)]
for name, w in zoo:
    m = w.meta
    print(f"{name:23s} {m['num_params'] / 1e6:6.1f} M params  {m['_ops']:6.2f} GFLOPs  "
          f"ImageNet top-1 {m['_metrics']['ImageNet-1K']['acc@1']:.1f}%")
# --8<-- [end:zoo]


# --8<-- [start:residual]
class ResidualBlock(nn.Module):
    """y = relu(x + F(x)): the block only has to learn a correction F to the identity."""

    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.f(x))


def first_layer_grad(net):
    torch.manual_seed(0)
    x = torch.randn(16, 16, 32, 32)                 # (batch, channels, H, W) random input
    net(x).pow(2).mean().backward()                 # any scalar loss will do
    return net[0].weight.grad.norm().item() if isinstance(net[0], nn.Conv2d) else net[0].f[0].weight.grad.norm().item()


for depth in (4, 16, 48):
    torch.manual_seed(0)
    plain = nn.Sequential(*[m for _ in range(depth) for m in (nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())])
    torch.manual_seed(0)
    resnet = nn.Sequential(*[ResidualBlock(16) for _ in range(depth // 2)])
    print(f"{depth:2d} conv layers: gradient norm at the first layer, plain {first_layer_grad(plain):.2e}, "
          f"residual {first_layer_grad(resnet):.2e}")
# --8<-- [end:residual]

# --8<-- [start:data]
ds = datasets.EuroSAT(str(U.CACHE), download=True)           # 27,000 Sentinel-2 RGB tiles, 64x64, 10 classes
labels = np.array(ds.targets)
rng = np.random.default_rng(0)
idx = {s: [] for s in ("train", "val", "test")}
for c in range(10):
    i = rng.permutation(np.flatnonzero(labels == c))
    idx["train"] += i[:100].tolist()                        # only 100 labelled images per class
    idx["val"] += i[100:150].tolist()
    idx["test"] += i[150:350].tolist()


def load(ids):
    """(N, 3, 64, 64) float32 tensor in [0, 1] and (N,) int64 labels."""
    X = torch.stack([TF.to_tensor(ds[i][0]) for i in ids])
    return X, torch.from_numpy(labels[ids])


(Xtr, ytr), (Xva, yva), (Xte, yte) = (load(idx[s]) for s in ("train", "val", "test"))
print("classes:", ds.classes)
print("train", tuple(Xtr.shape), "val", tuple(Xva.shape), "test", tuple(Xte.shape), Xtr.dtype)
# --8<-- [end:data]

fig, axes = U.figure(10, 1, size=(1.3, 1.5))
for c, ax in enumerate(axes[0]):
    ax.imshow(Xtr[ytr == c][0].permute(1, 2, 0))
    ax.set_title(ds.classes[c].replace("Vegetation", "Veg.").replace("Industrial", "Industry"), fontsize=7)
    ax.axis("off")
U.finish("eurosat", fig)


# --8<-- [start:augment]
def augment(x):
    """Satellite tiles have no 'up': random flips and 90-degree rotations are free extra data."""
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))


class Tiles(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.t = X, y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        return (self.t(x) if self.t else x), self.y[i]
# --8<-- [end:augment]


# --8<-- [start:scratch]
def scratch_cnn():
    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), *block(128, 128),   # 64 -> 4 px
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 10))


torch.manual_seed(0)
t0 = time.time()
scratch = scratch_cnn()
h_scratch = U.train_classifier(scratch, DataLoader(Tiles(Xtr, ytr, augment), batch_size=32, shuffle=True),
                               Xva, yva, epochs=40, lr=2e-3, weight_decay=1e-4, device=device, verbose=False)
print("CNN from scratch: val acc %.3f, test acc %.3f (%.0f s)"
      % (h_scratch[-1, 3], U.evaluate(scratch, Xte, yte, device)[1], time.time() - t0))
# --8<-- [end:scratch]

# --8<-- [start:features]
weights = models.ResNet18_Weights.IMAGENET1K_V1
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def prep(x):
    """What the pretrained network expects: ImageNet statistics, and a bigger image than 64x64."""
    x = nn.functional.interpolate(x, size=112, mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


backbone = models.resnet18(weights=weights)
backbone.fc = nn.Identity()                         # drop the 1000-class ImageNet head: outputs 512 features
backbone.eval().to(device)


@torch.no_grad()
def features(X):
    return torch.cat([backbone(prep(X[i:i + 200]).to(device)).cpu() for i in range(0, len(X), 200)]).numpy()


t0 = time.time()
Ftr, Fva, Fte = features(Xtr), features(Xva), features(Xte)      # (1000, 512), (500, 512), (2000, 512)
probe = LogisticRegression(C=0.5, max_iter=3000).fit(Ftr, ytr.numpy())
print("frozen ResNet-18 + logistic regression: val acc %.3f, test acc %.3f (%.0f s)"
      % (probe.score(Fva, yva.numpy()), probe.score(Fte, yte.numpy()), time.time() - t0))
# --8<-- [end:features]

# --8<-- [start:few-shot]
for n in (5, 10, 25, 50, 100):
    sel = np.concatenate([np.flatnonzero(ytr.numpy() == c)[:n] for c in range(10)])
    acc = LogisticRegression(C=0.5, max_iter=3000).fit(Ftr[sel], ytr.numpy()[sel]).score(Fte, yte.numpy())
    print(f"{n:3d} labelled images per class -> test acc {acc:.3f}")
# --8<-- [end:few-shot]

# --8<-- [start:finetune]
torch.manual_seed(0)
model = models.resnet18(weights=weights)
model.fc = nn.Linear(512, 10)                       # new head for our 10 classes, randomly initialised


class WithPrep(nn.Module):
    """Put the resizing and ImageNet normalisation inside the model, so it takes raw tiles."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):                           # x: (B, 3, 64, 64) in [0, 1]
        return self.m(prep(x))


net = WithPrep(model)
t0 = time.time()
h_ft = U.train_classifier(net, DataLoader(Tiles(Xtr, ytr, augment), batch_size=32, shuffle=True), Xva, yva,
                          epochs=8, lr=3e-4, weight_decay=1e-4, device=device, verbose=False)
_, ft_acc = U.evaluate(net, Xte, yte, device)
print("fine-tuned ResNet-18: val acc %.3f, test acc %.3f (%.0f s)" % (h_ft[-1, 3], ft_acc, time.time() - t0))
# --8<-- [end:finetune]

fig, axes = U.figure(2, 1, size=(4.2, 3.2))
axes[0, 0].plot(range(1, 41), h_scratch[:, 3], color=U.GRAY, label="CNN from scratch")
axes[0, 0].plot(range(1, 9), h_ft[:, 3], "o-", ms=3, color=U.RED, label="fine-tuned ResNet-18")
axes[0, 0].axhline(probe.score(Fva, yva.numpy()), color=U.BLUE, ls="--", label="frozen features + log. reg.")
axes[0, 0].set_xlabel("epoch")
axes[0, 0].set_ylabel("validation accuracy")
axes[0, 0].set_ylim(0.3, 1)
axes[0, 0].legend(fontsize=7)
axes[0, 0].set_title("1,000 training images")
with torch.no_grad():
    pred = torch.cat([net(Xte[i:i + 200].to(device)).argmax(1).cpu() for i in range(0, 2000, 200)])
wrong = np.flatnonzero(pred.numpy() != yte.numpy())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, pred)
ax = axes[0, 1]
ax.imshow(np.log1p(cm), cmap="Blues")
short = [c[:6] for c in ds.classes]
ax.set_xticks(range(10), short, rotation=60, fontsize=7)
ax.set_yticks(range(10), short, fontsize=7)
for (r, c), v in np.ndenumerate(cm):
    if v:
        ax.text(c, r, str(v), ha="center", va="center", fontsize=6, color="w" if v > 50 else "k")
ax.grid(False)
ax.set_title(f"fine-tuned, test: {len(wrong)} errors / 2000")
U.finish("results", fig)

examples = [w for c in range(10) for w in wrong[yte.numpy()[wrong] == c][:1]][:6]   # one error per true class
fig, axes = U.figure(6, 1, size=(1.6, 2.15))
for ax, i in zip(axes[0], examples):
    ax.imshow(Xte[i].permute(1, 2, 0))
    ax.set_title(f"true: {ds.classes[yte[i]][:10]}\npred: {ds.classes[pred[i]][:10]}", fontsize=7)
    ax.axis("off")
U.finish("errors", fig)
