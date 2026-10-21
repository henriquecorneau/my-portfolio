"""Lesson 13 - Convolutional networks on Fashion-MNIST: MLP vs CNN, overfitting, regularisation, what filters learn."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import mlutils as U

U.setup("13")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --8<-- [start:data]
x_train, y_train, x_test, y_test = U.load_mnist(fashion=True)
X = torch.from_numpy(x_train).float().unsqueeze(1) / 255.0     # (60000, 1, 28, 28) float32 in [0, 1]
Y = torch.from_numpy(y_train)                                  # (60000,) int64
X_test = torch.from_numpy(x_test).float().unsqueeze(1) / 255.0
Y_test = torch.from_numpy(y_test)
perm = torch.randperm(len(X))
tr, va = perm[:54000], perm[54000:]                            # 10% validation, like validation_split=0.1
print("train", tuple(X[tr].shape), "val", tuple(X[va].shape), "test", tuple(X_test.shape))
# --8<-- [end:data]

# --8<-- [start:conv]
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
print("conv weight", tuple(conv.weight.shape), "-> parameters:", sum(p.numel() for p in conv.parameters()))
x = X[:8]                                                      # (8, 1, 28, 28) a batch of 8 images
print("input", tuple(x.shape), "-> conv", tuple(conv(x).shape), "-> max-pool",
      tuple(nn.MaxPool2d(2)(conv(x)).shape))
dense = nn.Linear(28 * 28, 28 * 28 * 32)
print("a dense layer with the same output size would need %.1f M parameters"
      % (sum(p.numel() for p in dense.parameters()) / 1e6))
# --8<-- [end:conv]


# --8<-- [start:models]
def mlp():
    """The tutorial's first model: flatten the image, one hidden layer."""
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def small_cnn():
    """The class's 'version 2': one convolution + pooling in front of the same classifier."""
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # (32, 14, 14)
                         nn.Flatten(), nn.Linear(32 * 14 * 14, 128), nn.ReLU(), nn.Linear(128, 10))


def regularised_cnn():
    """Two conv stages with batch norm, dropout before the classifier."""
    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
    return nn.Sequential(*block(1, 32), *block(32, 32), nn.MaxPool2d(2),                 # (32, 14, 14)
                         *block(32, 64), nn.MaxPool2d(2),                                 # (64, 7, 7)
                         nn.Flatten(), nn.Dropout(0.4), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                         nn.Dropout(0.4), nn.Linear(128, 10))


for name, f in [("MLP", mlp), ("small CNN", small_cnn), ("regularised CNN", regularised_cnn)]:
    print(f"{name:16s} {sum(p.numel() for p in f().parameters()):9,d} parameters")
# --8<-- [end:models]


# --8<-- [start:augment]
class Augmented(torch.utils.data.Dataset):
    """Random horizontal flip and a random shift of up to 2 pixels, drawn anew every epoch."""

    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]                                          # (1, 28, 28)
        if torch.rand(1) < 0.5:
            x = x.flip(-1)
        dx, dy = torch.randint(-2, 3, (2,)).tolist()
        x = torch.roll(nn.functional.pad(x, (2, 2, 2, 2)), shifts=(dy, dx), dims=(1, 2))[:, 2:30, 2:30]
        return x, self.Y[i]
# --8<-- [end:augment]


# --8<-- [start:train]
EPOCHS = 15
plain = DataLoader(TensorDataset(X[tr], Y[tr]), batch_size=128, shuffle=True)
augmented = DataLoader(Augmented(X[tr], Y[tr]), batch_size=128, shuffle=True)
results, models = {}, {}
for name, make, loader, reg in [("MLP", mlp, plain, False), ("small CNN", small_cnn, plain, False),
                                ("regularised CNN + augmentation", regularised_cnn, augmented, True)]:
    torch.manual_seed(0)
    models[name] = make()
    print(name)
    # the first two as in class (Adam, constant lr 1e-3); the third adds weight decay and a cosine schedule
    results[name] = U.train_classifier(models[name], loader, X[va], Y[va], EPOCHS, lr=1e-3,
                                       weight_decay=1e-4 if reg else 0.0, cosine=reg, device=device,
                                       verbose=False)
    h = results[name]
    best = int(h[:, 2].argmin())
    _, test_acc = U.evaluate(models[name], X_test, Y_test, device)
    print(f"  final: train acc {h[-1, 1]:.4f}, val acc {h[-1, 3]:.4f}, test acc {test_acc:.4f} | "
          f"lowest val loss {h[best, 2]:.4f} at epoch {best + 1}, final val loss {h[-1, 2]:.4f}")
# --8<-- [end:train]
torch.save(models["regularised CNN + augmentation"].state_dict(), U.CACHE / "fashion_cnn.pt")

fig, axes = U.figure(2, 1, size=(4.3, 3.2))
for (name, h), c in zip(results.items(), (U.GRAY, U.RED, U.BLUE)):
    ep = np.arange(1, EPOCHS + 1)
    axes[0, 0].plot(ep, h[:, 0], ls="--", color=c, lw=1)
    axes[0, 0].plot(ep, h[:, 2], color=c, label=name)
    axes[0, 1].plot(ep, h[:, 1], ls="--", color=c, lw=1)
    axes[0, 1].plot(ep, h[:, 3], color=c, label=name)
axes[0, 0].set_title("loss (dashed: training, solid: validation)")
axes[0, 1].set_title("accuracy (dashed: training, solid: validation)")
axes[0, 1].set_ylim(0.84, 1.0)
for ax in axes[0]:
    ax.set_xlabel("epoch")
axes[0, 0].legend(fontsize=7)
U.finish("curves", fig)

# --8<-- [start:errors]
model = models["regularised CNN + augmentation"].eval()
with torch.no_grad():
    pred = torch.cat([model(X_test[i:i + 1000].to(device)).argmax(1).cpu() for i in range(0, 10000, 1000)])
cm = confusion_matrix(Y_test, pred)
per_class = cm.diagonal() / cm.sum(1)
for k in np.argsort(per_class)[:3]:
    other = np.argsort(cm[k])[-2]
    print(f"{U.FASHION_CLASSES[k]:12s} accuracy {per_class[k]:.3f}, most often mistaken for {U.FASHION_CLASSES[other]}"
          f" ({cm[k, other]} times)")
# --8<-- [end:errors]

fig, axes = U.figure(1, 1, size=(6.2, 5.2))
ax = axes[0, 0]
ax.imshow(np.log1p(cm), cmap="Blues")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", fontsize=7, color="w" if v > 300 else "k")
ax.set_xticks(range(10), U.FASHION_CLASSES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(10), U.FASHION_CLASSES, fontsize=8)
ax.set_xlabel("prediction")
ax.set_ylabel("truth")
ax.grid(False)
ax.set_title("confusion matrix on the 10,000 test images")
U.finish("confusion", fig)

# --8<-- [start:filters]
first = model[0].weight.detach().cpu()                          # (32, 1, 3, 3) learnt 3x3 filters
img = X_test[0:1].to(device)                                    # (1, 1, 28, 28) an ankle boot
with torch.no_grad():
    maps = model[:3](img)[0].cpu()                              # (32, 28, 28) after conv + BN + ReLU
active = maps.flatten(1).mean(1).argsort(descending=True)[:8]
print("first-layer filters", tuple(first.shape), "| feature maps", tuple(maps.shape))
# --8<-- [end:filters]

fig, axes = U.figure(9, 2, size=(1.25, 1.35))
axes[0, 0].imshow(img[0, 0].cpu(), cmap="gray_r")
axes[0, 0].set_title("input", fontsize=8)
axes[1, 0].axis("off")
for i, k in enumerate(active):
    f = first[k, 0]
    axes[0, i + 1].imshow(f, cmap="RdBu_r", vmin=-f.abs().max(), vmax=f.abs().max())
    axes[0, i + 1].set_title(f"filter {int(k)}", fontsize=8)
    axes[1, i + 1].imshow(maps[k], cmap="viridis")
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
U.finish("filters", fig)
