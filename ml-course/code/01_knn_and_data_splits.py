"""Lesson 01 - Data, splits and a first classifier: k-nearest neighbours on wine quality."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import mlutils as U

U.setup("01")

# --8<-- [start:load]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(U.fetch(URL, "winequality-white.csv"), sep=";")
print(df.shape)                                   # (4898, 12): 11 inputs + the quality grade
print(df["quality"].value_counts().sort_index().to_dict())

X = df.drop(columns="quality").to_numpy()         # (4898, 11) float64
y = (df["quality"] >= 6).astype(int).to_numpy()   # 1 = good wine (grade 6+), 0 = bad
print("good wines: %d, bad wines: %d" % (y.sum(), (1 - y).sum()))
# --8<-- [end:load]

# --8<-- [start:stats]
stats = pd.DataFrame({"mean": X.mean(0), "std": X.std(0)}, index=df.columns[:-1])
print(stats.round(4).to_string())
corr = np.corrcoef(X, rowvar=False)
i, j = np.unravel_index(np.argmax(np.abs(corr - np.eye(11))), corr.shape)
print("strongest correlation: %s / %s = %.2f" % (df.columns[i], df.columns[j], corr[i, j]))
# --8<-- [end:stats]

fig, axes = U.figure(2, 1, size=(5.2, 3.6))
names = [c.replace(" ", "\n", 1) for c in df.columns[:-1]]
axes[0, 0].boxplot(X, showfliers=False)
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("raw features (log scale!)")
axes[0, 1].boxplot((X - X.mean(0)) / X.std(0), showfliers=False)
axes[0, 1].set_title("standardised features")
for ax in axes[0]:
    ax.set_xticks(range(1, 12), names, rotation=90, fontsize=7)
U.finish("features", fig)

# --8<-- [start:split]
# 1/3 train, 1/3 validation, 1/3 test, each with the same proportion of good wines
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train,
                                                  random_state=0)
for name, part in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(f"{name:5s} {len(part):5d} samples, {part.mean():.1%} good")
# --8<-- [end:split]


# --8<-- [start:knn]
def knn_predict(X_train, y_train, X_query, k):
    """Majority vote among the k closest training samples (Euclidean distance)."""
    # squared distances between every query and every training sample: (n_query, n_train)
    d2 = (X_query ** 2).sum(1)[:, None] - 2 * X_query @ X_train.T + (X_train ** 2).sum(1)[None, :]
    nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]   # indices of the k smallest, unordered
    votes = y_train[nearest]                               # (n_query, k) labels of the neighbours
    return (votes.mean(1) > 0.5).astype(int)               # binary labels, odd k: no ties
# --8<-- [end:knn]

# --8<-- [start:check]
mine = knn_predict(X_train, y_train, X_val, k=3)
ref = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train).predict(X_val)
print("k=3: validation error %.3f, agreement with scikit-learn %.1f%%"
      % (np.mean(mine != y_val), 100 * np.mean(mine == ref)))
# --8<-- [end:check]


# --8<-- [start:curve]
def error_curves(Xa, Xv, ks):
    tr, va = [], []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xa, y_train)
        tr.append(np.mean(clf.predict(Xa) != y_train))
        va.append(np.mean(clf.predict(Xv) != y_val))
    return np.array(tr), np.array(va)


ks = np.arange(1, 60, 2)
tr_raw, va_raw = error_curves(X_train, X_val, ks)

scaler = StandardScaler().fit(X_train)       # mean and std of the TRAINING set only
Xs_train, Xs_val, Xs_test = (scaler.transform(a) for a in (X_train, X_val, X_test))
tr_std, va_std = error_curves(Xs_train, Xs_val, ks)

for name, va in [("raw", va_raw), ("standardised", va_std)]:
    print(f"{name:13s} best k = {ks[va.argmin()]:2d}, validation error {va.min():.3f}")
# --8<-- [end:curve]

fig, axes = U.figure(2, 1, size=(4.4, 3.2))
for ax, tr, va, title in [(axes[0, 0], tr_raw, va_raw, "raw features"),
                          (axes[0, 1], tr_std, va_std, "standardised features")]:
    ax.plot(ks, tr, "o-", ms=3, color=U.BLUE, label="training error")
    ax.plot(ks, va, "o-", ms=3, color=U.RED, label="validation error")
    ax.axvline(ks[va.argmin()], color=U.GRAY, ls="--", lw=1)
    ax.set_xlabel("k (number of neighbours)")
    ax.set_title(title)
    ax.set_ylim(0, 0.42)
axes[0, 0].set_ylabel("error rate")
axes[0, 0].legend()
U.finish("error-curves", fig)

# --8<-- [start:test]
k_star = ks[va_std.argmin()]
clf = KNeighborsClassifier(n_neighbors=k_star).fit(Xs_train, y_train)
print("k* = %d -> test error %.3f (majority-class baseline: %.3f)"
      % (k_star, np.mean(clf.predict(Xs_test) != y_test), min(y_test.mean(), 1 - y_test.mean())))
# --8<-- [end:test]

# a 2-D picture of what k does: two standardised features only
cols = [list(df.columns).index(c) for c in ("alcohol", "volatile acidity")]
X2, y2 = Xs_train[:, cols], y_train
fig, axes = U.figure(3, 1, size=(3.4, 3.2))
for ax, k in zip(axes[0], (1, 15, 101)):
    clf2 = KNeighborsClassifier(n_neighbors=k).fit(X2, y2)
    U.plot_decision(ax, clf2.predict, X2, y2, title=f"k = {k}", resolution=250)
    ax.set_xlabel("alcohol (standardised)")
axes[0, 0].set_ylabel("volatile acidity (standardised)")
axes[0, 0].legend(["bad", "good"], loc="upper right", fontsize=7)
U.finish("boundaries", fig)
