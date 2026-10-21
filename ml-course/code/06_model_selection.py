"""Lesson 06 - Model selection and evaluation: cross-validation, pipelines, metrics, learning curves."""
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, roc_curve,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mlutils as U

U.setup("06")

# --8<-- [start:data]
BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
X, y = load_svmlight_file(str(U.fetch(BASE + "splice", "splice")))
Xt, yt = load_svmlight_file(str(U.fetch(BASE + "splice.t", "splice.t")), n_features=X.shape[1])
X, Xt = X.toarray(), Xt.toarray()                  # (1000, 60) and (2175, 60), labels in {-1, +1}
print("train", X.shape, "test", Xt.shape, "| class balance: %.1f%% / %.1f%% positive"
      % (100 * np.mean(y > 0), 100 * np.mean(yt > 0)))
# --8<-- [end:data]


# --8<-- [start:kfold]
def kfold_indices(n, k, rng):
    """Shuffle once, cut into k folds; each fold is the validation set exactly once."""
    folds = np.array_split(rng.permutation(n), k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma=0.01))
scores = []
for tr, va in kfold_indices(len(y), 5, np.random.default_rng(0)):
    model.fit(X[tr], y[tr])                       # the scaler is refitted on each training fold
    scores.append(model.score(X[va], y[va]))
print("5-fold accuracy from scratch: %s -> %.3f ± %.3f" % (np.round(scores, 3), np.mean(scores), np.std(scores)))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
s = cross_val_score(model, X, y, cv=cv)
print("cross_val_score (stratified):  %.3f ± %.3f" % (s.mean(), s.std()))
# --8<-- [end:kfold]

# --8<-- [start:grid]
Cs, gammas = np.logspace(-0.5, 2, 11), np.logspace(-3, 0, 13)
search = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                      {"svc__C": Cs, "svc__gamma": gammas}, cv=cv, n_jobs=-1)
search.fit(X, y)
print("best:", {k: round(float(v), 4) for k, v in search.best_params_.items()},
      "cross-validated accuracy %.3f" % search.best_score_)
grid = search.cv_results_["mean_test_score"].reshape(len(Cs), len(gammas))   # rows: C, columns: gamma
print("test accuracy of the refitted best model: %.3f" % search.score(Xt, yt))

lin = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="linear")), {"svc__C": np.logspace(-3, 1, 9)}, cv=cv)
lin.fit(X, y)
print("linear SVM: C = %g, CV accuracy %.3f, test accuracy %.3f"
      % (lin.best_params_["svc__C"], lin.best_score_, lin.score(Xt, yt)))
# --8<-- [end:grid]

fig, axes = U.figure(1, 1, size=(5.2, 3.6))
ax = axes[0, 0]
im = ax.pcolormesh(gammas, Cs, grid, cmap="viridis", shading="nearest")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(search.best_params_["svc__gamma"], search.best_params_["svc__C"], "r*", ms=12)
ax.set_xlabel("γ (log)")
ax.set_ylabel("C (log)")
ax.set_title("5-fold CV accuracy, RBF SVM on splice")
ax.grid(False)
fig.colorbar(im, ax=ax)
U.finish("grid", fig)

# --8<-- [start:metrics]
best = search.best_estimator_
pred = best.predict(Xt)
cm = confusion_matrix(yt, pred)                    # rows: truth, columns: prediction
tn, fp, fn, tp = cm.ravel()
print("confusion matrix [[TN FP] [FN TP]] =", cm.tolist())
print("accuracy %.3f | precision %.3f | recall %.3f | F1 %.3f"
      % ((tp + tn) / cm.sum(), precision_score(yt, pred), recall_score(yt, pred), f1_score(yt, pred)))
score = best.decision_function(Xt)                 # signed distance to the boundary
print("ROC AUC %.3f" % roc_auc_score(yt, score))
# --8<-- [end:metrics]

fig, axes = U.figure(2, 1, size=(3.6, 3.3))
ax = axes[0, 0]
ax.imshow(cm, cmap="Blues")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", color="w" if v > cm.max() / 2 else "k", fontsize=12)
ax.set_xticks([0, 1], ["−1", "+1"])
ax.set_yticks([0, 1], ["−1", "+1"])
ax.set_xlabel("prediction")
ax.set_ylabel("truth")
ax.set_title("confusion matrix (test)")
ax.grid(False)
ax = axes[0, 1]
fpr, tpr, thr = roc_curve(yt, score)
ax.plot(fpr, tpr, color=U.BLUE, label=f"RBF SVM, AUC = {roc_auc_score(yt, score):.3f}")
lin_score = lin.best_estimator_.decision_function(Xt)
f2, t2, _ = roc_curve(yt, lin_score)
ax.plot(f2, t2, color=U.ORANGE, label=f"linear SVM, AUC = {roc_auc_score(yt, lin_score):.3f}")
ax.plot([0, 1], [0, 1], "--", color=U.GRAY, lw=1)
k = np.argmin(np.abs(thr))
ax.plot(fpr[k], tpr[k], "o", color=U.BLUE)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate (recall)")
ax.set_title("ROC curve")
ax.legend(fontsize=7, loc="lower right")
U.finish("metrics", fig)

# --8<-- [start:learning]
sizes = np.linspace(0.1, 1.0, 8)
fig, axes = U.figure(2, 1, size=(3.9, 3.1))
for ax, (name, est) in zip(axes[0], [("linear SVM (high bias)", lin.best_estimator_),
                                     ("RBF SVM, γ = 1 (high variance)", make_pipeline(StandardScaler(), SVC(C=10, gamma=1)))]):
    n, tr, va = learning_curve(est, X, y, cv=cv, train_sizes=sizes)
    print(f"{name:32s} n = {n[-1]}: train {tr[-1].mean():.3f}, validation {va[-1].mean():.3f}")
    ax.plot(n, 1 - tr.mean(1), "o-", ms=3, color=U.BLUE, label="training error")
    ax.plot(n, 1 - va.mean(1), "o-", ms=3, color=U.RED, label="validation error")
    ax.fill_between(n, 1 - va.mean(1) - va.std(1), 1 - va.mean(1) + va.std(1), color=U.RED, alpha=0.15)
    ax.set_xlabel("training set size")
    ax.set_title(name)
    ax.set_ylim(-0.02, 0.55)
axes[0, 0].set_ylabel("error rate")
axes[0, 0].legend()
U.finish("learning-curves", fig)
# --8<-- [end:learning]
