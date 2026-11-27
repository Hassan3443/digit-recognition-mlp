"""Handwritten digit recognition: a small neural network (MLP) vs classic models.
Data: scikit-learn's bundled 8x8 digits set (1,797 real scanned digits, UCI)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "k-NN (k=3)": make_pipeline(StandardScaler(), KNeighborsClassifier(3)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    "Neural network (MLP 64-32)": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=500, random_state=SEED)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
res = {}
for n, m in models.items():
    cvs = cross_val_score(m, Xtr, ytr, cv=cv)
    m.fit(Xtr, ytr)
    res[n] = dict(cv_mean=round(float(cvs.mean()), 4), cv_std=round(float(cvs.std()), 4),
                  test_acc=round(float(accuracy_score(yte, m.predict(Xte))), 4))
    print(f"{n:28s} CV {cvs.mean():.4f} +- {cvs.std():.4f}   test {res[n]['test_acc']:.4f}")

mlp = models["Neural network (MLP 64-32)"]
pred = mlp.predict(Xte)
print(classification_report(yte, pred, digits=3))
cm = confusion_matrix(yte, pred)
errs = np.where(pred != yte)[0]
print("MLP test errors:", len(errs), "of", len(yte))
json.dump(res, open("results.json", "w"), indent=1)

# Plot 1: sample digits
fig, ax = plt.subplots(2, 10, figsize=(11, 2.8))
for d in range(10):
    for r in range(2):
        ax[r, d].imshow(X[y == d][r].reshape(8, 8), cmap="gray_r"); ax[r, d].axis("off")
    ax[0, d].set_title(str(d))
fig.suptitle("Dataset samples (8x8 pixels, real scanned digits)"); fig.tight_layout()
fig.savefig("images/samples.png", dpi=150); plt.close(fig)

# Plot 2: model comparison + confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
names = list(res); vals = [res[n]["test_acc"] * 100 for n in names]
b = ax[0].barh(names, vals, color=["#2563eb" if v == max(vals) else "#9aa5b1" for v in vals])
ax[0].set_xlim(90, 100); ax[0].set_xlabel("Test accuracy (%)"); ax[0].set_title("Model comparison (same 25% hold-out)")
for r, v in zip(b, vals): ax[0].text(v + 0.1, r.get_y() + r.get_height() / 2, f"{v:.1f}", va="center")
im = ax[1].imshow(cm, cmap="Blues"); ax[1].set_title("MLP confusion matrix (test)")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True"); ax[1].set_xticks(range(10)); ax[1].set_yticks(range(10))
for i in range(10):
    for j in range(10):
        if cm[i, j]: ax[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
fig.tight_layout(); fig.savefig("images/results.png", dpi=150); plt.close(fig)

# Plot 3: misclassified
k = min(len(errs), 10)
fig, ax = plt.subplots(1, max(k, 1), figsize=(1.4 * max(k, 1), 2.1))
ax = np.atleast_1d(ax)
for a, i in zip(ax, errs[:k]):
    a.imshow(Xte[i].reshape(8, 8), cmap="gray_r"); a.axis("off"); a.set_title(f"true {yte[i]}\npred {pred[i]}", fontsize=8)
fig.suptitle("MLP mistakes on the test set", fontsize=9); fig.tight_layout()
fig.savefig("images/errors.png", dpi=150); plt.close(fig)
