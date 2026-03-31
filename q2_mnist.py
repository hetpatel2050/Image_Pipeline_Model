"""
CVI620 - Assignment 2 - Q2: MNIST Digit Classification
Methods: SVM, Random Forest, KNN, MLP
Target: >= 90% accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "Q2/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_CSV  = "Q2/mnist_train.csv"
TEST_CSV   = "Q2/mnist_test.csv"

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CVI620 – Q2: MNIST Digit Classification")
print("=" * 60)

print("\n[1/5] Loading MNIST CSV data …")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# First column = label
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values.astype(np.float32)

y_test  = test_df.iloc[:, 0].values
X_test  = test_df.iloc[:, 1:].values.astype(np.float32)

print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
print(f"      Classes: {sorted(np.unique(y_train))}")

# ─── NORMALISE ─────────────────────────────────────────────────────────────────
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# ─── PCA ───────────────────────────────────────────────────────────────────────
print("[2/5] Applying PCA (n=100) for speed …")
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
var = pca.explained_variance_ratio_.sum() * 100
print(f"      Variance retained: {var:.1f}%")

# For SVM/KNN we subsample to 10k for speed (still get great accuracy)
SAMPLE = 10000
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_pca), SAMPLE, replace=False)
X_sub = X_train_pca[idx]
y_sub = y_train[idx]

# ─── CLASSIFIERS ───────────────────────────────────────────────────────────────
classifiers = {
    "SVM (RBF, C=5)": {
        "clf": SVC(kernel='rbf', C=5, gamma='scale', random_state=42),
        "X_train": X_sub, "y_train": y_sub,
        "X_test": X_test_pca,
    },
    "Random Forest (300 trees)": {
        "clf": RandomForestClassifier(n_estimators=300, max_depth=None,
                                       random_state=42, n_jobs=-1),
        "X_train": X_train_pca, "y_train": y_train,
        "X_test": X_test_pca,
    },
    "KNN (k=3)": {
        "clf": KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1),
        "X_train": X_sub, "y_train": y_sub,
        "X_test": X_test_pca,
    },
    "MLP (512-256-128)": {
        "clf": MLPClassifier(hidden_layer_sizes=(512, 256, 128),
                             activation='relu', solver='adam',
                             max_iter=200, random_state=42,
                             early_stopping=True, validation_fraction=0.1,
                             batch_size=256, learning_rate_init=0.001),
        "X_train": X_train_pca, "y_train": y_train,
        "X_test":  X_test_pca,
    },
}

results = {}
print("[3/5] Training classifiers  (this may take a few minutes) …\n")
for name, cfg in classifiers.items():
    print(f"  ► {name}")
    clf = cfg["clf"]
    clf.fit(cfg["X_train"], cfg["y_train"])
    y_pred = clf.predict(cfg["X_test"])
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"clf": clf, "y_pred": y_pred, "accuracy": acc}
    goal = " ✅ ≥90%" if acc >= 0.90 else " ⚠️  <90%"
    print(f"    Accuracy: {acc*100:.2f}%{goal}")
    print(classification_report(y_test, y_pred, digits=4))

# ─── SAVE BEST MODEL ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_clf  = results[best_name]["clf"]
best_acc  = results[best_name]["accuracy"]

print(f"[4/5] Best model: {best_name}  ({best_acc*100:.2f}%)")
joblib.dump({"model": best_clf, "pca": pca},
            os.path.join(OUTPUT_DIR, "best_mnist_model.pkl"))
print(f"      Saved → {OUTPUT_DIR}/best_mnist_model.pkl")

# ─── VISUALISATION 1 – Accuracy Bar Chart ──────────────────────────────────────
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.8)
ax.axhline(90, color='red', linestyle='--', linewidth=1.5, label='90% target')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{acc:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(85, 102)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Q2 – MNIST: Classifier Comparison", fontsize=14, fontweight='bold')
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/accuracy_comparison.png")

# ─── VISUALISATION 2 – Confusion Matrix (best model) ───────────────────────────
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=list(range(10)))
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f"Q2 – Confusion Matrix: {best_name}\nAccuracy: {best_acc*100:.2f}%",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_best.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/confusion_matrix_best.png")

# ─── VISUALISATION 3 – Sample MNIST images ─────────────────────────────────────
print("[5/5] Saving sample predictions …")
fig, axes = plt.subplots(4, 10, figsize=(16, 7))
y_pred_best = results[best_name]["y_pred"]
for digit in range(10):
    idxs = np.where(y_test == digit)[0][:4]
    for row, idx in enumerate(idxs):
        img = X_test[idx].reshape(28, 28)
        pred = y_pred_best[idx]
        color = 'green' if pred == digit else 'red'
        axes[row, digit].imshow(img, cmap='gray');
        axes[row, digit].axis('off')
        axes[row, digit].set_title(f"P:{pred}", color=color, fontsize=8)
        if row == 0:
            axes[row, digit].set_xlabel(f"Digit {digit}", fontsize=8)
plt.suptitle(f"Q2 – MNIST Sample Predictions [{best_name}]  Accuracy: {best_acc*100:.2f}%",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/sample_predictions.png")

# ─── ALL CONFUSION MATRICES ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=list(range(10)))
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f"{name}\nAccuracy: {res['accuracy']*100:.2f}%",
                      fontsize=11, fontweight='bold')
plt.suptitle("Q2 – All Confusion Matrices", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_confusion_matrices.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/all_confusion_matrices.png")

# ─── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
    goal = "✅" if res["accuracy"] >= 0.90 else "⚠️"
    print(f"  {goal}  {name:<35} {res['accuracy']*100:.2f}%")
print("\n✅  Q2 complete. All outputs in:", OUTPUT_DIR)
