"""
CVI620 - Assignment 2 - Q1: Cat vs Dog Classification
Methods: SVM, Random Forest, KNN, MLP (Neural Network)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
import joblib
import requests
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE   = (64, 64)
TRAIN_DIR  = "Q1/train"
TEST_DIR   = "Q1/test"
OUTPUT_DIR = "Q1/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_features(image_path, img_size=IMG_SIZE):
    """
    Combine HOG-like features via raw pixels + color histograms.
    Returns a 1-D feature vector.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, img_size)

    # 1. Flattened grayscale pixels (normalised)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_feat = gray.flatten().astype(np.float32) / 255.0

    # 2. RGB colour histogram (32 bins per channel)
    hist_feat = []
    for ch in range(3):
        h = cv2.calcHist([img], [ch], None, [32], [0, 256])
        hist_feat.extend(h.flatten() / (img_size[0] * img_size[1]))

    # 3. Edge features via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_feat = edges.flatten().astype(np.float32) / 255.0

    return np.concatenate([pixel_feat, hist_feat, edge_feat])


def load_dataset(data_dir, label_map={"Cat": 0, "Dog": 1}):
    X, y, paths = [], [], []
    for label_name, label_val in label_map.items():
        folder = os.path.join(data_dir, label_name)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, fname)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat)
                y.append(label_val)
                paths.append(path)
    return np.array(X), np.array(y), paths


# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CVI620 – Q1: Cat vs Dog Classification")
print("=" * 60)

print("\n[1/5] Loading training data …")
X_train, y_train, train_paths = load_dataset(TRAIN_DIR)
print(f"      Loaded {len(X_train)} images  (shape: {X_train.shape})")

print("[2/5] Loading test data …")
X_test, y_test, test_paths = load_dataset(TEST_DIR)
print(f"      Loaded {len(X_test)} images")

# ─── SCALE ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── PCA (optional dimensionality reduction for speed) ──────────────────────────
pca = PCA(n_components=150, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)
print(f"[3/5] PCA: kept 150 components "
      f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

# ─── CLASSIFIERS ───────────────────────────────────────────────────────────────
classifiers = {
    "SVM (RBF kernel)": SVC(kernel='rbf', C=10, gamma='scale',
                             probability=True, random_state=42),
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200,
                                   max_depth=20, random_state=42, n_jobs=-1),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric='euclidean',
                                       n_jobs=-1),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(256, 128),
                            activation='relu', max_iter=300,
                            random_state=42, early_stopping=True),
}

results = {}
print("[4/5] Training & evaluating classifiers …\n")
for name, clf in classifiers.items():
    print(f"  ► {name}")
    if name == "Random Forest (200 trees)":
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
    else:
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    results[name] = {"clf": clf, "y_pred": y_pred, "accuracy": acc}
    print(f"    Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                 target_names=["Cat", "Dog"], digits=4))

# ─── SAVE BEST MODEL ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_clf  = results[best_name]["clf"]
best_acc  = results[best_name]["accuracy"]

print(f"\n[5/5] Best model: {best_name}  ({best_acc*100:.2f}%)")
joblib.dump({"model": best_clf, "scaler": scaler, "pca": pca,
             "use_pca": best_name != "Random Forest (200 trees)"},
            os.path.join(OUTPUT_DIR, "best_cat_dog_model.pkl"))
print(f"      Model saved → {OUTPUT_DIR}/best_cat_dog_model.pkl")

# ─── VISUALISATION 1 – Accuracy Bar Chart ──────────────────────────────────────
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
              edgecolor="black", linewidth=0.8)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{acc:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Q1 – Cat vs Dog: Classifier Comparison", fontsize=14, fontweight='bold')
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/accuracy_comparison.png")

# ─── VISUALISATION 2 – Confusion Matrices ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Cat", "Dog"])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f"{name}\nAccuracy: {res['accuracy']*100:.2f}%",
                      fontsize=11, fontweight='bold')
plt.suptitle("Q1 – Confusion Matrices", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/confusion_matrices.png")

# ─── VISUALISATION 3 – Sample Training Images ──────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
sample_cats = sorted(os.listdir(os.path.join(TRAIN_DIR, "Cat")))[:5]
sample_dogs = sorted(os.listdir(os.path.join(TRAIN_DIR, "Dog")))[:5]
for j, fname in enumerate(sample_cats):
    img = cv2.imread(os.path.join(TRAIN_DIR, "Cat", fname))
    img = cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_BGR2RGB)
    axes[0, j].imshow(img); axes[0, j].axis('off')
    axes[0, j].set_title("Cat", fontsize=10)
for j, fname in enumerate(sample_dogs):
    img = cv2.imread(os.path.join(TRAIN_DIR, "Dog", fname))
    img = cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_BGR2RGB)
    axes[1, j].imshow(img); axes[1, j].axis('off')
    axes[1, j].set_title("Dog", fontsize=10)
plt.suptitle("Q1 – Sample Training Images", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=150)
plt.close()
print(f"  Plot saved → {OUTPUT_DIR}/sample_images.png")

# ─── INTERNET IMAGE TESTING ────────────────────────────────────────────────────
print("\n─── Testing on internet images ───")
INTERNET_IMAGES = [
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",  "cat_internet_1.jpg", "Cat"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg", "dog_internet_1.jpg", "Dog"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cat_poster_1.jpg/240px-Cat_poster_1.jpg", "cat_internet_2.jpg", "Cat"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/320px-Collage_of_Nine_Dogs.jpg", "dog_internet_2.jpg", "Dog"),
]

LABEL_MAP_INV = {0: "Cat", 1: "Dog"}
internet_results = []

for url, fname, true_label in INTERNET_IMAGES:
    try:
        resp = requests.get(url, timeout=10)
        img_path = os.path.join(OUTPUT_DIR, fname)
        with open(img_path, "wb") as f:
            f.write(resp.content)
        feat = extract_features(img_path)
        if feat is not None:
            feat_s = scaler.transform([feat])
            if results[best_name].get("use_pca", True) and best_name != "Random Forest (200 trees)":
                feat_in = pca.transform(feat_s)
            else:
                feat_in = feat_s
            pred = best_clf.predict(feat_in)[0]
            pred_label = LABEL_MAP_INV[pred]
            correct = "✓" if pred_label == true_label else "✗"
            internet_results.append((fname, true_label, pred_label, correct, img_path))
            print(f"  {correct} {fname}: true={true_label}, predicted={pred_label}")
    except Exception as e:
        print(f"  ✗ {fname}: FAILED ({e})")

# ── Plot internet image predictions ──
if internet_results:
    n = len(internet_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (fname, true_lbl, pred_lbl, correct, img_path) in zip(axes, internet_results):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.resize(img, (160, 160)), cv2.COLOR_BGR2RGB)
        ax.imshow(img); ax.axis('off')
        color = 'green' if correct == '✓' else 'red'
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}  {correct}",
                     fontsize=11, color=color, fontweight='bold')
    plt.suptitle(f"Q1 – Internet Image Predictions  [{best_name}]",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "internet_predictions.png"), dpi=150)
    plt.close()
    print(f"  Plot saved → {OUTPUT_DIR}/internet_predictions.png")

print("\n✅  Q1 complete. All outputs in:", OUTPUT_DIR)
