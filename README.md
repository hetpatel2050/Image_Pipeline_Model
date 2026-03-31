# Image Pipeline Model

This project is a complete image processing and machine learning pipeline developed for CVI620 - Assignment 2. It explores and compares multiple traditional machine learning and neural network models using two distinct image classification datasets.

## Project Structure

The assignment is divided into two primary parts:

### Part 1: Cat vs. Dog Classification (`q1_cat_dog.py`)
This script implements a classification pipeline on a subset of Cat and Dog images.
* **Feature Extraction**: Extracts normalized grayscale pixels, RGB color histograms, and Canny edge features from raw images.
* **Dimensionality Reduction**: Implements PCA (Principal Component Analysis) to retain components capturing the majority of the variance for speed optimization.
* **Models Trained**:
  * Support Vector Machine (RBF Kernel)
  * Random Forest (200 Trees)
  * K-Nearest Neighbors (k=5)
  * Multi-Layer Perceptron (Neural Network)
* **Outputs**: Accuracy comparison charts, confusion matrices, a sample of images, and internet prediction results on newly downloaded images. All plots and the best performing model are saved to `Q1/output/`.

### Part 2: MNIST Digit Classification (`q2_mnist.py`)
This script attacks the classic MNIST Handwritten Digits dataset with a target of >=90% classification accuracy.
* **Preprocessing**: Normalizes pixel values and applies PCA to drastically reduce training times.
* **Models Trained**:
  * SVM (RBF, C=5)
  * Random Forest (300 Trees)
  * KNN (k=3)
  * Multi-Layer Perceptron (Neural Net: 512-256-128 configuration)
* **Outputs**: Training results summarizing whether each model crossed the 90% threshold, along with sample predictions, all confusion matrices, and accuracy comparison plots. Saved to `Q2/output/`.

## Dependencies

You need to have the following libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python` (cv2)
- `scikit-learn`
- `joblib`
- `requests`

You can install the requirements with:
```bash
pip install numpy pandas matplotlib opencv-python scikit-learn joblib requests
```

## How to Run

1. Clone the repository and ensure your dataset folders (`Q1/train`, `Q1/test` and `Q2/mnist_train.csv`, `Q2/mnist_test.csv`) are located inside their primary folders.
2. Run the scripts using Python:

```bash
# To run Cat vs. Dog Classification:
python q1_cat_dog.py

# To run MNIST Classification:
python q2_mnist.py
```

The scripts will print metrics to the console and dump visualizations into `Q1/output/` and `Q2/output/` respectively.
