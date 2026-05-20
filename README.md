# MNIST Handwritten Digit Classification using CNN and Keras Sequential API

This project implements a powerful **Convolutional Neural Network (CNN)** using the **TensorFlow Keras Sequential API** to classify handwritten digits (0–9) from the **MNIST dataset**.  

The project focuses on:
- Strong generalization performance
- Efficient CNN architecture design
- Robust evaluation using multiple metrics
- Detailed visualization and error analysis
- Reproducible deep learning experimentation

The final model achieves **99.50% test accuracy** with exceptionally strong probabilistic performance and class separability.

---

## Project Overview

- Loaded the MNIST dataset from OpenML  
- Reshaped and normalized grayscale images  
- Built a custom deep CNN architecture using:
  - Convolutional layers
  - Batch Normalization
  - MaxPooling
  - Dropout regularization
- Trained using Adam optimizer with dynamic learning rate scheduling  
- Evaluated performance using extensive classification metrics  
- Visualized confusion matrices, ROC curves, PR curves, confidence histograms, and misclassified digits  
- Saved the trained model using Joblib  

---

## Dataset

- **Dataset:** MNIST Handwritten Digit Dataset  
- **Source:** OpenML (`mnist_784`)  
- **Total Samples:** 70,000  
- **Classes:** 10 (Digits 0–9)  
- **Image Size:** 28×28 grayscale  
- **Train/Test Split:** 80% / 20%  

### Preprocessing
- Reshaped images to:
  ```python
  (28, 28, 1)
  ```
- Normalized pixel values to:
  ```python
  [0, 1]
  ```
- Enabled deterministic TensorFlow operations for reproducibility

---

## CNN Model Architecture

The CNN is implemented using the **Keras Sequential API** and consists of multiple convolutional and dense blocks.

## Model Architecture Visualization

The complete CNN architecture diagram generated using TensorFlow/Keras has been included in this repository for better understanding of the layer-wise model structure and data flow.

## Training Strategy

The model was trained with:

- **Epochs:** 25  
- **Validation Split:** 10%  
- **Optimizer:** Adam (`learning_rate = 1e-3`)  
- **Loss Function:** `sparse_categorical_crossentropy`

### Learning Rate Scheduling
Used:
- `ReduceLROnPlateau`

This dynamically reduces the learning rate when validation loss plateaus, improving convergence and generalization.

---

## Model Evaluation Metrics

| Metric                          | Value      |
|---------------------------------|------------|
| **Test Accuracy**               | *99.50%*   |
| **Precision (Weighted)**        | *99.5003%* |
| **Recall (Weighted)**           | *99.50%*   |
| **F1-Score (Weighted)**         | *99.4999%* |
| **ROC-AUC Score (OvR)**         | *99.9965%* |
| **Log Loss**                    | *0.0249*   |
| **Cohen’s Kappa Score**         | *0.9944*   |
| **Matthews Corr. Coefficient**  | *0.9944*   |
| **Top-3 Accuracy**              | *99.95%*   |

---

## Key Highlights

- Deep CNN architecture with strong generalization capability  
- Extensive use of Batch Normalization for stable training  
- Dropout-based regularization to reduce overfitting  
- Extremely high ROC-AUC indicating excellent class separability  
- Very low log loss demonstrating confident probabilistic predictions  
- Detailed multiclass evaluation pipeline  
- Deterministic reproducibility setup for consistent experimentation  

---

## Visualizations Produced

### Dataset Visualization
- Single handwritten digit visualization  
- 10×10 digit sample grid  

### Training Diagnostics
- Training vs Validation Accuracy  
- Training vs Validation Loss  

### Evaluation Visualizations
- Confusion Matrix  
- Per-Class Accuracy Plot  
- ROC Curves (per digit)  
- Precision-Recall Curves (per digit)  
- Classification Report Heatmap  
- Histogram of Prediction Confidence  

### Error Analysis
- First 25 misclassified digit samples  
- Per-class performance inspection  

These visualizations provide deeper insight into model behavior, strengths, and failure patterns.

---

## Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras (Sequential API)  
- **Data Manipulation:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Utilities:** Scikit-learn  
- **Model Persistence:** Joblib  

---

## Saving the Model

The trained CNN model is serialized using Joblib:

```python
import joblib
joblib.dump(model,'CNN_model_mnist.pkl')
```

---

## Author  

**Lavan Kumar Konda**  
-  Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
