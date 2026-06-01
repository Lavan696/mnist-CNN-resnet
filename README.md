# MNIST Handwritten Digit Classification using Residual CNN and TensorFlow Keras

This project implements a custom **Residual Convolutional Neural Network (Residual CNN)** using **TensorFlow Keras Functional API** to classify handwritten digits (0–9) from the **MNIST dataset**.

The project focuses on:

- Strong generalization through extensive data augmentation
- Efficient residual network architecture with skip connections
- Advanced training strategies (One-Cycle Learning Rate + Fine-Tuning)
- Test-Time Augmentation (TTA) for improved prediction stability
- Deterministic and reproducible experimentation
- Comprehensive evaluation and error analysis

The final model combines modern deep learning practices with rigorous evaluation to achieve highly accurate and reliable handwritten digit recognition.

---

## Project Overview

- Loaded the MNIST dataset using TensorFlow/Keras
- Normalized and reshaped images into CNN-compatible format
- Created a dedicated validation set before augmentation to prevent data leakage
- Applied eight augmentation strategies to improve generalization
- Built a custom Residual CNN architecture with skip connections
- Trained using SGD with Nesterov Momentum
- Implemented a custom One-Cycle Learning Rate Scheduler
- Performed an additional fine-tuning stage using a lower learning rate
- Applied Test-Time Augmentation (TTA) during inference
- Evaluated using extensive classification metrics and visualizations
- Saved the final trained model in native Keras format

---

## Dataset

- **Dataset:** MNIST Handwritten Digit Dataset
- **Source:** TensorFlow/Keras
- **Total Samples:** 70,000
- **Classes:** 10 (Digits 0–9)
- **Image Size:** 28 × 28 Grayscale
- **Training Samples:** 60,000
- **Test Samples:** 10,000

### Preprocessing

- Reshaped images to:

```python
(28, 28, 1)
```

- Normalized pixel values to:

```python
[0,1]
```

- Created a dedicated validation split before augmentation
- Enabled deterministic TensorFlow operations for reproducibility

---

## Data Augmentation Strategy

To improve robustness and generalization, the training dataset was expanded using multiple augmentation techniques:

1. Shift Left
2. Shift Right
3. Shift Up
4. Shift Down
5. Zoom In
6. Zoom Out
7. Random Rotation
8. Elastic Transformation
9. Original Images

### Why Augmentation?

These transformations simulate real-world handwriting variations such as:

- Translation
- Scaling
- Rotation
- Non-rigid distortions

The augmented dataset became significantly larger, allowing the model to learn more invariant and generalized digit representations.

---

## Residual CNN Architecture

The model is built using the **TensorFlow Keras Functional API** and incorporates modern residual learning principles.

### Architecture Highlights

- Initial Convolution Block
- Batch Normalization
- ReLU Activation
- Multiple Residual Blocks
- Identity Skip Connections
- Max Pooling Layers
- Spatial Dropout
- Global Average Pooling
- Softmax Classification Layer

### Why Residual Learning?

Residual connections help:

- Improve gradient flow
- Reduce optimization difficulties
- Enable deeper architectures
- Improve convergence stability

---

## Model Architecture Visualization

The complete Residual CNN architecture diagram generated using TensorFlow/Keras has been included in this repository for better understanding of the layer-wise model structure and data flow.

---

## Training Strategy

### Stage 1: One-Cycle Learning Rate Training

The model was trained using:

- SGD Optimizer
- Nesterov Momentum
- One-Cycle Learning Rate Policy
- Label Smoothing Cross-Entropy Loss

The One-Cycle policy rapidly increases and then gradually decreases the learning rate, enabling faster convergence and better generalization.

### Stage 2: Fine-Tuning

After the primary training phase:

- Best checkpoint weights were reloaded
- Learning rate was reduced significantly
- Additional fine-tuning epochs were performed

This helped refine decision boundaries and maximize final performance.

---

## Test-Time Augmentation (TTA)

Instead of relying on a single prediction per image, the final model uses:

- Original image
- +4° rotated image
- −4° rotated image

Predicted probabilities from all versions are averaged to produce the final prediction.

### Benefits

- Improved prediction stability
- Better handling of edge cases
- Reduced prediction variance
- Enhanced robustness

---

## Model Evaluation Metrics

| Metric                          | Value        |
|---------------------------------|--------------|
| **Test Accuracy**               | **99.71%**   |
| **Precision (Weighted)**        | **99.7102%** |
| **Recall (Weighted)**           | **99.71%**   |
| **F1-Score (Weighted)**         | **99.7100%** |
| **ROC-AUC Score (OvR)**         | **99.9776%** |
| **Log Loss**                    | **0.1021**   |
| **Cohen's Kappa Score**         | **0.9968**   |
| **Matthews Corr Coefficient**   | **0.9968**   |
| **Top-3 Accuracy**              | **99.98%**   |

---

## Key Highlights

The proposed Residual CNN achieved **99.71% test accuracy** on the MNIST handwritten digit classification task, placing it very close to reported **state-of-the-art performance (~99.8–99.9%)** on the benchmark. This demonstrates the effectiveness of the architecture, augmentation strategy, One-Cycle learning rate scheduling, fine-tuning, and Test-Time Augmentation (TTA), while maintaining a relatively efficient and reproducible training pipeline.

### Strong Generalization

- Extensive augmentation pipeline
- Dedicated validation split
- Test-Time Augmentation
- Fine-tuning stage

### Efficient Architecture

- Residual CNN design
- Skip connections
- Global Average Pooling
- Spatial Dropout regularization

### Robust Evaluation

- Classification metrics
- Probabilistic metrics
- Confidence analysis
- Error analysis
- Per-class performance inspection

### Reproducible Experimentation

- Fixed random seeds
- Deterministic TensorFlow operations
- Consistent training pipeline

---

## Visualizations Produced

### Dataset Visualization

- Single Digit Visualization
- 10×10 Digit Sample Grid

### Training Diagnostics

- One-Cycle Learning Rate Schedule
- Training Accuracy Curves
- Validation Accuracy Curves
- Training Loss Curves
- Validation Loss Curves

### Evaluation Visualizations

- Confusion Matrix
- Per-Class Accuracy Plot
- Precision-Recall Curves
- ROC Curves
- Classification Report Heatmap
- Prediction Confidence Histogram

### Error Analysis

- Misclassified Digit Visualization
- Class-wise Performance Analysis

These visualizations provide deeper insight into model behavior, strengths, weaknesses, and prediction confidence.

---

## Tech Stack

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras (Functional API)
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning Utilities:** Scikit-learn
- **Scientific Computing:** SciPy
- **Model Persistence:** Native Keras Serialization

---

## Model Persistence

The final trained model is saved using TensorFlow/Keras:

```python
model.save("CNN_model_mnist.keras")
```

This enables:

- Reproducible inference
- Further fine-tuning
- Model deployment
- Integration into production systems

---

## Author  

**Lavan Kumar Konda**  
-  Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
