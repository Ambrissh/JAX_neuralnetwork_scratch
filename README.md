#  JAX Neural Network from Scratch

This repository contains a **from-scratch implementation of a feedforward neural network** built entirely using [JAX](https://github.com/google/jax) and `jax.numpy`.  
The model is trained on the classic **Iris dataset** and demonstrates how deep learning architectures can be constructed manually — without high-level frameworks like TensorFlow or PyTorch.

---

##  Why JAX?

[JAX](https://github.com/google/jax) is a modern high-performance deep learning library built by Google.  
It combines the simplicity of NumPy with **just-in-time compilation (XLA)** and **automatic differentiation**, allowing extremely fast and flexible numerical computing.

**Advantages of JAX:**
-  **Speed:** Uses XLA compiler for optimized execution  
-  **Automatic Differentiation:** No need to manually derive gradients  
-  **NumPy Compatibility:** Drop-in replacement for `numpy`  
- **GPU/TPU acceleration:** Effortlessly runs on high-performance hardware  
-  Widely used in research environments (DeepMind, Google Research, etc.)

---

##  Project Overview

In this project, I have built a **multi-layer neural network** from scratch and train it on the Iris dataset using JAX.

### Model Architecture
nput Layer (4 features)
↓
Hidden Layer 1 (16 neurons, ReLU)
↓
Hidden Layer 2 (8 neurons, ReLU)
↓
Output Layer (3 neurons, Softmax)

---

##  Key Features

-  **Manual Forward & Backward Pass**
-  **Cross-Entropy Loss** implemented mathematically (no prebuilt functions)
-  **L2 Regularization** for weight control
-  **Mini-Batch Gradient Descent**
-  **Train/Test Accuracy Tracking**
-  **Matplotlib-based Accuracy & Loss Visualization**
-  **Modular Structure**  
  Organized cleanly into sections for model, data, loss, training, and evaluation.

---



## 📈 Results

After 200 epochs of training, the model achieves the following:

| Metric | Value |
|--------|--------|
| **Final Train Accuracy** | 0.9833 |
| **Final Test Accuracy**  | 1.0000 |

---

## 🖼️ Training Curves

Below is the training vs test **accuracy and loss** plot generated during model training.

<p align="center">
  <img src="results/fitting_curve.png" width="600" alt="Training and Test Accuracy/Loss curves">
</p>

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/JAX_neuralnetwork_scratch.git
cd JAX_neuralnetwork_scratch

