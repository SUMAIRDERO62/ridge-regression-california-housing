# Progree — Machine Learning Internship Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)](https://www.tensorflow.org/)

A collection of end-to-end machine learning projects completed during my **Machine Learning Internship at Progree**.

The portfolio covers three core machine learning workflows: **regularized regression, deep learning-based image classification, and highly imbalanced clinical classification**. Each task is organized as an independent project with reproducible workflows and supporting documentation.

---

## Overview

This repository demonstrates practical implementation of machine learning concepts across different problem types and model families.

| Task  | Project                                                          | Key Techniques                                                                                              |
| ----- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **2** | [Ridge Regression Pipeline](./task2-ridge-regression)            | Preprocessing, imputation, scaling, categorical encoding, Ridge Regression, GridSearchCV, residual analysis |
| **3** | [CNN Image Classification](./task3-cnn-image-classification)     | CNN, convolution, max-pooling, dense layers, dropout, batch normalization, data augmentation                |
| **4** | [Clinical Disease Predictor](./task4-clinical-disease-predictor) | SMOTE, SVM, Random Forest, XGBoost, hyperparameter tuning, recall-focused evaluation, cross-validation      |

**Task 1** was completed as the professional LinkedIn announcement for the internship.

---

## Repository Structure

```text
progree-ml-internship/
│
├── README.md
├── GUIDE.md
├── .gitignore
│
├── task2-ridge-regression/
│   ├── main.py
│   ├── src/
│   ├── tests/
│   ├── results/
│   ├── requirements.txt
│   └── README.md
│
├── task3-cnn-image-classification/
│   ├── main.py
│   ├── src/
│   ├── results/
│   ├── requirements.txt
│   └── README.md
│
└── task4-clinical-disease-predictor/
    ├── main.py
    ├── src/
    ├── tests/
    ├── report/
    ├── results/
    ├── requirements.txt
    └── README.md
```

---

## Task 2 — Ridge Regression Pipeline

### Objective

Build a supervised regression pipeline capable of predicting a continuous numerical target using a standard machine learning dataset.

### Implementation

The project uses a structured Scikit-learn pipeline covering:

* Missing-value imputation
* Numerical feature scaling
* Categorical feature encoding
* Ridge regression
* Baseline comparison
* Hyperparameter optimization with `GridSearchCV`
* Residual analysis
* Model evaluation using:

  * **R²**
  * **RMSE**

### Project

[View Task 2 →](./task2-ridge-regression)

---

## Task 3 — CNN Image Classification

### Objective

Develop a multi-class image classification pipeline using a convolutional neural network to classify raw image pixel data.

### Implementation

The deep learning pipeline includes:

* TensorFlow/Keras
* Convolutional layers
* Max-pooling layers
* Fully connected layers
* Dropout regularization
* Batch normalization
* Data augmentation
* Training and validation monitoring
* Augmentation analysis
* Accuracy curve visualization

The project is designed around standard image classification datasets such as **CIFAR-10 / MNIST**.

### Project

[View Task 3 →](./task3-cnn-image-classification)

---

## Task 4 — High-Imbalance Clinical Disease Predictor

### Objective

Build an end-to-end classification pipeline for detecting physiological abnormalities in a highly imbalanced clinical dataset.

Because medical classification problems can contain substantial class imbalance, the project focuses particularly on **recall/sensitivity** rather than relying only on overall accuracy.

### Implementation

The pipeline includes:

* Imbalanced clinical dataset preparation
* SMOTE-based synthetic minority oversampling
* Cross-validation
* SVM
* Random Forest
* XGBoost
* Hyperparameter tuning
* Recall-focused model evaluation
* SMOTE ablation analysis
* Comparative model evaluation

### Project

[View Task 4 →](./task4-clinical-disease-predictor)

---

## Engineering Practices

The projects follow a consistent engineering approach:

* **Modular architecture** using dedicated `src/` packages
* **Thin command-line interfaces** through `main.py`
* **Configurable execution** through command-line arguments
* **Leak-free preprocessing** within training workflows
* **SMOTE applied within cross-validation workflows**
* **Baseline comparisons** to establish meaningful reference points
* **Ablation experiments** to evaluate the effect of specific techniques
* **Reproducible experiments** using fixed random seeds
* **Saved evaluation results** in structured JSON/CSV formats
* **Training and evaluation plots** for model analysis
* **Unit tests** for applicable regression and classification components

---

## Reproducibility

Each task is independently executable and contains its own dependency configuration.

### Task 2

```bash
cd task2-ridge-regression
pip install -r requirements.txt
python main.py
```

### Task 3

```bash
cd task3-cnn-image-classification
pip install -r requirements.txt
python main.py --subset 2000 --epochs 2
```

### Task 4

```bash
cd task4-clinical-disease-predictor
pip install -r requirements.txt
python main.py --quick
```

For detailed instructions, refer to the project-specific `README.md` files and the repository [GUIDE.md](./GUIDE.md).

---

## Technologies

**Programming**

* Python 3.10+

**Machine Learning**

* Scikit-learn
* XGBoost
* Imbalanced-learn

**Deep Learning**

* TensorFlow
* Keras

**Data Processing**

* NumPy
* Pandas

**Evaluation & Visualization**

* Matplotlib
* Scikit-learn metrics

**Testing**

* Pytest

---

## Learning Outcomes

This internship portfolio provided practical experience across the machine learning lifecycle, including:

* Designing preprocessing pipelines
* Building regularized regression models
* Performing hyperparameter optimization
* Developing CNN-based image classifiers
* Applying data augmentation to vision tasks
* Handling severe class imbalance
* Comparing multiple classification algorithms
* Optimizing models for recall-sensitive applications
* Using cross-validation for more reliable evaluation
* Structuring machine learning projects for reproducibility

---

## Author

**Sumair Ahmed**

Machine Learning Intern @ Progree
BS Artificial Intelligence Student | AI/ML Engineer

* GitHub: [SUMAIRDERO7](https://github.com/SUMAIRDERO762)
* LinkedIn: [Sumair Ahmed Dero](https://www.linkedin.com/in/sumair-ahmed-dero-70ba852a0)

---

## Internship Tasks

This repository corresponds to the technical project work completed as part of the **Progree Machine Learning Internship**.

The internship task sequence covered:

1. Professional internship announcement
2. Predictive Linear/Ridge Regression Pipeline
3. Multi-Class CNN Image Classification
4. High-Imbalance Clinical Disease Predictor

Each technical task is maintained as a separate, reproducible project within this repository.
