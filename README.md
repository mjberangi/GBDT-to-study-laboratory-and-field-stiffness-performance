Excellent content — this is already **strong**, but it suffers from two issues that *immediately break professional perception*:

1. **Duplication / conflict** (the `=======` section looks like a merge artifact)
2. Lack of a **clean narrative hierarchy** (problem → method → rigor → relevance)

Below is a **fully cleaned, recruiter-grade README**, rewritten to match how **senior ML engineers / applied researchers** present work.
This is **copy–paste ready**.

---

# Gradient Boosting Decision Trees for Asphalt Stiffness Prediction

**Laboratory–Field Performance Modeling**

## 🔍 Problem Context

Accurate prediction of asphalt mixture stiffness is essential for pavement design, performance assessment, and lifecycle management. Traditional mechanistic or regression-based approaches struggle to generalize across laboratory and field conditions due to nonlinear material behavior and heterogeneous mixture properties.

## 🎯 Objective

This repository provides a **reproducible machine learning implementation** of the methodology presented in:

> **Berangi et al. (2024)** – *Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures*

The project focuses on predicting the **Stiffness Modulus (Sm)** of asphalt mixtures and rigorously comparing advanced and classical machine learning models.

> ⚠️ Due to data confidentiality constraints, **synthetic datasets** are generated to statistically mirror the original laboratory and field measurements.

---

## 🧠 Models Evaluated

* **CatBoost (Gradient Boosting Decision Trees)**
* **Random Forest (RF)**
* **Support Vector Machines (SVM)**

CatBoost is emphasized due to its ability to:

* Handle nonlinear interactions
* Process categorical variables effectively
* Maintain strong generalization under limited data regimes

---

## 🚀 Key Technical Features

This repository goes beyond basic model training and reflects **research-grade ML practice**:

* **Bayesian Hyperparameter Optimization**
  Hyperparameters are optimized using **Optuna**, minimizing RMSE through efficient search strategies.

* **Order Target Encoding**
  Demonstrates advanced categorical feature handling (e.g., mixture type), particularly relevant for material classification problems.

* **Model Interpretability (SHAP)**
  SHapley Additive exPlanations are used to quantify feature importance and explain nonlinear model behavior.

* **Robust Evaluation Protocol**
  Uses **5-fold cross-validation** to ensure statistically reliable performance estimates.

* **Confidentiality-Preserving Design**
  A synthetic data generator reproduces the statistical structure of real-world datasets without exposing sensitive data.

---

## 📊 Input Parameters

The models use laboratory-based IT-CY test parameters:

1. **Temperature** (°C)
2. **Loading Frequency** (Hz)
3. **Bitumen Content (Pb)** (%)
4. **Voids in Mineral Aggregate (VMA)** (%)
5. **Air Voids (Va)** (%)
6. **Voids Filled with Asphalt (VFA)** (%)
7. **Mixture Type** (categorical; used to demonstrate target encoding)

---

## 🔬 Modeling Pipeline

The full workflow is executed via `src/main.py`:

1. **Preprocessing**

   * Standard scaling for numerical features
   * Order Target Encoding for categorical variables

2. **Hyperparameter Optimization**

   * Bayesian optimization (10–15 trials per model)
   * Objective: minimize RMSE

3. **Model Training**

   * Train optimized CatBoost, RF, and SVM models

4. **Evaluation & Interpretation**

   * Performance comparison on test data
   * SHAP-based explanation of feature effects

---

## 📁 Repository Structure

```
├── data/
│   └── generate_synthetic_data.py
├── src/
│   ├── preprocessing.py
│   ├── optimization.py
│   ├── visualization.py
│   ├── predict.py
│   └── main.py
├── notebooks/
│   ├── 1_Training_Experiment.ipynb
│   └── 2_Inference_Demo.ipynb
├── outputs/
│   ├── models/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Generate synthetic dataset

```bash
python data/generate_synthetic_data.py
```

### 3️⃣ Run full training pipeline

```bash
python src/main.py
```

This will:

* Preprocess data
* Optimize hyperparameters
* Train all models
* Save trained models and plots to `outputs/`

### 4️⃣ Run inference

```bash
python src/predict.py
```

---

## 📈 Results Summary

Consistent with the findings of **Berangi et al. (2024)**:

* **CatBoost** consistently achieves:

  * Higher R²
  * Lower RMSE
* Particularly effective when modeling:

  * Nonlinear material behavior
  * Interactions between volumetric and loading parameters
  * Categorical mixture descriptors

---

## 🏗️ Engineering & Research Relevance

This work demonstrates how interpretable machine learning can support:

* Asphalt mixture design optimization
* Performance-based pavement engineering
* Bridging laboratory test results with field performance
* Transparent ML adoption in civil infrastructure decision-making

---

## 📄 Reference

Berangi, M., *et al.* (2024).
**Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures.**
*Computer-Aided Civil and Infrastructure Engineering.*
