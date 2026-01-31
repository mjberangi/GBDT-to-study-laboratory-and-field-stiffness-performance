# Gradient Boosting Decision Trees for Asphalt Stiffness Prediction

## Overview
This repository implements the simple version of machine learning models presented in the research paper **"Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures"** (Berangi et al., 2024).

The project predicts the **Stiffness Modulus (Sm)** of asphalt mixtures using various input parameters. It compares the performance of **Random Forest (RF)**, **CatBoost (CB)**, and **Support Vector Machines (SVM)**. 

**Note:** in this repository a synthetic dataset is used.

## Methodology
As detailed in the paper, the study utilizes experimental data from laboratory tests (IT-CY) on various asphalt mixtures.

### Input Parameters
1. **Temperature** (°C)
2. **Loading Frequency** (Hz)
3. **Bitumen Content** (Pb %)
4. **Voids in Mineral Aggregate** (VMA %)
5. **Air Voids** (Va %)
6. **Voids Filled with Asphalt** (VFA %)

### Models Implemented
* **CatBoost:** A gradient boosting algorithm that handles categorical data effectively and reduces overfitting.
* **Random Forest:** An ensemble method using bagging.
* **SVM:** Support Vector Regression with an RBF kernel.

## Results
According to the study, the **CatBoost** model outperformed others, showing higher accuracy (R²) and lower error (RMSE) in predicting pavement stiffness compared to RF and SVM.

## Usage
**Note:** The real dataset is confidential. A synthetic data generator is provided to demonstrate the code functionality.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Synthetic Data:**
    ```bash
    python data/generate_synthetic_data.py
    ```

3.  **Train Models:**
    ```bash
    python src/train.py
    ```

## Citation
> Berangi, M., et al. (2024). Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures. *Computer-Aided Civil and Infrastructure Engineering*.
