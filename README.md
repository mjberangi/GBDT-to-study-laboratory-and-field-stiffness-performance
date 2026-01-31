# Gradient Boosting Decision Trees for Asphalt Stiffness Prediction

## Overview

This repository implements the machine learning methodology presented in the research paper **"Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures"** (Berangi et al., 2024).

The project predicts the **Stiffness Modulus (Sm)** of asphalt mixtures using various input parameters. It performs a rigorous comparison between **CatBoost (GBDT)**, **Random Forest (RF)**, and **Support Vector Machines (SVM)**.

## Key Features
This implementation goes beyond basic modeling to include advanced techniques described in the study:
* **Bayesian Optimization:** Automates hyperparameter tuning using **Optuna** to find the optimal configuration for each model.
* **Order Target Encoding:** Implements advanced handling of categorical variables (demonstrated on synthetic mixture types) to improve prediction power.
* **SHAP Analysis:** Provides model interpretability through SHapley Additive exPlanations, visualizing feature importance and impact.
* **Robust Evaluation:** Uses **5-Fold Cross-Validation** to ensure reliable performance metrics.
* **Confidentiality Preserved:** Includes a statistical data generator to create synthetic datasets that mirror the properties of the confidential field data.
=======
This repository implements the simple version of machine learning models presented in the research paper **"Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures"** (Berangi et al., 2024).

The project predicts the **Stiffness Modulus (Sm)** of asphalt mixtures using various input parameters. It compares the performance of **Random Forest (RF)**, **CatBoost (CB)**, and **Support Vector Machines (SVM)**. 

**Note:** in this repository a synthetic dataset is used.
>>>>>>> 7c9ad62657673725868b16f88475bb6a669320fc

## Methodology

### Input Parameters
The model uses the following parameters based on laboratory tests (IT-CY):
1.  **Temperature** (°C)
2.  **Loading Frequency** (Hz)
3.  **Bitumen Content** (Pb %)
4.  **Voids in Mineral Aggregate** (VMA %)
5.  **Air Voids** (Va %)
6.  **Voids Filled with Asphalt** (VFA %)
7.  **Mixture Type** (Categorical variable added to demonstrate Order Target Encoding)

### Modeling Pipeline
The `src/main.py` script executes the following workflow:
1.  **Preprocessing:** Applies Order Target Encoding to categorical features and Standard Scaling to numerical inputs.
2.  **Optimization:** Runs Bayesian Optimization (10-15 trials) to minimize RMSE.
3.  **Training:** Trains the best version of CatBoost, RF, and SVM on the full training set.
4.  **Inference & Explanation:** Evaluates on the test set and generates SHAP summary plots.

## Repository Structure
* `data/`: Contains the synthetic data generator.
* `src/preprocessing.py`: Handles scaling and target encoding.
* `src/optimization.py`: Defines the Bayesian Optimization logic with Optuna.
* `src/visualization.py`: Generates and saves SHAP and comparison plots.
* `src/predict.py`: Utilities for loading saved models and making new predictions.
* `src/main.py`: The entry point that runs the full training pipeline.
* `notebooks/`:
    * `1_Training_Experiment.ipynb`: Step-by-step interactive demo of the training process.
    * `2_Inference_Demo.ipynb`: Shows how to load the saved model and make predictions.
* `outputs/`: Stores saved models (`.cbm`) and performance plots.

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Synthetic Data:**
    ```bash
    python data/generate_synthetic_data.py
    ```

3.  **Run Full Pipeline:**
    ```bash
    python src/main.py
    ```
    *This will preprocess data, optimize hyperparameters, train models, and save results to the `outputs/` folder.*

4.  **Make Predictions (Inference):**
    ```bash
    python src/predict.py
    ```

## Results
Consistent with the findings in Berangi et al. (2024), the **CatBoost** model typically demonstrates superior performance (higher R² and lower RMSE) compared to Random Forest and SVM, particularly when handling complex non-linear relationships and categorical data.

## Citation
> Berangi, M., et al. (2024). Gradient boosting decision trees to study laboratory and field performance in asphalt mixtures. *Computer-Aided Civil and Infrastructure Engineering*.
