import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models import StiffnessModels

def train_and_evaluate():
    # 1. Load Data
    data_path = 'data/synthetic_stiffness_data.csv'
    if not os.path.exists(data_path):
        print("Please run data/generate_synthetic_data.py first!")
        return

    df = pd.read_csv(data_path)
    X = df.drop('Sm', axis=1)
    y = df['Sm']

    # 2. Preprocessing (Standardization is crucial for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Split (80/20 as per paper methodology)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Initialize Models
    models = {
        "Random Forest": StiffnessModels.get_random_forest(),
        "CatBoost": StiffnessModels.get_catboost(),
        "SVM": StiffnessModels.get_svm()
    }

    # 5. Train and Predict
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        results[name] = {"R2": r2, "RMSE": rmse}
        print(f"  -> R2: {r2:.4f}, RMSE: {rmse:.2f}")

    # 6. Plot Comparison
    names = list(results.keys())
    r2_scores = [results[n]['R2'] for n in names]
    
    plt.bar(names, r2_scores, color=['blue', 'green', 'orange'])
    plt.title("Model Performance Comparison (R²)")
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()