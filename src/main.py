import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Local imports
from preprocessing import preprocess_data
from optimization import BayesianOptimizer
from visualization import plot_shap_summary, plot_model_comparison

def run_pipeline():
    # 1. Load Data
    data_path = 'data/synthetic_stiffness.csv'
    if not os.path.exists(data_path):
        print("Data not found. Generating it now...")
        sys.path.append('data')
        import generate_synthetic_data
        generate_synthetic_data.generate_data()

    df = pd.read_csv(data_path)
    X = df.drop('Sm', axis=1)
    y = df['Sm']
    
    # Identify categorical columns
    cat_cols = ['Mixture_Type'] if 'Mixture_Type' in X.columns else None

    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Preprocessing (Encoding + Scaling)
    print("Preprocessing data...")
    X_train_proc, X_test_proc, scaler = preprocess_data(X_train, X_test, y_train, cat_cols)

    # 4. Optimization & Training
    # We pass the PROCESSED data to the optimizer
    optimizer = BayesianOptimizer(X_train_proc, y_train, n_trials=10)
    
    models = {}
    models['SVM'] = optimizer.optimize_svm()
    models['Random Forest'] = optimizer.optimize_rf()
    models['CatBoost'] = optimizer.optimize_catboost()

    # 5. Evaluation & SHAP
    results = {}
    for name, model in models.items():
        print(f"\nTraining best {name} on full train set...")
        model.fit(X_train_proc, y_train)
        
        preds = model.predict(X_test_proc)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"  -> {name} R2: {r2:.4f}, RMSE: {rmse:.2f}")
        
        # Generate SHAP plots
        plot_shap_summary(model, X_test_proc, name)

    # 6. Final Comparison
    plot_model_comparison(results)
    
    # Save Best Model (CatBoost)
    models['CatBoost'].save_model('outputs/best_catboost_model.cbm')
    print("\nPipeline Completed. Check 'outputs/' folder for results.")

if __name__ == "__main__":
    run_pipeline()