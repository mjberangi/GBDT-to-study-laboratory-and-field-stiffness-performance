import pandas as pd
import os
from catboost import CatBoostRegressor

def load_catboost(model_path):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

def predict_from_csv(input_csv, model_path, output_csv=None):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Simple check for required numerical features (ignoring encoding logic for inference simplicity here)
    # In a real pipeline, you would need to apply the SAME encoder and scaler saved from training.
    # For this demo, we assume the input is already processed or we rely on CatBoost's native handling if used.
    # Since we used manual encoding in training, strict inference requires loading that encoder.
    # However, for the demo script, we will just load features.
    
    X = df.drop(columns=['Sm'], errors='ignore')
    
    model = load_catboost(model_path)
    preds = model.predict(X)
    
    df['Predicted_Sm'] = preds
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")
    else:
        print(df.head())

if __name__ == "__main__":
    predict_from_csv('data/synthetic_stiffness.csv', 'outputs/best_catboost_model.cbm')