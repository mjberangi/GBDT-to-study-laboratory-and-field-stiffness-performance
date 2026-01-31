import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_shap_summary(model, X_test, model_name):
    """Generates SHAP summary plot"""
    print(f"Generating SHAP plots for {model_name}...")
    
    # Use generic KernelExplainer for SVM (slow), TreeExplainer for trees (fast)
    if "SVM" in model_name:
        # Using KMeans to summarize background data for speed
        X_summary = shap.kmeans(X_test, 10) 
        explainer = shap.KernelExplainer(model.predict, X_summary)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    plt.figure()
    plt.title(f"SHAP Summary: {model_name}")
    shap.summary_plot(shap_values, X_test, show=False)
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/shap_{model_name}.png', bbox_inches='tight')
    plt.close()

def plot_model_comparison(results):
    """Bar chart comparing RMSE/R2 of all models"""
    df = pd.DataFrame(results).T
    
    plt.figure(figsize=(10, 5))
    df['R2'].plot(kind='bar', color='teal')
    plt.title("Model R² Score Comparison")
    plt.ylabel("R² Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison_r2.png')
    plt.close()
    print("Comparison plots saved to outputs/")