import pandas as pd
import numpy as np
import os

def generate_data(n_samples=100):
    """
    Generates synthetic asphalt mixture data based on Table 2 statistics
    from Berangi et al. (2024).
    """
    np.random.seed(42)
    
    # Ranges based on Table 2 in the paper
    data = {
        # Feature 1: Temperature (C) - Range -5 to 40
        'Temperature': np.random.uniform(-5, 40, n_samples),
        
        # Feature 2: Loading Frequency (Hz) - Logarithmic scale often used, but uniform for synthetic
        'Frequency': np.random.choice([0.1, 0.5, 1, 5, 10, 25], n_samples),
        
        # Feature 3: Bitumen Content (%) - Range 3.0 to 6.0
        'Pb': np.random.uniform(3.0, 6.0, n_samples),
        
        # Feature 4: Void in Mineral Aggregate (%) - Range 12 to 22
        'VMA': np.random.uniform(12, 22, n_samples),
        
        # Feature 5: Air Voids (%) - Range 2 to 10
        'Va': np.random.uniform(2, 10, n_samples),
        
        # Feature 6: Voids Filled with Asphalt (%) - Range 50 to 90
        'VFA': np.random.uniform(50, 90, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate Target: Stiffness Modulus (Sm)
    # Using a simplified approximation of the physical relationships described in the paper
    # Sm decreases with Temp, increases with Freq
    base_stiffness = 15000
    df['Sm'] = (
        base_stiffness 
        - (300 * df['Temperature']) 
        + (100 * np.log1p(df['Frequency'])) 
        - (500 * (df['Pb'] - 4.5)) 
        + np.random.normal(0, 1000, n_samples) # Add noise
    )
    
    # Ensure positive stiffness
    df['Sm'] = df['Sm'].clip(lower=1000)
    
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'synthetic_stiffness_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Synthetic data generated at {csv_path}")

if __name__ == "__main__":
    generate_data()