import pandas as pd
import numpy as np
import os

def generate_data(n_samples=300):
    """
    Generates synthetic asphalt mixture data based on Berangi et al. (2024).
    Includes a categorical variable to demonstrate Order Target Encoding.
    """
    np.random.seed(42)
    
    # Continuous Features (Ranges based on paper Table 2)
    data = {
        'Temperature': np.random.uniform(-5, 40, n_samples),
        'Frequency': np.random.choice([0.1, 0.5, 1, 5, 10, 25], n_samples),
        'Pb': np.random.uniform(3.0, 6.0, n_samples),   # Bitumen Content
        'VMA': np.random.uniform(12, 22, n_samples),    # Voids in Mineral Aggregate
        'Va': np.random.uniform(2, 10, n_samples),      # Air Voids
        'VFA': np.random.uniform(50, 90, n_samples),    # Voids Filled with Asphalt
    }
    
    # Categorical Feature (For Target Encoding demonstration)
    data['Mixture_Type'] = np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], n_samples)

    df = pd.DataFrame(data)
    
    # Generate Target: Stiffness Modulus (Sm)
    # Approx: Sm decreases with Temp, increases with Freq
    base_stiffness = 15000
    
    # Simulate effect of categorical variable
    mix_effect = {'Type_A': 0, 'Type_B': 2000, 'Type_C': -1500, 'Type_D': 500}
    mix_noise = [mix_effect[x] for x in df['Mixture_Type']]

    df['Sm'] = (
        base_stiffness 
        - (350 * df['Temperature']) 
        + (1200 * np.log1p(df['Frequency'])) 
        - (600 * (df['Pb'] - 4.5)) 
        + np.array(mix_noise)
        + np.random.normal(0, 1000, n_samples) # Noise
    )
    
    # Ensure positive stiffness
    df['Sm'] = df['Sm'].clip(lower=1000)
    
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'synthetic_stiffness.csv')
    df.to_csv(csv_path, index=False)
    print(f"Synthetic data generated: {csv_path}")

if __name__ == "__main__":
    generate_data()