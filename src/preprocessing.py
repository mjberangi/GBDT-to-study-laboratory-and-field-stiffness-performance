import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(X_train, X_test, y_train, cat_cols=None):
    """
    Applies Order Target Encoding to categorical features and 
    Standard Scaling to numerical features.
    """
    # 1. Target Encoding for Categorical Variables
    # While CatBoost handles this natively, we implement it explicitly 
    # for SVM/RF comparison equality.
    if cat_cols:
        encoder = ce.TargetEncoder(cols=cat_cols)
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
    else:
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()

    # 2. Scaling (Critical for SVM)
    scaler = StandardScaler()
    # Fit scaler on encoded training data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=X_train_enc.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns)
    
    return X_train_scaled, X_test_scaled, scaler