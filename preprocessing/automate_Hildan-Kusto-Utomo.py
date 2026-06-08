import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
import os

def preprocess_data(raw_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Preprocess diabetes dataset automatically.
    
    Args:
        raw_path: Path to raw diabetes CSV
        output_dir: Directory to save preprocessed files
        test_size: Proportion for test split
        random_state: Random state for reproducibility
    """
    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_allowed:
        df[col] = df[col].replace(0, np.nan)
    
    for col in zero_not_allowed:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    print(f"[INFO] Missing values after handling: {df.isnull().sum().sum()}")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = X_train_scaled.copy()
    train_df['Outcome'] = y_train.values
    train_path = os.path.join(output_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    
    test_df = X_test_scaled.copy()
    test_df['Outcome'] = y_test.values
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    
    # Save scaler for inference
    import joblib
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"[INFO] Train saved: {train_path} ({train_df.shape})")
    print(f"[INFO] Test saved: {test_path} ({test_df.shape})")
    print(f"[INFO] Scaler saved: {scaler_path}")
    print("[INFO] Preprocessing completed successfully!")
    
    return train_path, test_path, scaler_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automate Diabetes Data Preprocessing')
    parser.add_argument('--raw-path', type=str, default='../diabetes_raw/diabetes.csv',
                        help='Path to raw diabetes CSV')
    parser.add_argument('--output-dir', type=str, default='diabetes_preprocessing',
                        help='Output directory for preprocessed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state')
    
    args = parser.parse_args()
    preprocess_data(args.raw_path, args.output_dir, args.test_size, args.random_state)
