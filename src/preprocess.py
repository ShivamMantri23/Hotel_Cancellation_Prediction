import pandas as pd
import numpy as np

def clean_data(df):
    print("--- Cleaning Data ---")
    df = df.drop_duplicates()
    df = df.dropna()
    
    # 1. EXPLICIT DROP: Look for 'Booking_ID' specifically
    cols_to_drop = [c for c in df.columns if 'booking_id' in c.lower()]
    if cols_to_drop:
        print(f"Explicitly dropping ID column(s): {cols_to_drop}")
        df = df.drop(cols_to_drop, axis=1)

    # 2. HEURISTIC DROP: Check ALL columns for high cardinality
    for col in df.columns:
        if col in ['is_canceled', 'lead_time', 'adr', 'total_stay_nights', 'total_guests']:
            continue
            
        n_unique = df[col].nunique()
        n_rows = len(df)
        
        if n_unique > 0.95 * n_rows:
            print(f"Dropping High Cardinality Column (Likely ID): {col}")
            df = df.drop(col, axis=1)

    return df

def handle_outliers(df):
    target_cols = ['lead_time', 'adr']
    for col in target_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            lower = Q1 - 1.5 * IQR
            
            df[col] = np.where(df[col] > upper, upper, df[col])
            df[col] = np.where(df[col] < lower, lower, df[col])
    
    return df

def encode_data(df):
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"Encoding columns: {list(categorical_cols)}")
    
    # Drop massive columns if they sneak through
    for col in categorical_cols:
        if df[col].nunique() > 500:
             print(f"WARNING: Column '{col}' has too many unique values. Dropping it.")
             df = df.drop(col, axis=1)

    # Re-select and Encode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Encoding complete. New shape: {df.shape}")
    
    # --- CRITICAL: Ensure this line exists and is aligned with 'def' ---
    return df