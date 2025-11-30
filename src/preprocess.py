import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    # Task 2: Data Cleaning
    df = df.drop_duplicates()
    df = df.dropna() # Simple drop for this dataset as missingness is low
    return df

def handle_outliers(df):
    # Task 4: Outlier Detection & Treatment (Capping)
    for col in ['lead_time', 'adr']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        
        # Capping
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
    
    return df

def encode_data(df):
    # Task 5: Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Using Label Encoding for simplicity in tree models, 
    # but OneHot is better for Logistic Regression. 
    # Here we use pd.get_dummies for true categorical handling.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df