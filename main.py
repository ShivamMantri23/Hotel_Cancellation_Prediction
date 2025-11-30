from src.data_loader import load_data
from src.eda import perform_eda
from src.preprocess import clean_data, handle_outliers, encode_data
from src.feature_engineering import create_features
from src.train import train_models
from src.evaluate import evaluate_models
import os

def main():
    DATA_PATH = "data/hotel_reservations.csv"
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load
    df = load_data(DATA_PATH)
    if df is None: return

    # 2. EDA
    perform_eda(df)

    # 3. Cleaning & Outliers
    df = clean_data(df)
    print(f"After Cleaning: {df.shape}") # Debug check

    df = handle_outliers(df)
    print(f"After Outliers: {df.shape}") # Debug check

    # 4. Feature Engineering
    df = create_features(df)
    print(f"After Features: {df.shape}") # Debug check

    # 5. Encoding
    df = encode_data(df)
    
    # *** SAFETY CHECK ***
    if df is None:
        print("ERROR: Data became None after encoding! Check preprocess.py")
        return
    print(f"After Encoding: {df.shape}") 

    # 6. Train
    models, X_test, y_test = train_models(df)

    # 7. Evaluate
    evaluate_models(models, X_test, y_test)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()