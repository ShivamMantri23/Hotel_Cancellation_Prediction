from src.data_loader import load_data
from src.eda import perform_eda
from src.preprocess import clean_data, handle_outliers, encode_data
from src.feature_engineering import create_features
from src.train import train_models
from src.evaluate import evaluate_models
import os

def main():
    # Setup
    DATA_PATH = "data/hotel_reservations.csv" # Ensure this matches your file name
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(e)
        return

    # 2. EDA
    perform_eda(df)

    # 3. Cleaning & Outliers
    df = clean_data(df)
    df = handle_outliers(df)

    # 4. Feature Engineering
    df = create_features(df)

    # 5. Encoding
    df = encode_data(df)

    # 6. Train
    models, X_test, y_test = train_models(df)

    # 7. Evaluate
    evaluate_models(models, X_test, y_test)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()