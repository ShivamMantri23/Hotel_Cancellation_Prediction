import os
import logging
import sys
from datetime import datetime

# Import your modules
from src.data_loader import load_data
from src.eda import perform_eda
from src.preprocess import clean_data, handle_outliers, encode_data
from src.feature_engineering import create_features
from src.train import train_models
from src.evaluate import evaluate_models

# --- Task 10: Implement Proper Logging ---
def setup_logging():
    """Sets up logging to file and console."""
    os.makedirs("outputs", exist_ok=True)
    log_filename = f"outputs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename), # Save to file
            logging.StreamHandler(sys.stdout)  # Print to console
        ]
    )
    logging.info(f"Logging setup complete. Log file: {log_filename}")

def main():
    setup_logging()
    logging.info("Starting Hotel Cancellation Prediction Pipeline...")
    
    DATA_PATH = "data/hotel_reservations.csv"
    
    try:
        # --- Step 1: Data Loading ---
        logging.info("Step 1: Loading Data...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        
        df = load_data(DATA_PATH)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # --- Step 2: EDA ---
        logging.info("Step 2: Performing EDA...")
        perform_eda(df)

        # --- Step 3: Preprocessing ---
        logging.info("Step 3: Cleaning Data and Handling Outliers...")
        df = clean_data(df)
        df = handle_outliers(df)
        
        # --- Step 4: Feature Engineering ---
        logging.info("Step 4: Engineering Features...")
        df = create_features(df)
        
        # --- Step 5: Encoding ---
        logging.info("Step 5: Encoding Categorical Variables...")
        df = encode_data(df)
        
        # Safety Check
        if df is None:
            raise ValueError("Dataframe became None after encoding.")
        
        logging.info(f"Data preparation complete. Final Shape: {df.shape}")

        # --- Step 6: Training ---
        logging.info("Step 6: Training Models (Logistic Regression, RF, XGBoost)...")
        # Note: We are using Class Weights in train.py to handle imbalance (Task 6)
        models, X_test, y_test = train_models(df)

        # --- Step 7: Evaluation ---
        logging.info("Step 7: Evaluating Models...")
        evaluate_models(models, X_test, y_test)
        
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        # --- Task 10: Error Handling ---
        logging.error("Pipeline Failed!", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}")
        print("Check 'outputs/' folder for the log file details.")

if __name__ == "__main__":
    main()