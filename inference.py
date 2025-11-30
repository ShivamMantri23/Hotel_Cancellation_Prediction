import joblib
import pandas as pd
import numpy as np
import os
from src.preprocess import clean_data, handle_outliers, encode_data
from src.feature_engineering import create_features

def make_prediction(sample_data):
    """
    Loads the saved model and predicts cancellation for a single booking.
    """
    model_path = 'outputs/best_model.pkl'
    
    # 1. Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model file not found. Run main.py first.")
        return

    # 2. Load the Model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # 3. Preprocess the Input (Simulating the Pipeline)
    # In a real production system, you would save the 'encoder' object. 
    # For this assignment, we will process the single row and align it.
    
    # Convert dict to DataFrame
    df = pd.DataFrame([sample_data])
    
    # Apply the same engineering steps
    df = create_features(df)
    
    # Note: Encoding a single row often fails to create all columns (e.g., if meal='Meal Plan 1', 
    # we won't get columns for 'Meal Plan 2'). 
    # To fix this for the assignment, we align with the model's expected features.
    
    # Get the features the model expects (from the trained model object)
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
    else:
        # Fallback if model doesn't store feature names
        print("Model does not store feature names. Cannot validate input shape.")
        return

    # Create a DataFrame with all expected columns, initialized to 0
    df_processed = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # We can't easily run full preprocessing on one row without the original dataset context.
    # So for this DEMO, we will assume the input is already clean and just fill known values.
    # (In a real job, you would save a Scikit-Learn Pipeline object to handle this automatically).
    
    print("Aligning input features...")
    # Fill in numerical values that match column names
    for col in df.columns:
        if col in df_processed.columns:
            df_processed[col] = df[col]

    # 4. Predict
    try:
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0][1]
        
        result = "CANCELLED" if prediction == 1 else "NOT CANCELLED"
        print(f"\n--- Prediction Result ---")
        print(f"Status: {result}")
        print(f"Cancellation Probability: {probability:.2f}")
        
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    # Example "New Booking" Data
    new_booking = {
        'no_of_adults': 2,
        'no_of_children': 1,
        'no_of_weekend_nights': 2,
        'no_of_week_nights': 5,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 1,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 45,
        'arrival_year': 2018,
        'arrival_month': 10,
        'arrival_date': 15,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 120.50,
        'no_of_special_requests': 1,
        'booking_status': 'Not_Canceled' # This is just for shape, won't be used
    }
    
    # We must rename keys to match our internal standard (from data_loader.py)
    # Or simply input data using the names we know the model uses:
    formatted_input = {
        'adults': 2,
        'children': 0,
        'stays_in_weekend_nights': 1,
        'stays_in_week_nights': 3,
        'lead_time': 150,  # Long lead time increases chance of cancellation
        'adr': 150.0,
        'market_segment': 'Online',
        'meal': 'Meal Plan 1',
        'reserved_room_type': 'Room_Type 1'
    }

    make_prediction(formatted_input)