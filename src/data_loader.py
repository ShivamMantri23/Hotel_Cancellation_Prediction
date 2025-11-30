import pandas as pd
import os

def load_data(filepath):
    """
    Loads data and renames columns to match standard assignment terminology.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Mapping Kaggle dataset columns to Assignment/UCI standard names
    column_mapping = {
        'no_of_adults': 'adults',
        'no_of_children': 'children',
        'no_of_weekend_nights': 'stays_in_weekend_nights',
        'no_of_week_nights': 'stays_in_week_nights',
        'avg_price_per_room': 'adr',
        'booking_status': 'is_canceled',
        'market_segment_type': 'market_segment',
        'type_of_meal_plan': 'meal',
        'room_type_reserved': 'reserved_room_type'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Encode target variable immediately for consistency (Canceled=1, Not_Canceled=0)
    # The Kaggle dataset uses 'Canceled' and 'Not_Canceled' strings
    if df['is_canceled'].dtype == 'O':
        df['is_canceled'] = df['is_canceled'].map({'Canceled': 1, 'Not_Canceled': 0})
        
    print(f"Data Loaded. Shape: {df.shape}")
    return df