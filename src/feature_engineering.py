import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates new features as per assignment requirements.
    """
    df = df.copy()
    
    # 1. Total Stay Nights
    df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    # 2. Total Guests (Handle missing babies if column absent)
    if 'babies' not in df.columns:
        df['total_guests'] = df['adults'] + df['children']
    else:
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        
    # 3. Booking Lead Time Category
    df['lead_time_category'] = pd.cut(
        df['lead_time'], 
        bins=[-1, 7, 30, 9999], 
        labels=['Short', 'Medium', 'Long']
    )
    
    # 4. Average ADR per person (Avoid division by zero)
    df['adr_per_person'] = df.apply(
        lambda x: x['adr'] / x['total_guests'] if x['total_guests'] > 0 else 0, axis=1
    )
    
    # 5. Weekend Booking Flag
    df['is_weekend_booking'] = (df['stays_in_weekend_nights'] > 0).astype(int)
    
    print("Feature Engineering completed.")
    return df