import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def train_models(df):
    print("--- Training Models ---")
    
    # Separate Features and Target
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate scale_pos_weight for XGBoost (Ratio of Negatives / Positives)
    # This helps XGBoost handle the imbalance without SMOTE
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # Define Models with Class Weights (Replaces SMOTE)
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',  # Handle imbalance
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',  # Handle imbalance
            random_state=42,
            n_jobs=-1  # Use all CPU cores to speed up
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight, # Handle imbalance
            random_state=42
        )
    }
    
    trained_models = {}
    
    # Loop through models to train them
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} trained successfully.")
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    # Hyperparameter Tuning (Example on Random Forest)
    # We use a small grid to save time/memory
    print("Tuning Random Forest...")
    rf_params = {
        'max_depth': [10, 20],
        'min_samples_split': [5, 10]
    }
    
    # Use the trained RF from above as base
    grid_search = GridSearchCV(
        estimator=models['RandomForest'], 
        param_grid=rf_params, 
        cv=3, 
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    # Update the dictionary with the tuned model
    trained_models['RandomForest'] = best_rf
    print(f"Best RF Params: {grid_search.best_params_}")
    
    # Save the best model
    joblib.dump(best_rf, 'outputs/best_model.pkl')
    print("Best model saved to outputs/best_model.pkl")
    
    return trained_models, X_test, y_test