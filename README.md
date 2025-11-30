Model Deployment Instructions
1. Model Artifact The trained model is serialized and saved as a Python pickle file:

 Path: outputs/best_model.pkl

 Library: joblib

 Algorithm: Random Forest Classifier (optimized via GridSearchCV).

2. Prerequisites To load the model in a production environment, ensure the following dependencies are installed:

 pip install pandas scikit-learn joblib xgboost

3. Loading the Model Use the following Python code to load the model into memory:

 import joblib
 model = joblib.load('outputs/best_model.pkl')

4. Preparing Input Data The model expects a Pandas DataFrame with the exact same feature columns used during training (including One-Hot Encoded columns).

 Total Features: (Matches model.feature_names_in_)

 Preprocessing: Any new raw data must undergo the same cleaning and feature engineering (e.g., calculating total_stay_nights, adr_per_person) before prediction.

5. Making a Prediction

 # Assuming 'input_df' is your preprocessed data
 prediction = model.predict(input_df)       # Returns 0 or 1
 probability = model.predict_proba(input_df) # Returns probability