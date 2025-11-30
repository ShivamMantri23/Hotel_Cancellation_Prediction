from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

def train_models(df):
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Task 6: Handle Class Imbalance (SMOTE on Train only)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    trained_models = {}
    
    # Task 7 & 8: Training and Hyperparameter Tuning
    print("Training models...")
    
    # Tuning Random Forest as the example
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None]
    }
    grid_search = GridSearchCV(models['RandomForest'], rf_params, cv=3, scoring='f1')
    grid_search.fit(X_train_res, y_train_res)
    
    best_rf = grid_search.best_estimator_
    trained_models['RandomForest'] = best_rf
    print(f"Best RF Params: {grid_search.best_params_}")
    
    # Train others
    models['LogisticRegression'].fit(X_train_res, y_train_res)
    trained_models['LogisticRegression'] = models['LogisticRegression']
    
    models['XGBoost'].fit(X_train_res, y_train_res)
    trained_models['XGBoost'] = models['XGBoost']
    
    # Task 12: Save best model (assuming RF is selected for demo)
    joblib.dump(best_rf, 'outputs/best_model.pkl')
    print("Best model saved to outputs/best_model.pkl")
    
    return trained_models, X_test, y_test