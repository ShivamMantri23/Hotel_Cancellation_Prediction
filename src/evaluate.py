from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_models(models, X_test, y_test):
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec, 
            'Recall': rec, 'F1': f1, 'AUC': auc
        })
        
        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'outputs/cm_{name}.png')
        plt.close()

    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)
    results_df.to_csv('outputs/model_comparison.csv', index=False)
    
    # Feature Importance (for Random Forest)
    if 'RandomForest' in models:
        rf = models['RandomForest']
        importances = pd.Series(rf.feature_importances_, index=X_test.columns)
        top_features = importances.nlargest(10)
        print("\nTop 10 Features (RF):")
        print(top_features)
        
        plt.figure(figsize=(10,6))
        top_features.plot(kind='barh')
        plt.title("Top 10 Feature Importance")
        plt.savefig("outputs/feature_importance.png")
        plt.close()