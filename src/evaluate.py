# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

if __name__ == "__main__":
    # Charger les données de test
    test_df = pd.read_csv(os.path.join('data', 'test.csv'))
    X_test = test_df['text'].astype(str)
    y_test = test_df['sentiment']
    
    # --- Évaluation du modèle de Régression Logistique ---
    print("\n=== Évaluation: Régression Logistique ===")
    lr_model = joblib.load(os.path.join('models', 'logistic_regression_pipeline.joblib'))
    y_pred_lr = lr_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    
    # --- Évaluation du modèle Naive Bayes ---
    print("\n=== Évaluation: Naive Bayes ===")
    nb_model = joblib.load(os.path.join('models', 'naive_bayes_pipeline.joblib'))
    y_pred_nb = nb_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nb))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_nb))

