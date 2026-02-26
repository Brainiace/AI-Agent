import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Mathematical and Mechatronic Context:
# 1. Feature Engineering:
#    - Power [kW] = (Torque [Nm] * Rotational speed [rpm]) / 9550
#    - Temp_Diff = Process temperature [K] - Air temperature [K]
#    These features capture physical failure triggers related to energy dissipation and thermal stress.
#
# 2. SMOTE (Synthetic Minority Over-sampling Technique):
#    Used to balance the 'Machine failure' class, allowing the Random Forest to learn
#    the characteristics of rare failure events more effectively without simple duplication.
#
# 3. GridSearchCV (Hyperparameter Optimization):
#    Finds the optimal set of parameters (max_depth, min_samples_split, etc.) to
#    maximize the F1-score while ensuring generalization (preventing overfitting).

def build_model(file_path):
    """
    Trains an optimized Random Forest model to predict machine failure.
    """
    print("--- Starting Predictive Maintenance Model Training ---")

    # Load dataset
    df = pd.read_csv(file_path)

    # 1. Feature Engineering
    print("Engineering features: Power_kW and Temp_Diff...")
    df['Power_kW'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # Preprocessing
    # Drop non-predictive columns and potential leak sources (failure modes)
    drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(drop_cols, axis=1)
    y = df['Machine failure']

    # Encode categorical 'Type'
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

    feature_names = X.columns.tolist()

    # 2. Split Data (using stratify to handle imbalance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. SMOTE for Class Imbalance
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 4. GridSearchCV for Optimization
    print("Optimizing Random Forest via GridSearchCV...")
    rf_base = RandomForestClassifier(random_state=42)
    param_grid = {
        'max_depth': [5, 8, 12],
        'min_samples_split': [10, 20],
        'n_estimators': [100, 200],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_res, y_train_res)

    rf = grid_search.best_estimator_
    print(f"Best Parameters found: {grid_search.best_params_}")

    # Save the model, label encoder, and feature order
    joblib.dump(rf, 'maintenance_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(feature_names, 'feature_order.joblib')
    print("Model artifacts saved successfully.")

    # 5. Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Optimized Model)')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion Matrix saved to confusion_matrix.png")

    # Feature Importance
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Optimized Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature Importance plot saved to feature_importance.png")

if __name__ == "__main__":
    build_model('datasets/ai4i2020.csv')
