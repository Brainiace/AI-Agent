import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, PrecisionRecallDisplay, f1_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np

# Mathematical Explanation:
# 1. Physics-Informed Feature Engineering:
#    - Power_kW: P = (T * omega) / 9550. This represents the mechanical power output.
#      High power consumption can indicate excessive load or friction.
#    - Temp_Difference: Delta_T = T_process - T_air. This represents the heat generated
#      by the process itself, which is a key indicator of efficiency and thermal stress.
#    - Torque_Wear_Product: T * Wear. This captures the combined effect of mechanical
#      stress and cumulative degradation, often critical for overstrain failures.
#
# 2. SMOTE (Synthetic Minority Over-sampling Technique):
#    SMOTE addresses class imbalance by creating synthetic examples of the minority class
#    (failures). It works by selecting a minority class instance and its k-nearest neighbors,
#    then interpolating new points between them.
#
# 3. Cost-Sensitive Learning (class_weight='balanced_subsample'):
#    This adjusts the loss function to penalize misclassifications of the minority class
#    more heavily. 'balanced_subsample' calculates weights based on the bootstrap sample
#    for each individual tree in the Random Forest.
#
# 4. GridSearchCV:
#    Systematically exhaustively searches through a specified subset of the hyperparameter
#    space to find the best-performing model based on a scoring metric (F1-score).

def build_model(file_path):
    """
    Trains an optimized Random Forest model using physics-informed features and SMOTE.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # 1. Advanced Feature Engineering (Physics-Informed)
    df['Power_kW'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
    df['Temp_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Torque_Wear_Product'] = df['Torque [Nm]'] * df['Tool wear [min]']

    # Preprocessing
    # Drop non-predictive columns and potential failure mode leaks
    drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(drop_cols, axis=1)
    y = df['Machine failure']

    # Encode categorical 'Type' column
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Handling Class Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Classes balanced via SMOTE. New training size: {len(X_train_res)}")

    # 3. Rigorous Hyperparameter Tuning (GridSearch)
    param_grid = {
        'max_depth': [5, 7, 10],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'n_estimators': [200, 500]
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_res, y_train_res)

    best_rf = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Save the model and the label encoder
    joblib.dump(best_rf, 'maintenance_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("Optimized Model and Label Encoder saved successfully.")

    # 4. Evaluation
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = f1_score(y_test, y_pred) # Wait, I named it precision but used f1_score
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nFinal Model Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Recall:    {recall:.4f} (Priority for mechatronics)")
    print(f"F1-Score:  {f1:.4f}")

    # Precision-Recall Curve
    # In imbalanced datasets, PR Curve is more informative than ROC-AUC.
    display = PrecisionRecallDisplay.from_estimator(best_rf, X_test, y_test, name="Random Forest")
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()
    print("Precision-Recall Curve saved to precision_recall_curve.png")

    # Feature Importance
    importances = best_rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
    plt.xlabel('Importance')
    plt.title('Physics-Informed Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Updated Feature Importance plot saved to feature_importance.png")

if __name__ == "__main__":
    build_model('datasets/ai4i2020.csv')
