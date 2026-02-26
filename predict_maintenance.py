import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Mathematical and Mechatronic Context:
# 1. Feature Engineering (Physics-Informed):
#    - Power_kW: Calculated as (Torque * RPM) / 9550. This represents the mechanical power
#      delivered by the motor. Excessive power consumption can indicate internal friction
#      or overloading.
#    - Temp_Difference: Process Temperature - Air Temperature. This measures the thermal
#      gradient. A high gradient often precedes thermal failure or seizure.
#    - Torque_Wear_Product: Torque * Tool Wear. This captures the synergistic effect of
#      a dull tool operating under high load, a common precursor to breakdown.
#
# 2. Random Forest with Class Weight:
#    The 'balanced_subsample' class weight adjusts weights based on the bootstrap sample
#    for every tree grown, which is crucial for the imbalanced nature of failure data
#    (failures are rare events).
#
# 3. SMOTE (Synthetic Minority Over-sampling Technique):
#    SMOTE creates synthetic examples of the minority class (failures) by interpolating
#    between existing instances. This helps the model learn the characteristics of failures
#    better than simple oversampling.
#
# 4. Optimized Hyperparameters:
#    n_estimators=500, max_depth=10, min_samples_leaf=5, max_features='sqrt' are used
#    to provide a robust, stable model that generalizes well across different operating conditions.

def engineer_features(df):
    """
    Applies physics-informed feature engineering to the dataset.
    """
    # Power (kW) = (Torque * RPM) / 9550
    df['Power_kW'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550

    # Temperature Difference
    df['Temp_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # Torque-Wear Interaction
    df['Torque_Wear_Product'] = df['Torque [Nm]'] * df['Tool wear [min]']

    return df

def build_model(file_path):
    """
    Trains an optimized Random Forest model to predict machine failure.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Engineering Features
    df = engineer_features(df)

    # Preprocessing
    # Drop non-predictive columns and failure mode labels (data leaks)
    X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']

    # Encode categorical 'Type' column
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle Class Imbalance with SMOTE on Training Set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train Optimized Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42
    )
    rf.fit(X_train_res, y_train_res)

    # Save the model and the label encoder
    joblib.dump(rf, 'maintenance_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("Optimized Model and Label Encoder saved successfully.")

    # Predict and Evaluate
    y_pred = rf.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Optimized Model)')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Feature Importance
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance (Mean Decrease in Impurity)')
    plt.title('Feature Importance for Optimized Machine Failure Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    build_model('datasets/ai4i2020.csv')
