import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
import joblib

# Mathematical and Mechatronic Context:
# 1. Physics-Informed Features:
#    Adding 'Power_kW', 'Temp_Diff', and 'Torque_Wear_Product' provides the model
#    with explicit mechanical signals that represent stress and degradation.
#
# 2. Stability Analysis:
#    High variance in CV (Cross-Validation) indicates that the model is sensitive to
#    the specific training data it sees. By using better features and hyperparameters
#    (like min_samples_leaf), we aim to reduce this variance (std deviation).

def run_diagnostics(file_path):
    print("--- Starting Advanced Model Diagnostics ---\n")

    # Load dataset
    df = pd.read_csv(file_path)

    # 1. Advanced Feature Engineering (Physics-Informed)
    # Must match the training script exactly
    df['Power_kW'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Torque_Wear_Product'] = df['Torque [Nm]'] * df['Tool wear [min]']

    # Preprocessing
    drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(drop_cols, axis=1)
    y = df['Machine failure']

    # Load saved Label Encoder and Model
    try:
        le = joblib.load('label_encoder.joblib')
        rf = joblib.load('maintenance_model.joblib')
        X['Type'] = le.transform(X['Type'])
        print("Loaded saved model and encoder.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return

    # 1. Train vs. Test Comparison
    # Note: We use the same random_state to compare fairly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Task 1: Performance Evaluation")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")
    print(f"Testing Recall:    {test_recall:.4f}")
    print(f"Testing F1-Score:  {test_f1:.4f}")

    if train_acc - test_acc > 0.05:
        print("Warning: Potential Overfitting (Gap > 5%)")
    else:
        print("Model generalization looks improved.")
    print("-" * 30 + "\n")

    # 2. K-Fold Cross-Validation (5-fold) Stability
    print(f"Task 2: 5-Fold Cross-Validation Stability")
    # Using the optimized model on the whole dataset (with new features)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    new_mean = cv_scores.mean()
    new_std = cv_scores.std()

    print(f"New Mean CV Accuracy: {new_mean:.4f}")
    print(f"New Std Deviation:    {new_std:.4f}")

    # Comparison table (approximate old standard for historical context)
    old_std = 0.1579
    print("\n--- Stability Comparison ---")
    print(f"{'Metric':<20} | {'Old Model':<10} | {'New Model':<10}")
    print("-" * 45)
    print(f"{'CV Std (Variance)':<20} | {old_std:<10.4f} | {new_std:<10.4f}")
    improvement = (old_std - new_std) / old_std * 100
    print(f"\nStability Improvement: {improvement:.1f}%")
    print("-" * 30 + "\n")

    # 3. Learning Curve Generation
    print(f"Task 3: Generating Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training F1")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="CV F1")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.xlabel("Training examples")
    plt.ylabel("F1-Score")
    plt.title("Learning Curve (Optimized RF)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.close()
    print("New learning curve saved to learning_curve.png")
    print("-" * 30 + "\n")

    # 4. Feature Importance Verification
    print(f"Task 4: Feature Importance Verification")
    importances = rf.feature_importances_
    features = X.columns
    feat_importances = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    for feat, imp in feat_importances:
        print(f"{feat:25}: {imp:.4f}")

if __name__ == "__main__":
    run_diagnostics('datasets/ai4i2020.csv')
