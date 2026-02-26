import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Mathematical and Mechatronic Context:
# 1. Feature Engineering Impact:
#    Adding Power_kW and Temp_Diff provides the model with direct physical failure triggers.
#    This reduces the need for the model to "guess" these relationships from raw speed and torque,
#    improving generalization.
#
# 2. Generalization vs. Overfitting:
#    The goal is to reduce the gap between training and testing accuracy. A training accuracy
#    too close to 100% often indicates the model has memorized noise. By constraining
#    max_depth and increasing min_samples_split, we force the model to learn broader patterns.
#
# 3. SMOTE & Stability:
#    Class imbalance often leads to high variance in cross-validation because the model's
#    performance depends heavily on whether the few minority samples are in the training or
#    test fold. SMOTE stabilizes this by ensuring the minority class is well-represented.

def run_diagnostics(file_path):
    print("--- Starting Enhanced Model Diagnostics ---\n")

    # Load dataset
    df = pd.read_csv(file_path)

    # Preprocessing with Feature Engineering
    print("Applying Feature Engineering...")
    df['Power_kW'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

    drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(drop_cols, axis=1)
    y = df['Machine failure']

    # Encode categorical 'Type'
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

    # Load the optimized model
    print("Loading optimized model...")
    try:
        rf = joblib.load('maintenance_model.joblib')
    except:
        print("Warning: Could not load 'maintenance_model.joblib'. Falling back to default configuration for diagnostic.")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 1. Train vs. Test Comparison
    # Note: Use stratify to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Re-evaluating on the actual data (not resampled) to see true performance
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"Task 1: Train vs. Test Comparison")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

    if train_acc < 0.99:
        print("Success: Training Accuracy is below 99% (preventing overfitting).")
    else:
        print("Warning: Training Accuracy is still >= 99%.")
    print("-" * 30 + "\n")

    # 2. K-Fold Cross-Validation (5-fold)
    print(f"Task 2: 5-Fold Cross-Validation")
    # We use the whole dataset X, y for CV
    cv_scores = cross_val_score(rf, X, y, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"Mean CV Accuracy: {cv_mean:.4f}")
    print(f"Std Deviation:    {cv_std:.4f}")

    if cv_std < 0.05:
        print(f"Success: CV Standard Deviation is {cv_std:.4f} (< 5%).")
    else:
        print(f"Warning: CV Standard Deviation is {cv_std:.4f} (>= 5%).")
    print("-" * 30 + "\n")

    # 3. Stability Comparison Table
    print("Task 3: Stability Comparison (Before vs. After)")
    metrics = {
        "Metric": ["Training Accuracy", "CV Std Deviation"],
        "Before": ["100.0%", "15.8%"],
        "After": [f"{train_acc*100:.1f}%", f"{cv_std*100:.1f}%"]
    }
    comparison_df = pd.DataFrame(metrics)
    print(comparison_df.to_string(index=False))
    print("-" * 30 + "\n")

    # 4. Learning Curve Generation
    print(f"Task 4: Generating Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Optimized Random Forest)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('learning_curve_optimized.png')
    plt.close()
    print("Learning curve saved to learning_curve_optimized.png")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    run_diagnostics('datasets/ai4i2020.csv')
