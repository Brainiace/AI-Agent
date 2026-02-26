import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Mathematical and Mechatronic Context:
# 1. Overfitting:
#    Overfitting occurs when a model learns the training data too well, including its noise,
#    resulting in poor generalization to new, unseen data. In mechatronics, this could lead to
#    false positives or missing actual failure signals in a real-world environment like a
#    syrup dispenser.
#
# 2. Learning Curve:
#    A learning curve shows the relationship between training size and the model's performance on
#    training and validation sets.
#    - If the training score is much higher than the validation score, the model is likely overfitting.
#    - If both scores are low and close, the model is likely underfitting (high bias).
#
# 3. K-Fold Cross-Validation:
#    This technique splits the dataset into K subsets (folds). The model is trained on K-1 folds
#    and tested on the remaining fold. This is repeated K times. It provides a more robust
#    estimate of model performance and stability.
#
# 4. Feature Importance:
#    Random Forest calculates importance based on how much each feature contributes to reducing
#    impurity (Gini) across all trees. If an ID or index shows high importance, it indicates
#    a data leak, as these shouldn't have predictive power.

def run_diagnostics(file_path):
    print("--- Starting Model Diagnostics ---\n")

    # Load dataset
    df = pd.read_csv(file_path)

    # Preprocessing (matches predict_maintenance.py)
    # Dropping non-predictive columns and potential leak sources (failure modes)
    drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(drop_cols, axis=1)
    y = df['Machine failure']

    # Encode categorical 'Type'
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

    # 1. Train vs. Test Comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"Task 1: Train vs. Test Comparison")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")
    if train_acc - test_acc > 0.05:
        print("Warning: Potential Overfitting detected (Gap > 5%)")
    else:
        print("Model seems to generalize well.")
    print("-" * 30 + "\n")

    # 2. K-Fold Cross-Validation (5-fold)
    print(f"Task 2: 5-Fold Cross-Validation")
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Std Deviation:    {cv_scores.std():.4f}")
    print("-" * 30 + "\n")

    # 3. Learning Curve Generation
    print(f"Task 3: Generating Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, cv=5, n_jobs=-1,
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
    plt.title("Learning Curve (Random Forest)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.close()
    print("Learning curve saved to learning_curve.png")
    print("-" * 30 + "\n")

    # 4. Feature Importance Verification
    print(f"Task 4: Feature Importance Verification")
    importances = rf.feature_importances_
    features = X.columns
    feat_importances = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    for feat, imp in feat_importances:
        print(f"{feat:25}: {imp:.4f}")

    # Check for leaks
    top_feat, top_val = feat_importances[0]
    if top_val > 0.9:
        print(f"Warning: {top_feat} might be a data leak (Importance > 0.9)")
    else:
        print("No single feature dominates excessively. Model looks robust.")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    run_diagnostics('datasets/ai4i2020.csv')
