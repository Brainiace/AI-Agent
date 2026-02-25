import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Mathematical Explanation:
# 1. Random Forest Classifier:
#    A Random Forest is an ensemble learning method that constructs a multitude of decision trees
#    at training time. For classification tasks, the output of the random forest is the class
#    selected by most trees. It uses "bagging" (bootstrap aggregating) to improve stability
#    and accuracy by reducing variance without increasing bias.
#
# 2. Gini Impurity (used for Feature Importance):
#    Decision trees in the forest are built by splitting nodes to minimize Gini impurity.
#    Gini Impurity G = 1 - sum(pi^2), where pi is the probability of an object being classified
#    to a particular class. Feature importance is calculated by the Mean Decrease in Impurity (MDI),
#    which is the total decrease in node impurity (weighted by the probability of reaching that node)
#    averaged over all trees of the ensemble.
#
# 3. Train/Test Split:
#    We split the data to evaluate the model on unseen data, preventing overfitting.
#    Overfitting occurs when a model learns the noise in the training data rather than the actual
#    underlying pattern, leading to poor generalization.

def build_model(file_path):
    """
    Trains a Random Forest model to predict machine failure and evaluates its performance.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Preprocessing
    # Drop non-predictive columns: UDI (Index), Product ID (Identifier)
    # Drop failure mode columns: TWF, HDF, PWF, OSF, RNF (these are labels for failure types, not features)
    X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']

    # Encode categorical 'Type' column
    # Machine types L (Low), M (Medium), H (High) have different failure rates.
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])

    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    # n_estimators=100: Number of trees in the forest.
    # random_state=42: Ensures reproducibility.
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy:.4f}")

    # Confusion Matrix
    # A confusion matrix shows the number of correct and incorrect predictions
    # broken down by each class (True Positives, True Negatives, False Positives, False Negatives).
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion Matrix saved to confusion_matrix.png")

    # Feature Importance
    # Higher importance indicates that the feature is more useful for predicting the target.
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance (Mean Decrease in Impurity)')
    plt.title('Feature Importance for Machine Failure Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature Importance plot saved to feature_importance.png")

    top_feature = feature_importance_df.iloc[-1]['Feature']
    print(f"\nThe most critical sensor is: {top_feature}")

    # Mechanical Perspective Explanation (Mechatronics context)
    print("\n--- Mechanical Perspective (Mechatronics) ---")
    if "Torque" in top_feature:
        print("Torque is the rotational equivalent of linear force. In this machine,")
        print("high or fluctuating torque is the strongest indicator of failure because:")
        print("1. Tool Wear: As cutting tools dull, friction increases, requiring more torque.")
        print("2. Overload: Pushing the machine beyond its limits increases mechanical stress.")
        print("3. Jamming: Sudden spikes in torque often signal a mechanical jam or bearing failure.")
    elif "Rotational speed" in top_feature:
        print("Rotational speed is critical because excessive speeds can lead to:")
        print("1. Overheating: High speeds increase friction and thermal degradation.")
        print("2. Bearing Stress: Centrifugal forces at high RPM can lead to premature bearing failure.")
    elif "temperature" in top_feature:
        print("Temperature (Air or Process) is critical because:")
        print("1. Thermal Expansion: High heat causes parts to expand, potentially leading to seized components.")
        print("2. Material Degradation: Excessive heat weakens the structural integrity of tools.")
    else:
        print(f"The sensor '{top_feature}' is the most critical for predicting breakdown.")

if __name__ == "__main__":
    build_model('datasets/ai4i2020.csv')
