import pandas as pd
import joblib

# Mathematical Explanation:
# 1. Prediction using Random Forest:
#    The model consists of 100 decision trees. When we input new sensor data,
#    each tree in the forest makes a prediction (0 for Safe, 1 for Failure).
#    The final output is determined by a majority vote among all trees.
#
# 2. Probability and Threshold:
#    The Random Forest can also output a probability score. In this binary
#    classification, if more than 50% of the trees predict '1', the result
#    is "High Risk of Failure".

# Mechatronic Context:
# - Torque [Nm]: Represents the mechanical stress on the motor and drive system.
#   High torque often indicates increased friction due to tool wear or mechanical
#   obstruction.
# - Rotational Speed [rpm]: Influences centrifugal forces and heat generation.
# - Tool Wear [min]: Cumulative time the tool has been used; as it increases,
#   the cutting efficiency drops, leading to higher torque requirements.
# - Motor Couplings: These components connect the motor shaft to the machine
#   spindle. High torque (>60Nm) can exceed the rated capacity of standard
#   couplings, leading to slippage or mechanical failure.

def main():
    """
    Interactively takes sensor inputs and predicts machine failure risk.
    """
    try:
        # Load the saved model and label encoder
        # joblib is used for efficient serialization of scikit-learn estimators.
        model = joblib.load('maintenance_model.joblib')
        le = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        print("Error: Model files not found. Please run predict_maintenance.py first.")
        return

    print("--- AI4I 2020 Machine Failure Prediction ---")
    print("Please enter the following sensor readings:")

    try:
        torque = float(input("Enter Torque [Nm]: "))
        rpm = float(input("Enter Rotational speed [rpm]: "))
        tool_wear = float(input("Enter Tool wear [min]: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Constants/Defaults for other features
    # These values are chosen based on the dataset averages to provide a baseline.
    air_temp = 300.0  # Average Air temperature [K]
    proc_temp = 310.0 # Average Process temperature [K]
    machine_type = 'L' # Defaulting to 'Low' type (most common)

    # Prepare data for prediction
    # Feature order must match the order used during model training:
    # ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # Encode 'Type' using the same label encoder used in training
    type_encoded = le.transform([machine_type])[0]

    input_data = pd.DataFrame([[type_encoded, air_temp, proc_temp, rpm, torque, tool_wear]],
                              columns=['Type', 'Air temperature [K]', 'Process temperature [K]',
                                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

    # Make prediction
    # .predict() returns an array, we take the first element.
    prediction = model.predict(input_data)[0]

    print("\n--- Prediction Result ---")
    if prediction == 0:
        print("Result: Machine is Safe")
    else:
        print("Result: WARNING: High Risk of Failure!")

    # Specific Torque Warning as requested
    if torque > 60:
        print("\n[!] CRITICAL WARNING: Torque is over 60Nm. CHECK MOTOR COUPLINGS immediately for fatigue or misalignment.")

    print("\n--- Analysis Note ---")
    print(f"Input: Torque={torque} Nm, Speed={rpm} RPM, Tool Wear={tool_wear} min.")
    print(f"Assumption: Air Temp=300K, Process Temp=310K, Machine Type={machine_type}.")

if __name__ == "__main__":
    main()
