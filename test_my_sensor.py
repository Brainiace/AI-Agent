import pandas as pd
import joblib

# Mathematical Explanation:
# 1. Prediction using Random Forest:
#    The model consists of 500 decision trees. When we input new sensor data,
#    each tree in the forest makes a prediction (0 for Safe, 1 for Failure).
#    The final probability is determined by the percentage of trees predicting '1'.
#
# 2. Physics-Informed Feature Engineering:
#    The model now expects 9 features. We must calculate Power_kW, Temp_Difference,
#    and Torque_Wear_Product from the raw sensor inputs to match the training data.

def main():
    """
    Interactively takes sensor inputs and predicts machine failure risk using the optimized model.
    """
    try:
        # Load the saved model and label encoder
        model = joblib.load('maintenance_model.joblib')
        le = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        print("Error: Model files not found. Please run predict_maintenance.py first.")
        return

    print("--- AI4I 2020 Optimized Machine Failure Prediction ---")
    print("Please enter the following sensor readings:")

    try:
        torque = float(input("Enter Torque [Nm]: "))
        rpm = float(input("Enter Rotational speed [rpm]: "))
        tool_wear = float(input("Enter Tool wear [min]: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Constants/Defaults for other features
    air_temp = 300.0  # Average Air temperature [K]
    proc_temp = 310.0 # Average Process temperature [K]
    machine_type = 'L' # Defaulting to 'Low' type

    # Calculate engineered features
    power_kw = (torque * rpm) / 9550
    temp_diff = proc_temp - air_temp
    torque_wear = torque * tool_wear

    # Prepare data for prediction (9 features)
    type_encoded = le.transform([machine_type])[0]

    input_data = pd.DataFrame([[type_encoded, air_temp, proc_temp, rpm, torque, tool_wear, power_kw, temp_diff, torque_wear]],
                              columns=['Type', 'Air temperature [K]', 'Process temperature [K]',
                                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                                       'Power_kW', 'Temp_Difference', 'Torque_Wear_Product'])

    # Make prediction using probability
    prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if prob >= 0.5 else 0

    print("\n--- Prediction Result ---")
    print(f"Failure Probability: {prob*100:.1f}%")
    if prediction == 0:
        print("Result: Machine is Safe")
    else:
        print("Result: WARNING: High Risk of Failure!")

    # Specific Torque Warning
    if torque > 60:
        print("\n[!] CRITICAL WARNING: Torque is over 60Nm. CHECK MOTOR COUPLINGS immediately.")

if __name__ == "__main__":
    main()
