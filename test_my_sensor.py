import pandas as pd
import joblib

# Mathematical Explanation:
# 1. Physics-Informed Feature Calculation:
#    The model now expects three additional features derived from the raw sensors:
#    - Power_kW = (Torque * RPM) / 9550
#    - Temp_Difference = Process_Temp - Air_Temp
#    - Torque_Wear_Product = Torque * Tool_Wear
#
# 2. Strategic Importance:
#    By explicitly calculating these products and differences, we guide the model
#    towards the mechanical physics of the machine (e.g., work done and thermal energy).

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
    air_temp = 300.0
    proc_temp = 310.0
    machine_type = 'L'

    # 1. Calculate Physics-Informed Features
    power_kw = (torque * rpm) / 9550
    temp_diff = proc_temp - air_temp
    torque_wear = torque * tool_wear

    # 2. Prepare data for prediction
    # Feature order MUST match the sequence the model was trained on.
    # Training columns were: ['Type', 'Air temperature [K]', 'Process temperature [K]',
    #                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    #                         'Power_kW', 'Temp_Difference', 'Torque_Wear_Product']

    type_encoded = le.transform([machine_type])[0]

    input_data = pd.DataFrame([[
        type_encoded, air_temp, proc_temp, rpm, torque, tool_wear,
        power_kw, temp_diff, torque_wear
    ]], columns=[
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Power_kW', 'Temp_Difference', 'Torque_Wear_Product'
    ])

    # 3. Make prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    print("\n--- Prediction Result ---")
    if prediction == 0:
        print(f"Result: Machine is Safe (Failure Probability: {proba:.2%})")
    else:
        print(f"Result: WARNING: High Risk of Failure! (Failure Probability: {proba:.2%})")

    # Specific Torque Warning
    if torque > 60:
        print("\n[!] CRITICAL WARNING: Torque is over 60Nm. CHECK MOTOR COUPLINGS.")

    print("\n--- Analysis Note ---")
    print(f"Engineered Features: Power={power_kw:.2f} kW, Temp Diff={temp_diff:.1f} K, Stress Product={torque_wear:.1f}")

if __name__ == "__main__":
    main()
