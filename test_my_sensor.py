import pandas as pd
import joblib

# Mathematical Explanation:
# 1. Physics-Informed Feature Calculation:
#    The model expects three engineered features derived from the raw sensors:
#    - Power_kW = (Torque * RPM) / 9550. Measures the mechanical workload.
#    - Temp_Diff = Process_Temp - Air_Temp. Measures thermal energy accumulation.
#    - Torque_Wear_Product = Torque * Tool_Wear. Measures the combined impact of
#      current load and historical degradation.
#
# 2. Predictive Context:
#    By providing these pre-calculated features, we help the Random Forest
#    efficiently navigate the decision boundaries of mechatronic failure modes.

def main():
    """
    Interactively takes sensor inputs and predicts machine failure risk using the
    optimized "Clean Slate" model.
    """
    try:
        # Load the saved model and label encoder
        model = joblib.load('maintenance_model.joblib')
        le = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        print("Error: Model files not found. Please run predict_maintenance.py first.")
        return

    print("\n--- AI4I 2020 Optimized Predictive Maintenance ---")
    print("Please enter the following sensor readings:")

    try:
        torque = float(input("Enter Torque [Nm]: "))
        rpm = float(input("Enter Rotational speed [rpm]: "))
        tool_wear = float(input("Enter Tool wear [min]: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Constants/Defaults for features not interactively queried
    air_temp = 300.0
    proc_temp = 310.0
    machine_type = 'L'

    # 1. Calculate Physics-Informed Features
    power_kw = (torque * rpm) / 9550
    temp_diff = proc_temp - air_temp
    torque_wear = torque * tool_wear

    # 2. Prepare data for prediction
    # Feature order MUST match X.columns from training
    type_encoded = le.transform([machine_type])[0]

    input_data = pd.DataFrame([[
        type_encoded, air_temp, proc_temp, rpm, torque, tool_wear,
        power_kw, temp_diff, torque_wear
    ]], columns=[
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Power_kW', 'Temp_Diff', 'Torque_Wear_Product'
    ])

    # 3. Make prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    print("\n--- Prediction Result ---")
    if prediction == 0:
        print(f"Result: Machine is Safe (Failure Probability: {proba:.2%})")
    else:
        print(f"Result: WARNING: High Risk of Failure! (Failure Probability: {proba:.2%})")

    # Specific Torque Warning (Heuristic)
    if torque > 60:
        print("[!] CRITICAL WARNING: Torque exceeds 60Nm. High risk of mechanical shear.")
    if tool_wear > 200:
        print("[!] TOOL WEAR ALERT: Accumulated wear over 200 mins. Schedule replacement.")

    print(f"\nEngineered Metrics: Power={power_kw:.2f} kW, Temp Diff={temp_diff:.1f} K, Stress Product={torque_wear:.1f}")

if __name__ == "__main__":
    main()
