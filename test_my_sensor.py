import pandas as pd
import joblib

# Mechatronic Context with New Features:
# - Power_kW: Calculated as (Torque * RPM) / 9550. It indicates the total mechanical load.
# - Temp_Diff: Process Temperature minus Air Temperature. High values indicate inefficiency
#   or localized heating at the tool interface.

def main():
    """
    Predicts machine failure risk based on optimized features.
    """
    try:
        model = joblib.load('maintenance_model.joblib')
        le = joblib.load('label_encoder.joblib')
        feature_order = joblib.load('feature_order.joblib')
    except FileNotFoundError:
        print("Error: Model files not found. Please run predict_maintenance.py first.")
        return

    print("--- AI4I 2020 Optimized Failure Prediction ---")

    try:
        torque = float(input("Enter Torque [Nm]: "))
        rpm = float(input("Enter Rotational speed [rpm]: "))
        tool_wear = float(input("Enter Tool wear [min]: "))
        air_temp = float(input("Enter Air temperature [K] (default 300): ") or 300.0)
        proc_temp = float(input("Enter Process temperature [K] (default 310): ") or 310.0)
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Calculate engineered features
    power_kw = (torque * rpm) / 9550
    temp_diff = proc_temp - air_temp
    machine_type = 'L' # Default

    # Create input dictionary
    data = {
        'Type': machine_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Power_kW': power_kw,
        'Temp_Diff': temp_diff
    }

    # Convert to DataFrame and align with feature order
    input_df = pd.DataFrame([data])
    input_df['Type'] = le.transform(input_df['Type'])
    input_df = input_df[feature_order]

    # Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    print("\n--- Prediction Result ---")
    if prediction == 0:
        print(f"Result: Machine is Safe (Risk Probability: {prob:.2%})")
    else:
        print(f"Result: WARNING: High Risk of Failure! (Risk Probability: {prob:.2%})")

    if torque > 60:
        print("[!] CRITICAL: High torque detected. Inspect motor couplings.")

if __name__ == "__main__":
    main()
