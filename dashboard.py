import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the AI model and encoder
model = joblib.load('maintenance_model.joblib')
encoder = joblib.load('label_encoder.joblib')

st.set_page_config(page_title="Mechatronics Maintenance", page_icon="🛠️")

st.title("🛠️ Mechatronics Predictive Maintenance")
st.markdown("""
Predict machine failure based on sensor readings using an optimized Random Forest model.
This dashboard incorporates **physics-informed features** to improve predictive accuracy.
""")

# Sidebar for inputs
st.sidebar.header("Sensor Input Panel")

# Inputs
machine_type = st.sidebar.selectbox("Machine Type", options=['L', 'M', 'H'], index=0)
air_temp = st.sidebar.slider("Air Temperature [K]", 295, 305, 300)
proc_temp = st.sidebar.slider("Process Temperature [K]", 305, 315, 310)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, 1500)
torque = st.sidebar.slider("Torque [Nm]", 0, 80, 40)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 50)

st.sidebar.divider()
risk_threshold = st.sidebar.slider("Risk Sensitivity (Threshold)", 0.0, 1.0, 0.5, 0.05)

# 1. Physics-Informed Feature Engineering
# These calculations must match the training pipeline in predict_maintenance.py
power_kw = (torque * rpm) / 9550
temp_diff = proc_temp - air_temp
torque_wear = torque * tool_wear

# Prepare encoded data
type_encoded = encoder.transform([machine_type])[0]

# Create the data for the model to read (Correct Column Order)
input_data = pd.DataFrame([[
    type_encoded, air_temp, proc_temp, rpm, torque, tool_wear,
    power_kw, temp_diff, torque_wear
]], columns=[
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Power_kW', 'Temp_Diff', 'Torque_Wear_Product'
])

# Make the prediction
# We use the probability and the user-defined threshold for risk sensitivity
prob = model.predict_proba(input_data)[0][1]
prediction = 1 if prob >= risk_threshold else 0

# Display results
st.subheader("Predictive Diagnostics")
col1, col2 = st.columns(2)

with col1:
    if prediction == 0:
        st.success(f"✅ Machine Status: Healthy")
    else:
        st.error(f"⚠️ Machine Status: FAILURE RISK!")
    st.write(f"Failure Probability: **{prob:.1%}**")

with col2:
    st.write("### Physics-Informed Metrics")
    st.write(f"- **Power Output:** {power_kw:.2f} kW")
    st.write(f"- **Thermal Stress:** {temp_diff:.1f} K")
    st.write(f"- **Wear-Stress Product:** {torque_wear:.1f}")

# Visual Alert for critical conditions
if torque > 60:
    st.warning("[!] High Torque Alert: Risk of mechanical shear is elevated.")
if tool_wear > 200:
    st.warning("[!] High Wear Alert: Tool is near end-of-life.")

st.divider()
st.info("Mathematical Note: The model uses SMOTE and GridSearchCV to ensure stability and high recall (>85%) for failure detection.")
