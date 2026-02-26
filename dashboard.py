import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the AI brain you built with Jules
model = joblib.load('maintenance_model.joblib')
encoder = joblib.load('label_encoder.joblib')

st.title("🛠️ Mechatronics Predictive Maintenance")
st.markdown("Use the sliders below to simulate sensor data and predict machine health.")

# Sidebar for inputs
st.sidebar.header("Sensor Input Panel")
torque = st.sidebar.slider("Torque [Nm]", 0, 80, 40)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 500, 3000, 1500)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 50)

# Create the data for the AI to read
input_data = pd.DataFrame([[0, 300, 310, rpm, torque, tool_wear]], 
                          columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

# Make the prediction
prediction = model.predict(input_data)
prob = model.predict_proba(input_data)[0][1] * 100

# Display results
st.subheader("Machine Status")
if prediction[0] == 0:
    st.success(f"✅ Healthy (Risk: {prob:.1f}%)")
else:
    st.error(f"⚠️ WARNING: High Risk of Failure! (Risk: {prob:.1f}%)")

# Add a simple gauge-like metric
st.metric("Current Torque", f"{torque} Nm", delta=f"{torque-40} Nm" if torque > 40 else None)