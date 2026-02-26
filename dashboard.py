import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load optimized artifacts
model = joblib.load('maintenance_model.joblib')
encoder = joblib.load('label_encoder.joblib')
feature_order = joblib.load('feature_order.joblib')

st.set_page_config(page_title="Machine Health Monitor", page_icon="🛠️")

st.title("🛠️ Optimized Predictive Maintenance Dashboard")
st.markdown("""
This dashboard uses an optimized Random Forest model to predict machine failures.
New features **Power [kW]** and **Thermal Gradient (Temp Diff)** are calculated in real-time.
""")

# Sidebar for inputs
st.sidebar.header("Real-time Sensor Readings")
machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.sidebar.slider("Air Temperature [K]", 295, 305, 300)
proc_temp = st.sidebar.slider("Process Temperature [K]", 305, 315, 310)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 500, 3000, 1500)
torque = st.sidebar.slider("Torque [Nm]", 0, 80, 40)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 50)

# Calculate Engineered Features
power_kw = (torque * rpm) / 9550
temp_diff = proc_temp - air_temp

# Prepare input data
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

input_df = pd.DataFrame([data])
input_df['Type'] = encoder.transform(input_df['Type'])
input_df = input_df[feature_order]

# Prediction
prediction = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]
risk_score = probs[1] * 100

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analysis")
    if prediction == 0:
        st.success(f"✅ Machine State: Healthy")
    else:
        st.error(f"⚠️ Machine State: FAILURE RISK")

    st.metric("Failure Probability", f"{risk_score:.1f}%")

with col2:
    st.subheader("Computed Metrics")
    st.write(f"**Power output:** {power_kw:.2f} kW")
    st.write(f"**Thermal Gradient:** {temp_diff:.1f} K")

if torque > 60:
    st.warning("Critical Torque Level! (>60Nm) Check mechanical integrity.")
