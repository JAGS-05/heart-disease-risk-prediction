import streamlit as st
import requests
import time

st.set_page_config(page_title="IoT Health Monitor", layout="wide")

st.title("💓 Real-Time Heart Disease Monitoring Dashboard")

placeholder = st.empty()

while True:
    try:
        res = requests.get("http://localhost:8000/latest")
        result = res.json()

        data = result["data"]
        prediction = result["prediction"]

        with placeholder.container():

            st.subheader("👤 Patient Information")

            col1, col2, col3 = st.columns(3)

            col1.metric("Age", data.get("age", "-"))
            col1.metric("BMI", data.get("BMI", "-"))
            col1.metric("Cholesterol", data.get("totChol", "-"))

            col2.metric("Systolic BP", data.get("sysBP", "-"))
            col2.metric("Diastolic BP", data.get("diaBP", "-"))
            col2.metric("Heart Rate", data.get("heartRate", "-"))

            col3.metric("Glucose", data.get("glucose", "-"))
            col3.metric("Smoker", data.get("currentSmoker", "-"))
            col3.metric("Diabetes", data.get("diabetes", "-"))

            st.divider()

            st.subheader("⚠️ Risk Prediction")

            if prediction == 1:
                st.error("🔴 HIGH RISK of Heart Disease in 10years")
            else:
                st.success("🟢 LOW RISK of Heart Disease in 10years")

    except Exception as e:
        st.error(f"Error: {e}")

    time.sleep(3)