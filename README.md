# IoT-Based Heart Disease Risk Prediction System

This project presents a real-time heart disease risk prediction system that integrates **IoT simulation, MQTT communication, and Machine Learning**. It demonstrates how AI can enhance IoT-based healthcare monitoring systems.

---

## Project Overview

The system simulates real-time patient health data using an ESP32 (via Wokwi), transmits it using MQTT, processes it through a backend ML model, and visualizes the results on a live dashboard. Random Forest achieves the best predictive performance with an accuracy of 0.928, AUC of 0.988, and F1-score of 0.932. Beyond model accuracy, the system is assessed under streaming conditions, achieving an average latency of 3.13 ms, throughput of 53.66 messages per second, and zero packet loss.

---

## Key Features

- 📡 Real-time data simulation using ESP32 (Wokwi)
- 🔄 MQTT-based communication (lightweight IoT protocol)
- 🤖 Machine Learning model for prediction
- 🧾 Patient-specific data handling
- 📊 Live dashboard visualization (Streamlit)
- ⚡ Event-driven backend (FastAPI + MQTT subscriber)

---

## 🏗️ System Architecture

![System Architecture](https://github.com/JAGS-05/heart-disease-risk-prediction/blob/main/assets/System%20Architecture.png)


---

## Dataset

- Dataset used: Framingham Heart Study Dataset  
- Source: Kaggle [Link](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)
- Target: `TenYearCHD` (10-year coronary heart disease risk)

---

## Machine Learning Model

- Preprocessing:
  - Missing value handling
  - Feature scaling (StandardScaler)
  - Class imbalance handling (SMOTE)
- Output:
  - `0` → Low Risk  
  - `1` → High Risk  

---

## IoT Simulation

- Platform: Wokwi
- Device: ESP32 (MicroPython)
- Simulated parameters:
  - Systolic BP
  - Diastolic BP
  - Heart Rate
  - Glucose

---

## MQTT Communication

- Protocol: MQTT
- Broker: Public broker (Mosquitto)
- Topic: `health/patient/data`

---
