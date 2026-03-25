from fastapi import FastAPI
import json
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import pandas as pd
import threading

app = FastAPI()

# -----------------------------
# Load ML model
# -----------------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Load patients DB
# -----------------------------
with open("patient.json") as f:
    patients_db = json.load(f)

latest_data = {}
latest_prediction = None

# -----------------------------
# MQTT CONFIG
# -----------------------------
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "health/patient/data"

# -----------------------------
# Feature names 
# -----------------------------
feature_names = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

# -----------------------------
# MQTT CALLBACKS
# -----------------------------
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global latest_data, latest_prediction

    try:
        data = json.loads(msg.payload.decode())
        print("Received:", data)

        patient = patients_db[str(data["patient_id"])]

        # Create dataframe (to avoid warning)
        features = pd.DataFrame([[
            patient["male"],
            patient["age"],
            patient["education"],
            patient["currentSmoker"],
            patient["cigsPerDay"],
            patient["BPMeds"],
            patient["prevalentStroke"],
            patient["prevalentHyp"],
            patient["diabetes"],
            patient["totChol"],
            data["sysBP"],
            data["diaBP"],
            patient["BMI"],
            data["heartRate"],
            data["glucose"]
        ]], columns=feature_names)

        # Scale + Predict
        features = scaler.transform(features)
        prediction = model.predict(features)[0]

        latest_data = {**patient, **data}
        latest_prediction = int(prediction)

        print("Prediction:", prediction)

    except Exception as e:
        print("Error:", e)

# -----------------------------
# Start MQTT in background
# -----------------------------
def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.loop_forever()

# Run MQTT 
threading.Thread(target=start_mqtt, daemon=True).start()

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.get("/")
def home():
    return {"message": "Backend running"}

@app.get("/latest")
def latest():
    return {
        "data": latest_data,
        "prediction": latest_prediction
    }