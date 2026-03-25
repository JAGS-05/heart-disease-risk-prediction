from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("heart_model.pkl")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["male"],
        data["age"],
        data["education"],
        data["currentSmoker"],
        data["cigsPerDay"],
        data["BPMeds"],
        data["prevalentStroke"],
        data["prevalentHyp"],
        data["diabetes"],
        data["totChol"],
        data["sysBP"],
        data["diaBP"],
        data["BMI"],
        data["heartRate"],
        data["glucose"]
    ]])

    prediction = model.predict(features)[0]

    return {"risk": int(prediction)}