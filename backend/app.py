from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()

# -----------------------------
# CORS (Frontend Connection)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Static Files (CSS, JS, Video)
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Load ML Model Files
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "model" / "parkinsons_model.pkl"
features_path = BASE_DIR / "model" / "top_features.pkl"
threshold_path = BASE_DIR / "model" / "threshold.pkl"

model = joblib.load(model_path)
top_features = joblib.load(features_path)
threshold = joblib.load(threshold_path)

# -----------------------------
# Input Schema
# -----------------------------
class PatientData(BaseModel):
    spread1: float
    PPE: float
    spread2: float
    HNR: float
    MDVP_Shimmer: float
    voice_amplitude_instability: float
    MDVP_APQ: float
    MDVP_Flo_Hz: float
    Shimmer_APQ5: float
    D2: float
    MDVP_Shimmer_dB: float
    Shimmer_DDA: float
    Shimmer_APQ3: float
    MDVP_Jitter_Abs: float
    MDVP_Fo_Hz: float


# -----------------------------
# HOME PAGE
# -----------------------------
@app.get("/")
def home():
    return FileResponse("templates/index.html")


# -----------------------------
# ANALYSIS PAGE
# -----------------------------
@app.get("/analysis")
def analysis():
    return FileResponse("templates/analysis.html")


# -----------------------------
# PREDICTION API
# -----------------------------
@app.post("/predict")
def predict(data: PatientData):

    input_data = {
        "spread1": data.spread1,
        "PPE": data.PPE,
        "spread2": data.spread2,
        "HNR": data.HNR,
        "MDVP:Shimmer": data.MDVP_Shimmer,
        "voice_amplitude_instability": data.voice_amplitude_instability,
        "MDVP:APQ": data.MDVP_APQ,
        "MDVP:Flo(Hz)": data.MDVP_Flo_Hz,
        "Shimmer:APQ5": data.Shimmer_APQ5,
        "D2": data.D2,
        "MDVP:Shimmer(dB)": data.MDVP_Shimmer_dB,
        "Shimmer:DDA": data.Shimmer_DDA,
        "Shimmer:APQ3": data.Shimmer_APQ3,
        "MDVP:Jitter(Abs)": data.MDVP_Jitter_Abs,
        "MDVP:Fo(Hz)": data.MDVP_Fo_Hz
    }

    input_df = pd.DataFrame([input_data])

    # Feature order
    input_df = input_df[top_features]

    # Prediction probability
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Threshold prediction
    prediction = (
        1 if probability >= float(threshold)
        else 0
    )

    # Result text
    result = (
        "Parkinson Risk Detected"
        if prediction == 1
        else "No Strong Parkinson Indicators"
    )

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(probability * 100, 2)
    }