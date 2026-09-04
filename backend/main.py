from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ai_maintenance_optimization.data.loader import SENSOR_COLUMNS


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="AI Maintenance Optimization API",
    description="Backend API for predictive maintenance",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# SENSOR INPUT SCHEMA
# --------------------------------------------------

class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    rotation_speed: float
    voltage: float
    current: float
    oil_level: float
    load: float
    motor_temperature: float
    gearbox_temperature: float
    sound_level: float
    fan_speed: float
    reactive_power: float
    active_power: float


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "AI Maintenance Optimization API is running"
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_failure(sensor_data: SensorData):

    # Convert input into dictionary
    data = sensor_data.model_dump()

    # Create DataFrame using exact training column order
    input_df = pd.DataFrame(
        [[data[column] for column in SENSOR_COLUMNS]],
        columns=SENSOR_COLUMNS
    )

    # Scale the sensor values
    input_scaled = scaler.transform(input_df)

    # Predict failure class
    prediction = model.predict(input_scaled)[0]

    # Predict probability of machine failure
    probability = model.predict_proba(input_scaled)[0][1]

    # Determine risk level
    if probability < 0.30:
        risk_level = "LOW"
    elif probability < 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
    "predicted_failure": int(prediction),
    "failure_probability": round(float(probability), 4),
    "risk_level": risk_level,
    "recommendation": (
        "Continue normal monitoring. "
        "No immediate maintenance action is required."
        if risk_level == "LOW"
        else
        "Schedule a maintenance inspection "
        "and continue monitoring sensor readings."
        if risk_level == "MEDIUM"
        else
        "Inspect the machine as soon as possible "
        "and closely monitor sensor conditions."
    )
}
    
    # --------------------------------------------------
# DATA SUMMARY ENDPOINT
# --------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simulated_iiot_dataset.csv"
)


@app.get("/data/summary")
def data_summary():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    total_records = len(df)
    failure_count = int(df["machine_failure"].sum())
    normal_count = total_records - failure_count

    failure_rate = (
        failure_count / total_records
        if total_records > 0
        else 0
    )

    return {
        "total_records": total_records,
        "normal_records": normal_count,
        "failure_records": failure_count,
        "failure_rate": round(failure_rate, 4),
        "sensor_count": len(SENSOR_COLUMNS),
        "sensors": SENSOR_COLUMNS
    }
    
    # --------------------------------------------------
# MODEL METRICS ENDPOINT
# --------------------------------------------------

METRICS_PATH = MODELS_DIR / "metrics.json"


@app.get("/model/metrics")
def model_metrics():

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    import json

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    return metrics


# --------------------------------------------------
# MAINTENANCE RECOMMENDATION ENDPOINT
# --------------------------------------------------

@app.post("/maintenance/recommendation")
def maintenance_recommendation(sensor_data: SensorData):

    # Convert input data into DataFrame
    data = sensor_data.model_dump()

    input_df = pd.DataFrame(
        [[data[column] for column in SENSOR_COLUMNS]],
        columns=SENSOR_COLUMNS
    )

    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # Predict failure probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Determine risk level
    if probability < 0.30:
        risk_level = "LOW"
        recommendation = (
            "Continue normal monitoring. "
            "No immediate maintenance action is required."
        )

    elif probability < 0.60:
        risk_level = "MEDIUM"
        recommendation = (
            "Schedule a maintenance inspection "
            "and continue monitoring sensor readings."
        )

    else:
        risk_level = "HIGH"
        recommendation = (
            "Inspect the machine as soon as possible "
            "and closely monitor sensor conditions."
        )

    return {
        "failure_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "recommendation": recommendation
    }