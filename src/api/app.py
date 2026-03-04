import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --------------------------------------------------
# Fix path
# --------------------------------------------------

root_path = Path(__file__).resolve().parents[2]


from src.features.feature_engineering import FeatureEngineer

# --------------------------------------------------
# Load model + metadata
# --------------------------------------------------

artifacts_path = root_path / "src" / "artifacts"

model_path = artifacts_path / "xgb_pipeline.pkl"
metadata_path = artifacts_path / "xgb_metadata.json"

pipeline = joblib.load(model_path)

with open(metadata_path) as f:
    metadata = json.load(f)

BEST_THRESHOLD = metadata["best_threshold"]

# --------------------------------------------------
# FastAPI App Metadata
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using a trained XGBoost model with engineered features.",
    version="1.0.0",
)

# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/", tags=["Health Check"])
def home():
    return {"message": "Churn Prediction API is running 🚀"}

# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class Customer(BaseModel):
    tenure: int = Field(..., example=12)
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=850.2)
    Contract: str = Field(..., example="Month-to-month")
    InternetService: str = Field(..., example="Fiber optic")
    PaymentMethod: str = Field(..., example="Electronic check")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    PaperlessBilling: str = Field(..., example="Yes")

# --------------------------------------------------
# Response Schema
# --------------------------------------------------

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., example=0.7421)
    prediction: int = Field(..., example=1)
    threshold_used: float = Field(..., example=0.61)

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict customer churn",
    description="Returns churn probability and binary prediction using trained model."
)
def predict(customer: Customer):

    input_df = pd.DataFrame([customer.dict()])

    prob = pipeline.predict_proba(input_df)[:, 1][0]

    prediction = int(prob > BEST_THRESHOLD)

    return {
        "churn_probability": round(float(prob), 4),
        "prediction": prediction,
        "threshold_used": round(BEST_THRESHOLD, 4)
    }