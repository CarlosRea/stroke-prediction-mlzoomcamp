"""Web service for Stroke Risk Prediction using FastAPI.

ML Zoomcamp Midterm Project.
This service:
1. Loads model.bin containing the DictVectorizer, trained XGBoost model,
   imputation median for BMI, and calibrated decision threshold.
2. Exposes /health and /predict endpoints.
3. Automatically documents the API with interactive Swagger UI at /docs.
"""

import pickle
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.bin")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found at {MODEL_PATH}. Run 'python train.py' first."
    )

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

dv = artifact["dv"]
model = artifact["model"]
bmi_median = float(artifact["bmi_median"])
threshold = float(artifact["threshold"])

app = FastAPI(
    title="Stroke Risk Prediction API",
    description=(
        "Production REST API that estimates stroke risk based on clinical, "
        "demographic, and lifestyle indicators. Developed for DataTalksClub ML Zoomcamp Midterm."
    ),
    version="1.0.0",
)


class PatientData(BaseModel):
    gender: str = Field(..., description="Gender (Male, Female)", examples=["Male"])
    age: float = Field(..., ge=0, le=120, description="Age in years", examples=[67.0])
    hypertension: int = Field(
        ..., ge=0, le=1, description="0 = No hypertension, 1 = Hypertension", examples=[0]
    )
    heart_disease: int = Field(
        ..., ge=0, le=1, description="0 = No heart disease, 1 = Heart disease", examples=[1]
    )
    ever_married: str = Field(..., description="Ever married ('Yes' or 'No')", examples=["Yes"])
    work_type: str = Field(
        ...,
        description="Work type (Private, Self-employed, Govt_job, children, Never_worked)",
        examples=["Private"],
    )
    residence_type: str = Field(
        ..., description="Residence type (Urban or Rural)", examples=["Urban"]
    )
    avg_glucose_level: float = Field(
        ..., gt=0, description="Average blood glucose level (mg/dL)", examples=[228.69]
    )
    bmi: Optional[float] = Field(
        None, gt=0, description="Body Mass Index (kg/m^2). If null, train median will be used.", examples=[36.6]
    )
    smoking_status: str = Field(
        ...,
        description="Smoking status (formerly smoked, never smoked, smokes, Unknown)",
        examples=["formerly smoked"],
    )


class PredictionResponse(BaseModel):
    stroke_probability: float = Field(..., description="Predicted probability of stroke (0.0 to 1.0)")
    high_stroke_risk: bool = Field(..., description="True if probability >= calibrated threshold")
    decision_threshold: float = Field(..., description="Optimal decision threshold calibrated on validation set")
    imputed_bmi_used: bool = Field(..., description="Indicates if BMI was missing and imputed using train median")


def preprocess_patient(patient: PatientData) -> tuple[dict, bool]:
    """Cleans and normalizes patient record for DictVectorizer."""
    raw = patient.model_dump()
    clean = {}
    bmi_was_imputed = False

    for key, val in raw.items():
        norm_key = key.strip().lower().replace(" ", "_")
        if isinstance(val, str):
            clean[norm_key] = val.strip().lower().replace(" ", "_")
        else:
            clean[norm_key] = val

    if clean.get("gender") == "other":
        clean["gender"] = "female"

    if clean.get("bmi") is None:
        clean["bmi"] = bmi_median
        bmi_was_imputed = True

    return clean, bmi_was_imputed


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Stroke Risk Prediction API is running. Visit /docs for interactive Swagger API documentation.",
        "status": "healthy",
        "model": "XGBoost Classifier",
        "threshold": round(threshold, 4),
    }


@app.get("/health", tags=["General"])
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_stroke_risk(patient: PatientData):
    try:
        clean_record, bmi_was_imputed = preprocess_patient(patient)
        X = dv.transform([clean_record])
        prob = float(model.predict_proba(X)[0, 1])
        high_risk = bool(prob >= threshold)

        return PredictionResponse(
            stroke_probability=round(prob, 4),
            high_stroke_risk=high_risk,
            decision_threshold=round(threshold, 4),
            imputed_bmi_used=bmi_was_imputed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=False)
