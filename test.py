"""Test client script for the Stroke Risk Prediction web service.

ML Zoomcamp Midterm Project.
Sends sample patient JSON payloads to http://localhost:9696/predict
and prints formatted predictions.
"""

import sys
import requests

API_URL = "http://localhost:9696/predict"
HEALTH_URL = "http://localhost:9696/health"

# High-risk profile: Elderly male, hypertension, heart disease, high glucose, high BMI, smoking history
high_risk_patient = {
    "gender": "Male",
    "age": 78.0,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}

# Low-risk profile: Young female, no hypertension, no heart disease, normal glucose, missing BMI, never smoked
low_risk_patient = {
    "gender": "Female",
    "age": 25.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "No",
    "work_type": "Private",
    "residence_type": "Rural",
    "avg_glucose_level": 82.5,
    "bmi": None,  # Missing BMI to test automatic train median imputation
    "smoking_status": "never smoked",
}


def test_api():
    print(f"Testing API health at {HEALTH_URL}...")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=5)
        print(f"Health check status: {health_resp.status_code} -> {health_resp.json()}\n")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {HEALTH_URL}.")
        print("Please ensure the web service is running (e.g. 'python predict.py' or Docker container).")
        sys.exit(1)

    test_cases = [
        ("High-Risk Patient Profile", high_risk_patient),
        ("Low-Risk Patient Profile (with missing BMI)", low_risk_patient),
    ]

    for label, payload in test_cases:
        print(f"--- Sending request for: {label} ---")
        response = requests.post(API_URL, json=payload, timeout=5)
        print(f"HTTP Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            prob = data["stroke_probability"]
            risk = data["high_stroke_risk"]
            thresh = data["decision_threshold"]
            imputed = data["imputed_bmi_used"]
            print(f" -> Stroke Probability : {prob:.4f} ({prob*100:.2f}%)")
            print(f" -> High Risk Flag     : {'YES [HIGH RISK]' if risk else 'NO [LOW RISK]'}")
            print(f" -> Decision Threshold : {thresh:.4f}")
            print(f" -> BMI Was Imputed    : {imputed}")
        else:
            print("Error response:", response.text)
        print()


if __name__ == "__main__":
    test_api()
