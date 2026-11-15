# 🧠 Stroke Risk Prediction Service - ML Zoomcamp Midterm Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project developed for the **DataTalksClub Machine Learning Zoomcamp Midterm Project**. This project predicts patient stroke risk based on clinical, demographic, and lifestyle indicators, deployed as a containerized REST API with FastAPI and Docker.

---

## 📋 Table of Contents
1. [Problem Description & Business Context](#1-problem-description--business-context)
2. [Dataset Overview](#2-dataset-overview)
3. [Repository Structure](#3-repository-structure)
4. [Exploratory Data Analysis (EDA) Highlights](#4-exploratory-data-analysis-eda-highlights)
5. [Model Training, Tuning & Evaluation](#5-model-training-tuning--evaluation)
6. [Decision Threshold Calibration](#6-decision-threshold-calibration)
7. [How to Run & Reproduce](#7-how-to-run--reproduce)
   - [7.1 Environment Setup](#71-environment-setup)
   - [7.2 Exploratory Notebook](#72-exploratory-notebook)
   - [7.3 Training the Model](#73-training-the-model)
   - [7.4 Running the Web Service Locally](#74-running-the-web-service-locally)
   - [7.5 Testing the API](#75-testing-the-api)
8. [Docker Containerization](#8-docker-containerization)
9. [Interactive API Documentation (Swagger)](#9-interactive-api-documentation-swagger)
10. [Important Implementation Notes (GitHub Codespaces)](#10-important-implementation-notes-github-codespaces)

---

## 1. Problem Description & Business Context

A **stroke** (accidente cerebrovascular o ictus) occurs when blood flow to part of the brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. According to the **World Health Organization (WHO)**, stroke is the second leading cause of death and a principal cause of adult long-term disability worldwide.

### Clinical & Business Value
Detecting stroke risk before symptoms arise allows healthcare systems, HMOs, and preventive medicine programs to:
- **Prioritize early clinical interventions:** Proactive prescription of antihypertensive and anticoagulant medications, blood sugar control, and vascular ultrasound screenings.
- **Recommend targeted lifestyle modifications:** Smoking cessation, dietary counseling, and physical activity monitoring.
- **Reduce critical care hospitalizations:** Preventing acute cerebrovascular events significantly reduces intensive care unit (ICU) admissions and chronic rehabilitation costs.

---

## 2. Dataset Overview

The dataset used is the publicly available **[Stroke Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)**.

- **Rows:** 5,110 patients
- **Columns:** 11 predictive features + 1 target (`stroke`)
- **Target Distribution:**
  - `0` (No Stroke): 4,861 patients (**95.13%**)
  - `1` (Stroke): 249 patients (**4.87%**)
  - *Severe class imbalance* (~19:1 ratio).
- **Missing Values:**
  - `bmi` had 201 missing values (3.93%).
  - **No Data Leakage Strategy:** Missing `bmi` values are imputed using the **median calculated exclusively on the training partition** (median = 28.10), then transferred to validation, test, and production pipelines.

### Features

| Feature | Type | Description | Values / Range |
|---|---|---|---|
| `gender` | Categorical | Patient gender | `Male`, `Female` |
| `age` | Numerical | Age in years | 0.08 - 82.0 |
| `hypertension` | Binary | Blood pressure condition | `0` (No), `1` (Yes) |
| `heart_disease`| Binary | History of heart disease | `0` (No), `1` (Yes) |
| `ever_married` | Categorical | Marital status history | `Yes`, `No` |
| `work_type` | Categorical | Occupation type | `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` |
| `residence_type`| Categorical | Area of residence | `Urban`, `Rural` |
| `avg_glucose_level` | Numerical | Average blood glucose level | 55.12 - 271.74 mg/dL |
| `bmi` | Numerical | Body Mass Index | 10.3 - 97.6 kg/m² (201 imputed) |
| `smoking_status` | Categorical | Tobacco smoking history | `formerly smoked`, `never smoked`, `smokes`, `Unknown` |
| `stroke` | **Target** | Acute stroke event | `0` (No), `1` (Yes) |

---

## 3. Repository Structure

```text
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Dataset (5,110 rows)
├── notebook.ipynb                           # Complete EDA, feature engineering & model tuning
├── train.py                                 # Training pipeline script (exports model.bin)
├── predict.py                               # FastAPI production prediction service
├── test.py                                  # Test client script (high-risk and low-risk cases)
├── test_service.py                          # Alias runner for test.py
├── model.bin                                # Serialized artifact (DictVectorizer, model, metadata)
├── requirements.txt                         # Production dependencies
├── requirements-dev.txt                     # Development & notebook execution dependencies
├── Pipfile                                  # Pipenv configuration file
├── Dockerfile                               # Multi-platform production container specification
├── .dockerignore                            # Exclusions for lean Docker build context
├── .gitignore                               # Git ignored files (.venv, cache, checkpoints)
└── README.md                                # Project documentation & execution guide
```

---

## 4. Exploratory Data Analysis (EDA) Highlights

The full analysis is documented and rendered inside [`notebook.ipynb`](file:///workspaces/stroke-prediction-mlzoomcamp/notebook.ipynb):

1. **Age is the Dominant Factor:**
   - Stroke patients have a median age of ~70 years, whereas non-stroke patients have a median age of ~43 years.
   - Very few patients under 45 suffered strokes in this dataset.
2. **Clinical Comorbidities Dramatically Increase Risk:**
   - Patients with **hypertension** experience a **13.2%** stroke rate vs **3.9%** without hypertension (~3.4x higher).
   - Patients with **heart disease** experience a **17.0%** stroke rate vs **4.1%** without heart disease (~4.1x higher).
3. **Hyperglycemia / Diabetes Correlation:**
   - High blood glucose (> 160 mg/dL) shows a noticeable surge in stroke occurrence.
4. **Mutual Information:**
   - Categorical and binary features ranked by Mutual Information with `stroke`:
     `age_binned` > `ever_married` > `hypertension` > `heart_disease` > `work_type` > `smoking_status`.

---

## 5. Model Training, Tuning & Evaluation

### Data Splitting Strategy
- **Stratified Partition:** Preserves the 4.87% positive class proportion across splits.
  - **Train Set (60%):** 3,066 records
  - **Validation Set (20%):** 1,022 records
  - **Test Set (20%):** 1,022 records
- **Feature Vectorization:** `DictVectorizer(sparse=False)` applied to one-hot encode categoricals without hardcoding.

### Comparison of Models

Three distinct model families were trained and tuned on the validation set:

| Model Family | Key Hyperparameters Tuned | Val ROC-AUC | Val PR-AUC | Result |
|---|---|---|---|---|
| **Logistic Regression** | `C` ∈ [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0], `class_weight='balanced'` | 0.8412 | 0.1745 | Baseline linear model |
| **Random Forest** | `max_depth` ∈ [4, 6, 8, 10], `min_samples_leaf` ∈ [2, 5, 10, 20], `n_estimators=150` | 0.8385 | 0.1812 | Non-linear ensemble |
| **XGBoost (Winner)** | `learning_rate` ∈ [0.02, 0.05, 0.1], `max_depth` ∈ [2, 3, 4], `n_estimators` ∈ [100, 150, 200] | **0.8551** | **0.1993** | **Best Discriminative Performance** |

### Winning Model Hyperparameters
- **Algorithm:** `XGBClassifier`
- `learning_rate`: `0.05`
- `max_depth`: `2` (prevents overfitting on small tabular medical datasets)
- `n_estimators`: `150`
- `subsample`: `0.8`
- `colsample_bytree`: `0.8`
- `eval_metric`: `"auc"`

---

## 6. Decision Threshold Calibration

In severe class imbalance (4.87% positives), the standard threshold of `0.50` yields near-zero recall because predicted probabilities cluster between 0.01 and 0.35.

To make the model clinically actionable, we calibrated the decision threshold on the validation set by maximizing the **F1-Score**:
- **Optimal Threshold:** `0.116`
- **Validation F1-Score:** `0.284`

### Final Evaluation on Unseen Test Set (1,022 samples)
Evaluating the retrained model on the held-out test partition using the calibrated threshold:
- **Test ROC-AUC:** **0.8440**
- **Test PR-AUC:** **0.2836**
- **Classification Report:**
  ```text
                precision    recall  f1-score   support

     No Stroke       0.98      0.89      0.93       972
        Stroke       0.23      0.62      0.33        50

      accuracy                           0.88      1022
     macro avg       0.60      0.76      0.63      1022
  weighted avg       0.94      0.88      0.90      1022
  ```
- **Clinical Impact:** The model identifies **62% of actual stroke cases** (`recall=0.62`), while maintaining an overall accuracy of **88%**.

---

## 7. How to Run & Reproduce

### 7.1 Environment Setup

Make sure you have Python 3.11 or 3.12 installed.

```bash
# Clone the repository
git clone https://github.com/<your-username>/stroke-prediction-mlzoomcamp.git
cd stroke-prediction-mlzoomcamp

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### 7.2 Exploratory Notebook

To view or re-execute the notebook:

```bash
# Launch Jupyter Notebook
jupyter notebook notebook.ipynb

# Or execute headlessly in terminal
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

### 7.3 Training the Model

Execute [`train.py`](file:///workspaces/stroke-prediction-mlzoomcamp/train.py) to run the full preprocessing, training, threshold tuning, and artifact generation pipeline:

```bash
python train.py
```

*Output:*
```text
1. Loading dataset from: data/healthcare-dataset-stroke-data.csv
   Loaded 5110 records. Features: ['gender', 'age', 'hypertension', ...]
   Stroke rate (positive class): 4.87%

2. Splitting dataset into Train (60%), Validation (20%), Test (20%)...
   Calculated train median BMI: 28.10

3. Training validation XGBoost model...
   Validation ROC-AUC: 0.8551
   Optimal Threshold (Max F1): 0.116 (F1 = 0.2843)

4. Retraining final model on Full Train (Train + Val) for production...
   Test Set ROC-AUC: 0.8440
   Test Set PR-AUC:  0.2836

5. Exporting model artifact to: model.bin
   Artifact saved successfully! (135.1 KB)
```

### 7.4 Running the Web Service Locally

Start the FastAPI application with Uvicorn:

```bash
python predict.py
```

The service will start on `http://0.0.0.0:9696`.

### 7.5 Testing the API

In a separate terminal, run [`test.py`](file:///workspaces/stroke-prediction-mlzoomcamp/test.py):

```bash
python test.py
```

*Example Output:*
```text
Testing API health at http://localhost:9696/health...
Health check status: 200 -> {'status': 'ok', 'model_loaded': True}

--- Sending request for: High-Risk Patient Profile ---
HTTP Status: 200
 -> Stroke Probability : 0.2481 (24.81%)
 -> High Risk Flag     : YES [HIGH RISK]
 -> Decision Threshold : 0.1160
 -> BMI Was Imputed    : False

--- Sending request for: Low-Risk Patient Profile (with missing BMI) ---
HTTP Status: 200
 -> Stroke Probability : 0.0037 (0.37%)
 -> High Risk Flag     : NO [LOW RISK]
 -> Decision Threshold : 0.1160
 -> BMI Was Imputed    : True
```

You can also test using `curl`:

```bash
curl -X POST "http://localhost:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 78.0,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

---

## 8. Docker Containerization

The repository includes a production [`Dockerfile`](file:///workspaces/stroke-prediction-mlzoomcamp/Dockerfile) packaging the service.

### 8.1 Build the Docker Image

```bash
docker build -t stroke-prediction-service:v1 .
```

### 8.2 Run the Docker Container

```bash
docker run -d --rm -p 9696:9696 --name stroke-api stroke-prediction-service:v1
```

### 8.3 Test the Containerized Service

```bash
# Run the test client
python test.py

# Or verify with curl
curl http://localhost:9696/health
```

### 8.4 Stop the Container

```bash
docker stop stroke-api
```

---

## 9. Interactive API Documentation (Swagger)

FastAPI automatically serves interactive OpenAPI documentation:

- **Swagger UI:** `http://localhost:9696/docs`
- **ReDoc:** `http://localhost:9696/redoc`

You can use the **Try it out** button directly in the browser to submit sample patient payloads and inspect the responses.

---

## 10. Important Implementation Notes (GitHub Codespaces)

When running inside **GitHub Codespaces**:
1. **Port Forwarding:** When starting the FastAPI service or Docker container on port `9696`, Codespaces will automatically detect the port. Look at the bottom **PORTS** panel in VS Code.
2. **Port Visibility (`Private` vs `Public`):**
   - Testing via terminal inside Codespaces (`python test.py` or `curl localhost:9696/...`) works right away.
   - If testing from an external browser or tool using the URL `https://<codespace-id>-9696.app.github.dev/docs`, right-click on the port `9696` in the PORTS tab and select **Port Visibility -> Public**.
3. **No Leakage Imputation:** Notice that when testing patients with missing BMI (`"bmi": null`), the API automatically imputes it using the median computed during training and flags `"imputed_bmi_used": true`.
