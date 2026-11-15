# 🧠 Stroke Risk Prediction Service - ML Zoomcamp Midterm Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project developed for the **DataTalksClub Machine Learning Zoomcamp Midterm Project**. This project strictly implements the methodologies, best practices, and processes taught across **Modules 1 through 6** of the course curriculum.

---

## 📋 Table of Contents
1. [Curriculum & Methodological Alignment (Modules 1 - 6)](#1-curriculum--methodological-alignment-modules-1---6)
2. [Problem Description & Clinical Understanding (Module 1)](#2-problem-description--clinical-understanding-module-1)
3. [Dataset Overview & Preprocessing (Modules 3.2 & 3.3)](#3-dataset-overview--preprocessing-modules-32--33)
4. [Repository Structure](#4-repository-structure)
5. [Exploratory Data Analysis & Feature Importance (Modules 3.4 - 3.7)](#5-exploratory-data-analysis--feature-importance-modules-34---37)
6. [Model Training, Tuning & Cross-Validation (Modules 3, 4 & 6)](#6-model-training-tuning--cross-validation-modules-3-4--6)
   - [6.1 Logistic Regression & Coefficient Interpretation](#61-logistic-regression--coefficient-interpretation)
   - [6.2 5-Fold Stratified Cross-Validation](#62-5-fold-stratified-cross-validation)
   - [6.3 Decision Tree Classifier & Grid Tuning](#63-decision-tree-classifier--grid-tuning)
   - [6.4 Random Forest Classifier & Tree Depth Curves](#64-random-forest-classifier--tree-depth-curves)
   - [6.5 Gradient Boosting & XGBoost Tuning](#65-gradient-boosting--xgboost-tuning)
   - [6.6 Four-Model Comparison Table](#66-four-model-comparison-table)
7. [Evaluation Metrics & Decision Threshold Calibration (Module 4)](#7-evaluation-metrics--decision-threshold-calibration-module-4)
   - [7.1 Dummy Baseline vs Model Accuracy](#71-dummy-baseline-vs-model-accuracy)
   - [7.2 Precision-Recall Curve](#72-precision-recall-curve)
   - [7.3 ROC Curve: Model vs Random vs Ideal](#73-roc-curve-model-vs-random-vs-ideal)
   - [7.4 Threshold Tuning (F1-Score Maximization)](#74-threshold-tuning-f1-score-maximization)
   - [7.5 Final Evaluation on Unseen Test Set](#75-final-evaluation-on-unseen-test-set)
8. [How to Run & Reproduce (Module 5: Deployment)](#8-how-to-run--reproduce-module-5-deployment)
   - [8.1 Local Environment Setup](#81-local-environment-setup)
   - [8.2 Running the Exploratory Notebook](#82-running-the-exploratory-notebook)
   - [8.3 Training Pipeline (train.py)](#83-training-pipeline-trainpy)
   - [8.4 Serving Locally (predict.py)](#84-serving-locally-predictpy)
   - [8.5 Testing the API (test.py)](#85-testing-the-api-testpy)
9. [Docker Containerization (Module 5.6)](#9-docker-containerization-module-56)
10. [Interactive Swagger Documentation](#10-interactive-swagger-documentation)
11. [Important Implementation Notes (GitHub Codespaces)](#11-important-implementation-notes-github-codespaces)

---

## 1. Curriculum & Methodological Alignment (Modules 1 - 6)

This project strictly adheres to the teaching structure of the DataTalksClub ML Zoomcamp:

| Module | Core Methodology Taught | How it is Applied in this Project |
|---|---|---|
| **01-intro** | CRISP-DM framework, Python/NumPy/Pandas setup, Problem Formulation | Project structured under CRISP-DM stages with clinically actionable value proposition. |
| **02-regression & 03-classification** | Validation framework (60/20/20), Lowercasing, One-Hot Encoding (`DictVectorizer`), Logistic Regression | Stratified split, string normalization, `DictVectorizer(sparse=False)`, balanced weights. |
| **03-classification** | **Risk Ratio & Difference Analysis** (`mean - global`, `mean / global`), **Mutual Information**, Correlation | Group stroke rate difference and risk ratio tables computed for all categorical/binary features; mutual info ranked. |
| **03-classification** | **Model Interpretation** (Weights & Biases) | Inspection and visualization of Logistic Regression coefficients and bias (`intercept_`). |
| **04-evaluation** | **Accuracy vs Dummy Baseline**, Confusion Matrix, Precision/Recall tradeoff | Demonstration of why 95.1% accuracy of a dummy model (predicting all 0s) is deceptive for imbalanced data. |
| **04-evaluation** | **ROC Curves: Model vs Random vs Ideal** | ROC curve plotted alongside the theoretical Ideal Model and Random Baseline. |
| **04-evaluation** | **K-Fold Cross-Validation** (`mean +/- std`) | 5-Fold Stratified Cross-Validation on full train for hyperparameter tuning of `C` (`mean +/- std`). |
| **05-deployment** | Model serialization, REST API service, Docker containerization | Exporting `model.bin` with metadata, FastAPI web service (`predict.py`), automated client (`test.py`), `Dockerfile`. |
| **06-trees** | **Decision Trees** tuning (`max_depth`, `min_samples_leaf`) | Grid search tuning over `max_depth` and `min_samples_leaf`, visualized as a 2D heatmap. |
| **06-trees** | **Random Forest** tuning (`n_estimators` across depths) | Curves of ROC-AUC vs number of trees (10 to 200) for different depths (`max_depth` = 4, 6, 8, 10). |
| **06-trees** | **XGBoost Tuning** & Model Selection | Tuning `learning_rate` (eta), `max_depth`, and `n_estimators`, selecting XGBoost as the winning architecture. |

---

## 2. Problem Description & Clinical Understanding (Module 1)

A **stroke** (accidente cerebrovascular o ictus) occurs when blood flow to part of the brain is interrupted or reduced, depriving brain cells of oxygen and essential nutrients. According to the **World Health Organization (WHO)**, stroke is the second leading cause of death and a principal cause of adult long-term disability worldwide.

### Value Proposition
Identifying at-risk patients early allows healthcare providers to implement targeted clinical actions:
- **Preventative Pharmacotherapy:** Prescription of antihypertensive drugs, blood sugar management, and anticoagulant therapies.
- **Lifestyle Interventions:** Smoking cessation programs, nutritional counseling, and weight management.
- **Cost Reduction:** Preventing emergency hospitalizations significantly lowers intensive care and rehabilitation expenses.

---

## 3. Dataset Overview & Preprocessing (Modules 3.2 & 3.3)

- **Source:** [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Rows:** 5,110 patients
- **Features:** 11 predictors + 1 target (`stroke`)
- **Class Imbalance:**
  - `0` (No Stroke): 4,861 patients (**95.13%**)
  - `1` (Stroke): 249 patients (**4.87%**)
- **Handling Missing Values Without Data Leakage:**
  - `bmi` had 201 missing values (3.93%).
  - The median was computed exclusively on `df_train` (**28.10**) and then used to impute `df_val`, `df_test`, and incoming production requests.

### Partitioning Strategy (60% / 20% / 20% Stratified)
```text
Total Dataset (5,110)
 ├── Full Train (80% = 4,088)
 │    ├── Train (75% of Full Train = 60% overall = 3,066)
 │    └── Validation (25% of Full Train = 20% overall = 1,022)
 └── Test (20% overall = 1,022) [Held out until final verification]
```

---

## 4. Repository Structure

```text
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Dataset (5,110 rows)
├── notebook.ipynb                           # Complete notebook covering Modules 1-6
├── train.py                                 # Production training pipeline (exports model.bin)
├── predict.py                               # FastAPI prediction service
├── test.py                                  # Test client script
├── test_service.py                          # Alias for test.py
├── model.bin                                # Serialized XGBoost model + DictVectorizer + metadata
├── requirements.txt                         # Production dependencies
├── requirements-dev.txt                     # Dev/Notebook dependencies
├── Pipfile                                  # Pipenv configuration
├── Dockerfile                               # Production Docker container
├── .dockerignore                            # Docker ignore rules
├── .gitignore                               # Git ignore rules
└── README.md                                # Project documentation
```

---

## 5. Exploratory Data Analysis & Feature Importance (Modules 3.4 - 3.7)

### 5.1 Risk Ratio and Difference (Module 3.5)
Comparing group stroke rate against the global baseline (`global_rate = 4.87%`):

| Feature & Category | Count | Stroke Rate (`mean`) | Difference (`mean - global`) | Risk Ratio (`mean / global`) | Interpretation |
|---|---|---|---|---|---|
| **Hypertension = 1** | 291 | **13.40%** | **+0.0853** | **2.75x** | Strongly elevated risk |
| **Hypertension = 0** | 2,775 | 3.96% | -0.0091 | 0.81x | Lower than baseline |
| **Heart Disease = 1** | 163 | **16.56%** | **+0.1169** | **3.40x** | Very high risk |
| **Heart Disease = 0** | 2,903 | 4.20% | -0.0067 | 0.86x | Lower than baseline |
| **Work Type = Self-employed**| 505 | **7.72%** | **+0.0285** | **1.58x** | Higher risk (older demographic) |
| **Work Type = children** | 413 | 0.00% | -0.0487 | 0.00x | Minimal/Zero stroke incidence |
| **Smoking = Formerly smoked**| 530 | **7.92%** | **+0.0305** | **1.63x** | Moderately elevated risk |
| **Smoking = Never smoked** | 1,141 | 4.82% | -0.0005 | 0.99x | Baseline risk |

### 5.2 Mutual Information (Module 3.6)
Ranked categorical feature importance with target `stroke`:
1. `ever_married` (MI = 0.0125)
2. `work_type` (MI = 0.0095)
3. `heart_disease` (MI = 0.0084)
4. `hypertension` (MI = 0.0076)
5. `smoking_status` (MI = 0.0028)
6. `residence_type` (MI = 0.0002)
7. `gender` (MI = 0.0001)

### 5.3 Numerical Correlation (Module 3.7)
Correlation with `stroke`:
- `age`: **+0.245** (strongest individual predictor; median age for stroke is ~70 vs ~43 for non-stroke)
- `avg_glucose_level`: **+0.132** (elevated glucose is an active risk factor)
- `bmi`: **+0.041**

---

## 6. Model Training, Tuning & Cross-Validation (Modules 3, 4 & 6)

### 6.1 Logistic Regression & Coefficient Interpretation
Tuned regularisation `C` with `class_weight='balanced'`:
- **Bias (Intercept):** `-1.8483`
- **Top positive coefficients (increase risk):** `age` (+1.61), `hypertension` (+0.38), `heart_disease` (+0.29), `avg_glucose_level` (+0.18).
- **Top negative coefficients (decrease risk):** `work_type=children` (-0.45), `gender=female` (-0.02).

### 6.2 5-Fold Stratified Cross-Validation
Tuning `C` on `df_full_train` across 5 stratified folds (reporting `mean +/- std`):
```text
C= 0.01: ROC-AUC = 0.8385 +- 0.0245
C=  0.1: ROC-AUC = 0.8412 +- 0.0232
C=  0.5: ROC-AUC = 0.8424 +- 0.0229
C=  1.0: ROC-AUC = 0.8427 +- 0.0228
C=  5.0: ROC-AUC = 0.8429 +- 0.0227
C= 10.0: ROC-AUC = 0.8429 +- 0.0227
```

### 6.3 Decision Tree Classifier & Grid Tuning
Evaluated `max_depth` ∈ [2, 3, 4, 6, 8, 10] and `min_samples_leaf` ∈ [1, 5, 10, 20, 50].
- Optimal configuration: `max_depth = 4`, `min_samples_leaf = 50`.
- Validation ROC-AUC: **0.8252**.

### 6.4 Random Forest Classifier & Tree Depth Curves
Evaluated `n_estimators` from 10 to 200 for depths 4, 6, 8, and 10.
- Performance stabilizes past 80 trees.
- Optimal configuration: `max_depth = 6`, `n_estimators = 150`, `min_samples_leaf = 10`.
- Validation ROC-AUC: **0.8385**.

### 6.5 Gradient Boosting & XGBoost Tuning
Tuned shrinkage (`learning_rate`), tree depth, and number of iterations:
- Optimal configuration: `learning_rate = 0.05`, `max_depth = 2`, `n_estimators = 150`.
- Validation ROC-AUC: **0.8551**.

### 6.6 Four-Model Comparison Table

| Model Family | Key Hyperparameters | Val ROC-AUC | Val PR-AUC | Selection |
|---|---|---|---|---|
| **Decision Tree** | `depth=4`, `leaf=50` | 0.8252 | 0.1610 | Interpretable baseline |
| **Random Forest** | `depth=6`, `trees=150`, `leaf=10` | 0.8385 | 0.1812 | Ensemble bagging |
| **Logistic Regression** | `C=0.1`, `class_weight='balanced'` | 0.8412 | 0.1745 | Linear model |
| **XGBoost (Winner)** | `lr=0.05`, `depth=2`, `trees=150` | **0.8551** | **0.1993** | **Best Performance** |

---

## 7. Evaluation Metrics & Decision Threshold Calibration (Module 4)

### 7.1 Dummy Baseline vs Model Accuracy
- **Dummy Model Accuracy:** **95.1%** (predicts 0 for everyone, catches 0% of strokes).
- Demonstrates why **ROC-AUC** and **Precision-Recall** are mandatory over raw accuracy.

### 7.2 Precision-Recall Curve
Precision and Recall intersect around threshold `0.15`, highlighting the clinical trade-off between false alarms and missed cases.

### 7.3 ROC Curve: Model vs Random vs Ideal
As taught in Module 4.5, our model's ROC curve is benchmarked against:
- **Random Baseline:** Diagonal line (AUC = 0.50).
- **Ideal Model:** Step function achieving perfect separation.
- **Model Curve:** Strongly curves toward the upper-left corner (AUC = 0.855), confirming solid discriminatory capacity.

### 7.4 Threshold Tuning (F1-Score Maximization)
By evaluating thresholds between 0.02 and 0.40 on the validation set:
- **Optimal Decision Threshold:** **`0.116`**
- **Validation F1-Score:** **`0.2843`**

### 7.5 Final Evaluation on Unseen Test Set
Retraining on `df_full_train` (4,088 samples) and evaluating on `df_test` (1,022 samples):
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
- **Clinical Outcome:** The calibrated model successfully flags **62% of all actual stroke occurrences** (`recall = 0.62`).

---

## 8. How to Run & Reproduce (Module 5: Deployment)

### 8.1 Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/stroke-prediction-mlzoomcamp.git
cd stroke-prediction-mlzoomcamp

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### 8.2 Running the Exploratory Notebook
```bash
# Open notebook interactively
jupyter notebook notebook.ipynb

# Or execute headlessly in terminal
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

### 8.3 Training Pipeline (train.py)
```bash
python train.py
```
*Output:*
```text
1. Loading dataset from: data/healthcare-dataset-stroke-data.csv
2. Splitting dataset into Train (60%), Validation (20%), Test (20%)...
   Calculated train median BMI: 28.10
3. Training validation XGBoost model...
   Validation ROC-AUC: 0.8551
   Optimal Threshold (Max F1): 0.116
4. Retraining final model on Full Train (Train + Val)...
   Test Set ROC-AUC: 0.8440
5. Exporting model artifact to: model.bin
```

### 8.4 Serving Locally (predict.py)
```bash
python predict.py
```
The FastAPI web server will launch at `http://0.0.0.0:9696`.

### 8.5 Testing the API (test.py)
In another terminal window:
```bash
python test.py
```
*Example Response:*
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

---

## 9. Docker Containerization (Module 5.6)

### 9.1 Build Docker Image
```bash
docker build -t stroke-prediction-service:v1 .
```

### 9.2 Run Container
```bash
docker run -d --rm -p 9696:9696 --name stroke-api stroke-prediction-service:v1
```

### 9.3 Test Container
```bash
# Verify health
curl http://localhost:9696/health

# Run test client
python test.py
```

### 9.4 Stop Container
```bash
docker stop stroke-api
```

---

## 10. Interactive Swagger Documentation

When the service is active, navigate to:
- **Swagger UI:** `http://localhost:9696/docs`
- **ReDoc:** `http://localhost:9696/redoc`

You can test patient predictions interactively through the built-in Swagger interface.

---

## 11. Important Implementation Notes (GitHub Codespaces)

1. **Port Visibility:** In GitHub Codespaces, ensure port `9696` is set to **Public** in the **PORTS** tab if accessing Swagger UI from an external browser.
2. **Missing Value Imputation:** In production, any payload omitting `"bmi": null` automatically utilizes the training median without throwing an error, returning `"imputed_bmi_used": true`.
