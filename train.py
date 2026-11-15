"""Training script for Stroke Risk Prediction model.

ML Zoomcamp Midterm Project.
This script:
1. Loads and preprocesses the dataset (healthcare-dataset-stroke-data.csv).
2. Performs stratified Train/Validation/Test splits.
3. Imputes missing values without data leakage using train set median.
4. Trains the selected XGBoost model with tuned hyperparameters.
5. Evaluates on validation set and calibrates the optimal decision threshold.
6. Evaluates final performance on test set.
7. Saves the trained model artifact (DictVectorizer, model, median BMI, threshold) to model.bin.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path("data/healthcare-dataset-stroke-data.csv")
OUTPUT_MODEL_PATH = Path("model.bin")

CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "residence_type",
    "smoking_status",
]
NUMERICAL_COLS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]


def load_and_preprocess_data(file_path: Path) -> pd.DataFrame:
    """Loads CSV and applies standard data cleaning."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Drop non-predictive id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Clean categorical text
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")

    # Harmonize rare 'other' gender (1 sample) to majority category
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({"other": "female"})

    return df


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """Finds probability threshold that maximizes F1-Score on validation set."""
    thresholds = np.linspace(0.02, 0.40, 100)
    scores = [f1_score(y_true, y_probs >= t) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def train():
    print(f"1. Loading dataset from: {DATA_PATH}")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"   Loaded {len(df)} records. Features: {list(df.columns)}")
    stroke_pct = df["stroke"].mean() * 100
    print(f"   Stroke rate (positive class): {stroke_pct:.2f}%")

    print("\n2. Splitting dataset into Train (60%), Validation (20%), Test (20%)...")
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["stroke"]
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42, stratify=df_full_train["stroke"]
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Impute missing BMI using train set median only (prevents data leakage)
    bmi_median_train = float(df_train["bmi"].median())
    print(f"   Calculated train median BMI: {bmi_median_train:.2f}")

    for split_df in (df_train, df_val):
        split_df["bmi"] = split_df["bmi"].fillna(bmi_median_train)

    y_train = df_train["stroke"].values
    y_val = df_val["stroke"].values

    df_train_feat = df_train.drop(columns=["stroke"])
    df_val_feat = df_val.drop(columns=["stroke"])

    dv_val = DictVectorizer(sparse=False)
    X_train = dv_val.fit_transform(df_train_feat.to_dict(orient="records"))
    X_val = dv_val.transform(df_val_feat.to_dict(orient="records"))

    best_params = {
        "learning_rate": 0.05,
        "max_depth": 2,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    print("\n3. Training validation XGBoost model...")
    val_model = XGBClassifier(**best_params)
    val_model.fit(X_train, y_train)

    val_probs = val_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_pr_auc = average_precision_score(y_val, val_probs)
    best_threshold, best_f1 = find_optimal_threshold(y_val, val_probs)

    print(f"   Validation ROC-AUC: {val_auc:.4f}")
    print(f"   Validation PR-AUC:  {val_pr_auc:.4f}")
    print(f"   Optimal Threshold (Max F1): {best_threshold:.3f} (F1 = {best_f1:.4f})")

    print("\n4. Retraining final model on Full Train (Train + Val) for production...")
    bmi_median_full = float(df_full_train["bmi"].median())
    df_full_train = df_full_train.copy()
    df_test = df_test.copy()

    df_full_train["bmi"] = df_full_train["bmi"].fillna(bmi_median_full)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median_full)

    y_full_train = df_full_train["stroke"].values
    y_test = df_test["stroke"].values

    df_full_train_feat = df_full_train.drop(columns=["stroke"])
    df_test_feat = df_test.drop(columns=["stroke"])

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(df_full_train_feat.to_dict(orient="records"))
    X_test = dv.transform(df_test_feat.to_dict(orient="records"))

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_full_train, y_full_train)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    test_preds = (test_probs >= best_threshold).astype(int)

    print(f"   Test Set ROC-AUC: {test_auc:.4f}")
    print(f"   Test Set PR-AUC:  {test_pr_auc:.4f}")
    print("\n   Test Set Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["No Stroke", "Stroke"]))

    print(f"\n5. Exporting model artifact to: {OUTPUT_MODEL_PATH}")
    artifact = {
        "dv": dv,
        "model": final_model,
        "bmi_median": bmi_median_full,
        "threshold": best_threshold,
    }

    with open(OUTPUT_MODEL_PATH, "wb") as f_out:
        pickle.dump(artifact, f_out)

    print(f"   Artifact saved successfully! ({OUTPUT_MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print("Training pipeline finished successfully.")


if __name__ == "__main__":
    train()
