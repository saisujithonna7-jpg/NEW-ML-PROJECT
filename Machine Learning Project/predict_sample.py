"""
Chronic Kidney Disease (CKD) - Inference / Prediction Script
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

def load_inference_artifacts(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    features_path = os.path.join(models_dir, "feature_names.json")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)

    return model, scaler, feature_names

def predict_patient(patient_dict, model, scaler, feature_names):
    """
    Accepts a dictionary of patient measurements, formats them to match
    the training feature schema, scales them, and returns prediction + probability.
    """
    df_raw = pd.DataFrame([patient_dict])
    
    # One-hot encode categoricals matching the training format
    cat_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
    df_encoded = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True)
    
    # Reindex to guarantee exact feature order and fill missing dummy columns with 0
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Scale numerical features
    features_scaled = scaler.transform(df_aligned)
    
    # Predict
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1] if hasattr(model, "predict_proba") else None
    
    return pred, proba

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model, scaler, feature_names = load_inference_artifacts(base_dir)
    print("Successfully loaded trained model and scaler pipeline!\n")

    # Sample Case 1: High-risk patient profile (elevated serum creatinine, blood urea, albumin, hypertension)
    sample_ckd_patient = {
        "age": 62.0, "bp": 80.0, "sg": 1.010, "al": 2.0, "su": 3.0,
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "bgr": 423.0, "bu": 53.0, "sc": 1.8, "sod": 135.0, "pot": 4.4,
        "hemo": 9.6, "pcv": 31.0, "wc": 7500.0, "rc": 3.9,
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor",
        "pe": "no", "ane": "yes"
    }

    # Sample Case 2: Healthy patient profile (normal hemoglobin, creatinine, no albumin, no hypertension)
    sample_healthy_patient = {
        "age": 40.0, "bp": 80.0, "sg": 1.025, "al": 0.0, "su": 0.0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 140.0, "bu": 10.0, "sc": 1.2, "sod": 135.0, "pot": 5.0,
        "hemo": 15.0, "pcv": 48.0, "wc": 6300.0, "rc": 5.3,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no"
    }

    print("--- Evaluating Patient Case 1 (Suspected CKD) ---")
    pred1, proba1 = predict_patient(sample_ckd_patient, model, scaler, feature_names)
    diagnosis1 = "Chronic Kidney Disease Detected (CKD)" if pred1 == 1 else "Healthy (Not CKD)"
    print(f"Prediction: {diagnosis1}")
    if proba1 is not None:
        print(f"CKD Risk Probability: {proba1 * 100:.2f}%\n")

    print("--- Evaluating Patient Case 2 (Suspected Healthy) ---")
    pred2, proba2 = predict_patient(sample_healthy_patient, model, scaler, feature_names)
    diagnosis2 = "Chronic Kidney Disease Detected (CKD)" if pred2 == 1 else "Healthy (Not CKD)"
    print(f"Prediction: {diagnosis2}")
    if proba2 is not None:
        print(f"CKD Risk Probability: {proba2 * 100:.2f}%\n")

if __name__ == "__main__":
    main()
