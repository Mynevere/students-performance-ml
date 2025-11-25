from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Student Performance Prediction API")

MODELS = {
    "LinearRegression": joblib.load("models/LinearRegression/student_performance_linearregression_model.pkl"),
    "RandomForest": joblib.load("models/RandomForest/student_performance_randomforest_model.pkl"),
    "SVM": joblib.load("models/SVM/student_performance_svm_model.pkl"),
}

MODEL_COLUMNS = {
    "LinearRegression": joblib.load("models/LinearRegression/linearregression_model_columns.pkl"),
    "RandomForest": joblib.load("models/RandomForest/randomforest_model_columns.pkl"),
    "SVM": joblib.load("models/SVM/svm_model_columns.pkl"),
}

SCALERS = {
    "SVM": joblib.load("models/SVM/svm_model_scaler.pkl")
}

class StudentInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_education: str
    lunch: str
    test_preparation_course: str

@app.post("/predict/{model_name}")
def predict(model_name: str, student: StudentInput):
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    model = MODELS[model_name]
    columns = MODEL_COLUMNS[model_name]

    data = pd.DataFrame([student.dict()])
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=columns, fill_value=0)

    if model_name == "SVM":
        scaler = SCALERS["SVM"]
        data_encoded = scaler.transform(data_encoded)

    prediction = model.predict(data_encoded)
    return {"prediction": float(prediction[0])}

@app.get("/feature-importance")
def feature_importance():
    try:
        model = MODELS["RandomForest"]
        columns = MODEL_COLUMNS["RandomForest"]
        importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        return importance.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gender-comparison")
def gender_comparison():
    df = pd.read_csv("StudentsPerformance.csv")
    result = (
        df.groupby("gender")[["math score", "reading score", "writing score"]]
        .mean()
        .round(2)
        .to_dict()
    )
    return result
