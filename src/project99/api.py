import os
from contextlib import asynccontextmanager
from io import StringIO

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from project99.preprocess import input_preprocessing
from project99.type import BatchPredictionResponse, HealthResponse, ModelInfoResponse, PredictionResponse, RawPointInput

model: xgb.XGBClassifier | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("Loading model")

    global model
    model_path = os.getenv("AIP_MODEL_DIR", "models/xgboost_model.json")
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Prediction endpoint will fail.")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    yield

    print("Cleaning up")
    del model

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Edit later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Project 99 API is running"}

@app.post("/predict")
def predict(input_data: RawPointInput, response_model=PredictionResponse):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw_point = input_data.model_dump()
        features = input_preprocessing(raw_point)

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(probability[0][1])
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check(response_model=HealthResponse):
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=os.getenv("AIP_MODEL_DIR", "models/xgboost_model.json") if model is not None else None
    )

@app.get("/model/info")
def model_info(response_model=ModelInfoResponse):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = model.get_booster().feature_names
    return ModelInfoResponse(
        model_type="XGBoost",
        model_loaded=True,
        model_path=os.getenv("AIP_MODEL_DIR", "models/xgboost_model.json"),
        feature_count=len(feature_names),
        feature_names=feature_names
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    required_columns = [
        'PointServer', 'P1Score', 'P2Score', 'P1GamesWon', 'P2GamesWon',
        'P1PointsWon', 'P2PointsWon', 'P1SetsWon', 'P2SetsWon',
        'SetNo', 'GameNo', 'PointNumber', 'ServeIndicator',
        'P1Momentum', 'P2Momentum'
    ]

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )

        prediction_values = []
        probability_values = []

        for _, row in df.iterrows():
            raw_point = row.to_dict()
            features = input_preprocessing(raw_point)

            prediction = model.predict(features)
            probability = model.predict_proba(features)

            pred_val = int(prediction[0])
            prob_val = float(probability[0][1])

            prediction_values.append(pred_val)
            probability_values.append(prob_val)

        df_with_predictions = df.copy()
        df_with_predictions['Prediction'] = prediction_values
        df_with_predictions['Probability'] = probability_values

        csv_output = StringIO()
        df_with_predictions.to_csv(csv_output, index=False)
        csv_string = csv_output.getvalue()

        return BatchPredictionResponse(
            total_predictions=len(prediction_values),
            csv_with_predictions=csv_string
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
