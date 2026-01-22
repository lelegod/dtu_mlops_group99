import os
from contextlib import asynccontextmanager
from io import StringIO
from typing import List

import numpy as np
import pandas as pd  # type: ignore
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage  # type: ignore
from loguru import logger
from pydantic import BaseModel

from project99.constants import GCS_MODEL_PATH, LOCAL_MODEL_PATH
from project99.logging_utils import setup_logging
from project99.preprocess import input_preprocessing
from project99.type import BatchPredictionResponse, HealthResponse, ModelInfoResponse, PredictionResponse, RawPointInput

model: xgb.XGBClassifier | None = None


class VertexRequest(BaseModel):
    instances: List[RawPointInput]


class VertexResponse(BaseModel):
    predictions: List[PredictionResponse]


setup_logging(log_file="reports/api.log")


def download_model_from_gcs(gcs_path: str, local_path: str) -> bool:
    try:
        path_parts = gcs_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded model from {gcs_path} to {local_path}")
            return True
        else:
            logger.warning(f"Model not found in GCS: {gcs_path}")
            return False
    except Exception as e:
        logger.exception(f"Error downloading from GCS: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model")
    global model

    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.warning(f"Model not found locally at {LOCAL_MODEL_PATH}, trying GCS...")
        download_model_from_gcs(GCS_MODEL_PATH, LOCAL_MODEL_PATH)

    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.error(f"Model not found at {LOCAL_MODEL_PATH}. API will start but predictions will fail.")
        yield
        return

    model = xgb.XGBClassifier()
    model.load_model(LOCAL_MODEL_PATH)
    logger.info(f"Model loaded successfully from {LOCAL_MODEL_PATH}")
    yield

    logger.info("Cleaning up API resources")
    if model is not None:
        del model


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Edit later
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "Project 99 API is running"}


@app.post("/predict", response_model=VertexResponse)
def predict(request: VertexRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    try:
        for input_data in request.instances:
            raw_point = input_data.model_dump()
            features = input_preprocessing(raw_point)

            prediction = model.predict(features)
            probability = model.predict_proba(features)

            results.append(PredictionResponse(prediction=int(prediction[0]), probability=float(probability[0][1])))

        return VertexResponse(predictions=results)
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=LOCAL_MODEL_PATH if model is not None else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = model.get_booster().feature_names
    return ModelInfoResponse(
        model_type="XGBoost",
        model_loaded=True,
        model_path=LOCAL_MODEL_PATH,
        feature_count=len(feature_names),
        feature_names=feature_names,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    required_columns = [
        "PointServer",
        "P1Score",
        "P2Score",
        "P1GamesWon",
        "P2GamesWon",
        "P1PointsWon",
        "P2PointsWon",
        "P1SetsWon",
        "P2SetsWon",
        "SetNo",
        "GameNo",
        "PointNumber",
        "ServeIndicator",
        "P1Momentum",
        "P2Momentum",
    ]

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_cols)}")

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
        df_with_predictions["Prediction"] = prediction_values
        df_with_predictions["Probability"] = probability_values

        csv_output = StringIO()
        df_with_predictions.to_csv(csv_output, index=False)
        csv_string = csv_output.getvalue()

        return BatchPredictionResponse(total_predictions=len(prediction_values), csv_with_predictions=csv_string)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
