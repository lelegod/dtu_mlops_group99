import os
import datetime
import json
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from project99.constants import GCS_MODEL_PATH, LOCAL_MODEL_PATH
from project99.logging_utils import setup_logging
from project99.preprocess import input_preprocessing
from project99.type import PredictionResponse, RawPointInput

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "http_status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time spent processing prediction")
MONITORING_BUCKET = "dtumlopsgroup99-monitoring"

model: xgb.XGBClassifier | None = None
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
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def log_prediction_data(data_dict: dict, prediction: int, probability: float):
    try:
        client = storage.Client()
        bucket = client.bucket(MONITORING_BUCKET)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        blob = bucket.blob(f"logs/prediction_{timestamp}.json")
        
        log_entry = {**data_dict, "prediction": prediction, "probability": probability}
        blob.upload_from_string(json.dumps(log_entry), content_type="application/json")
    except Exception as e:
        logger.error(f"Failed to log monitoring data: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model_from_gcs(GCS_MODEL_PATH, LOCAL_MODEL_PATH)

    if os.path.exists(LOCAL_MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(LOCAL_MODEL_PATH)
        logger.info("Model loaded successfully.")
    yield
    if model is not None:
        del model

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"status": "Project 99 API is running"}

@app.post("/predict")
def predict(input_data: RawPointInput):
    if model is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with PREDICTION_LATENCY.time():
        try:
            # Preprocess and Predict
            raw_point = input_data.model_dump()
            features = input_preprocessing(raw_point)
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0].max()

            # Monitoring
            log_prediction_data(raw_point, int(prediction), float(probability))
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status="200").inc()
            
            return PredictionResponse(
                prediction=int(prediction), 
                probability=float(probability)
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status="400").inc()
            raise HTTPException(status_code=400, detail=str(e))