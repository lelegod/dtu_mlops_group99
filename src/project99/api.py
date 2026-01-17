import os
from fastapi import FastAPI, HTTPException
import xgboost as xgb
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Global variable to hold the model
model = None

class PredictionInput(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_model():
    global model
    # Check GCP path first, then local fallback
    model_path = os.getenv("AIP_MODEL_DIR", "models/xgboost_model.json")
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Prediction endpoint will fail.")
        return
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

@app.get("/")
def read_root():
    return {"status": "Project 99 API is running"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(data)
        probability = model.predict_proba(data)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))