# Tennis Point Predictor

Welcome to the documentation for **Project 99** — a machine learning application that predicts the outcome of individual tennis points.

## Overview

This project uses an XGBoost gradient boosting model to predict the probability that the **server wins** a given point in a professional tennis match. Instead of predicting entire match outcomes, the model focuses on point-by-point dynamics of the game.


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Train the model
python src/project99/train.py

# Start the API server
uvicorn project99.api:app --reload
```

## Project Structure

```
dtu_mlops_group99/
├── src/project99/          # Main source code
│   ├── api.py              # FastAPI backend
│   ├── frontend.py         # Streamlit frontend
│   ├── train.py            # Model training script
│   ├── data.py             # Data processing pipeline
│   ├── model.py            # Model definition
│   ├── preprocess.py       # Feature preprocessing
│   └── evaluate.py         # Model evaluation script
├── configs/                # Hydra configuration files
├── data/                   # Raw and processed data
├── dockerfiles/            # Docker configurations
├── tests/                  # Unit and integration tests
└── cloudbuild.yaml         # CI/CD pipeline definition
```

## Data Source

The project uses Jeff Sackmann's publicly available tennis point-by-point datasets from professional ATP and Grand Slam matches.

## Model

The XGBoost classifier is trained with:

- **500 estimators** with learning rate 0.05
- **Max depth 6** for balanced complexity
- **Log loss** objective for calibrated probabilities
- **Early stopping** to prevent overfitting

## Deployment

The application is deployed on Google Cloud Platform:

- **Model Training**: Vertex AI Custom Jobs
- **Model Serving**: Vertex AI Endpoints
- **Frontend**: Cloud Run (Streamlit)
- **CI/CD**: Cloud Build
