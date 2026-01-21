import os
from pathlib import Path

import typer
import xgboost as xgb
from loguru import logger
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from project99.data import tennis_data
from project99.logging_utils import setup_logging

setup_logging(log_file="reports/eval.log")

def evaluate(model_checkpoint: str = None) -> None:
    logger.info("Started evaluation...")
    if model_checkpoint is None:
        storage_path = os.getenv("AIP_MODEL_DIR")
        if storage_path:
            model_checkpoint = os.path.join(storage_path, "model.json")
        else:
            model_checkpoint = "models/xgboost_model.json"
    logger.info(f"Loading model from: {model_checkpoint}")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_checkpoint)
    (_, _), (X_test, y_test) = tennis_data(data_type='numpy')
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = xgb_model.predict(X_test)
    logger.info("Model performance on test set:")
    logger.info(f"Log Loss:    {log_loss(y_test, y_prob):.4f}")
    logger.info(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")
    logger.info(f"AUC Score:   {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    typer.run(evaluate)
