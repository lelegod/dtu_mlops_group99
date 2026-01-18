import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from project99.data import tennis_data
from project99.model import model

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def train(cfg: DictConfig):
    print("Started training...")
    (X_train, y_train), (X_test, y_test) = tennis_data(data_type='numpy')
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state
    )
    print("Training XGBoost Model...")
    xgb_model = model(cfg)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    y_prob = xgb_model.predict_proba(X_val)[:, 1]
    y_pred = xgb_model.predict(X_val)
    print("\n--- Model Performance ---")
    print(f"Log Loss:    {log_loss(y_val, y_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_val, y_prob):.4f}")
    print(f"AUC Score:   {roc_auc_score(y_val, y_prob):.4f}")
    print(f"Accuracy:    {accuracy_score(y_val, y_pred):.4f}")
    storage_path = os.getenv("AIP_MODEL_DIR")
    if storage_path:
        print(f"Cloud environment detected. Target: {storage_path}")
        model_output_path = os.path.join(storage_path, "model.json")
    else:
        print("Local environment detected.")
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_output_path = str(models_dir / "xgboost_model.json")
    xgb_model.save_model(model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train()
