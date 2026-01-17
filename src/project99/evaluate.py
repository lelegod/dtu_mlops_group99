import os
from pathlib import Path
import typer
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from project99.data import tennis_data

def evaluate(model_checkpoint: str = None) -> None:
    print("Started evaluation...")
    if model_checkpoint is None:
        storage_path = os.getenv("AIP_MODEL_DIR")
        if storage_path:
            model_checkpoint = os.path.join(storage_path, "model.json")
        else:
            model_checkpoint = "models/xgboost_model.json"
    print(f"Loading model from: {model_checkpoint}")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_checkpoint)
    (_, _), (X_test, y_test) = tennis_data(data_type='numpy')
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = xgb_model.predict(X_test)
    print("\n--- Model Performance ---")
    print(f"Log Loss:    {log_loss(y_test, y_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")
    print(f"AUC Score:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    typer.run(evaluate)