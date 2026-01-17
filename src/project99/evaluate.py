from pathlib import Path

import typer
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from project99.data import tennis_data
from project99.model import model


def evaluate(model_checkpoint: str) -> None:
    """Evaluate the trained XGBoost model"""
    print("Started evaluation...")

    if not Path(model_checkpoint).exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint}")

    model = xgb.XGBClassifier()
    model.load_model(model_checkpoint)

    (X_train, y_train), (X_test, y_test) = tennis_data(data_type='numpy')

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n--- Model Performance ---")
    print(f"Log Loss:    {log_loss(y_test, y_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")
    print(f"AUC Score:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    typer.run(evaluate)
