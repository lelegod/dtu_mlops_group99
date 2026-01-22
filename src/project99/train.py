import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

import wandb
from project99.constants import GCS_MODEL_PATH, LOCAL_MODEL_PATH
from project99.data import tennis_data
from project99.logging_utils import setup_logging
from project99.model import model

setup_logging(log_file="reports/app.log")
load_dotenv()

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def upload_to_gcs(local_path: str, gcs_path: str):
    try:
        from google.cloud import storage

        path_parts = gcs_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(local_path)
        logger.info(f"Model uploaded to {gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload model to GCS: {e}")
        raise e


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def train(cfg: DictConfig):
    logger.info("Started training")

    wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "disabled"
    if wandb_mode == "disabled":
        logger.warning("WANDB_API_KEY not found. Running WandB in disabled mode.")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "project99"),
        mode=wandb_mode,
        config={
            "data": {
                "test_size": float(cfg.data.test_size),
                "random_state": int(cfg.data.random_state),
            },
            "model": {
                "max_depth": int(cfg.model.params.max_depth),
                "learning_rate": float(cfg.model.params.learning_rate),
                "n_estimators": int(cfg.model.params.n_estimators),
                "objective": str(cfg.model.params.objective),
                "eval_metric": str(cfg.model.params.eval_metric),
            },
        },
    )

    try:
        (X_train, y_train), (X_test, y_test) = tennis_data(data_type="numpy")
        logger.info(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    except Exception as e:
        logger.error(f"Error loading tennis data: {e}")
        raise
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )
    logger.info("Training XGBoost model")
    xgb_model = model(cfg)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    y_prob = xgb_model.predict_proba(X_val)[:, 1]
    y_pred = xgb_model.predict(X_val)
    logger.info("Model performance on validation set")
    logger.info(f"Log loss: {log_loss(y_val, y_prob):.4f}")
    logger.info(f"Brier score: {brier_score_loss(y_val, y_prob):.4f}")
    logger.info(f"AUC score: {roc_auc_score(y_val, y_prob):.4f}")
    logger.info(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")

    wandb.log(
        {
            "val/log_loss": float(log_loss(y_val, y_prob)),
            "val/brier_score": float(brier_score_loss(y_val, y_prob)),
            "val/auc": float(roc_auc_score(y_val, y_prob)),
            "val/accuracy": float(accuracy_score(y_val, y_pred)),
        }
    )

    Path(LOCAL_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(LOCAL_MODEL_PATH)
    upload_to_gcs(LOCAL_MODEL_PATH, GCS_MODEL_PATH)

    artifact = wandb.Artifact("xgboost_model", type="model")
    artifact.add_file(LOCAL_MODEL_PATH)
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    train()
