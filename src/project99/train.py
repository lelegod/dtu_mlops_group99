from pathlib import Path

import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from project99.constants import GCS_MODEL_PATH, LOCAL_MODEL_PATH
from project99.data import tennis_data
from project99.model import model


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
        print(f"Model uploaded to {gcs_path}")
    except Exception as e:
        print(f"Warning: Failed to upload to GCS: {e}")


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
    
    Path(LOCAL_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(LOCAL_MODEL_PATH)
    upload_to_gcs(LOCAL_MODEL_PATH, GCS_MODEL_PATH)


if __name__ == "__main__":
    train()
