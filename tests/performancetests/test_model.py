import os
import time
from pathlib import Path

import pytest
import xgboost as xgb

import wandb
from project99.data import tennis_data


def _download_wandb_artifact(artifact_path: str) -> Path:
    """
    artifact_path typically looks like:
      entity/project/artifact_name:version
    """
    # wandb.Api reads WANDB_API_KEY from env; no entity env required
    api = wandb.Api()
    artifact = api.artifact(artifact_path)

    artifact_dir = Path("artifacts") / artifact.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    downloaded_dir = Path(artifact.download(root=str(artifact_dir)))

    # Find a model file in the artifact directory
    # Your training saves via xgb_model.save_model(LOCAL_MODEL_PATH)
    candidates = (
        list(downloaded_dir.rglob("*.json"))
        + list(downloaded_dir.rglob("*.ubj"))
        + list(downloaded_dir.rglob("*.model"))
    )
    if not candidates:
        # fallback: any single file in artifact
        all_files = [p for p in downloaded_dir.rglob("*") if p.is_file()]
        if len(all_files) == 1:
            return all_files[0]
        raise RuntimeError(f"No model file found in artifact directory: {downloaded_dir}")

    # Prefer smallest set: pick the first candidate
    return candidates[0]


def _load_xgb_model(model_file: Path) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_file))
    return clf


def test_model_speed():
    model_name = os.getenv("MODEL_NAME")

    if not model_name:
        pytest.skip("MODEL_NAME not set (not running in CML context)")

    model_file = _download_wandb_artifact(model_name)
    clf = _load_xgb_model(model_file)

    (X_train, _), (X_test, _) = tennis_data(data_type="numpy")

    # Make this stable: fixed batch size
    batch = X_test[:1024] if len(X_test) >= 1024 else X_test

    start = time.time()
    for _ in range(100):
        _ = clf.predict_proba(batch)
    elapsed = time.time() - start

    # Choose a threshold that makes sense on GitHub runners.
    # Start conservative; tighten once you see typical timings.
    assert elapsed < 5.0, f"Inference too slow: {elapsed:.3f}s for 100x predict_proba on batch size {len(batch)}"
