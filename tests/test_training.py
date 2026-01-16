import numpy as np
from hydra import initialize, compose
from project99.model import model


def test_training_like_flow_saves_model(tmp_path):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    # tiny dataset
    X = np.random.randn(50, 5)
    y = (np.random.rand(50) > 0.5).astype(int)

    m = model(cfg)
    m.fit(X, y)

    out = tmp_path / "xgboost_model.json"
    m.save_model(out)
    assert out.exists(), "Expected model artifact to be saved."
