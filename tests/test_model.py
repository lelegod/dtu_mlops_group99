import pytest
import numpy as np
from hydra import initialize, compose
import xgboost as xgb

from project99.model import model


def test_model_constructs_from_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    m = model(cfg)
    assert isinstance(m, xgb.XGBClassifier)
    params = m.get_params()

    assert params["max_depth"] == cfg.model.params.max_depth
    assert params["learning_rate"] == cfg.model.params.learning_rate
    assert params["n_estimators"] == cfg.model.params.n_estimators

def test_model_import():
    import project99.model  # noqa: F401

@pytest.mark.parametrize("n_features", [5, 20])
def test_xgb_predict_proba_shape_parametrized(n_features):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    m = model(cfg)
    X = np.random.randn(30, n_features)
    y = (np.random.rand(30) > 0.5).astype(int)

    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (30, 2)
