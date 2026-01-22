import xgboost as xgb
from omegaconf import DictConfig


def model(cfg: DictConfig) -> xgb.XGBClassifier:
    """Create XGBoost classifier from config.
    Args:
        cfg: Hydra config containing model parameters
    """
    model = xgb.XGBClassifier(
        max_depth=cfg.model.params.max_depth,
        learning_rate=cfg.model.params.learning_rate,
        n_estimators=cfg.model.params.n_estimators,
        objective=cfg.model.params.objective,
        eval_metric=cfg.model.params.eval_metric,
    )
    return model


if __name__ == "__main__":
    from omegaconf import OmegaConf

    # Test model creation with dummy config
    cfg = OmegaConf.create(
        {
            "model": {
                "params": {
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                }
            }
        }
    )

    model = model(cfg)
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {model.get_params()}")
