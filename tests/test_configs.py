from hydra import initialize, compose


def test_config_composes_and_has_expected_fields():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    assert "data" in cfg
    assert "model" in cfg

    # data section
    assert cfg.data.test_size == 0.2
    assert cfg.data.random_state == 42

    # model section
    assert cfg.model.name == "xgboost"
    assert "params" in cfg.model
    assert cfg.model.params.objective == "binary:logistic"
    assert cfg.model.params.eval_metric == "logloss"
