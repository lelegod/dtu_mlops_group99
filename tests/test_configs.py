from hydra import compose, initialize


def test_config_composes_and_has_expected_fields():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    assert "data" in cfg, "Config missing required top-level section 'data'."
    assert "model" in cfg, "Config missing required top-level section 'model'."

    # data section
    assert cfg.data.test_size == 0.2, f"Expected cfg.data.test_size == 0.2, got {cfg.data.test_size}"
    assert cfg.data.random_state == 42, f"Expected cfg.data.random_state == 42, got {cfg.data.random_state}"

    # model section
    assert cfg.model.name == "xgboost", f"Expected model name 'xgboost', got '{cfg.model.name}'"
    assert "params" in cfg.model, "Model config missing 'params' section."
    assert cfg.model.params.objective == "binary:logistic", (
        f"Expected objective 'binary:logistic', got {cfg.model.params.objective}"
    )
    assert cfg.model.params.eval_metric == "logloss", (
        f"Expected eval_metric 'logloss', got {cfg.model.params.eval_metric}"
    )
