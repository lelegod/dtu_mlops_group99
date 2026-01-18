from hydra import compose, initialize
from omegaconf import OmegaConf


def test_xgboost_params_convert_to_dict():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    assert isinstance(params, dict), (
        f"Expected model params to be dict, got {type(params)}"
    )
    assert params["objective"] == "binary:logistic", (
        f"Expected objective 'binary:logistic', got {params['objective']}"
    )
