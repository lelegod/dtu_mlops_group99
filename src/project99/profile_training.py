import cProfile
import io
import pstats
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split  # type: ignore

from project99.data import TennisDataProcessor, tennis_data
from project99.model import model as create_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILING_DIR = PROJECT_ROOT / "docs" / "profiling"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def timing_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        return result, elapsed

    return wrapper


@timing_decorator
def profile_data_loading() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (X_train, y_train), (X_test, y_test) = tennis_data(data_type="numpy")  # type: ignore
    return X_train, y_train, X_test, y_test


@timing_decorator
def profile_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@timing_decorator
def profile_model_creation(cfg) -> Any:
    return create_model(cfg)


@timing_decorator
def profile_model_training(
    model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Any:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


@timing_decorator
def profile_prediction(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return y_prob, y_pred


def run_cprofile(func: Callable, *args, **kwargs) -> tuple[Any, pstats.Stats]:
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    stats = pstats.Stats(profiler)
    return result, stats


def save_profile_stats(stats: pstats.Stats, filename: str) -> None:
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    stream = io.StringIO()
    stats.stream = stream  # type: ignore
    stats.sort_stats("cumulative")
    stats.print_stats(50)

    output_path = PROFILING_DIR / filename
    with open(output_path, "w") as f:
        f.write(stream.getvalue())


def create_summary_report(timings: dict, memory_info: dict | None = None) -> str:
    total_time = sum(timings.values())

    report = []

    for step, elapsed in timings.items():
        percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
        bar = "█" * int(percentage / 2)
        report.append(f"{step:25s}: {elapsed:8.3f}s ({percentage:5.1f}%) {bar}")

    report.append("-" * 40)
    report.append(f"{'TOTAL':25s}: {total_time:8.3f}s")
    report.append("")

    return "\n".join(report)


def profile_full_pipeline():
    print("RUNNING TRAINING PIPELINE PROFILER\n")

    timings = {}

    config_path = CONFIGS_DIR / "config.yaml"
    cfg = OmegaConf.load(config_path)

    print("Profiling data loading...")
    (data, load_time) = profile_data_loading()
    X_train, y_train, X_test, y_test = data
    timings["Data Loading"] = load_time
    print(f"    - Data shape: X_train={X_train.shape}, y_train={y_train.shape}")

    print("Profiling train/val split...")
    (split_data, split_time) = profile_train_test_split(
        X_train, y_train, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )
    X_train, X_val, y_train, y_val = split_data
    timings["Train/Val Split"] = split_time

    print("Profiling model creation...")
    (xgb_model, create_time) = profile_model_creation(cfg)
    timings["Model Creation"] = create_time

    print("Profiling model training (this may take a moment)...")

    def train_model():
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return xgb_model

    start = time.perf_counter()
    trained_model, training_stats = run_cprofile(train_model)
    train_time = time.perf_counter() - start
    timings["Model Training"] = train_time

    save_profile_stats(training_stats, "training_profile_stats.txt")

    print("Profiling prediction...")
    ((y_prob, y_pred), pred_time) = profile_prediction(trained_model, X_val)
    timings["Prediction"] = pred_time

    summary = create_summary_report(timings)
    print("\n" + summary)

    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PROFILING_DIR / "profile_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    return timings


if __name__ == "__main__":
    import sys

    processed_data = PROJECT_ROOT / "data" / "processed"
    if not (processed_data / "train_set.csv").exists():
        print("Processed data not found!")
        sys.exit(1)

    timings = profile_full_pipeline()

    print("Profiling complete!")
