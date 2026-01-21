from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from project99.data import PROJECT_ROOT
from project99.dataset import TennisDataset


def dataset_statistics(datadir: str = "data/processed") -> None:
    data_path = PROJECT_ROOT / datadir

    train_ds = TennisDataset(data_path / "train_set.csv")
    test_ds = TennisDataset(data_path / "test_set.csv")

    print(f"Train dataset: {train_ds.name}")
    print(f"Number of training samples: {len(train_ds)}")
    print(f"Feature dimension: {train_ds[0][0].shape}")
    print()

    print(f"Test dataset: {test_ds.name}")
    print(f"Number of test samples: {len(test_ds)}")
    print(f"Feature dimension: {test_ds[0][0].shape}")
    print()

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Class distribution
    train_labels = torch.stack([y for _, y in train_ds])
    test_labels = torch.stack([y for _, y in test_ds])

    plt.bar([0, 1], torch.bincount(train_labels, minlength=2))
    plt.title("Train label distribution")
    plt.xlabel("ServerWon")
    plt.ylabel("Count")
    plt.savefig(reports_dir / "train_label_distribution.png")
    plt.close()

    plt.bar([0, 1], torch.bincount(test_labels, minlength=2))
    plt.title("Test label distribution")
    plt.xlabel("ServerWon")
    plt.ylabel("Count")
    plt.savefig(reports_dir / "test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
