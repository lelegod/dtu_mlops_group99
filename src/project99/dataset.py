from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TennisDataset(Dataset):
    """Tennis dataset backed by processed CSV files."""

    name: str = "TennisDataset"

    def __init__(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)

        self.features: Tensor = torch.tensor(
            df.drop("ServerWon", axis=1).values,
            dtype=torch.float32,
        )
        self.targets: Tensor = torch.tensor(
            df["ServerWon"].values,
            dtype=torch.long,
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.targets[idx]

    def __len__(self) -> int:
        return self.features.shape[0]
