from pathlib import Path

import pandas as pd
import pytest

from project99.data import TennisDataProcessor


def test_missing_points_file_raises(tmp_path: Path):
    # create only matches file
    matches = pd.DataFrame({"match_id": [1], "player1": ["A"], "player2": ["B"]})
    match_file = tmp_path / "tourn-matches.csv"
    matches.to_csv(match_file, index=False)

    p = TennisDataProcessor(tmp_path)
    with pytest.raises(FileNotFoundError, match="Points file not found"):
        p._load_tournament_data(match_file)
