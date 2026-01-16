from pathlib import Path

import pandas as pd
import pytest

from project99.data import TennisDataProcessor


def _write_minimal_tournament(tmp_path: Path, name: str = "test_tournament") -> Path:
    """Create minimal matches/points CSVs that satisfy TennisDataProcessor expectations."""
    matches = pd.DataFrame(
        {
            "match_id": [1],
            "player1": ["PlayerA"],
            "player2": ["PlayerB"],
        }
    )
    # Minimal set of columns required by processing pipeline
    points = pd.DataFrame(
        {
            "match_id": [1, 1],
            "PointServer": [1, 2],
            "PointWinner": [1, 2],
            "ServeIndicator": [1, 2],  # 1st serve, 2nd serve
            "SetNo": [1, 1],
            "GameNo": [1, 1],
            "PointNumber": [1, 2],
            "P1GamesWon": [0, 0],
            "P2GamesWon": [0, 0],
            "P1Score": ["0", "15"],
            "P2Score": ["0", "0"],
            "P1PointsWon": [0, 1],
            "P2PointsWon": [0, 0],
            "P1Momentum": [0, 0],
            "P2Momentum": [0, 0],
            "SetWinner": [0, 0],  # no set winner yet
        }
    )

    matches_path = tmp_path / f"{name}-matches.csv"
    points_path = tmp_path / f"{name}-points.csv"
    matches.to_csv(matches_path, index=False)
    points.to_csv(points_path, index=False)
    return matches_path


def test_process_single_tournament_outputs_expected_columns(tmp_path: Path):
    match_file = _write_minimal_tournament(tmp_path)

    processor = TennisDataProcessor(tmp_path)
    df, tournament_name = processor._process_single_tournament(match_file)

    assert tournament_name == "test_tournament"
    assert "ServerWon" in df.columns, "Expected label column 'ServerWon' missing after preprocessing."
    assert df["ServerWon"].dropna().isin([0, 1]).all(), "ServerWon must be binary (0/1)."

    # Schema check: these are the columns your code promises
    expected_cols = [
        "SetNo", "GameNo", "PointNumber",
        "ServerGamesWon", "ReceiverGamesWon",
        "ServerScore", "ReceiverScore",
        "ServerPointsWon", "ReceiverPointsWon",
        "ServerMomentum", "ReceiverMomentum",
        "ServerSetsWon", "ReceiverSetsWon",
        "GameDiff", "SetDiff", "PointDiff", "MomentumDiff",
        "IsBreakPoint", "SetPressure", "IsSecondServe", "ServerWon",
    ]

    assert list(df.columns) == expected_cols, (
        "Dataframe schema mismatch.\n"
        f"Expected columns: {expected_cols}\n"
        f"Got columns: {list(df.columns)}"
    )
