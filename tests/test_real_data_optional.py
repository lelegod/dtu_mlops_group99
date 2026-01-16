import os
from pathlib import Path
import pytest

from tests import _PATH_DATA
from project99.data import TennisDataProcessor


RAW_DIR = Path(_PATH_DATA) / "raw"

@pytest.mark.skipif(not RAW_DIR.exists(), reason="Data files not found")
def test_preprocessor_finds_match_files():
    p = TennisDataProcessor(RAW_DIR)
    assert len(p.match_files) > 0, "No *-matches.csv files found in data/raw."

