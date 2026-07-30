from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_excel(tmp_path: Path) -> Path:
    path = tmp_path / "data.xlsx"
    df = pd.DataFrame(
        {
            "ticket_id": [1, 2, 3, 4],
            "price": [10.0, 12.5, 8.0, 15.0],
            "validated": [True, False, True, True],
        }
    )
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def sample_db_dir(tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "events.json").write_text(
        '[{"id": 1, "name": "Concert"}, {"id": 2, "name": "Match"}]', encoding="utf-8"
    )
    (db_dir / "users.json").write_text(
        '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Carol"}]',
        encoding="utf-8",
    )
    return db_dir


@pytest.fixture
def sample_data(sample_excel: Path) -> pd.DataFrame:
    return pd.read_excel(sample_excel)


@pytest.fixture
def sample_db(sample_db_dir: Path) -> dict:
    from smart_ticketing.db import load_database

    return load_database(sample_db_dir)
