import pytest

from smart_ticketing.db import load_database


def test_load_database_returns_collections(sample_db_dir):
    db = load_database(sample_db_dir)
    assert set(db.keys()) == {"events", "users"}
    assert len(db["events"]) == 2
    assert len(db["users"]) == 3


def test_load_database_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_database(tmp_path / "nope")


def test_load_database_empty_dir(tmp_path):
    empty = tmp_path / "empty_db"
    empty.mkdir()
    db = load_database(empty)
    assert db == {}
