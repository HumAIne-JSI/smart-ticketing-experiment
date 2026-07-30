import pandas as pd
import pytest

from smart_ticketing.data_loader import load_data


def test_load_data_returns_dataframe(sample_excel):
    df = load_data(sample_excel)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df.columns) == ["ticket_id", "price", "validated"]


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.xlsx")
