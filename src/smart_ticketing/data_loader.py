from pathlib import Path

import pandas as pd

from .config import DATA_PATH


def load_data(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_excel(path)
