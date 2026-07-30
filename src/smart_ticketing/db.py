import json
from pathlib import Path
from typing import Any

from .config import DATABASE_DIR


def load_database(path: Path | None = None) -> dict[str, Any]:
    db_path = Path(path) if path is not None else DATABASE_DIR
    if not db_path.exists():
        raise FileNotFoundError(f"Database directory not found: {db_path}")

    database: dict[str, Any] = {}
    for file in sorted(db_path.glob("*.json")):
        with file.open("r", encoding="utf-8") as f:
            database[file.stem] = json.load(f)
    return database
