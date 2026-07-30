from typing import Any, Callable

import pandas as pd


MetricFn = Callable[[pd.DataFrame, dict[str, Any]], Any]


def total_records(data: pd.DataFrame, db: dict[str, Any]) -> int:
    return len(data)


def database_collections(data: pd.DataFrame, db: dict[str, Any]) -> list[str]:
    return sorted(db.keys())


def records_per_collection(data: pd.DataFrame, db: dict[str, Any]) -> dict[str, int]:
    return {name: len(value) for name, value in db.items()}


ALL_METRICS: list[MetricFn] = [
    total_records,
    database_collections,
    records_per_collection,
]


def compute_all(
    data: pd.DataFrame,
    db: dict[str, Any],
    metrics: list[MetricFn] | None = None,
) -> dict[str, Any]:
    selected = metrics if metrics is not None else ALL_METRICS
    return {fn.__name__: fn(data, db) for fn in selected}
