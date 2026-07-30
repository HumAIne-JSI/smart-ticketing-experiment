from smart_ticketing.metrics import (
    ALL_METRICS,
    compute_all,
    database_collections,
    records_per_collection,
    total_records,
)


def test_total_records(sample_data, sample_db):
    assert total_records(sample_data, sample_db) == 4


def test_database_collections(sample_data, sample_db):
    assert database_collections(sample_data, sample_db) == ["events", "users"]


def test_records_per_collection(sample_data, sample_db):
    assert records_per_collection(sample_data, sample_db) == {"events": 2, "users": 3}


def test_compute_all_runs_every_metric(sample_data, sample_db):
    results = compute_all(sample_data, sample_db)
    assert set(results.keys()) == {fn.__name__ for fn in ALL_METRICS}


def test_compute_all_with_subset(sample_data, sample_db):
    results = compute_all(sample_data, sample_db, metrics=[total_records])
    assert results == {"total_records": 4}
