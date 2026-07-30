# smart-ticketing-experiment

Experiment for loading an Excel dataset and a folder-of-JSONs "database", then computing a set of metrics on top.

## Layout

```
.
├── data/                       # Excel input data (place your .xlsx here)
├── database/                   # Folder of JSON files acting as a lookup DB
├── src/smart_ticketing/        # Package
│   ├── config.py               # Default paths
│   ├── data_loader.py          # Excel loader
│   ├── db.py                   # JSON database loader
│   ├── metrics.py              # All metric functions
│   ├── cli.py                  # CLI entry point
│   └── __main__.py             # `python -m smart_ticketing`
└── tests/                      # Pytest suite
```

## Setup

```bash
uv sync
```

## Usage

```bash
# Compute all metrics using default data/ and database/ folders
uv run smart-ticketing compute

# Or via the module
uv run python -m smart_ticketing compute

# Custom paths
uv run smart-ticketing compute --data path/to/data.xlsx --db path/to/json_dir
```

## Tests

```bash
uv run pytest
```
