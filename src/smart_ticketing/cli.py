import argparse
import json
import sys
from pathlib import Path

from .data_loader import load_data
from .db import load_database
from .metrics import ALL_METRICS, compute_all


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-ticketing",
        description="Load data + JSON database and compute metrics.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the Excel data file (default: data/data.xlsx).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to the JSON database directory (default: database/).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compute = sub.add_parser("compute", help="Run all metrics and print results as JSON.")
    compute.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of metric function names to run (default: all).",
    )

    sub.add_parser("list-metrics", help="List available metric function names.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-metrics":
        print(json.dumps([fn.__name__ for fn in ALL_METRICS], indent=2))
        return 0

    if args.command == "compute":
        data = load_data(args.data)
        db = load_database(args.db)

        selected = ALL_METRICS
        if args.only:
            names = set(args.only)
            selected = [fn for fn in ALL_METRICS if fn.__name__ in names]
            missing = names - {fn.__name__ for fn in selected}
            if missing:
                parser.error(f"Unknown metric(s): {sorted(missing)}")

        results = compute_all(data, db, metrics=selected)
        print(json.dumps(results, indent=2, default=str))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
