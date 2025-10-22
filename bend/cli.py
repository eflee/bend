#!/usr/bin/env python3
"""Command-line interface for Bend."""

import argparse
import math

import pandas as pd

from . import Q, dump_csv, load_csv


def main():
    """Launch an interactive REPL session with optional CSV data loading."""
    parser = argparse.ArgumentParser(
        description="Bend: A precise CSV analysis tool with an intuitive query interface"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",  # Make it optional
        help="Path to CSV file or Google Sheets URL (optional)",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N rows before loading CSV (default: 0)",
    )

    args = parser.parse_args()

    # Load CSV if provided
    if args.csv_file:
        q = load_csv(args.csv_file, skip_rows=args.skip_rows)

        banner = (
            f"Bend REPL - Loaded {q.rows} rows × {len(q.columns)} columns as 'q'\n"
            "\n"
            "Available functions:\n"
            "  load_csv(path)     : Load a CSV file into a Q\n"
            "  dump_csv(q, path)  : Save a Q to a CSV file\n"
            "\n"
            "Examples:\n"
            "  q.columns, q.rows                           # discover shape\n"
            "  q.filter(lambda x: x.price > 100)           # filter rows\n"
            "  q.assign(total=lambda x: x.price * x.qty)   # add computed columns\n"
            "  for row in q: print(row.name, row.age)      # iterate rows\n"
            "  q2 = load_csv('other.csv')                  # load another file\n"
            "  q.filter(q2, on='id')                       # semi-join\n"
            "  dump_csv(q.filter(...), 'output.csv')       # export result\n"
            "\n"
            "Type 'q' to see your data!\n"
        )
    else:
        # No file loaded, just provide helpers
        q = None

        banner = (
            "Bend REPL - No data loaded\n"
            "\n"
            "Load data:\n"
            "  q = load_csv('data.csv')  # or URL\n"
            "\n"
            "With options:\n"
            "  q = load_csv('data.csv', dtype={'age': int, 'price': float})\n"
            "  q = load_csv('data.csv', skip_rows=3)\n"
            "\n"
            "Export data:\n"
            "  dump_csv(q, 'output.csv')\n"
            "\n"
            "Examples:\n"
            "  q.filter(lambda x: x.region == 'CA')        # filter rows\n"
            "  q.assign(total=lambda x: x.price * x.qty)   # add columns\n"
            "  q2 = load_csv('other.csv')                  # load another file\n"
            "  q.merge(q2, on='id', how='left')            # join datasets\n"
        )

    # Helper functions for the REPL
    def r():
        """Reload data from source file and re-apply all tracked changes.

        Shortcut for: q = q.reload()
        """
        nonlocal q
        if q is None:
            raise ValueError("No Q object loaded. Load a CSV first with: q = load_csv('file.csv')")
        q = q.reload()
        print(f"Reloaded {q.rows} rows × {len(q.columns)} columns")
        return q

    try:
        import builtins

        from IPython import start_ipython

        ns = {"q": q, "load_csv": load_csv, "dump_csv": dump_csv, "r": r, "Q": Q, "math": math, "pd": pd}
        builtins.__dict__.update(ns)
        start_ipython(argv=[], user_ns=ns, display_banner=True)
    except Exception:
        import code

        ns = {"q": q, "load_csv": load_csv, "dump_csv": dump_csv, "r": r, "Q": Q, "math": math, "pd": pd}
        code.interact(banner=banner, local=ns)


if __name__ == "__main__":
    main()
