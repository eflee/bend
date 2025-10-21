#!/usr/bin/env python3
"""Command-line interface for Bend."""

import sys
import argparse
import math
import pandas as pd

from .core import Q, load_csv, rows


def main():
    """Launch an interactive REPL session with optional CSV data loading."""
    parser = argparse.ArgumentParser(
        description="Bend: A precise CSV analysis tool with an intuitive query interface"
    )
    parser.add_argument(
        "csv_file",
        nargs='?',  # Make it optional
        help="Path to CSV file or Google Sheets URL (optional)"
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N rows before loading CSV (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Load CSV if provided
    if args.csv_file:
        df = load_csv(args.csv_file, skip_rows=args.skip_rows)
        q = Q(df, source_path=args.csv_file, skip_rows=args.skip_rows)
        r = list(rows(df))
        
        banner = (
            "Loaded as:\n"
            "  q   -> Q object with tracked change history\n"
            "  df  -> pandas.DataFrame (for pandas compatibility)\n"
            "Helpers:\n"
            "  rows(df), load_csv(url), reload(), replay()\n"
            "\n"
            "Examples:\n"
            "  q.columns, q.rows  # discover shape\n"
            "  q.assign(total=lambda x: x.price * x.qty)  # add computed columns\n"
            "  q.filter(lambda x: x.region == 'CA')  # filter rows\n"
            "  q.to_df()  # get DataFrame copy for pandas operations\n"
            "  for row in q: print(row.name)  # iterate over rows\n"
            "  q.dump('out.csv')  # write to CSV\n"
        )
    else:
        # No file loaded, just provide helpers
        df = None
        q = None
        r = None
        
        banner = (
            "Bend REPL - No data loaded\n"
            "\n"
            "Load data:\n"
            "  df = load_csv('data.csv')  # or URL\n"
            "  q = Q(df, source_path='data.csv')\n"
            "\n"
            "Or with type conversion:\n"
            "  df = load_csv('data.csv', dtype={'age': int, 'price': float})\n"
            "  q = Q(df)\n"
            "\n"
            "Examples:\n"
            "  q.columns, q.rows  # discover shape\n"
            "  q.assign(total=lambda x: x.price * x.qty)  # add computed columns\n"
            "  q.filter(lambda x: x.region == 'CA')  # filter rows\n"
        )
    
    # Helper functions for the REPL
    def reload():
        """Reload data from source file and re-apply all tracked changes.
        
        Returns the reloaded Q object. You can reassign to q:
            q = reload()
        """
        if q is None:
            raise ValueError("No Q object loaded. Load a CSV first.")
        return q.reload()
    
    def replay():
        """Re-apply all tracked changes to the in-memory base data.
        
        Returns the replayed Q object. You can reassign to q:
            q = replay()
        """
        if q is None:
            raise ValueError("No Q object loaded. Load a CSV first.")
        return q.replay()

    try:
        from IPython import start_ipython
        import builtins
        ns = dict(df=df, q=q, r=r, rows=rows, load_csv=load_csv, reload=reload, replay=replay, math=math, pd=pd, Q=Q)
        builtins.__dict__.update(ns)
        start_ipython(argv=[], user_ns=ns, display_banner=True)
    except Exception:
        import code
        ns = dict(df=df, q=q, r=r, rows=rows, load_csv=load_csv, reload=reload, replay=replay, math=math, pd=pd, Q=Q)
        code.interact(banner=banner, local=ns)


if __name__ == "__main__":
    main()

