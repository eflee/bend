#!/usr/bin/env python3
"""Command-line interface for Bend."""

import sys
import argparse
import math
import pandas as pd

from .core import Q, load_csv, rows


def main():
    """Launch an interactive REPL session with loaded CSV data."""
    parser = argparse.ArgumentParser(
        description="Bend: A precise CSV analysis tool with an intuitive query interface"
    )
    parser.add_argument(
        "csv_file",
        help="Path to CSV file or Google Sheets URL"
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N rows before loading CSV (default: 0)"
    )
    
    args = parser.parse_args()
    
    df = load_csv(args.csv_file, skip_rows=args.skip_rows)
    q = Q(df, source_path=args.csv_file, skip_rows=args.skip_rows)
    r = list(rows(df))
    
    # Helper functions for the REPL
    def reload():
        """Reload data from source file and re-apply all tracked changes.
        
        Returns the reloaded Q object. You can reassign to q:
            q = reload()
        """
        return q.reload()
    
    def refresh():
        """Re-apply all tracked changes to the in-memory base data.
        
        Returns the refreshed Q object. You can reassign to q:
            q = refresh()
        """
        return q.refresh()

    banner = (
        "Loaded as:\n"
        "  df  -> pandas.DataFrame (base data)\n"
        "  q   -> Q(df)  # tracked change history + current state\n"
        "  r   -> list of Row namedtuples (dot access)\n"
        "Helpers:\n"
        "  rows(df), load_csv(url), reload(), refresh()\n"
        "\n"
        "Examples:\n"
        "  q.extend(total=lambda x: x.price * x.qty)  # add computed columns\n"
        "  q.filter(lambda x: x.region == 'CA')  # filter rows\n"
        "  q.transform(lambda x: {'name': x.first + ' ' + x.last})  # reshape\n"
        "  q.hide('id', 'internal')  # hide columns from display only\n"
        "  q = reload()  # reload from source, re-apply all changes\n"
        "  q = refresh()  # re-apply changes to base (no file reload)\n"
        "  q = q.rebase()  # flatten: make current state the new base\n"
        "  q.sum('price'), q.mean('age')  # aggregations\n"
        "  for row in q: print(row.name)  # iterate over rows\n"
        "  q.dump('out.csv')  # write to CSV\n"
    )

    try:
        from IPython import start_ipython
        import builtins
        ns = dict(df=df, q=q, r=r, rows=rows, load_csv=load_csv, reload=reload, refresh=refresh, math=math, pd=pd, Q=Q)
        builtins.__dict__.update(ns)
        start_ipython(argv=[], user_ns=ns, display_banner=True)
    except Exception:
        import code
        code.interact(banner=banner, local=dict(df=df, q=q, r=r, rows=rows, load_csv=load_csv, reload=reload, refresh=refresh, math=math, pd=pd, Q=Q))


if __name__ == "__main__":
    main()

