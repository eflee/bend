#!/usr/bin/env python3
import sys, re, math, pandas as pd
from collections import defaultdict

def _gsheets_csv(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([^/]+)/", url)
    if not m:
        return url
    sheet_id = m.group(1)
    gid = "0"
    g = re.search(r"(?:[#?&]gid=)(\d+)", url)
    if g:
        gid = g.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def load_csv(path_or_url: str) -> pd.DataFrame:
    u = _gsheets_csv(path_or_url)
    return pd.read_csv(u)

def rows(df: pd.DataFrame):
    safe_cols = [re.sub(r'[^0-9a-zA-Z_]', '_', c) for c in df.columns]
    df2 = df.copy()
    df2.columns = safe_cols
    return df2.itertuples(index=False, name="Row")

class Q:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter(self, fn):
        mask = []
        for r in rows(self.df):
            try:
                mask.append(bool(fn(r)))
            except Exception:
                mask.append(False)
        return Q(self.df[mask])

    def select(self, fn):
        recs = []
        for r in rows(self.df):
            out = fn(r)
            if isinstance(out, dict):
                recs.append(out)
            elif isinstance(out, tuple):
                recs.append({f"c{i}": v for i, v in enumerate(out)})
            else:
                recs.append({"value": out})
        return Q(pd.DataFrame.from_records(recs))

    def mutate(self, **newcols):
        add = {}
        for k, f in newcols.items():
            add[k] = [f(r) for r in rows(self.df)]
        return Q(pd.concat([self.df.reset_index(drop=True), pd.DataFrame(add)], axis=1))

    def groupby(self, keyfn, **aggs):
        buckets = defaultdict(list)
        for r in rows(self.df):
            buckets[keyfn(r)].append(r)
        out = []
        for k, grp in buckets.items():
            rec = {"key": k}
            for name, fn in aggs.items():
                rec[name] = fn(grp)
            out.append(rec)
        return Q(pd.DataFrame(out))

    def head(self, n=5):
        return Q(self.df.head(n))

    def sort(self, *cols, ascending=False):
        return Q(self.df.sort_values(list(cols) or self.df.columns.tolist(), ascending=ascending))

    def show(self, n=20):
        print(self.df.head(n).to_string(index=False))
        return self

    def to_df(self):
        return self.df

    def dump(self, filename):
        """Write DataFrame to a new CSV file."""
        self.df.to_csv(filename, index=False)
        print(f"Wrote {len(self.df)} rows to {filename}")
        return self

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: csvrepl.py <csv_or_google_sheets_url>")
        sys.exit(2)
    src = sys.argv[1]
    df = load_csv(src)
    q = Q(df)
    r = list(rows(df))

    banner = (
        "Loaded as:\n"
        "  df  -> pandas.DataFrame\n"
        "  q   -> Q(df)  # filter/mutate/select/groupby without SQL\n"
        "  r   -> list of Row namedtuples (dot access)\n"
        "Helpers:\n"
        "  rows(df), load_csv(url)\n"
        "\n"
        "Examples:\n"
        "  q.mutate(total=lambda x: float(x.price)*int(x.qty)).show()\n"
        "  q.filter(lambda x: x.region=='CA').show()\n"
        "  q.groupby(lambda x: x.seller, gross=lambda g: sum(t.total for t in g)).show()\n"
        "  q.dump('out.csv')\n"
    )

    try:
        from IPython import start_ipython
        import builtins
        ns = dict(df=df, q=q, r=r, rows=rows, load_csv=load_csv, math=math, pd=pd, Q=Q)
        builtins.__dict__.update(ns)
        start_ipython(argv=[], user_ns=ns, display_banner=True)
    except Exception:
        import code
        code.interact(banner=banner, local=dict(df=df, q=q, r=r, rows=rows, load_csv=load_csv, math=math, pd=pd, Q=Q))
