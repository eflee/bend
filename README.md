# Bend

**Bend is essentially pandas with training wheels + change tracking + CLI convenience.**

A CSV analysis tool that combines the power of pandas with an intuitive functional interface, immutable operations, and automatic change tracking—all in an interactive REPL.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Launch REPL with your data
bend data.csv

# Or start empty and load data interactively
bend
>>> df = load_csv('sales.csv')
>>> q = Q(df)
```

Your data is automatically loaded as `q` and you're ready to explore.

## Why Bend?

Bend fills the gap between simple command-line CSV tools and full Jupyter notebooks:

### Key Features

- **🐍 IPython REPL** - Full Python environment, not just command flags
- **🔗 Method Chaining** - Fluent, readable pipelines
- **🔒 Immutable Operations** - Safe experimentation, no accidental mutations
- **📝 Change Tracking** - Replay transformations when data updates
- **🎯 Pythonic API** - Lambda functions with dot-accessible columns
- **♻️ Reproducibility** - Track deterministic vs non-deterministic operations

### Comparison with Other Tools

| Tool | Approach | Best For |
|------|----------|----------|
| **Bend** | Interactive Python REPL | Exploratory analysis with Python |
| VisiData | Visual TUI spreadsheet | Visual browsing and editing |
| csvkit | Unix-style utilities | Shell scripting and SQL queries |
| Miller | Stream processing | One-pass transformations |
| xsv | Fast single commands | Quick operations on large files |
| q/textql | SQL on CSV | SQL-centric analysis |

**Bend is the sweet spot** for data scientists who want more than `csvkit` but less overhead than Jupyter.

## Core Concepts

### Immutability

All operations return a **new Q object**—originals are never modified:

```python
q1 = Q(df)
q2 = q1.filter(lambda x: x.age >= 18)
# q1 unchanged, q2 is new
```

### Change Tracking

Every operation is tracked and can be replayed:

```python
q = Q(load_csv('data.csv'), source_path='data.csv')
q2 = q.filter(lambda x: x.active).assign(total=lambda x: x.price * x.qty)

# CSV updated externally...
q3 = q2.reload()  # Reloads CSV and replays all changes
```

### Row Namedtuples

Lambda functions receive rows with dot-accessible columns:

```python
q.filter(lambda x: x.age >= 18)  # Not x['age']
q.assign(full_name=lambda x: f"{x.first} {x.last}")
```

### Reproducibility Tracking

Bend tracks whether your pipeline is deterministic:

```python
q2 = q.sample(100)  # Non-deterministic
print(q2.reproducible)  # False

q3 = q.sample(100, random_state=42)  # Deterministic
print(q3.reproducible)  # True
```

## Basic Examples

### Data Exploration

```python
# Check shape
print(f"{q.rows} rows × {len(q.columns)} columns")

# Quick stats
q.mean('age')
q.sum('revenue')
q.nunique('customer_id')
```

### Filtering and Transformation

```python
# Clean and enrich data
result = (q
    .filter(lambda x: x.age >= 18)
    .assign(total=lambda x: x.price * x.qty)
    .filter(lambda x: x.total > 100)
    .sort('total', ascending=False)
)
```

### Multi-Source Join

```python
customers = Q(load_csv('customers.csv'))
orders = Q(load_csv('orders.csv'))

enriched = customers.merge(orders, on='customer_id', how='left')
```

### Grouping and Aggregation

```python
by_region = q.groupby(
    lambda x: x.region,
    total_sales=lambda g: sum(r.amount for r in g),
    avg_order=lambda g: sum(r.amount for r in g) / len(g),
    count=lambda g: len(g)
).rename(key='region')
```

## Method Documentation

Complete reference for all 37 methods and 4 properties: **[📖 Full Documentation](docs/README.md)**

### Quick Reference by Category

#### 🔧 Data Manipulation
Transform and filter your data:
- **[`assign()`](docs/data-manipulation/assign.md)** - Add computed columns
- **[`filter()`](docs/data-manipulation/filter.md)** - Filter rows by condition  
- **[`map()`](docs/data-manipulation/map.md)** - Restructure rows completely

#### 🔗 Multi-Q Operations
Combine multiple datasets:
- **[`merge()`](docs/multi-q-operations/merge.md)** - Join with conflict resolution
- **[`join()`](docs/multi-q-operations/join.md)** - Simple join wrapper
- **[`concat()`](docs/multi-q-operations/concat.md)** - Vertical stacking
- **[`union()`](docs/multi-q-operations/union.md)** - Concat + deduplicate
- **[`intersect()`](docs/multi-q-operations/intersect.md)** - Rows in both
- **[`difference()`](docs/multi-q-operations/difference.md)** - Rows in self but not other

#### 📊 Column Operations
Manage columns:
- **[`drop()`](docs/column-operations/drop.md)** - Remove columns permanently
- **[`select()`](docs/column-operations/select.md)** - Keep only specified columns
- **[`rename()`](docs/column-operations/rename.md)** - Rename columns
- **[`hide()`](docs/column-operations/hide.md)** - Hide from display only
- **[`unhide()`](docs/column-operations/unhide.md)** - Unhide columns

#### 📏 Row Operations
Select and order rows:
- **[`head()`](docs/row-operations/head.md)** - First n rows
- **[`tail()`](docs/row-operations/tail.md)** - Last n rows
- **[`sample()`](docs/row-operations/sample.md)** - Random sample
- **[`sort()`](docs/row-operations/sort.md)** - Sort by columns
- **[`distinct()`](docs/row-operations/distinct.md)** - Remove duplicates

#### 📈 Aggregations
Compute statistics:
- **[`sum()`, `mean()`, `median()`, `min()`, `max()`, `count()`, `std()`, `var()`, `unique()`, `nunique()`](docs/aggregations/aggregations.md)** - All aggregation methods
- **[`groupby()`](docs/aggregations/groupby.md)** - Group and aggregate

#### 🔄 Lifecycle
Manage state and history:
- **[`reload()`](docs/lifecycle/reload.md)** - Reload from disk (recursive)
- **[`refresh()`](docs/lifecycle/refresh.md)** - Re-apply changes from memory
- **[`rebase()`](docs/lifecycle/rebase.md)** - Flatten history
- **[`memory_usage()`](docs/lifecycle/memory_usage.md)** - Memory breakdown

#### 📤 Output
Export and display:
- **[`show()`](docs/output/show.md)** - Print preview
- **[`to_df()`](docs/output/to_df.md)** - Export to pandas DataFrame
- **[`dump()`](docs/output/dump.md)** - Export to CSV

#### 🏷️ Properties
Read-only attributes:
- **[`columns` / `cols`](docs/properties/properties.md)** - Column names
- **[`rows`](docs/properties/properties.md)** - Row count
- **[`reproducible`](docs/properties/properties.md)** - Determinism flag

## Common Patterns

### ETL Pipeline

```python
result = (Q(load_csv('raw.csv'))
    .filter(lambda x: x.valid)
    .assign(normalized=lambda x: x.value / x.count)
    .select('id', 'normalized', 'category')
    .dump('clean.csv')
)
```

### Data Quality

```python
clean = (q
    .filter(lambda x: x.email and '@' in x.email)
    .filter(lambda x: 0 <= x.age <= 120)
    .distinct('customer_id')
)
```

### Customer Segmentation

```python
customer_stats = orders.groupby(
    lambda x: x.customer_id,
    total_spent=lambda g: sum(r.total for r in g),
    order_count=lambda g: len(g),
    avg_order=lambda g: sum(r.total for r in g) / len(g)
).rename(key='customer_id')

high_value = customer_stats.filter(lambda x: x.total_spent > 10000)
```

### Change Detection

```python
before = Q(load_csv('snapshot_before.csv'))
after = Q(load_csv('snapshot_after.csv'))

deleted = before.difference(after)
added = after.difference(before)
unchanged = before.intersect(after)
```

## CLI Features

### REPL Commands

- `q` - Your loaded Q object
- `r()` - Reload data from source
- `df` - Raw pandas DataFrame (avoid using)
- `load_csv(path)` - Load additional files

### Type Conversion

```python
# Convert columns on load
df = load_csv('data.csv', dtype={'age': int, 'price': float, 'active': bool})
q = Q(df)
```

### Skip Header Rows

```bash
bend data.csv --skip-rows 3
```

Or in Python:
```python
df = load_csv('data.csv', skip_rows=3)
```

### Google Sheets Integration

```python
df = load_csv('https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0')
q = Q(df)
```

## Performance Tips

1. **Filter early** - Reduce dataset size before expensive operations
   ```python
   q.filter(lambda x: x.region == 'CA').assign(...)  # Good
   ```

2. **Use `rebase()`** - After multi-Q operations to reduce memory
   ```python
   result = q1.merge(huge_q, on='id').rebase()
   ```

3. **Monitor memory** - Check usage periodically
   ```python
   usage = q.memory_usage()
   print(f"Using {usage['total_mb']} MB")
   ```

4. **Drop to pandas** - For very large datasets or complex operations
   ```python
   df = q.to_df()
   # Use pandas directly for heavy lifting
   result = df.groupby(...).agg(...)
   q2 = Q(result)
   ```

## Architecture Highlights

### Idempotency

Operations are idempotent by default—replaying produces identical results:

```python
q2 = q.filter(lambda x: x.active)
q3 = q2.refresh()  # Re-applies filter
assert q2.to_df().equals(q3.to_df())
```

### Deep Copy for Multi-Q Operations

By default, `merge()`, `concat()`, etc. deep copy other Q objects for full reproducibility:

```python
q1.merge(q2, on='id', deep_copy=True)  # Default, reproducible
q1.merge(huge_q, on='id', deep_copy=False)  # Faster, non-reproducible
```

### Change History as Tree

For multi-Q operations, change history becomes a tree. `reload()` and `refresh()` recursively process the entire tree.

## Troubleshooting

### "AttributeError: 'Row' object has no attribute 'x'"
Column doesn't exist or was dropped. Check `q.columns`.

### "ValueError: Column conflicts detected"
Use `resolve` parameter in `merge()`:
```python
q1.merge(q2, on='id', resolve={'status': lambda left, right: left})
```

### "Cannot reload: no source path available"
Provide `source_path` when creating Q:
```python
q = Q(df, source_path='data.csv')
```

### Pipeline is non-reproducible
Check `q.reproducible`. Use `random_state` in `sample()` and `deep_copy=True` in merge/concat.

## Version

**2.0.0** - Multi-DataFrame operations, reproducibility tracking

### Changelog

**2.0.0:**
- **BREAKING**: `extend()` renamed to `assign()` (aligns with pandas)
- **BREAKING**: `transform()` renamed to `map()` (clearer semantics)
- Added multi-Q operations: `merge()`, `join()`, `concat()`, `union()`, `intersect()`, `difference()`
- Added `reproducible` property for determinism tracking
- Changed `sample()` to non-deterministic by default (pass `random_state` for reproducibility)
- Added deep/recursive `reload()` for multi-Q pipelines
- Added `deep_copy` parameter to all multi-Q operations

**1.1.0:**
- Added `distinct()`, `rename()`, `tail()`, `sample()`, `drop()`, `select()`
- Added `dtype` parameter to `load_csv()`
- Added `columns`, `rows`, `memory_usage()` methods
- Changed `hide_cols()`/`show_cols()` to `hide()`/`unhide()`
- Changed `sort()` default to `ascending=True`

## License

MIT

## Contributing

Bend is designed to be simple and focused. Before adding features, consider:
- Does it fit the "pandas with training wheels" philosophy?
- Can it be done easily with pandas via `to_df()`?
- Does it maintain immutability and change tracking?

See [TODO.md](TODO.md) for planned features.
