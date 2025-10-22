# Bend Documentation

Complete reference for all Bend REPL capabilities.

## Quick Start

```python
# Load a CSV
from bend import Q, load_csv
q = Q(load_csv('data.csv'))

# Explore and transform
result = (q
    .filter(lambda x: x.age >= 18)
    .assign(total=lambda x: x.price * x.qty)
    .sort('total', ascending=False)
    .head(10)
)

# Export
result.dump('output.csv')
```

## Documentation Categories

### 🔧 [Data Manipulation](data-manipulation/README.md)

Core operations for transforming your data:

- [`assign()`](data-manipulation/assign.md) - Add computed columns
- [`filter()`](data-manipulation/filter.md) - Filter rows by condition
- [`map()`](data-manipulation/map.md) - Restructure rows completely

### 🔗 [Multi-Q Operations](multi-q-operations/README.md)

Combining multiple Q objects:

- [`merge()`](multi-q-operations/merge.md) - Join with conflict resolution
- [`join()`](multi-q-operations/join.md) - Simple join wrapper
- [`concat()`](multi-q-operations/concat.md) - Vertical stacking
- [`union()`](multi-q-operations/union.md) - Concat + deduplicate
- [`intersect()`](multi-q-operations/intersect.md) - Rows in both
- [`difference()`](multi-q-operations/difference.md) - Rows in self but not other

### 📊 [Column Operations](column-operations/README.md)

Managing columns:

- [`drop()`](column-operations/drop.md) - Remove columns (structural)
- [`select()`](column-operations/select.md) - Keep only specified columns
- [`rename()`](column-operations/rename.md) - Rename columns
- [`hide()`](column-operations/hide.md) - Hide from display only
- [`unhide()`](column-operations/unhide.md) - Unhide columns

### 📏 [Row Operations](row-operations/README.md)

Selecting and ordering rows:

- [`head()`](row-operations/head.md) - First n rows
- [`tail()`](row-operations/tail.md) - Last n rows
- [`sample()`](row-operations/sample.md) - Random sample
- [`sort()`](row-operations/sort.md) - Sort by columns
- [`distinct()`](row-operations/distinct.md) - Remove duplicates

### 📈 [Aggregations](aggregations/README.md)

Computing summary statistics:

- [All Aggregations](aggregations/aggregations.md) - Quick reference
- [`groupby()`](aggregations/groupby.md) - Group and aggregate
- Individual methods: `sum()`, `mean()`, `median()`, `min()`, `max()`, `count()`, `std()`, `var()`, `unique()`, `nunique()`

### 🔄 [Lifecycle](lifecycle/README.md)

Managing Q state and history:

- [`reload()`](lifecycle/reload.md) - Reload from disk (deep/recursive)
- [`replay()`](lifecycle/replay.md) - Re-apply changes from memory
- [`rebase()`](lifecycle/rebase.md) - Flatten history
- [`memory_usage()`](lifecycle/memory_usage.md) - Memory breakdown

### 📤 [Output](output/README.md)

Exporting and displaying data:

- [`show()`](output/show.md) - Print preview
- [`to_df()`](output/to_df.md) - Export to pandas DataFrame
- [`dump()`](output/dump.md) - Export to CSV

### 🏷️ [Properties](properties/properties.md)

Read-only attributes:

- [All Properties](properties/properties.md) - `columns`, `cols`, `rows`, `deterministic`

## Key Concepts

**[📘 Concepts](concepts/README.md)** - In-depth guides on determinism, reloadability, and other core concepts

### Immutability

All operations return a **new Q object**. Original is never modified:

```python
q1 = Q(df)
q2 = q1.filter(lambda x: x.active)
# q1 is unchanged, q2 is new
```

### Change Tracking

Every operation is tracked and can be replayed:

```python
q = Q(load_csv('data.csv'), source_path='data.csv')
q2 = q.filter(...).assign(...)
# ... CSV updated ...
q3 = q2.reload()  # Reloads CSV and replays changes
```

### Reproducibility

Track whether your pipeline is deterministic:

```python
q2 = q.sample(100)  # Non-deterministic
print(q2.deterministic)  # False

q3 = q.sample(100, random_state=42)  # Deterministic
print(q3.deterministic)  # True
```

### Row Namedtuples

Lambda functions receive rows as namedtuples with dot-accessible columns:

```python
q.filter(lambda x: x.age >= 18)  # x.age instead of x['age']
q.assign(full_name=lambda x: f"{x.first} {x.last}")
```

### Method Chaining

Fluent API for readable pipelines:

```python
result = (q
    .filter(lambda x: x.region == 'CA')
    .assign(total=lambda x: x.price * x.qty)
    .sort('total', ascending=False)
    .head(10)
    .dump('top_10_ca.csv')
)
```

## Common Patterns

### ETL Pipeline

```python
result = (Q(load_csv('raw.csv'))
    .filter(lambda x: x.valid)  # Clean
    .assign(normalized=lambda x: x.value / x.count)  # Transform
    .select('id', 'normalized', 'category')  # Select
    .dump('clean.csv')  # Load
)
```

### Data Quality

```python
clean = (q
    .filter(lambda x: x.email and '@' in x.email)
    .filter(lambda x: x.age >= 0 and x.age <= 120)
    .distinct('customer_id')  # Remove duplicates
)
```

### Multi-Source Join

```python
customers = Q(load_csv('customers.csv'))
orders = Q(load_csv('orders.csv'))
products = Q(load_csv('products.csv'))

enriched = (orders
    .merge(customers, on='customer_id', how='left')
    .merge(products, on='product_id', how='left')
    .assign(revenue=lambda x: x.price * x.quantity)
)
```

### Time Series Aggregation

```python
daily_stats = (transactions
    .groupby(
        lambda x: x.timestamp.date(),
        count=lambda g: len(g),
        revenue=lambda g: sum(r.amount for r in g),
        avg_transaction=lambda g: sum(r.amount for r in g) / len(g)
    )
    .rename(key='date')
    .sort('date')
)
```

## CLI Usage

### Interactive REPL

```bash
# Start with file
bend data.csv

# Start empty
bend

# In REPL
>>> q.filter(lambda x: x.active).show()
>>> q.columns
>>> q.rows
```

### REPL Commands

- `q` - Your loaded Q object
- `r()` - Reload (alias for `q.reload()`)
- `df` - Raw pandas DataFrame (avoid using)
- `load_csv()` - Load additional files

## Performance Tips

1. **Filter Early**: Reduce dataset size before expensive operations

   ```python
   q.filter(lambda x: x.region == 'CA').assign(...)  # Good
   q.assign(...).filter(lambda x: x.region == 'CA')  # Slower
   ```

2. **Use rebase()**: After multi-Q operations with large datasets

   ```python
   result = q1.merge(huge_q, on='id').rebase()  # Drops deep copy
   ```

3. **Monitor Memory**: Check memory usage periodically

   ```python
   usage = q.memory_usage()
   print(f"Using {usage['total_mb']} MB")
   ```

4. **Use pandas for Heavy Lifting**: For very large datasets or complex operations

   ```python
   df = q.to_df()
   # Use pandas directly for performance
   result = df.groupby(...).agg(...)
   q2 = Q(result)
   ```

## Troubleshooting

### "AttributeError: 'Row' object has no attribute 'x'"

Your column name doesn't exist or was dropped earlier in the pipeline.

```python
# Check available columns
print(q.columns)
```

### "ValueError: Column conflicts detected"

When merging, both Q objects have the same column names. Use `resolve`:

```python
q1.merge(q2, on='id', resolve={'status': lambda l, r: l})
```

### "Cannot reload: no source path available"

Q was created without a source path, or after a terminal operation like `groupby()`:

```python
# Fix: provide source_path at creation
q = Q(df, source_path='data.csv')
```

### Non-deterministic pipeline

Check `q.deterministic`. Use `random_state` in `sample()` and `deep_copy=True` in merge/concat.

## See Also

- [Main README](../README.md) - Project overview and installation
- [TODO](../TODO.md) - Roadmap and future features
- [Understanding Determinism & Reloadability](concepts/understanding-determinism-reloadability.md) - Core concepts guide

## Philosophy

Bend is designed to make CSV analysis **simple, safe, and deterministic**:

- **Simple**: Pandas with training wheels - intuitive API, readable code
- **Safe**: Immutable operations, explicit conflict resolution, validation
- **Deterministic**: Change tracking, replay capability, determinism tracking

When Bend feels limiting, that's a feature not a bug - drop to pandas via `to_df()` for unrestricted power, then bring results back into Bend with `Q(df)`.

