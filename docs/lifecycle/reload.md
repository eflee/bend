# reload()

Reload data from source CSV and recursively reload all referenced Q objects.

## Signature

```python
q.reload() -> Q
```

## Returns

A new Q object with reloaded base data and re-applied changes.

## What It Does

**Deep/recursive reload**:
1. Reloads this Q's source CSV from disk
2. Recursively reloads any Q objects in change history (from `concat`, `merge`, etc.)
3. Re-applies all changes to the newly reloaded data

## Basic Usage

```python
q = Q(load_csv('sales.csv', source_path='sales.csv'))
q2 = q.filter(lambda x: x.amount > 100)

# ... CSV file is updated ...

q3 = q2.reload()  # Reads new CSV, re-applies filter
```

## Use Cases

### 1. External Data Updates
```python
# Initial load
sales = Q(load_csv('daily_sales.csv'), source_path='daily_sales.csv')
pipeline = sales.filter(lambda x: x.verified).assign(tax=lambda x: x.total * 0.08)

# CSV is updated by another process
# Reload to get fresh data
updated_pipeline = pipeline.reload()
```

### 2. Iterative Development
```python
# Build pipeline
q = Q(load_csv('data.csv'), source_path='data.csv')
result = q.filter(...).assign(...).sort(...)

# Fix data issues in CSV
# Reload to reprocess
result = result.reload()
```

### 3. Multi-Q Pipelines
```python
q1 = Q(load_csv('customers.csv'), source_path='customers.csv')
q2 = Q(load_csv('orders.csv'), source_path='orders.csv')
combined = q1.merge(q2, on='customer_id')

# Both CSVs updated
# Deep reload gets fresh data from BOTH sources
fresh = combined.reload()
```

## Deep Reload (Recursive)

`reload()` recursively reloads the entire Q tree:

```python
q1 = Q(load_csv('a.csv'), source_path='a.csv')
q2 = Q(load_csv('b.csv'), source_path='b.csv')
q3 = Q(load_csv('c.csv'), source_path='c.csv')

result = q1.concat(q2).merge(q3, on='id')

# reload() will:
# 1. Reload a.csv (q1's source)
# 2. Reload b.csv (q2's source, from concat)
# 3. Reload c.csv (q3's source, from merge)
# 4. Re-apply concat and merge
fresh = result.reload()
```

## Column Validation

Reload validates that all original columns still exist:

```python
q = Q(df, source_path='data.csv')  # Has columns: a, b, c
q2 = q.filter(...)

# CSV is edited, column 'b' removed
q2.reload()  # ValueError: required columns missing from source: b
```

New columns and rows are allowed - only column removal causes errors.

## Errors

### No Source Path
```python
q = Q(df)  # No source_path
q.reload()  # ValueError: Cannot reload: no source path available
```

### After groupby()
```python
q = Q(df, source_path='data.csv')
q2 = q.groupby(...)  # Terminal operation, resets history
q2.reload()  # ValueError: no source path
```

## reload() vs refresh()

| `reload()` | `refresh()` |
|------------|-------------|
| Reads from disk | Uses in-memory base |
| Recursive (reloads other Qs) | Recursive (refreshes other Qs) |
| Gets updated data | Recomputes current data |
| Requires source_path | Always works |

```python
# reload() - fresh from disk
q2 = q.reload()

# refresh() - recompute from memory
q2 = q.refresh()
```

## Idempotency

✅ **Yes** - If all source files haven't changed, produces same result.

## See Also

- [`refresh()`](refresh.md) - Re-apply changes without disk reload
- [`rebase()`](rebase.md) - Flatten history
