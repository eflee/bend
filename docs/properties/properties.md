# Properties

Quick reference for Q object properties (read-only attributes).

## columns / cols

Get list of column names.

```python
cols = q.columns  # ['name', 'age', 'email']
cols = q.cols     # Alias for columns
```

**Use in lambdas:**
```python
# Check what columns are available
print(q.columns)

# Use in operations
q.assign(total=lambda x: x.price * x.qty)
```

**Includes hidden columns** - `hide()` doesn't affect this property.

## rows

Get count of rows.

```python
count = q.rows  # 1523
```

Equivalent to `len(q)`:
```python
assert q.rows == len(q)
```

**Use for reporting:**
```python
print(f"{q.rows} rows × {len(q.columns)} columns")
```

## reproducible

Check if Q's history is fully deterministic.

```python
is_repro = q.reproducible  # True or False
```

### What Makes a Q Non-Reproducible?

```python
# These make reproducible=False:
q.sample(100)  # No random_state
q.merge(other, deep_copy=False)  # Reference mode
q.concat(other, deep_copy=False)

# These keep reproducible=True:
q.sample(100, random_state=42)  # Explicit seed
q.filter(lambda x: x.value > 100)
q.assign(total=lambda x: x.price * x.qty)
```

### Why It Matters

```python
q2 = q.filter(...).sample(100).assign(...)
print(q2.reproducible)  # False (sample without seed)

# reload() will produce different results!
q3 = q2.reload()  # Different 100 rows sampled

# To fix:
q2 = q.filter(...).sample(100, random_state=42).assign(...)
print(q2.reproducible)  # True
q3 = q2.reload()  # Same 100 rows
```

### Propagation

`reproducible` propagates like a flag through all operations:

```python
q1 = Q(df)  # True
q2 = q1.filter(...)  # True
q3 = q2.sample(100)  # False (non-deterministic sample)
q4 = q3.assign(...)  # False (inherited from q3)
```

**Once False, stays False** (unless you rebase from a reproducible Q).

## Usage Examples

```python
# Column discovery
print(f"Available columns: {', '.join(q.columns)}")

# Data shape
print(f"Dataset: {q.rows} rows × {len(q.columns)} columns")

# Reproducibility check
if not q.reproducible:
    print("⚠️ Warning: Pipeline is non-reproducible")
    print("reload() may produce different results")
```

## See Also

- [`sample()`](sample.md) - Non-reproducible by default
- [`merge()`](merge.md), [`concat()`](concat.md) - `deep_copy=False` makes non-reproducible
- [`reload()`](reload.md), [`refresh()`](refresh.md) - Replay operations
