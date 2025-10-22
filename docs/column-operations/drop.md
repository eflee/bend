# drop()

Remove specified columns from the DataFrame (structural change).

## Signature

```python
q.drop(*cols) -> Q
```

## Parameters

- `*cols`: Column names to remove

## Returns

A new Q object without the specified columns.

## Basic Usage

```python
# Drop single column
q2 = q.drop('id')

# Drop multiple columns
q2 = q.drop('id', 'internal_field', 'temp')
```

## Use Cases

### 1. Remove Unnecessary Columns

```python
# Clean dataset for export
export = q.drop('internal_id', 'created_at_timestamp', 'debug_flag')
```

### 2. Before Merge

```python
# Remove conflicting columns before merge
q1_clean = q1.drop('status', 'priority')
result = q1_clean.merge(q2, on='id')
```

### 3. Privacy/Security

```python
# Remove sensitive data
public = q.drop('ssn', 'password_hash', 'credit_card')
```

## drop() vs hide()

| `drop()` | `hide()` |
|----------|----------|
| **Removes** columns | **Hides** columns |
| Structural change | Display-only change |
| Cannot use in later operations | Can still use in operations |
| Tracked in history | Not tracked in history |
| Affects `to_df()` and `dump()` | Only affects print/show |

```python
# drop() - column is gone
q2 = q.drop('cost')
q3 = q2.assign(profit=lambda x: x.revenue - x.cost)  # ERROR!

# hide() - column still exists
q2 = q.hide('cost')
q3 = q2.assign(profit=lambda x: x.revenue - x.cost)  # Works!
```

## Gotchas

### Dropping Non-Existent Columns

```python
# Silently succeeds - no error if column doesn't exist
q2 = q.drop('nonexistent_column')  # OK, no-op
```

### Cannot Access After Drop

```python
q2 = q.drop('price')
q3 = q2.filter(lambda x: x.price > 100)  # AttributeError!
```

### Order Doesn't Matter

```python
q.drop('a', 'b', 'c')  # Same as
q.drop('c', 'a', 'b')
```

## Chaining

```python
result = (q
    .drop('id', 'created_at')
    .filter(lambda x: x.active)
    .assign(full_name=lambda x: f"{x.first} {x.last}")
)
```

## Idempotency

✅ **Yes** - Fully idempotent.

## See Also

- [`select()`](select.md) - Keep only specific columns
- [`hide()`](hide.md) - Hide columns from display only
- [`rename()`](rename.md) - Rename columns
