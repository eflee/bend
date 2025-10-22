# select()

Keep only specified columns, removing all others (inverse of drop).

## Signature

```python
q.select(*cols) -> Q
```

## Parameters

- `*cols`: Column names to keep

## Returns

A new Q object with only the specified columns.

## Basic Usage

```python
# Keep specific columns
q2 = q.select('name', 'email', 'age')

# Single column
q2 = q.select('revenue')
```

## Use Cases

### 1. Simplify Data

```python
# Keep only what you need
simple = q.select('customer_id', 'order_date', 'total')
```

### 2. Export Subset

```python
# Create export with specific fields
export = (q
    .filter(lambda x: x.status == 'complete')
    .select('order_id', 'customer_email', 'total')
)
```

### 3. Privacy

```python
# Remove sensitive data
safe = q.select('id', 'name', 'email')  # Drops SSN, etc.
```

## select() vs drop()

```python
# These are equivalent
q.select('a', 'b', 'c')
q.drop('d', 'e', 'f')  # If d,e,f are the only other columns
```

Use `select()` when you know what to keep.
Use `drop()` when you know what to remove.

## Gotchas

### Non-Existent Columns Ignored

```python
# Only keeps columns that exist
q2 = q.select('a', 'b', 'nonexistent')  # Gets a, b only
```

### Order Doesn't Matter

```python
q.select('age', 'name', 'email')  # Column order unchanged
# Result has columns in original order, not selection order
```

## Idempotency

✅ **Yes** - Fully idempotent.

## See Also

- [`drop()`](drop.md) - Remove specific columns
- [`hide()`](hide.md) - Hide from display only
- [`map()`](map.md) - Transform to specific structure
