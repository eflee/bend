# sort()

Sort rows by one or more columns.

## Signature

```python
q.sort(*cols, ascending: bool = True) -> Q
```

## Parameters

- `*cols`: Column names to sort by. If empty, sorts by all columns.
- `ascending`: Sort order (default: `True` for ascending)

## Basic Usage

```python
# Sort by single column (ascending)
q2 = q.sort('age')

# Descending
q2 = q.sort('price', ascending=False)

# Multiple columns
q2 = q.sort('last_name', 'first_name')

# Sort by all columns
q2 = q.sort()
```

## Use Cases

```python
# Find top/bottom
top_10 = q.sort('revenue', ascending=False).head(10)
bottom_10 = q.sort('score').head(10)

# Chronological
timeline = q.sort('timestamp')

# Alphabetical
alpha = q.sort('name')
```

## Multi-Column Sort

```python
# Sort by region, then by sales (descending) within each region
q.sort('region', 'sales')  # Both ascending

# For mixed order, chain sorts (later sorts are primary)
q.sort('sales', ascending=False).sort('region')  # Region primary, sales secondary
```

## Gotchas

### Empty Columns Parameter
```python
q.sort()  # Sorts by ALL columns (may be slow)
```

### Null Handling
Nulls sort to the end by default in pandas.

## Idempotency

✅ **Yes**

## See Also

- [`head()`](head.md), [`tail()`](tail.md) - Used after sort
