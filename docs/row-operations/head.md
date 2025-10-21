# head()

Return the first n rows.

## Signature

```python
q.head(n: int = 5) -> Q
```

## Parameters

- `n`: Number of rows to return (default: 5)

## Returns

A new Q object with the first n rows.

## Basic Usage

```python
# First 10 rows
q2 = q.head(10)

# Default (5 rows)
q2 = q.head()
```

## Use Cases

```python
# Preview after sort
q.sort('date').head(20)  # Earliest 20

# Quick sample
q.head(100)  # First 100 for testing

# Top N after ranking
q.sort('revenue', ascending=False).head(10)  # Top 10 by revenue
```

## Idempotency

✅ **Yes**

## See Also

- [`tail()`](tail.md) - Last n rows
- [`sample()`](sample.md) - Random sample
- [`sort()`](sort.md) - Sort before head
