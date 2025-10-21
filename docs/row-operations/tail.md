# tail()

Return the last n rows.

## Signature

```python
q.tail(n: int = 5) -> Q
```

## Parameters

- `n`: Number of rows to return (default: 5)

## Returns

A new Q object with the last n rows.

## Basic Usage

```python
# Last 10 rows
q2 = q.tail(10)

# Most recent after sort
q.sort('date').tail(20)  # Most recent 20
```

## Use Cases

```python
# Latest records
recent = q.sort('timestamp').tail(100)

# Bottom performers
worst = q.sort('score', ascending=False).tail(10)
```

## Idempotency

✅ **Yes**

## See Also

- [`head()`](head.md), [`sample()`](sample.md), [`sort()`](sort.md)
