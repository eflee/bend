# show()

Print the first n rows to console (respects hidden columns).

## Signature

```python
q.show(n: int = 20) -> Q
```

## Parameters

- `n`: Number of rows to display (default: 20)

## Returns

Self (for chaining).

## Basic Usage

```python
q.show()  # Print 20 rows
q.show(50)  # Print 50 rows

# Chaining
q.filter(...).show().assign(...)  # Show intermediate result
```

## See Also

- [`hide()`](hide.md) - Hide columns from display
- [`to_df()`](to_df.md) - Export to DataFrame
