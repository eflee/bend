# dump()

Export Q to a CSV file.

## Signature

```python
q.dump(filename: str) -> Q
```

## Parameters

- `filename`: Path to output CSV file

## Returns

Self (for chaining).

## Basic Usage

```python
q.dump('output.csv')

# Chaining
q.filter(...).assign(...).dump('results.csv')
```

## Important Notes

- Writes **all columns**, including hidden ones
- Does not write row index
- Overwrites existing file

## See Also

- [`to_df()`](to_df.md) - Export to DataFrame
- [`hide()`](hide.md) - Note: hidden columns ARE included in dump
