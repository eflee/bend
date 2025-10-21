# Aggregation Methods

Quick reference for all aggregation methods (non-transforming, return scalars).

## sum(col)

```python
total = q.sum('revenue')  # Sum of revenue column
```

## mean(col)

```python
average = q.mean('age')  # Average age
```

## median(col)

```python
middle = q.median('price')  # Median price
```

## min(col)

```python
lowest = q.min('temperature')  # Minimum value
```

## max(col)

```python
highest = q.max('score')  # Maximum value
```

## count(col=None)

```python
total_rows = q.count()  # Total rows
non_null = q.count('email')  # Non-null emails
```

## std(col)

```python
std_dev = q.std('values')  # Standard deviation
```

## var(col)

```python
variance = q.var('measurements')  # Variance
```

## unique(col)

```python
values = q.unique('category')  # List of unique values
# Returns: ['A', 'B', 'C']
```

## nunique(col)

```python
count = q.nunique('customer_id')  # Count of unique customers
# Returns: 1523
```

## Usage

All aggregations are **informational** - they don't modify the Q object:

```python
q = Q(df)
total = q.sum('revenue')  # Returns a number
print(f"Total revenue: ${total}")

# q is unchanged
q2 = q.filter(lambda x: x.active)  # Can continue using q
```

## See Also

- [`groupby()`](groupby.md) - Grouped aggregations
