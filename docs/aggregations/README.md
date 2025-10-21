# Aggregations

Computing summary statistics and grouped aggregations.

## Quick Reference

See [aggregations.md](aggregations.md) for all methods: `sum()`, `mean()`, `median()`, `min()`, `max()`, `count()`, `std()`, `var()`, `unique()`, `nunique()`

## [groupby()](groupby.md)

Group rows and compute aggregations.

```python
q.groupby(
    lambda x: x.category,
    total=lambda g: sum(r.sales for r in g),
    count=lambda g: len(g),
    avg=lambda g: sum(r.sales for r in g) / len(g)
)
```

## Examples

```python
# Simple aggregation
total_revenue = q.sum('revenue')

# Multiple metrics
print(f"Mean: {q.mean('age')}, Median: {q.median('age')}")

# Grouped
by_region = q.groupby(
    lambda x: x.region,
    revenue=lambda g: sum(r.amount for r in g)
).rename(key='region')
```
