# Multi-Q Operations

Combining multiple Q objects through joins, unions, and set operations.

## Methods

### [merge()](merge.md)
Join Q objects with explicit conflict resolution.

**When to use**: Complex joins, column conflicts, custom merge logic

```python
q1.merge(q2, on='id', resolve={'status': lambda l, r: l})
```

### [join()](join.md)
Simple join wrapper (no conflicts).

**When to use**: Standard joins without column conflicts

```python
customers.join(orders, on='customer_id', how='left')
```

### [concat()](concat.md)
Vertical stacking (keeps duplicates).

**When to use**: Combining time series, appending batches

```python
jan.concat(feb).concat(mar)
```

### [union()](union.md)
Vertical stacking with deduplication.

**When to use**: Combining similar datasets, removing duplicates

```python
west.union(east)  # Unique rows from both
```

### [intersect()](intersect.md)
Rows present in both Q objects.

**When to use**: Finding common records, validation

```python
expected.intersect(actual)
```

### [difference()](difference.md)
Rows in self but not in other.

**When to use**: Finding missing/extra records, change detection

```python
before.difference(after)  # Deleted rows
```

## Join Types

| Method | Type | Duplicates | Use Case |
|--------|------|------------|----------|
| `merge()` | Horizontal | Depends | Complex joins |
| `join()` | Horizontal | Depends | Simple joins |
| `concat()` | Vertical | Keeps | Stacking |
| `union()` | Vertical | Removes | Set union |
| `intersect()` | Vertical | Removes | Set intersection |
| `difference()` | Vertical | Removes | Set difference |

## Reproducibility

All multi-Q operations support `deep_copy` parameter:

```python
# Default: deep copy (reproducible)
q1.merge(q2, on='id', deep_copy=True)

# Performance: reference (non-reproducible)
q1.merge(huge_q, on='id', deep_copy=False)
```

## Common Patterns

### Customer-Order Enrichment
```python
enriched = (customers
    .join(orders, on='customer_id', how='left')
    .join(products, on='product_id', how='left')
)
```

### Data Validation
```python
missing = expected.difference(actual)
extra = actual.difference(expected)
matches = expected.intersect(actual)
```

### Time Series Union
```python
all_data = (jan.union(feb).union(mar).union(apr))
```
