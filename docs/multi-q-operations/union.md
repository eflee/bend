# union()

Set union operation: combine rows from two Q objects and remove duplicates.

## Signature

```python
q.union(other: Q, deep_copy: bool = True) -> Q
```

## Parameters

- `other`: Another Q object to union with (must have same columns)
- `deep_copy`: If `True` (default), stores deep copy for reproducibility

## Returns

A new Q object with unique rows from both Q objects.

## Basic Usage

```python
q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))

result = q1.union(q2)
# Result: [1, 2, 3, 4] (duplicates removed)
```

## Behavior

`union()` is equivalent to `concat().distinct()`:

```python
# These are the same
result = q1.union(q2)
result = q1.concat(q2).distinct()
```

## Use Cases

### 1. Combining Similar Datasets
```python
west_customers = Q(load_csv('customers_west.csv'))
east_customers = Q(load_csv('customers_east.csv'))

# All unique customers
all_customers = west_customers.union(east_customers)
```

### 2. Deduplicating Across Sources
```python
source1 = Q(df1).filter(lambda x: x.status == 'active')
source2 = Q(df2).filter(lambda x: x.status == 'active')

# Combined without duplicates
active = source1.union(source2)
```

## Set Theory

Union follows mathematical set union: A ∪ B

```python
q1 = Q(pd.DataFrame({'x': [1, 2, 3]}))
q2 = Q(pd.DataFrame({'x': [2, 3, 4]}))
result = q1.union(q2)
# Result: {1, 2, 3, 4}
```

## Row-Level Deduplication

Duplicates are determined by comparing **all columns** row-by-row:

```python
q1 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 20]}))
q2 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 30]}))
result = q1.union(q2)
# Result: 3 rows (row [1, 10] is same, but [2, 20] ≠ [2, 30])
```

## Insertion Order

Union preserves insertion order (self first, then other's unique rows):

```python
q1 = Q(pd.DataFrame({'a': [3, 1, 2]}))
q2 = Q(pd.DataFrame({'a': [2, 4, 1]}))
result = q1.union(q2)
# Result order: [3, 1, 2, 4]
# (from q1: 3, 1, 2; from q2: only 4 is new)
```

## Gotchas

### Columns Must Match
```python
q1 = Q(pd.DataFrame({'a': [1, 2]}))
q2 = Q(pd.DataFrame({'b': [3, 4]}))  # Different columns

# union() will work but produce unexpected results
# All original values become NaN for mismatched columns
```

## Comparison with concat()

| `union()` | `concat()` |
|-----------|------------|
| Removes duplicates | Keeps duplicates |
| Slower (runs distinct()) | Faster |
| Set operation | Simple stacking |

## Idempotency

✅ **Conditional** - Same as [`concat()`](concat.md)

## See Also

- [`concat()`](concat.md) - Without deduplication
- [`intersect()`](intersect.md) - Rows in both
- [`difference()`](difference.md) - Rows in self but not other
- [`distinct()`](distinct.md) - Remove duplicates from single Q

