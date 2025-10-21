# join()

Simplified merge wrapper for joining Q objects when no column conflicts exist.

## Signature

```python
q.join(other: Q, on, how: str = 'inner', deep_copy: bool = True) -> Q
```

## Parameters

- `other`: Another Q object to join with
- `on`: Column name(s) to join on (string or list)
- `how`: Join type: `'inner'`, `'left'`, `'right'`, `'outer'` (default: `'inner'`)
- `deep_copy`: If `True` (default), stores deep copy for reproducibility

## Returns

A new Q object with joined data.

## Basic Usage

```python
customers = Q(customers_df)
orders = Q(orders_df)

# Simple join
result = customers.join(orders, on='customer_id', how='left')
```

## When to Use join() vs merge()

| Use `join()` when... | Use `merge()` when... |
|----------------------|----------------------|
| No column conflicts | Column conflicts exist |
| Simpler, cleaner code | Need explicit conflict resolution |
| Most common use case | Complex merge logic |

```python
# join() - simple case
result = q1.join(q2, on='id')

# merge() - handles conflicts
result = q1.merge(q2, on='id', resolve={'status': lambda l, r: l})
```

## Error on Conflicts

If column conflicts exist (same column name in both Q objects, excluding join keys), `join()` will **raise an error**:

```python
q1 = Q(pd.DataFrame({'id': [1, 2], 'status': ['active', 'inactive']}))
q2 = Q(pd.DataFrame({'id': [1, 2], 'status': ['pending', 'complete']}))

# ERROR - 'status' column conflict
# result = q1.join(q2, on='id')

# Use merge() with resolve parameter instead
result = q1.merge(q2, on='id', resolve={'status': lambda l, r: l})
```

## Use Cases

### 1. Standard Left Join
```python
# All customers with their orders (if any)
customers = Q(load_csv('customers.csv'))
orders = Q(load_csv('orders.csv'))

with_orders = customers.join(orders, on='customer_id', how='left')
```

### 2. Inner Join (Matching Only)
```python
# Only customers who have placed orders
active_customers = customers.join(orders, on='customer_id', how='inner')
```

### 3. Multiple Key Joins
```python
# Join on composite key
result = q1.join(q2, on=['country', 'state', 'city'])
```

## Idempotency

✅ **Conditional** - Same as [`merge()`](merge.md)

## See Also

- [`merge()`](merge.md) - Full merge with conflict resolution
- [`concat()`](concat.md) - Vertical stacking

