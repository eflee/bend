# merge()

Merge two Q objects based on key columns, similar to SQL JOIN or pandas merge.

## Signature

```python
q.merge(other: Q, on, how: str = 'inner', resolve: dict = None, deep_copy: bool = True) -> Q
```

## Parameters

- `other`: Another Q object to merge with
- `on`: Column name(s) to merge on. Can be:
  - A string for single column: `'customer_id'`
  - A list for multiple columns: `['first_name', 'last_name']`
- `how`: Type of merge (default: `'inner'`)
  - `'inner'`: Only rows with matching keys in both
  - `'left'`: All rows from left (self), matched from right
  - `'right'`: All rows from right (other), matched from left
  - `'outer'`: All rows from both, matched where possible
- `resolve`: **Required if column conflicts exist**. Dict mapping conflicting column names to resolution lambdas with signature `lambda left_val, right_val: result_val`
- `deep_copy`: If `True` (default), stores a deep copy of `other` for full reproducibility. If `False`, stores a reference (faster but non-deterministic)

## Returns

A new Q object with merged data from both Q objects.

## Basic Usage

```python
# Simple inner join on single column
customers = Q(customers_df)
orders = Q(orders_df)
result = customers.merge(orders, on='customer_id')

# Left join (keep all customers, match orders where available)
result = customers.merge(orders, on='customer_id', how='left')

# Merge on multiple columns
result = q1.merge(q2, on=['first_name', 'last_name'])
```

## Use Cases

### 1. Customer-Order Enrichment

```python
# Enrich orders with customer data
customers = load_csv('customers.csv')
orders = load_csv('orders.csv')

enriched_orders = (Q(orders)
    .merge(Q(customers), on='customer_id', how='left')
    .assign(customer_full_name=lambda x: f"{x.first_name} {x.last_name}")
)
```

### 2. Multiple Joins

```python
# Join orders with customers, products, and shipping
result = (Q(orders)
    .merge(Q(customers), on='customer_id', how='left')
    .merge(Q(products), on='product_id', how='left')
    .merge(Q(shipping), on='order_id', how='left')
)
```

### 3. Self-Joins (Employee-Manager)

```python
employees = Q(employee_df)

# Self-join to get manager names
# Note: self-joins automatically deep copy to avoid circular references
with_managers = employees.merge(
    employees,
    on='manager_id',
    resolve={
        'name': lambda emp_name, mgr_name: emp_name,  # Keep employee name
        'department': lambda emp_dept, mgr_dept: emp_dept  # Keep employee dept
    }
).rename(name_RIGHT='manager_name')  # Rename manager's name column
```

## Column Conflicts

When both Q objects have columns with the same name (excluding join keys), you **must** provide resolution lambdas.

### Conflict Resolution

```python
q1 = Q(pd.DataFrame({
    "id": [1, 2],
    "name": ["Alice", "Bob"],
    "status": ["active", "inactive"]
}))

q2 = Q(pd.DataFrame({
    "id": [1, 2],
    "status": ["pending", "complete"]
}))

# ERROR - 'status' column conflict without resolution
# q3 = q1.merge(q2, on='id')  # Raises ValueError

# CORRECT - provide resolution lambda
q3 = q1.merge(q2, on='id', resolve={
    'status': lambda left, right: left  # Keep left status
})

# Or combine values
q3 = q1.merge(q2, on='id', resolve={
    'status': lambda left, right: f"{left}/{right}"  # "active/pending"
})
```

### Resolution Lambda Signature

```python
lambda left_value, right_value: result_value
```

- Applied **row-by-row** during the merge
- Must handle all possible value combinations (including None)
- Return value becomes the final column value

### Common Resolution Patterns

```python
# Keep left value
resolve={'col': lambda l, r: l}

# Keep right value
resolve={'col': lambda l, r: r}

# Keep non-null value (coalesce)
resolve={'col': lambda l, r: l if l is not None else r}

# Take maximum
resolve={'col': lambda l, r: max(l, r)}

# Concatenate strings
resolve={'col': lambda l, r: f"{l}, {r}"}

# Custom logic
resolve={'col': lambda l, r: l if l > r else r * 2}
```

## Reproducibility & Deep Copy

### Deep Copy (Default)

```python
q1 = Q(df1)
q2 = Q(df2)

# Stores a deep copy of q2 in q3's history
q3 = q1.merge(q2, on='id', deep_copy=True)

# q3 is fully deterministic
print(q3.deterministic)  # True (if both q1 and q2 are deterministic)

# Modifying q2 after the merge doesn't affect q3
q2 = q2.filter(lambda x: x.value > 100)
q3_reloaded = q3.reload()  # Still works, uses original q2 state
```

### Reference Mode (Performance)

```python
# For large datasets, avoid deep copy overhead
huge_q = Q(very_large_df)
q3 = q1.merge(huge_q, on='id', deep_copy=False)

# Faster, but non-deterministic
print(q3.deterministic)  # False

# If huge_q is modified, reload() might fail or produce different results
huge_q = huge_q.filter(...)  # Changes the object q3 references
```

**When to use `deep_copy=False`:**

- Very large datasets (>1M rows) where memory is a concern
- One-off analysis where reproducibility isn't needed
- You're going to `rebase()` immediately after

## Gotchas

### 1. All Conflicts Must Be Resolved

```python
# If 3 columns conflict, must resolve all 3
q1.merge(q2, on='id', resolve={
    'status': lambda l, r: l,
    'priority': lambda l, r: max(l, r)
    # ERROR: Missing 'category' resolution
})
```

### 2. Join Keys Are Not Conflicted

```python
# Join keys never conflict, even if they exist in both
q1.merge(q2, on='id')  # 'id' appears once in result, no resolution needed
```

### 3. Null Handling in Resolution

```python
# Resolution lambdas receive None for non-matching rows
q1.merge(q2, on='id', how='outer', resolve={
    # Dangerous - TypeError if either is None
    'value': lambda l, r: l + r
    
    # Safe
    'value': lambda l, r: (l or 0) + (r or 0)
})
```

### 4. Performance with Large Merges

Deep copying large Q objects can be expensive:

```python
# Expensive - copies 10M rows
large_q = Q(huge_df)  # 10M rows
result = q.merge(large_q, on='id')

# More efficient - use reference mode + rebase
result = q.merge(large_q, on='id', deep_copy=False).rebase()
# Now result is self-contained, no reference to large_q
```

### 5. Self-Joins Automatically Deep Copy

```python
employees = Q(emp_df)

# Always makes a deep copy of self, even with deep_copy=False
with_managers = employees.merge(employees, on='manager_id')
# Prevents circular reference issues
```

## Merge Types

### Inner Join

```python
# Only matching rows
result = q1.merge(q2, on='id', how='inner')
```

| q1.id | q1.value | q2.id | q2.score |
|-------|----------|-------|----------|
| 1     | A        | 1     | 90       |
| 2     | B        | 3     | 85       |

Result:
| id | value | score |
|----|-------|-------|
| 1  | A     | 90    |

### Left Join

```python
# All rows from left, matched from right
result = q1.merge(q2, on='id', how='left')
```

Result:
| id | value | score |
|----|-------|-------|
| 1  | A     | 90    |
| 2  | B     | NaN   |

### Right Join

```python
# All rows from right, matched from left
result = q1.merge(q2, on='id', how='right')
```

Result:
| id | value | score |
|----|-------|-------|
| 1  | A     | 90    |
| 3  | NaN   | 85    |

### Outer Join

```python
# All rows from both
result = q1.merge(q2, on='id', how='outer')
```

Result:
| id | value | score |
|----|-------|-------|
| 1  | A     | 90    |
| 2  | B     | NaN   |
| 3  | NaN   | 85    |

## Performance Considerations

- Merge is implemented using pandas `pd.merge()` (efficient)
- Deep copy overhead is proportional to the size of `other`
- Resolution lambdas are applied row-by-row (not vectorized)
- For very large merges, consider using pandas directly

## Chaining

```python
result = (customers
    .filter(lambda x: x.active)
    .merge(orders, on='customer_id', how='left')
    .merge(products, on='product_id', how='left')
    .assign(total_value=lambda x: x.price * x.quantity)
    .filter(lambda x: x.total_value > 100)
)
```

## Idempotency

✅ **Conditional**

- **Yes** if both Q objects are deterministic and `deep_copy=True` (default)
- **No** if `deep_copy=False`

Check with `q.deterministic` property.

## See Also

- [`join()`](join.md) - Simpler merge wrapper (no conflict resolution)
- [`concat()`](concat.md) - Vertical stacking (union of rows)
- [`union()`](union.md) - Concat + deduplication

