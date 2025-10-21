# distinct()

Remove duplicate rows.

## Signature

```python
q.distinct(*cols) -> Q
```

## Parameters

- `*cols`: Optional columns to consider for uniqueness. If empty, uses all columns.

## Returns

A new Q object with duplicates removed (keeps first occurrence).

## Basic Usage

```python
# Remove completely duplicate rows
q2 = q.distinct()

# Keep first per customer
q2 = q.distinct('customer_id')

# Unique by multiple columns
q2 = q.distinct('email', 'phone')
```

## Use Cases

### 1. Data Quality
```python
# Remove duplicate records
clean = q.distinct()
```

### 2. Unique Entities
```python
# One row per customer
unique_customers = q.distinct('customer_id')
```

### 3. After Union
```python
# Combine and deduplicate
all_records = q1.concat(q2).distinct()

# Or use union() which does this automatically
all_records = q1.union(q2)
```

## Row-Level vs Column-Level

```python
# All columns must match for duplicate
q = Q(pd.DataFrame({'a': [1, 1, 2], 'b': [10, 20, 10]}))
q.distinct()  # Keeps all 3 rows (none are exact duplicates)

# Only 'a' must match
q.distinct('a')  # Keeps 2 rows: [1, 10] and [2, 10]
```

## Which Row Is Kept?

**First occurrence** is always kept:

```python
q = Q(pd.DataFrame({
    'id': [1, 2, 3, 2, 1],
    'value': [100, 200, 300, 999, 888]
}))
q.distinct('id')
# Keeps: [1, 100], [2, 200], [3, 300]
# Drops: [2, 999], [1, 888]
```

## Idempotency

✅ **Yes**

## See Also

- [`union()`](union.md) - Concat + distinct
- [`filter()`](filter.md) - Conditional removal
