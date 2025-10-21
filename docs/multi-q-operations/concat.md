# concat()

Concatenate (vertically stack) rows from two Q objects.

## Signature

```python
q.concat(other: Q, deep_copy: bool = True) -> Q
```

## Parameters

- `other`: Another Q object to concatenate
- `deep_copy`: If `True` (default), stores a deep copy of `other` for full reproducibility. If `False`, stores a reference (faster but non-deterministic)

## Returns

A new Q object containing rows from both Q objects, stacked vertically.

## Basic Usage

```python
q1 = Q(df1)  # 100 rows
q2 = Q(df2)  # 50 rows
combined = q1.concat(q2)  # 150 rows

# Columns are aligned by name
# Missing columns filled with NaN
```

## Use Cases

### 1. Combining Time Series Data
```python
jan_sales = load_csv('sales_jan.csv')
feb_sales = load_csv('sales_feb.csv')
mar_sales = load_csv('sales_mar.csv')

q1 = Q(jan_sales).concat(Q(feb_sales)).concat(Q(mar_sales))
```

### 2. Merging Multiple Sources
```python
# Combine data from different sources
online = load_csv('online_orders.csv')
retail = load_csv('retail_orders.csv')
phone = load_csv('phone_orders.csv')

all_orders = (Q(online)
    .assign(source='online')
    .concat(Q(retail).assign(source='retail'))
    .concat(Q(phone).assign(source='phone'))
)
```

### 3. Appending New Data
```python
existing = Q(existing_df)
new_batch = Q(new_df)

updated = existing.concat(new_batch)
```

## Column Handling

### Matching Columns
```python
# Both have same columns
q1 = Q(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
q2 = Q(pd.DataFrame({'a': [5, 6], 'b': [7, 8]}))
result = q1.concat(q2)
# Result: 4 rows, columns: a, b
```

### Mismatched Columns (Superset/Subset)
```python
q1 = Q(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
q2 = Q(pd.DataFrame({'a': [5, 6], 'c': [9, 10]}))
result = q1.concat(q2)
# Result: 4 rows, columns: a, b, c
# q2 rows have NaN for 'b', q1 rows have NaN for 'c'
```

### Completely Different Columns
```python
q1 = Q(pd.DataFrame({'x': [1, 2]}))
q2 = Q(pd.DataFrame({'y': [3, 4]}))
result = q1.concat(q2)
# Result: 4 rows, columns: x, y (all NaN except originals)
```

## Reproducibility & Deep Copy

### Deep Copy (Default)
```python
q1 = Q(df1)
q2 = Q(df2)

# Stores deep copy of q2
combined = q1.concat(q2, deep_copy=True)

# Fully deterministic
print(combined.deterministic)  # True

# Changes to q2 don't affect combined
q2_modified = q2.filter(lambda x: x.value > 100)
combined.reload()  # Still uses original q2 state
```

### Reference Mode (Performance)
```python
large_q = Q(huge_df)  # 10M rows

# Faster, no deep copy
combined = q1.concat(large_q, deep_copy=False)

# Non-deterministic
print(combined.deterministic)  # False

# Use rebase() to make self-contained
combined = combined.rebase()  # Now independent
```

## Self-Concatenation (Duplicate Rows)

```python
q = Q(df)  # 100 rows

# Duplicate all rows
doubled = q.concat(q)  # 200 rows (exact duplicates)

# Self-concat automatically deep copies to avoid circular references
```

## Row Index Behavior

Concatenation **ignores original row indices** and creates a new sequential index:

```python
q1 = Q(pd.DataFrame({'a': [1, 2]}, index=[10, 20]))
q2 = Q(pd.DataFrame({'a': [3, 4]}, index=[30, 40]))
result = q1.concat(q2)
# Result has sequential index: 0, 1, 2, 3 (not 10, 20, 30, 40)
```

## Gotchas

### 1. Duplicates Are Kept
```python
q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))
result = q1.concat(q2)
# Result has 6 rows: [1, 2, 3, 2, 3, 4]
# Values 2 and 3 appear twice

# Use distinct() to remove duplicates
result = q1.concat(q2).distinct()  # [1, 2, 3, 4]

# Or use union() which automatically deduplicates
result = q1.union(q2)  # [1, 2, 3, 4]
```

### 2. Column Type Mismatches
```python
q1 = Q(pd.DataFrame({'a': [1, 2]}))  # int
q2 = Q(pd.DataFrame({'a': ['x', 'y']}))  # string
result = q1.concat(q2)
# Pandas will try to find a common type (usually object)
# May cause unexpected behavior in later operations
```

### 3. Memory with Deep Copy
```python
# Each concat stores a full deep copy
result = q1.concat(q2).concat(q3).concat(q4)
# History contains deep copies of q2, q3, q4
# Can consume significant memory

# Solution: rebase() after concatenating
result = q1.concat(q2).concat(q3).concat(q4).rebase()
# Flattens history, drops deep copies
```

### 4. Order Matters
```python
# These produce different results
a = q1.concat(q2)  # q1 rows first, then q2 rows
b = q2.concat(q1)  # q2 rows first, then q1 rows
```

## Comparison with Union

| `concat()` | `union()` |
|------------|-----------|
| Keeps duplicates | Removes duplicates |
| Faster | Slower (runs distinct()) |
| Simple stacking | Set operation |
| Use for combining different data | Use for combining similar data |

```python
# concat keeps duplicates
q1.concat(q2)  # [1, 2, 3, 2, 3, 4]

# union removes duplicates
q1.union(q2)   # [1, 2, 3, 4]
```

## Chaining

```python
result = (Q(jan_df)
    .concat(Q(feb_df))
    .concat(Q(mar_df))
    .filter(lambda x: x.amount > 100)
    .distinct('customer_id')
)
```

## Performance Considerations

- Concatenation is fast (uses pandas `pd.concat()`)
- Deep copy overhead proportional to size of `other`
- For many concatenations, consider collecting into list and doing single concat:

```python
# Less efficient
result = q1
for q in [q2, q3, q4, q5]:
    result = result.concat(q)

# More efficient (but loses Bend's change tracking)
dfs = [q1.to_df()] + [q.to_df() for q in [q2, q3, q4, q5]]
result = Q(pd.concat(dfs, ignore_index=True))
```

## Idempotency

✅ **Conditional**
- **Yes** if both Q objects are deterministic and `deep_copy=True` (default)
- **No** if `deep_copy=False`

Check with `q.deterministic` property.

## See Also

- [`union()`](union.md) - Concat with automatic deduplication
- [`merge()`](merge.md) - Horizontal joining (columns)
- [`distinct()`](distinct.md) - Remove duplicates after concat

