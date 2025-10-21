# difference()

Set difference operation: return rows in self that are NOT in other.

## Signature

```python
q.difference(other: Q, deep_copy: bool = True) -> Q
```

## Parameters

- `other`: Another Q object to subtract (must have same columns)
- `deep_copy`: If `True` (default), stores deep copy for reproducibility

## Returns

A new Q object with rows from self that don't appear in other.

## Basic Usage

```python
q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))

result = q1.difference(q2)
# Result: [1] (in q1 but not in q2)
```

## Set Theory

Difference follows mathematical set difference: A - B (or A \ B)

```python
q1 = Q(pd.DataFrame({'x': [1, 2, 3, 4]}))
q2 = Q(pd.DataFrame({'x': [3, 4, 5, 6]}))
result = q1.difference(q2)
# Result: {1, 2}
```

## Row-Level Matching

Rows are matched by comparing **all columns**:

```python
q1 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 20]}))
q2 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 30]}))
result = q1.difference(q2)
# Result: 1 row [2, 20]
# ([1, 10] is in both, [2, 20] is only in q1)
```

## Use Cases

### 1. Finding Dropped Records
```python
last_week = Q(load_csv('active_users_2024_01_15.csv'))
this_week = Q(load_csv('active_users_2024_01_22.csv'))

# Users who became inactive
churned = last_week.difference(this_week)
```

### 2. Data Validation
```python
expected = Q(expected_df)
actual = Q(actual_df)

# Records missing from actual
missing = expected.difference(actual)

# Records that shouldn't be in actual
extra = actual.difference(expected)
```

### 3. Exclusion Lists
```python
all_users = Q(all_df)
blocked = Q(blocked_df)

# All users except blocked ones
allowed = all_users.difference(blocked)
```

### 4. Change Detection
```python
before = Q(snapshot_before_df)
after = Q(snapshot_after_df)

# Deleted records
deleted = before.difference(after)

# New records
added = after.difference(before)
```

## Self-Difference

Difference with self is always empty:

```python
q = Q(pd.DataFrame({'a': [1, 2, 3]}))
result = q.difference(q)
# Result: 0 rows, columns: ['a']
```

## Non-Commutative

Order matters! A - B ≠ B - A

```python
q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))

q1.difference(q2)  # [1]
q2.difference(q1)  # [4]
```

## Empty Results

If all rows in self are also in other:

```python
q1 = Q(pd.DataFrame({'a': [1, 2]}))
q2 = Q(pd.DataFrame({'a': [1, 2, 3, 4]}))
result = q1.difference(q2)
# Result: 0 rows (all of q1's rows are in q2)
```

## Gotchas

### Columns Must Match
```python
q1 = Q(pd.DataFrame({'a': [1, 2]}))
q2 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 20]}))

# May give unexpected results due to column mismatch
```

### Null Handling
```python
q1 = Q(pd.DataFrame({'a': [1, None, 3]}))
q2 = Q(pd.DataFrame({'a': [None, 2]}))

# NaN != NaN, so None rows don't match
result = q1.difference(q2)
# Result: [1, None, 3] (None in q1 ≠ None in q2)
```

### Duplicates in Self
```python
q1 = Q(pd.DataFrame({'a': [1, 1, 2, 2, 3]}))
q2 = Q(pd.DataFrame({'a': [2]}))
result = q1.difference(q2)
# Result: [1, 3] (deduplicated)
# Both copies of '1' are kept as one row
```

## Practical Pattern: Symmetric Difference

To find rows that are in either Q but not both (A Δ B):

```python
only_in_q1 = q1.difference(q2)
only_in_q2 = q2.difference(q1)
symmetric_diff = only_in_q1.union(only_in_q2)
```

## Performance

- Uses pandas `merge()` with indicator internally
- Automatically deduplicates result
- Efficient for large datasets

## Idempotency

✅ **Conditional** - Same as [`concat()`](concat.md)

## See Also

- [`union()`](union.md) - Rows in either (A ∪ B)
- [`intersect()`](intersect.md) - Rows in both (A ∩ B)
- [`filter()`](filter.md) - Conditional row removal

