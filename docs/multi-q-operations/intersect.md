# intersect()

Set intersection operation: return only rows that appear in both Q objects.

## Signature

```python
q.intersect(other: Q, deep_copy: bool = True) -> Q
```

## Parameters

- `other`: Another Q object to intersect with (must have same columns)
- `deep_copy`: If `True` (default), stores deep copy for reproducibility

## Returns

A new Q object with rows common to both Q objects.

## Basic Usage

```python
q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))

result = q1.intersect(q2)
# Result: [2, 3] (rows present in both)
```

## Set Theory

Intersection follows mathematical set intersection: A ∩ B

```python
q1 = Q(pd.DataFrame({'x': [1, 2, 3, 4]}))
q2 = Q(pd.DataFrame({'x': [3, 4, 5, 6]}))
result = q1.intersect(q2)
# Result: {3, 4}
```

## Row-Level Matching

Rows are matched by comparing **all columns**:

```python
q1 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 20]}))
q2 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 30]}))
result = q1.intersect(q2)
# Result: 1 row [1, 10] (only exact match)
# Row [2, 20] ≠ [2, 30]
```

## Use Cases

### 1. Finding Common Records

```python
active_last_month = Q(load_csv('active_march.csv'))
active_this_month = Q(load_csv('active_april.csv'))

# Customers active in both months
retained = active_last_month.intersect(active_this_month)
```

### 2. Data Validation

```python
expected = Q(expected_df)
actual = Q(actual_df)

# Find matching records
matches = expected.intersect(actual)

# Find mismatches
missing_from_actual = expected.difference(actual)
extra_in_actual = actual.difference(expected)
```

### 3. Set Membership

```python
whitelist = Q(approved_df)
submissions = Q(submission_df)

# Only approved submissions
approved = submissions.intersect(whitelist)
```

## Self-Intersection

Intersecting with self returns all unique rows:

```python
q = Q(pd.DataFrame({'a': [1, 2, 2, 3, 3, 3]}))
unique = q.intersect(q)
# Result: [1, 2, 3] (deduplicated)

# Equivalent to distinct()
unique = q.distinct()
```

## Empty Results

If no rows match, returns empty Q with same columns:

```python
q1 = Q(pd.DataFrame({'a': [1, 2]}))
q2 = Q(pd.DataFrame({'a': [3, 4]}))
result = q1.intersect(q2)
# Result: 0 rows, columns: ['a']
```

## Gotchas

### Columns Must Match

```python
q1 = Q(pd.DataFrame({'a': [1, 2]}))
q2 = Q(pd.DataFrame({'a': [1, 2], 'b': [10, 20]}))

# Works but may give unexpected results
# Pandas will try to match on common columns
```

### Null Handling

```python
q1 = Q(pd.DataFrame({'a': [1, None, 3]}))
q2 = Q(pd.DataFrame({'a': [None, 2, 3]}))

# NaN != NaN in pandas, so None rows don't match
result = q1.intersect(q2)
# Result: [3] only
```

## Performance

- Uses pandas `merge()` internally (efficient)
- Automatically deduplicates result
- For large datasets, consider pandas directly

## Idempotency

✅ **Conditional** - Same as [`concat()`](concat.md)

## See Also

- [`union()`](union.md) - Rows in either (A ∪ B)
- [`difference()`](difference.md) - Rows in self but not other (A - B)
- [`distinct()`](distinct.md) - Remove duplicates

