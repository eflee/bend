# rebase()

Flatten change history by making current state the new base.

## Signature

```python
q.rebase() -> Q
```

## Returns

A new Q object with current state as base and empty change list.

## What It Does

- Current DataFrame becomes new base
- Change history cleared
- Deep copies of other Qs (from merge/concat) are dropped

## Basic Usage

```python
q2 = q.filter(...).assign(...).filter(...).assign(...)
# q2 has 4 changes in history

q3 = q2.rebase()
# q3 has 0 changes, but same data
```

## Use Cases

### 1. Memory Management
```python
large_q = Q(huge_df)
result = q.merge(large_q, on='id')  # Stores deep copy of large_q

# Drop the deep copy
result = result.rebase()  # Much less memory
```

### 2. Performance
```python
# Long pipeline
result = q
for i in range(100):
    result = result.filter(...).assign(...)
# 200 changes in history!

# Flatten
result = result.rebase()  # Fast access, no replay needed
```

### 3. Before Serialization
```python
# Complex history with lambdas
result = q.filter(...).assign(...).merge(...)

# Rebase before pickling (if using standard pickle)
flat = result.rebase()  # No lambdas in history
```

## Trade-offs

**Benefits:**
- ✅ Reduced memory (drops deep copies)
- ✅ Faster operations (no change replay)
- ✅ Simpler state

**Costs:**
- ❌ Cannot `reload()` (no source path preserved)
- ❌ Cannot inspect change history
- ❌ Cannot `refresh()` back to base

## When to Rebase

- After merging large Q objects
- When memory is a concern
- Before long-term storage
- When you no longer need history

## Idempotency

✅ **Yes** - But only in terms of data. History is intentionally lost.

## See Also

- [`reload()`](reload.md), [`refresh()`](refresh.md)
- [`memory_usage()`](memory_usage.md)
