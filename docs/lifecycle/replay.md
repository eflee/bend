# replay()

Re-apply all tracked changes to the in-memory base DataFrame.

## Signature

```python
q.replay() -> Q
```

## Returns

A new Q object with changes re-applied to base (no disk access).

## What It Does

Recomputes current state from `base + changes` without loading from disk.

## Basic Usage

```python
q = Q(df)
q2 = q.filter(lambda x: x.active).assign(score=lambda x: x.value * 2)

# Re-apply all changes to base
q3 = q2.replay()  # Same as q2
```

## Use Cases

### 1. Verify Idempotency
```python
q2 = q.filter(...).assign(...)
q3 = q2.replay()

# q2 and q3 should be identical
assert q2.to_df().equals(q3.to_df())
```

### 2. After Manual DataFrame Manipulation
```python
# Don't do this, but if you did...
q._df['new_col'] = 123  # Manual mutation (bad!)

# Refresh to recompute proper state
q = q.replay()  # Back to correct state
```

## replay() vs reload()

| `replay()` | `reload()` |
|-------------|------------|
| Uses in-memory base | Reads from disk |
| Fast | Slower (disk I/O) |
| Always works | Requires source_path |
| Doesn't get new data | Gets updated data |

## Idempotency

✅ **Yes**

## See Also

- [`reload()`](reload.md) - Reload from disk
- [`rebase()`](rebase.md) - Flatten history
