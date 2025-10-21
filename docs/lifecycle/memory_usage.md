# memory_usage()

Get detailed memory usage breakdown.

## Signature

```python
q.memory_usage(deep: bool = True) -> dict
```

## Parameters

- `deep`: Introspect data for accurate memory (default: `True`)

## Returns

Dictionary with memory breakdown:
```python
{
    'current_df': 12345,        # bytes
    'base_df': 12345,           # bytes
    'changes': 5,               # count
    'changes_memory': 98765,    # bytes (includes deep copied Qs)
    'total': 123455,            # bytes
    'total_mb': 0.12            # megabytes
}
```

## Basic Usage

```python
usage = q.memory_usage()
print(f"Total memory: {usage['total_mb']} MB")
print(f"Changes tracked: {usage['changes']}")
```

## Use Cases

### 1. Monitor Memory Growth
```python
q1 = Q(df)
print(q1.memory_usage()['total_mb'])  # 10 MB

q2 = q1.merge(large_q, on='id')
print(q2.memory_usage()['total_mb'])  # 1010 MB (includes large_q deep copy!)

# Flatten to reduce
q3 = q2.rebase()
print(q3.memory_usage()['total_mb'])  # 10 MB
```

### 2. Debug Memory Issues
```python
usage = q.memory_usage()
if usage['changes_memory'] > usage['current_df']:
    print("History uses more memory than data! Consider rebase()")
```

## Deep Copied Qs

For multi-Q operations (merge, concat), `changes_memory` includes the full memory of deep-copied Q objects:

```python
q1 = Q(small_df)  # 1 MB
q2 = Q(huge_df)   # 1000 MB

combined = q1.concat(q2)
usage = combined.memory_usage()
# current_df: ~1001 MB
# changes_memory: ~1000 MB (deep copy of q2)
# total: ~2001 MB
```

## See Also

- [`rebase()`](rebase.md) - Reduce memory
