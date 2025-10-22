# Lifecycle

Managing Q state, history, and memory.

## Methods

### [reload()](reload.md)

Reload from source files (deep/recursive).

**When to use**: Source data updated, need fresh data

```python
q.reload()  # Reloads CSV and replays all changes
```

### [replay()](replay.md)

Re-apply changes to in-memory base.

**When to use**: Verify idempotency, reset after manual changes

```python
q.replay()  # Recomputes from base + changes
```

### [rebase()](rebase.md)

Flatten history, drop deep copies.

**When to use**: Memory management, before serialization

```python
q.rebase()  # Current state becomes new base
```

### [memory_usage()](memory_usage.md)

Get memory breakdown.

**When to use**: Monitoring, debugging memory issues

```python
usage = q.memory_usage()
print(f"Total: {usage['total_mb']} MB")
```

## When to Use What

| Need to... | Use |
|------------|-----|
| Get fresh data from disk | `reload()` |
| Verify pipeline works | `replay()` |
| Reduce memory usage | `rebase()` |
| Monitor memory | `memory_usage()` |
