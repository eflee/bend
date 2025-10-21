# unhide()

Unhide previously hidden columns for display.

## Signature

```python
q.unhide(*cols) -> Q
```

## Parameters

- `*cols`: Column names to unhide. If empty, unhides ALL columns.

## Basic Usage

```python
# Unhide specific columns
q2 = q.unhide('id', 'created_at')

# Unhide everything
q2 = q.unhide()
```

## Use Cases

```python
# Hide temporarily, then unhide
q2 = q.hide('debug_field')
# ... do some analysis ...
q3 = q2.unhide('debug_field')  # Show it again
```

## Idempotency

✅ **Yes**

## See Also

- [`hide()`](hide.md)
