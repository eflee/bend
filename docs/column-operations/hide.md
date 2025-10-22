# hide()

Hide columns from display (does NOT remove them).

## Signature

```python
q.hide(*cols) -> Q
```

## Parameters

- `*cols`: Column names to hide from display

## Returns

A new Q object with specified columns hidden from print/show/repr.

## Basic Usage

```python
# Hide sensitive columns
q2 = q.hide('ssn', 'credit_card')

# Still usable in operations!
q3 = q2.filter(lambda x: x.ssn.startswith('123'))  # Works!
```

## DISPLAY-ONLY

**Critical**: `hide()` is a **display-only** operation. Hidden columns:

- ✅ ARE included in `to_df()`
- ✅ ARE included in `dump()`
- ✅ CAN be used in `assign()`, `filter()`, etc.
- ❌ ARE NOT visible in `print(q)`, `q.show()`, `repr(q)`

## Use Cases

### 1. Hide Clutter While Exploring

```python
# Hide IDs and timestamps to see data better
q.hide('id', 'customer_id', 'product_id', 'created_at', 'updated_at')
```

### 2. Presentations/Demos

```python
# Hide internal fields
demo = q.hide('internal_status', 'debug_flag', 'processing_time_ms')
```

### 3. Keep But Don't Show

```python
# Hide intermediate calculation columns
result = (q
    .assign(temp_calc=lambda x: x.a * x.b)
    .assign(final=lambda x: x.temp_calc + x.c)
    .hide('temp_calc')  # Hide but keep for further use
)
```

## hide() vs drop()

| `hide()` | `drop()` |
|----------|----------|
| Display-only | Removes column |
| Can still use in operations | Cannot use after drop |
| Not in change history | Tracked in history |
| Doesn't affect `to_df()`/`dump()` | Affects everything |

```python
# hide() - column still exists
q2 = q.hide('cost')
q3 = q2.assign(profit=lambda x: x.revenue - x.cost)  # ✅ Works!
q3.to_df()  # ✅ Includes 'cost'

# drop() - column is gone
q2 = q.drop('cost')
q3 = q2.assign(profit=lambda x: x.revenue - x.cost)  # ❌ Error!
q3.to_df()  # ❌ No 'cost' column
```

## Gotchas

### Hidden Columns in Export

```python
q2 = q.hide('password')
q2.dump('export.csv')  # ⚠️ 'password' IS in the file!
```

Use `drop()` instead for true removal:

```python
q2 = q.drop('password')
q2.dump('export.csv')  # ✅ 'password' is NOT in the file
```

### Non-Existent Columns

```python
q.hide('nonexistent')  # OK, no error
```

## Chaining

```python
result = (q
    .hide('id', 'created_at')  # Hide clutter
    .filter(lambda x: x.active)
    .assign(score=lambda x: x.value / x.count)
)
```

## Idempotency

✅ **Yes** - But not tracked in change history (display preference, not data transformation).

## See Also

- [`unhide()`](unhide.md) - Reverse hide
- [`drop()`](drop.md) - Actually remove columns
- [`select()`](select.md) - Keep only specific columns
