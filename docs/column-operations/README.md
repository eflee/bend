# Column Operations

Managing, renaming, and controlling column visibility.

## Methods

### [drop()](drop.md) - Remove columns permanently
### [select()](select.md) - Keep only specified columns  
### [rename()](rename.md) - Rename columns
### [hide()](hide.md) - Hide from display only
### [unhide()](unhide.md) - Unhide columns

## Structural vs Display

| Operation | Type | Affects Data | Use Case |
|-----------|------|--------------|----------|
| `drop()` | Structural | ✅ Yes | Remove unwanted columns |
| `select()` | Structural | ✅ Yes | Keep only needed columns |
| `rename()` | Structural | ✅ Yes | Fix column names |
| `hide()` | Display | ❌ No | Clean up display |
| `unhide()` | Display | ❌ No | Show hidden columns |

## Quick Examples

```python
# Remove columns
q.drop('id', 'temp')

# Keep specific columns
q.select('name', 'email', 'age')

# Rename
q.rename(customer_id='cust_id')

# Hide from display (still usable!)
q.hide('debug_field').filter(lambda x: x.debug_field > 0)
```
