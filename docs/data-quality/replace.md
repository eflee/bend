# `replace()` - Replace Specific Values

Replace specific values in the dataset with new values.

## Signature

```python
def replace(self, to_replace, value=None) -> 'Q'
```

## Parameters

- **`to_replace`**: Either:
  - A scalar value to replace across all columns
  - A dict for column-specific replacements: `{'col': {'old': 'new'}}`
  - A dict for value mapping across all columns: `{'old': 'new'}`
- **`value`** (optional): Replacement value (only if `to_replace` is scalar)

## Returns

A new Q object with values replaced.

## Examples

### Scalar Replacement Across All Columns

```python
# Replace 0 with -1 everywhere
q.replace(0, -1)

# Replace NaN with 0
import numpy as np
q.replace(np.nan, 0)
```

### Column-Specific Replacements

```python
# Replace values in specific columns
q.replace({
    'region': {'CA': 'California', 'NY': 'New York'},
    'status': {'active': 'Active', 'inactive': 'Inactive'}
})
```

### Value Mapping Across All Columns

```python
# Replace these values wherever they appear
q.replace({
    'old': 'legacy',
    'new': 'current',
    0: -1
})
```

### Chaining

```python
result = (q
    .fillna({'status': 'unknown'})
    .replace({'status': {'old': 'inactive', 'new': 'active'}})
    .filter(lambda x: x.status != 'inactive')
)
```

## Use Cases

### Standardize Legacy Values

```python
# Update old codes to new standards
standardized = (data
    .replace({
        'status': {
            'old': 'inactive',
            'new': 'active',
            'pending': 'in_progress'
        },
        'region': {
            'CA': 'California',
            'NY': 'New York',
            'TX': 'Texas'
        }
    })
)
```

### Clean Categorical Data

```python
# Fix inconsistent categorical values
cleaned = (survey
    .replace({
        'response': {
            'yes': 'Yes',
            'YES': 'Yes',
            'y': 'Yes',
            'no': 'No',
            'NO': 'No',
            'n': 'No'
        }
    })
)
```

### Replace Sentinel Values

```python
# Replace sentinel/magic numbers with proper nulls
cleaned = (data
    .replace(-999, np.nan)  # -999 was "missing" sentinel
    .replace(-1, np.nan)    # -1 was "unknown" sentinel
    .fillna(0)              # Now fill proper nulls
)
```

### Status Transitions

```python
# Update status values in bulk
updated = (orders
    .replace({
        'status': {
            'processing': 'in_transit',
            'awaiting_payment': 'pending'
        }
    })
)
```

## Gotchas

### Replace vs fillna

```python
# replace() doesn't work on NaN like fillna does
q.replace(np.nan, 0)  # Works, but...

# This is more explicit for filling nulls:
q.fillna(0)
```

### Column-Specific vs Global

```python
# Be careful with dict syntax

# This replaces in specific columns:
q.replace({'region': {'CA': 'California'}})

# This replaces values globally:
q.replace({'CA': 'California'})  # Affects ALL columns
```

### Partial Replacements

```python
# Unmapped values remain unchanged
q.replace({'status': {'old': 'inactive'}})
# 'new' and other values remain as-is
```

### Type Coercion

```python
# Replacing with different type may cause issues
df = DataFrame({'count': [1, 2, 3]})
q.replace(2, 'two')  # May result in mixed types
```

### No Change If No Matches

```python
# If value doesn't exist, replace has no effect
q.replace(999, 0)  # No rows have 999, no changes made
```

## Performance

- **Efficiency**: O(n*m) where n=rows, m=columns affected
- **Memory**: Creates new DataFrame with replaced values
- **Best Practice**: Combine multiple replacements in one call rather than chaining

```python
# Better:
q.replace({'status': {'old': 'inactive', 'new': 'active'}})

# Worse (two operations):
q.replace({'status': {'old': 'inactive'}}).replace({'status': {'new': 'active'}})
```

## Common Patterns

### Data Normalization

```python
# Normalize categorical values to standard format
normalized = (raw_data
    .replace({
        'country': {
            'US': 'United States',
            'USA': 'United States',
            'United States of America': 'United States'
        }
    })
)
```

### Boolean Conversion

```python
# Convert string booleans to actual booleans
# (Note: this changes type, be careful)
converted = (data
    .replace({'active': {'true': True, 'false': False, '1': True, '0': False}})
)
```

### Redaction/Anonymization

```python
# Replace sensitive values
anonymized = (customer_data
    .replace({
        'email': {email: 'REDACTED' for email in sensitive_emails},
        'phone': {phone: 'XXX-XXX-XXXX' for phone in sensitive_phones}
    })
)
```

### Fix Data Entry Errors

```python
# Correct common typos/errors
corrected = (survey
    .replace({
        'city': {
            'New Yrok': 'New York',
            'Los Angelas': 'Los Angeles',
            'San Fransisco': 'San Francisco'
        }
    })
)
```

## Deterministic

✅ **Yes** - Same input and replacement mapping always produce same output.

## Preserves Flags

- **`deterministic`**: ✅ Inherited from parent Q
- **`reloadable`**: ✅ Inherited from parent Q

## Tracked in History

✅ **Yes** - Stored as `("replace", {mapping})` in change history.

This means:

- `replay()` will re-apply the replacement
- `reload()` will reload data then re-apply the replacement
- `rebase()` will bake the replacement into the base DataFrame

## Comparison with Other Methods

| Operation | Use When | Example |
|-----------|----------|---------|
| `replace()` | Changing specific known values | Replace 'CA' with 'California' |
| `fillna()` | Filling null values | Replace nulls with 0 |
| `assign()` | Computing new values from existing | Calculate full_name from first + last |
| `map()` | Complete row transformation | Restructure entire row |

## See Also

- [`fillna()`](fillna.md) - Fill null values
- [`dropna()`](dropna.md) - Remove rows with nulls
- [`assign()`](../data-manipulation/assign.md) - Add computed columns
- [`filter()`](../data-manipulation/filter.md) - Filter rows by condition
- [Understanding Determinism & Reloadability](../concepts/understanding-determinism-reloadability.md)

