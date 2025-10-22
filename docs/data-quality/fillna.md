# `fillna()` - Fill Missing Values

Fill null/missing values with specified values.

## Signature

```python
def fillna(self, value) -> 'Q'
```

## Parameters

- **`value`**: Either:
  - A scalar value to fill all nulls
  - A dict mapping column names to fill values

## Returns

A new Q object with nulls filled.

## Examples

### Fill All Nulls with Scalar

```python
# Fill all nulls with 0
q.fillna(0)

# Fill all nulls with empty string
q.fillna('')

# Fill all nulls with NaN (useful for standardization)
import numpy as np
q.fillna(np.nan)
```

### Fill Specific Columns

```python
# Column-specific fill values
q.fillna({
    'age': 0,
    'city': 'Unknown',
    'status': 'pending',
    'score': -1
})
```

### Chaining

```python
result = (q
    .dropna('id')                           # Remove rows without ID
    .fillna({'age': 0, 'city': 'Unknown'})  # Fill other nulls
    .replace({'status': {'old': 'inactive'}})
)
```

## Use Cases

### Default Values for Optional Fields

```python
# Set sensible defaults for optional data
users = (raw_users
    .fillna({
        'age': 0,
        'bio': 'No bio provided',
        'avatar': 'default.png',
        'preferences': '{}'
    })
)
```

### Zero-Filling Numeric Data

```python
# Fill all numeric nulls with 0 for calculations
sales = (orders
    .fillna({
        'discount': 0,
        'tax': 0,
        'shipping': 0
    })
    .assign(total=lambda x: x.subtotal - x.discount + x.tax + x.shipping)
)
```

### Categorical Defaults

```python
# Fill categorical columns with 'Unknown' or 'Other'
categorized = (data
    .fillna({
        'region': 'Unknown',
        'category': 'Other',
        'status': 'Pending'
    })
)
```

### Forward Fill Pattern

```python
# While fillna() doesn't support forward fill directly,
# you can use pandas for complex fills then wrap in Q:
import pandas as pd

df = q.to_df()
df['column'] = df['column'].fillna(method='ffill')
q_filled = Q(df, source_path=q._source_path)
```

## Gotchas

### Type Coercion

```python
# Filling string column with number may coerce types
df = DataFrame({'name': ['Alice', None, 'Charlie']})
q.fillna(0)  # Results in ['Alice', '0', 'Charlie'] or type error
```

### Partial Fills

```python
# Only specified columns in dict are filled
q.fillna({'age': 0})  # Other columns remain unchanged

# To fill all except specified:
# 1. Fill all first, then override specific columns
q.fillna(0).fillna({'name': 'Unknown'})
```

### Empty String vs Null

```python
# fillna() doesn't affect empty strings
df = DataFrame({'a': [None, '', 'value']})
q.fillna('EMPTY')  # Results in ['EMPTY', '', 'value']
```

### No Change If No Nulls

```python
# If column has no nulls, fillna has no effect
df = DataFrame({'a': [1, 2, 3]})
q.fillna({'a': 0})  # Result is identical to input
```

## Performance

- **Efficiency**: O(n*m) where n=rows, m=columns to fill
- **Memory**: Creates new DataFrame with filled values
- **Best Practice**: Fill nulls early to avoid null checks in later operations

## Common Patterns

### Fill Then Validate

```python
result = (q
    .fillna({'age': 0, 'income': 0})
    .filter(lambda x: x.age >= 18)    # Now age is never null
    .filter(lambda x: x.income > 0)   # Can safely compare
)
```

### Different Fills for Different Column Types

```python
# Identify column types and fill accordingly
numeric_fills = {col: 0 for col in ['age', 'salary', 'score']}
string_fills = {col: 'Unknown' for col in ['city', 'region']}
all_fills = {**numeric_fills, **string_fills}

q.fillna(all_fills)
```

### Fill Before Aggregation

```python
# Ensure aggregations don't skip nulls unexpectedly
result = (orders
    .fillna({'discount': 0, 'tax': 0})
    .groupby(
        lambda x: x.customer_id,
        total=lambda g: sum(r.amount - r.discount + r.tax for r in g)
    )
)
```

## Deterministic

✅ **Yes** - Same input and fill values always produce same output.

## Preserves Flags

- **`deterministic`**: ✅ Inherited from parent Q
- **`reloadable`**: ✅ Inherited from parent Q

## Tracked in History

✅ **Yes** - Stored as `("fillna", value)` in change history.

This means:

- `replay()` will re-apply the fill
- `reload()` will reload data then re-apply the fill
- `rebase()` will bake the fill into the base DataFrame

## See Also

- [`dropna()`](dropna.md) - Remove rows with nulls instead of filling
- [`replace()`](replace.md) - Replace specific values
- [`assign()`](../data-manipulation/assign.md) - Add computed columns
- [Understanding Determinism & Reloadability](../concepts/understanding-determinism-reloadability.md)

