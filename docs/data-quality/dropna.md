# `dropna()` - Remove Rows with Null Values

Remove rows containing null/missing values.

## Signature

```python
def dropna(self, *cols, how='any') -> 'Q'
```

## Parameters

- **`*cols`** (optional): Column name(s) to check for nulls. If not specified, checks all columns.
- **`how`** (str): Either `'any'` (default) or `'all'`:
  - `'any'`: Drop row if ANY specified column has a null
  - `'all'`: Drop row if ALL specified columns are null

## Returns

A new Q object with rows containing nulls removed.

## Examples

### Drop Rows with Any Null

```python
# Remove rows with null in any column
q.dropna()

# Remove rows with null in specific column
q.dropna('email')

# Remove rows with null in any of these columns
q.dropna('email', 'phone')
```

### Drop Only When All Are Null

```python
# Only drop rows where BOTH email and phone are null
q.dropna('email', 'phone', how='all')
```

### Chaining

```python
result = (q
    .dropna('customer_id')  # Must have customer_id
    .fillna({'age': 0})     # Fill other nulls
    .filter(lambda x: x.age >= 18)
)
```

## Use Cases

### Remove Incomplete Records

```python
# Remove rows missing critical fields
clean = orders.dropna('order_id', 'customer_id', 'total')
```

### Conditional Cleaning

```python
# Remove rows only if key contact info is completely missing
contacts = (customers
    .dropna('email', 'phone', how='all')  # Must have at least one
    .dropna('name')                       # Name is required
)
```

### Data Quality Pipeline

```python
validated = (raw_data
    .dropna('id')                  # ID is required
    .fillna({'status': 'pending'}) # Fill optional fields
    .filter(lambda x: x.id > 0)    # Additional validation
)
```

## Gotchas

### All Columns vs Specific Columns

```python
df = DataFrame({'a': [1, None], 'b': [2, 3]})
q = Q(df)

# This removes rows with null in ANY column (only row 0 remains)
q.dropna()

# This only checks column 'a' (still removes row 1)
q.dropna('a')
```

### String 'None' vs Actual None

```python
# dropna() only removes pandas null/NaN, not string 'None'
df = DataFrame({'a': [1, 'None', 3]})
q.dropna('a')  # Keeps all rows ('None' is not null)

# To remove string 'None':
q.filter(lambda x: x.a != 'None')
```

### Empty Result

```python
# If all rows have nulls, result is empty
df = DataFrame({'a': [None, None], 'b': [None, None]})
q = Q(df)
result = q.dropna()  # len(result) == 0
```

## Implementation

`dropna()` is a **wrapper around `filter()`**, building the appropriate lambda and delegating to the existing filter mechanism. This keeps the codebase DRY and leverages tested functionality.

```python
# Internally, this:
q.dropna('email')

# Becomes:
q.filter(lambda x: pd.notna(x.email))
```

## Performance

- **Efficiency**: O(n) where n is number of rows
- **Memory**: Creates new Q with filtered DataFrame
- **Best Practice**: Drop rows early in pipeline to reduce subsequent processing

## Deterministic

✅ **Yes** - Same input always produces same output.

## Preserves Flags

- **`deterministic`**: ✅ Inherited from parent Q
- **`reloadable`**: ✅ Inherited from parent Q

## See Also

- [`fillna()`](fillna.md) - Fill nulls instead of dropping
- [`filter()`](../data-manipulation/filter.md) - General row filtering
- [`replace()`](replace.md) - Replace specific values

