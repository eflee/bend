# map()

Transform each row into a completely new structure, replacing all columns.

## Signature

```python
q.map(fn: Callable) -> Q
```

## Parameters

- `fn`: A function that takes a Row and returns:
  - A **dict**: keys become column names, values become column values
  - A **tuple**: creates columns named `c0`, `c1`, `c2`, etc.
  - Any **other value**: creates a single column named `value`

## Returns

A new Q object with completely transformed data. Original columns are **not preserved**.

## Basic Usage

```python
# Return a dict (recommended)
q2 = q.map(lambda x: {
    'full_name': f"{x.first} {x.last}",
    'age': x.age
})

# Return a tuple (creates c0, c1, c2...)
q2 = q.map(lambda x: (x.year, x.month, x.day))

# Return a scalar (creates 'value' column)
q2 = q.map(lambda x: x.price * x.qty)
```

## Use Cases

### 1. Data Restructuring
```python
# Reshape complex data into simpler form
simplified = q.map(lambda x: {
    'customer': x.customer_name,
    'total': x.price * x.quantity,
    'date': x.order_date.strftime('%Y-%m-%d')
})
```

### 2. Column Subset with Transformation
```python
# Extract and transform specific fields
export = q.map(lambda x: {
    'id': x.customer_id,
    'name': x.full_name.upper(),
    'email': x.email.lower(),
    'lifetime_value': round(x.ltv, 2)
})
```

### 3. JSON/API Response Mapping
```python
# Transform database records into API response format
api_response = orders.map(lambda x: {
    'order_id': x.id,
    'customer': {
        'id': x.customer_id,
        'name': x.customer_name
    },
    'items': x.line_items_count,
    'total_usd': f"${x.total:.2f}",
    'status': x.status.upper()
})
```

### 4. Date/Time Decomposition
```python
# Break timestamps into components
time_series = q.map(lambda x: {
    'year': x.timestamp.year,
    'month': x.timestamp.month,
    'day': x.timestamp.day,
    'hour': x.timestamp.hour,
    'weekday': x.timestamp.strftime('%A')
})
```

### 5. Aggregating Multiple Columns
```python
# Combine multiple fields into summary
summary = q.map(lambda x: {
    'id': x.id,
    'total_score': x.math_score + x.english_score + x.science_score,
    'grade': 'A' if x.avg_score >= 90 else 'B' if x.avg_score >= 80 else 'C'
})
```

## When to Use map() vs assign()

| Use `map()` when... | Use `assign()` when... |
|---------------------|------------------------|
| You want to completely restructure the data | You want to add columns to existing data |
| Original columns are not needed | You need to keep original columns |
| Building an export/API response | Doing feature engineering |
| Simplifying complex nested data | Adding derived calculations |

```python
# Use assign() - keeps original columns
q.assign(total=lambda x: x.price * x.qty)
# Result: original columns + total

# Use map() - replaces everything
q.map(lambda x: {'total': x.price * x.qty})
# Result: only 'total' column
```

## Gotchas

### All Original Columns Are Lost

```python
# This loses all original columns!
q2 = q.map(lambda x: {'total': x.price * x.qty})
# q2 only has 'total', no 'price' or 'qty'

# If you need originals, use assign() instead
q2 = q.assign(total=lambda x: x.price * x.qty)
```

### Dict Keys Must Be Strings

```python
# Wrong - numeric keys not supported
q.map(lambda x: {0: x.value1, 1: x.value2})

# Right - use string keys
q.map(lambda x: {'col0': x.value1, 'col1': x.value2})
```

### Tuple Creates Generic Column Names

```python
# Tuple output creates c0, c1, c2...
q2 = q.map(lambda x: (x.year, x.month, x.day))
# Columns: c0, c1, c2

# Use dict for meaningful names
q2 = q.map(lambda x: {'year': x.year, 'month': x.month, 'day': x.day})
```

### Scalar Output Creates 'value' Column

```python
# Single value creates column named 'value'
q2 = q.map(lambda x: x.price * x.qty)
# Column: 'value'

# Use dict for custom name
q2 = q.map(lambda x: {'total': x.price * x.qty})
```

### Inconsistent Keys Across Rows

If your lambda returns different keys for different rows, missing keys become NaN:

```python
# Inconsistent structure
q2 = q.map(lambda x: {
    'name': x.name,
    'discount': x.discount if x.vip else None  # Key still present
})

# Better - consistent keys
q2 = q.map(lambda x: {
    'name': x.name,
    'discount': x.discount if x.vip else 0.0  # Always present
})
```

### Exception Handling

If your lambda raises an exception on any row, the entire operation fails:

```python
# Dangerous
q.map(lambda x: {'ratio': x.a / x.b})  # Fails on b=0

# Safe
q.map(lambda x: {
    'ratio': x.a / x.b if x.b != 0 else 0
})
```

## Complex Transformations

### Nested Data
```python
# Flatten nested structure
q.map(lambda x: {
    'order_id': x.id,
    'customer_id': x.customer['id'],
    'customer_name': x.customer['name'],
    'shipping_city': x.shipping_address['city']
})
```

### Conditional Structure
```python
# Different fields based on type
q.map(lambda x: {
    'id': x.id,
    'amount': x.amount,
    **({'refund_id': x.refund_id} if x.type == 'refund' else {}),
    **({'payment_method': x.method} if x.type == 'payment' else {})
})
```

## Performance Considerations

- `map()` processes rows one at a time (not vectorized)
- Cheaper than `assign()` if you're dropping most columns
- More expensive than `select()` if you just need to drop columns
- For large datasets (>1M rows), consider pandas directly

## Comparison with Pandas

```python
# Pandas - apply with dict return
df = df.apply(lambda row: {
    'name': row['first'] + ' ' + row['last'],
    'age': row['age']
}, axis=1, result_type='expand')

# Bend - cleaner syntax
q = q.map(lambda x: {
    'name': x.first + ' ' + x.last,
    'age': x.age
})
```

## Chaining

```python
result = (q
    .filter(lambda x: x.status == 'active')
    .map(lambda x: {
        'id': x.customer_id,
        'value': x.lifetime_value,
        'tier': 'platinum' if x.lifetime_value > 10000 else 'gold'
    })
    .sort('value', ascending=False)
)
```

## Idempotency

✅ **Yes** - `map()` is fully idempotent. The same transformation will always produce the same results when replayed.

## See Also

- [`assign()`](assign.md) - Add columns while keeping originals
- [`select()`](select.md) - Keep specific columns without transformation
- [`filter()`](filter.md) - Filter before mapping

