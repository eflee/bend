# filter()

Filter rows based on a predicate function, keeping only rows where the condition is True.

## Signature

```python
q.filter(fn: Callable) -> Q
```

## Parameters

- `fn`: A function that takes a Row and returns a boolean (True to keep the row, False to discard it)

## Returns

A new Q object containing only the rows where `fn(row)` returned True.

## Basic Usage

```python
# Simple equality filter
q2 = q.filter(lambda x: x.region == 'CA')

# Numeric comparison
q2 = q.filter(lambda x: x.age >= 18)

# Multiple conditions with 'and'
q2 = q.filter(lambda x: x.age >= 18 and x.status == 'active')

# Multiple conditions with 'or'
q2 = q.filter(lambda x: x.priority == 'high' or x.urgent == True)
```

## Use Cases

### 1. Data Quality Filtering

```python
# Remove invalid rows
clean_data = (q
    .filter(lambda x: x.age is not None)
    .filter(lambda x: x.age >= 0 and x.age <= 120)
    .filter(lambda x: '@' in x.email)
)
```

### 2. Business Logic Filtering

```python
# Active high-value customers
targets = q.filter(lambda x: 
    x.status == 'active' and 
    x.lifetime_value > 10000 and
    x.last_purchase_days < 90
)
```

### 3. Date Range Filtering

```python
from datetime import datetime, timedelta

# Recent orders
recent = q.filter(lambda x: 
    x.order_date >= datetime.now() - timedelta(days=30)
)
```

### 4. String Pattern Matching

```python
# Gmail users
gmail_users = q.filter(lambda x: x.email.endswith('@gmail.com'))

# Name contains 'smith' (case-insensitive)
smiths = q.filter(lambda x: 'smith' in x.last_name.lower())
```

### 5. Complex Conditions

```python
# Sophisticated filtering logic
qualified = q.filter(lambda x:
    (x.credit_score > 700 or x.income > 100000) and
    x.bankruptcy_flag == False and
    x.employment_status in ['full-time', 'self-employed'] and
    len(x.late_payments) == 0
)
```

## Gotchas

### Exception Handling

If your lambda raises an exception, that row is **silently excluded** (treated as False):

```python
# This won't crash, but rows with None ages will be filtered out
q.filter(lambda x: x.age > 18)

# Better - explicit None handling
q.filter(lambda x: x.age is not None and x.age > 18)
```

### Null/None Values

Be careful with null values in comparisons:

```python
# Dangerous - None > 0 raises TypeError, row gets excluded
q.filter(lambda x: x.value > 0)

# Safe
q.filter(lambda x: x.value is not None and x.value > 0)

# Or use a default
q.filter(lambda x: (x.value or 0) > 0)
```

### String Methods on None

```python
# Dangerous - AttributeError if email is None
q.filter(lambda x: x.email.endswith('@company.com'))

# Safe
q.filter(lambda x: x.email and x.email.endswith('@company.com'))
```

### Boolean Traps

```python
# Dangerous - only keeps rows where paid is explicitly True
# Rows where paid is None or missing are excluded
q.filter(lambda x: x.paid)

# Explicit is better
q.filter(lambda x: x.paid == True)
```

### Filter Order Matters

Filters are applied in sequence, so order can affect performance:

```python
# Inefficient - checks expensive condition on all rows first
result = (q
    .filter(lambda x: expensive_api_call(x.id))  # Slow!
    .filter(lambda x: x.region == 'CA')  # Fast, but too late
)

# Efficient - reduce dataset first
result = (q
    .filter(lambda x: x.region == 'CA')  # Fast filter first
    .filter(lambda x: expensive_api_call(x.id))  # Only on CA rows
)
```

## Multiple Filters vs. Single Complex Filter

These are equivalent:

```python
# Multiple filters (more readable)
result = (q
    .filter(lambda x: x.age >= 18)
    .filter(lambda x: x.region == 'CA')
    .filter(lambda x: x.active == True)
)

# Single complex filter (more efficient)
result = q.filter(lambda x: 
    x.age >= 18 and 
    x.region == 'CA' and 
    x.active == True
)
```

**Trade-off**: Multiple filters are more readable and debuggable, but create more change history entries. Single complex filters are more efficient but harder to debug.

## Debugging Filters

To see why rows are being filtered out, use intermediate steps:

```python
print(f"Starting rows: {q.rows}")

q2 = q.filter(lambda x: x.age >= 18)
print(f"After age filter: {q2.rows}")

q3 = q2.filter(lambda x: x.region == 'CA')
print(f"After region filter: {q3.rows}")
```

Or use `assign()` to add a debug column:

```python
# Add a column showing which condition failed
debug = q.assign(
    passes_age=lambda x: x.age >= 18,
    passes_region=lambda x: x.region == 'CA',
    passes_status=lambda x: x.status == 'active'
)
debug.show()
```

## Performance Considerations

- Each row is evaluated individually in Python (not vectorized)
- For very large datasets (>1M rows), consider using pandas directly
- Filtering is not lazy - it happens immediately when called
- Exception handling adds overhead (avoid exceptions in hot paths)

## Comparison with Pandas

```python
# Pandas (verbose)
df = df[df['age'] >= 18]
df = df[df['region'] == 'CA']

# Bend (cleaner with method chaining)
q = q.filter(lambda x: x.age >= 18).filter(lambda x: x.region == 'CA')
```

## Chaining

```python
result = (q
    .filter(lambda x: x.age >= 18)
    .assign(total=lambda x: x.price * x.qty)
    .filter(lambda x: x.total > 100)
    .sort('total', ascending=False)
)
```

## Idempotency

✅ **Yes** - `filter()` is fully idempotent. The same filter will always produce the same results when replayed.

## See Also

- [`assign()`](assign.md) - Add columns before filtering
- [`distinct()`](distinct.md) - Remove duplicates after filtering
- [`select()`](select.md) - Keep only specific columns

