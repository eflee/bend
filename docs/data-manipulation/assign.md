# assign()

Add new computed columns to your DataFrame based on existing columns.

## Signature

```python
q.assign(**newcols) -> Q
```

## Parameters

- `**newcols`: Keyword arguments where keys are new column names and values are lambda functions that take a Row and return the computed value

## Returns

A new Q object with the additional columns. Original columns are preserved.

## Basic Usage

```python
# Add a single computed column
q2 = q.assign(total=lambda x: x.price * x.qty)

# Add multiple columns at once
q2 = q.assign(
    total=lambda x: x.price * x.qty,
    tax=lambda x: x.total * 0.08,
    grand_total=lambda x: x.total + x.tax
)

# Reference newly created columns in same call
q2 = q.assign(
    full_name=lambda x: f"{x.first} {x.last}",
    name_length=lambda x: len(x.full_name)
)
```

## Use Cases

### 1. Business Calculations

```python
# E-commerce order totals
orders = q.assign(
    subtotal=lambda x: x.price * x.quantity,
    tax=lambda x: x.subtotal * 0.0875,
    shipping=lambda x: 10.0 if x.subtotal < 50 else 0.0,
    total=lambda x: x.subtotal + x.tax + x.shipping
)
```

### 2. Data Enrichment

```python
# Add derived fields for analysis
customers = q.assign(
    age_group=lambda x: 'senior' if x.age >= 65 else 'adult' if x.age >= 18 else 'minor',
    lifetime_value_tier=lambda x: 'platinum' if x.ltv > 10000 else 'gold' if x.ltv > 5000 else 'silver',
    days_since_signup=lambda x: (datetime.now() - x.signup_date).days
)
```

### 3. Feature Engineering

```python
# ML feature creation
dataset = q.assign(
    price_per_sqft=lambda x: x.price / x.square_feet,
    bed_bath_ratio=lambda x: x.bedrooms / x.bathrooms if x.bathrooms > 0 else 0,
    is_luxury=lambda x: 1 if x.price > 1000000 else 0
)
```

### 4. String Manipulation

```python
# Text processing
users = q.assign(
    email_domain=lambda x: x.email.split('@')[1] if '@' in x.email else '',
    username=lambda x: x.email.split('@')[0],
    display_name=lambda x: x.full_name.title()
)
```

## Gotchas

### Column Order Dependency

Columns are evaluated **left to right**, so you can reference earlier columns in the same `assign()` call:

```python
# This works - total is computed before tax
q.assign(
    total=lambda x: x.price * x.qty,
    tax=lambda x: x.total * 0.08  # Can reference total
)

# This fails - tax doesn't exist yet when computing total
q.assign(
    total=lambda x: x.price * x.qty + x.tax,  # Error: no attribute 'tax'
    tax=lambda x: x.price * 0.08
)
```

### Overwriting Existing Columns

If you assign to a column name that already exists, the original is preserved and you can reference it:

```python
# Double the price
q.assign(price=lambda x: x.price * 2)

# Transform based on original value
q.assign(status=lambda x: 'ACTIVE' if x.status == 'pending' else x.status)
```

### Error Handling

If your lambda raises an exception on any row, the entire operation fails:

```python
# Dangerous - division by zero
q.assign(ratio=lambda x: x.numerator / x.denominator)

# Safe - handle edge cases
q.assign(ratio=lambda x: x.numerator / x.denominator if x.denominator != 0 else 0)
```

### Null/Missing Values

Be careful with null values - they can cause exceptions:

```python
# Dangerous if age can be None
q.assign(is_adult=lambda x: x.age >= 18)

# Safe
q.assign(is_adult=lambda x: x.age >= 18 if x.age is not None else False)
```

## Performance Considerations

- `assign()` processes rows one at a time in Python
- For very large datasets (>1M rows), consider using pandas directly via `q.to_df()`
- Column evaluation happens during `assign()`, not lazily
- Each `assign()` call is a separate change in the history

## Comparison with Pandas

Bend's `assign()` aligns with pandas `DataFrame.assign()`:

```python
# Pandas
df = df.assign(total=df['price'] * df['qty'])

# Bend (more ergonomic with Row access)
q = q.assign(total=lambda x: x.price * x.qty)
```

## Chaining

`assign()` returns a Q, so it chains naturally:

```python
result = (q
    .assign(total=lambda x: x.price * x.qty)
    .filter(lambda x: x.total > 100)
    .assign(discount=lambda x: x.total * 0.1)
    .sort('total', ascending=False)
)
```

## Idempotency

✅ **Yes** - `assign()` is fully idempotent. The same `assign()` call will always produce the same result when replayed via `replay()` or `reload()`.

## See Also

- [`map()`](map.md) - For completely restructuring rows
- [`filter()`](filter.md) - For filtering after assignment
- [`select()`](select.md) - For keeping only specific columns

