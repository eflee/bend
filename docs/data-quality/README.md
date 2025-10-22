# Data Quality Operations

Methods for cleaning and handling missing or incorrect data.

## Available Methods

### [`dropna()`](dropna.md)
Remove rows with null/missing values.

### [`fillna()`](fillna.md)
Fill missing values with specified values.

### [`replace()`](replace.md)
Replace specific values in the dataset.

## Common Use Cases

### Removing Incomplete Records
```python
# Remove rows missing critical fields
clean = q.dropna('email', 'customer_id')
```

### Filling Missing Values
```python
# Fill nulls with sensible defaults
filled = q.fillna({
    'age': 0,
    'city': 'Unknown',
    'status': 'pending'
})
```

### Standardizing Values
```python
# Replace legacy values with standardized ones
standardized = q.replace({
    'status': {'old': 'inactive', 'new': 'active'},
    'region': {'CA': 'California', 'NY': 'New York'}
})
```

### Complete Cleaning Pipeline
```python
result = (q
    .dropna('email')                    # Remove rows without email
    .fillna({'age': 0, 'city': 'Unknown'})  # Fill other nulls
    .replace({'status': {'old': 'inactive'}})  # Standardize values
    .filter(lambda x: 0 <= x.age <= 120)    # Validate ranges
)
```

## See Also

- [Data Manipulation](../data-manipulation/) - Transform and filter data
- [Row Operations](../row-operations/) - Select and order rows
- [Understanding Determinism & Reloadability](../concepts/understanding-determinism-reloadability.md) - How these operations affect flags

