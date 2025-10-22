# Data Manipulation

Core operations for transforming your data.

## Methods

### [assign()](assign.md)

Add new computed columns based on existing columns.

**When to use**: Adding derived fields, calculations, feature engineering

```python
q.assign(total=lambda x: x.price * x.qty, tax=lambda x: x.total * 0.08)
```

### [filter()](filter.md)

Keep only rows that match a condition.

**When to use**: Data quality, business logic filtering, subsetting

```python
q.filter(lambda x: x.age >= 18 and x.status == 'active')
```

### [map()](map.md)

Transform each row into a completely new structure.

**When to use**: Restructuring data, building API responses, simplifying complex data

```python
q.map(lambda x: {'name': f"{x.first} {x.last}", 'age': x.age})
```

## Comparison

| Operation | Keeps Original Columns | Use Case |
|-----------|----------------------|----------|
| `assign()` | ✅ Yes | Add calculations |
| `filter()` | ✅ Yes | Remove rows |
| `map()` | ❌ No | Complete restructure |

## Common Patterns

### ETL Chain

```python
result = (q
    .filter(lambda x: x.valid)
    .assign(normalized=lambda x: x.value / x.total)
    .map(lambda x: {'id': x.id, 'score': x.normalized})
)
```

### Data Quality

```python
clean = (q
    .filter(lambda x: x.email and '@' in x.email)
    .filter(lambda x: 0 <= x.age <= 120)
    .assign(age_group=lambda x: 'senior' if x.age >= 65 else 'adult')
)
```
