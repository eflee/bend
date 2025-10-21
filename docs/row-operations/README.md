# Row Operations

Selecting, ordering, and sampling rows.

## Methods

### [head()](head.md) - First n rows
### [tail()](tail.md) - Last n rows
### [sample()](sample.md) - Random sample
### [sort()](sort.md) - Sort by columns
### [distinct()](distinct.md) - Remove duplicates

## Quick Reference

```python
q.head(10)                          # First 10
q.tail(10)                          # Last 10
q.sample(100)                       # 100 random (non-deterministic)
q.sample(100, random_state=42)      # 100 random (deterministic)
q.sort('date')                      # Sort ascending
q.sort('price', ascending=False)    # Sort descending
q.distinct()                        # Remove duplicate rows
q.distinct('customer_id')           # One per customer
```

## Common Patterns

```python
# Top 10
q.sort('revenue', ascending=False).head(10)

# Recent records
q.sort('date').tail(100)

# Quick preview
q.sample(frac=0.01)
```
