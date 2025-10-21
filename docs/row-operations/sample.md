# sample()

Return a random sample of rows.

## Signature

```python
q.sample(n: int = None, frac: float = None, random_state: int = None) -> Q
```

## Parameters

- `n`: Number of rows to sample
- `frac`: Fraction of rows (0.0 to 1.0)
- `random_state`: Seed for reproducibility (default: `None` = truly random)

Must specify exactly one of `n` or `frac`.

## Returns

A new Q object with sampled rows.

## Basic Usage

```python
# 100 random rows
q2 = q.sample(100)

# 10% of rows
q2 = q.sample(frac=0.1)

# Deterministic sample
q2 = q.sample(100, random_state=42)
```

## Reproducibility

**By default, `sample()` is NON-DETERMINISTIC**:

```python
q2 = q.sample(100)  # Different results each time
print(q2.deterministic)  # False

# For reproducibility, pass random_state
q2 = q.sample(100, random_state=42)  # Same results every time
print(q2.deterministic)  # True
```

## Use Cases

```python
# Quick analysis on subset
preview = q.sample(frac=0.01)  # 1% sample

# Train/test split (with seeds for reproducibility)
train = q.sample(frac=0.8, random_state=42)
test = q.difference(train)  # Remaining 20%

# A/B testing
group_a = q.sample(frac=0.5, random_state=123)
group_b = q.difference(group_a)
```

## Gotchas

### Sample Size > Dataset
```python
q.sample(1000)  # If q has 100 rows, raises ValueError
```

### Non-Deterministic by Default
```python
q2 = q.sample(100)
q3 = q2.reload()  # Different 100 rows!

# Fix: use random_state
q2 = q.sample(100, random_state=42)
q3 = q2.reload()  # Same 100 rows
```

## Idempotency

⚠️ **Conditional**
- **No** if `random_state=None` (default)
- **Yes** if `random_state` is specified

## See Also

- [`head()`](head.md), [`tail()`](tail.md) - Deterministic subsets
- [`filter()`](filter.md) - Conditional selection
