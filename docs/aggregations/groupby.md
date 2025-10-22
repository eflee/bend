# groupby()

Group rows and compute aggregations.

## Signature

```python
q.groupby(keyfn: Callable, **aggs) -> Q
```

## Parameters

- `keyfn`: Function that takes a Row and returns a grouping key
- `**aggs`: Aggregation functions (name=function pairs)

## Returns

A new Q object with one row per group. **Change history is reset** (terminal operation).

## Basic Usage

```python
# Group by category, compute sum and count
summary = q.groupby(
    lambda x: x.category,
    total_sales=lambda group: sum(r.sales for r in group),
    count=lambda group: len(group)
)
# Result columns: 'key', 'total_sales', 'count'
```

## The 'key' Column

The grouping key becomes a column named `key`:

```python
by_region = q.groupby(
    lambda x: x.region,
    revenue=lambda g: sum(r.amount for r in g)
)
# Columns: 'key' (region), 'revenue'

# Rename for clarity
by_region = by_region.rename(key='region')
```

## Aggregation Functions

Each aggregation receives a **list of Rows** for that group:

```python
q.groupby(
    lambda x: x.category,
    
    # Count
    count=lambda g: len(g),
    
    # Sum
    total=lambda g: sum(r.amount for r in g),
    
    # Average
    avg=lambda g: sum(r.value for r in g) / len(g),
    
    # Min/Max
    min_val=lambda g: min(r.score for r in g),
    max_val=lambda g: max(r.score for r in g),
    
    # Custom logic
    has_vip=lambda g: any(r.is_vip for r in g),
    all_complete=lambda g: all(r.status == 'complete' for r in g)
)
```

## Use Cases

### 1. Sales by Region

```python
regional_sales = orders.groupby(
    lambda x: x.region,
    total_revenue=lambda g: sum(r.amount for r in g),
    order_count=lambda g: len(g),
    avg_order=lambda g: sum(r.amount for r in g) / len(g)
).rename(key='region')
```

### 2. Customer Segmentation

```python
customer_stats = orders.groupby(
    lambda x: x.customer_id,
    total_spent=lambda g: sum(r.total for r in g),
    order_count=lambda g: len(g),
    first_order=lambda g: min(r.date for r in g),
    last_order=lambda g: max(r.date for r in g)
).rename(key='customer_id')
```

### 3. Time Series Aggregation

```python
daily = transactions.groupby(
    lambda x: x.timestamp.date(),
    transaction_count=lambda g: len(g),
    total_volume=lambda g: sum(r.amount for r in g),
    unique_users=lambda g: len(set(r.user_id for r in g))
).rename(key='date')
```

## Multi-Level Grouping

Use a tuple as the key:

```python
by_region_and_category = q.groupby(
    lambda x: (x.region, x.category),
    sales=lambda g: sum(r.amount for r in g)
)
# key column contains tuples: ('West', 'Electronics')
```

## Gotchas

### Terminal Operation

`groupby()` **resets the change history**:

```python
q = Q(df, source_path='data.csv')
q2 = q.filter(lambda x: x.active)
q3 = q2.groupby(lambda x: x.region, count=lambda g: len(g))

# q3 cannot reload() - no source path
# q3._changes is empty - history is reset
```

### Empty Groups

If no rows match a group, that group doesn't appear in results:

```python
# Only groups that exist in data appear in result
```

### Performance

Groupby processes all rows in Python (not vectorized). For large datasets (>1M rows), consider using pandas directly.

## Comparison with Pandas

```python
# Pandas
df.groupby('category').agg({'sales': 'sum', 'count': 'size'})

# Bend
q.groupby(
    lambda x: x.category,
    sales=lambda g: sum(r.sales for r in g),
    count=lambda g: len(g)
).rename(key='category')
```

## Idempotency

❌ **No** - Because it's a terminal operation that resets history. Cannot be replayed via `reload()`.

## See Also

- [Aggregation methods](aggregations.md) - Single-value aggregations
- [`distinct()`](distinct.md) - Unique values
