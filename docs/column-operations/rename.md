# rename()

Rename columns using keyword arguments.

## Signature

```python
q.rename(**mapping) -> Q
```

## Parameters

- `**mapping`: Keyword arguments where key=old name, value=new name

## Basic Usage

```python
# Rename single column
q2 = q.rename(customer_id='cust_id')

# Rename multiple
q2 = q.rename(customer_id='cust_id', order_date='date', total_amount='total')
```

## Use Cases

```python
# Fix bad CSV column names
clean = q.rename(
    CustomerID='customer_id',
    OrderDate='order_date',
    TotalAmount='total'
)

# Shorten long names
short = q.rename(customer_lifetime_value='ltv', average_order_value='aov')
```

## Gotchas

- Non-existent columns ignored (no error)
- Subsequent operations use new names
- History tracking persists through rename

## Idempotency

✅ **Yes**

## See Also

- [`drop()`](drop.md), [`select()`](select.md)
