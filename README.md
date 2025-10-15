# Bend

Bend is essentially pandas with training wheels + change tracking + CLI convenience.

Load any CSV and start analyzing with a functional query interface.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
bend data.csv
```

You'll get an interactive REPL with your data loaded as `q`.

## Why Bend?

Several CSV CLI tools exist (VisiData, csvkit, Miller, xsv, q/textql), but Bend fills a unique niche by combining pandas power with training wheels, change tracking, and CLI convenience.

### What Makes Bend Different

- **IPython REPL**: Full Python environment with pre-loaded data, not just command flags
- **Programmatic**: Lambda functions instead of SQL or command-line flags
- **Functional/Immutable**: Operations return new instances for safe experimentation
- **Change Tracking**: Replay transformations on refreshed data with `reload()` and `refresh()`
- **Method Chaining**: Fluent API for readable pipelines

### Comparison

| Tool | Approach | Best For |
|------|----------|----------|
| **Bend** | Interactive Python REPL | Exploratory analysis with Python |
| VisiData | Visual TUI spreadsheet | Visual browsing and editing |
| csvkit | Unix-style utilities | Shell scripting and SQL queries |
| Miller | Stream processing | One-pass transformations |
| xsv | Fast single commands | Quick operations on large files |
| q/textql | SQL on CSV | SQL-centric analysis |

**Bend is the sweet spot** between simple Unix tools and full Jupyter notebooks—perfect for data scientists who want more than `csvkit` but less overhead than a notebook environment.

## Examples

### Basic Data Exploration

```python
# Load sales data
$ bend sales.csv

# View first rows
q

# Iterate through data
for row in q:
    print(row.customer, row.amount)

# Quick statistics
q.sum('amount')      # 15750
q.mean('age')        # 34.5
q.count()            # 100 rows
q.nunique('region')  # 4 unique regions
```

### Adding Computed Columns

```python
# Calculate total from price and quantity
q2 = q.extend(total=lambda x: x.price * x.qty)

# Add multiple columns at once (from original columns)
q3 = q.extend(
    total=lambda x: x.price * x.qty,
    discount=lambda x: x.price * 0.1
)

# Chain extensions to reference previous extensions
q4 = (q
    .extend(revenue=lambda x: x.price * x.units)
    .extend(profit=lambda x: x.revenue - x.cost)
    .extend(margin=lambda x: x.profit / x.revenue))
```

### Filtering Data

```python
# High-value customers
q.filter(lambda x: x.purchase_amount > 1000)

# Multiple conditions
q.filter(lambda x: x.age > 25 and x.region == 'West')

# Computed column filters
q.extend(total=lambda x: x.price * x.qty).filter(lambda x: x.total > 100)

# Safe filtering (exceptions treated as False)
q.filter(lambda x: int(x.year) >= 2020)  # Non-numeric years excluded
```

### Sorting and Limiting

```python
# Top 10 by revenue
q.extend(revenue=lambda x: x.price * x.units).sort('revenue').head(10)

# Sort ascending
q.sort('age', ascending=True)

# Sort by multiple columns
q.sort('region', 'sales')

# Bottom performers
q.sort('performance', ascending=True).head(5)
```

### Transforming Data

```python
# Combine first and last name
q.transform(lambda x: {
    'full_name': f"{x.first} {x.last}",
    'email': x.email
})

# Create summary view
q.transform(lambda x: {
    'customer': f"{x.first} {x.last}",
    'total_spent': x.purchases * x.avg_price,
    'status': 'VIP' if x.purchases > 10 else 'Regular'
})

# Extract date components
q.transform(lambda x: {
    'year': x.date.split('-')[0],
    'month': x.date.split('-')[1],
    'amount': x.amount
})
```

### Grouping and Aggregating

```python
# Sales by region
q.groupby(
    lambda x: x.region,
    total_sales=lambda g: sum(r.amount for r in g),
    avg_sale=lambda g: sum(r.amount for r in g) / len(g),
    customers=lambda g: len(set(r.customer_id for r in g))
)

# Monthly revenue
q.groupby(
    lambda x: x.date[:7],  # YYYY-MM
    revenue=lambda g: sum(r.price * r.qty for r in g),
    orders=lambda g: len(g)
)

# Top product by category
q.groupby(
    lambda x: x.category,
    top_product=lambda g: max(g, key=lambda r: r.sales).product_name,
    max_sales=lambda g: max(r.sales for r in g)
)
```

### Working with External Data Updates

```python
# Initial load
q = Q(df, source_path='daily_sales.csv')
q2 = q.extend(commission=lambda x: x.sales * 0.15)

# Later, when daily_sales.csv is updated by another process...
q3 = q2.reload()  # Reloads CSV and re-applies commission calculation

# Or just re-apply transformations without reloading
q3 = q2.refresh()
```

### Handling Files with Metadata

```python
# Skip header rows
$ bend report.csv --skip-rows 5

# In Python
df = load_csv('report.csv', skip_rows=5)
```

### Managing Column Visibility

```python
# Hide internal columns
q.hide_cols('id', 'internal_code', 'audit_timestamp')

# Show only specific columns
q.show_cols('name', 'email', 'status')

# Unhide everything
q.show_cols()

# Hidden columns still work in calculations
q.hide_cols('cost').extend(profit=lambda x: x.revenue - x.cost)
```

### Real-World Pipeline Example

```python
# Sales analysis pipeline
result = (q
    # Add calculated fields (chain to reference previous extensions)
    .extend(total=lambda x: x.price * x.quantity)
    .extend(discount=lambda x: x.total * x.discount_pct)
    .extend(final=lambda x: x.total - x.discount)
    # Filter for this year
    .filter(lambda x: x.date.startswith('2024'))
    # Only successful transactions
    .filter(lambda x: x.status == 'completed')
    # Hide internal fields
    .hide_cols('internal_id', 'processor_code')
    # Sort by final amount
    .sort('final')
    # Top 100
    .head(100))

# Save results
result.dump('top_sales_2024.csv')
```

### Performance Optimization

```python
# Long pipeline with many operations
q2 = (q
    .extend(a=lambda x: x.x * 2)
    .filter(lambda x: x.a > 10)
    .extend(b=lambda x: x.a + 5)
    .filter(lambda x: x.b < 100)
    .extend(c=lambda x: x.b * 3))

# Check change history
len(q2._changes)  # 5 operations

# Flatten to improve performance
q3 = q2.rebase()  # Makes current state the base, clears history
len(q3._changes)  # 0

# Continue building on flattened state
q4 = q3.extend(d=lambda x: x.c / 2)
```

### Data Quality Checks

```python
# Check for nulls
q.count('email')  # Non-null count
q.count()         # Total rows

# Find unique values
q.unique('status')    # ['active', 'inactive', 'pending']
q.nunique('region')   # 4

# Value ranges
print(f"Age range: {q.min('age')} to {q.max('age')}")
print(f"Average: {q.mean('age'):.1f} (±{q.std('age'):.1f})")
```

### Working with Google Sheets

```python
# Load directly from Google Sheets
$ bend "https://docs.google.com/spreadsheets/d/abc123/edit#gid=0"

# Or in Python
df = load_csv("https://docs.google.com/spreadsheets/d/abc123/edit#gid=456")
```

### Complex Calculations

```python
# Customer lifetime value
q.extend(
    purchase_freq=lambda x: x.orders / x.months_active,
    avg_order=lambda x: x.total_spent / x.orders,
    lifetime_months=lambda x: 24,  # 2 year projection
    clv=lambda x: x.purchase_freq * x.avg_order * x.lifetime_months
)

# Percentile ranking
sales = q.extend(
    total_sales=lambda x: x.price * x.units
).to_df()
sales_sorted = sorted([r.total_sales for r in rows(sales)])

q.extend(
    percentile=lambda x: sum(1 for s in sales_sorted if s <= x.price * x.units) / len(sales_sorted) * 100
)

# Running calculations with context
data = []
running_total = 0
for row in q:
    running_total += row.amount
    data.append({'date': row.date, 'amount': row.amount, 'cumsum': running_total})

cumulative = Q(pd.DataFrame(data))
```

### Validation Pipeline

```python
# Data validation and cleaning
clean = (q
    # Remove test data
    .filter(lambda x: not x.email.endswith('@test.com'))
    # Valid dates only
    .filter(lambda x: len(x.date.split('-')) == 3)
    # Positive amounts
    .filter(lambda x: x.amount > 0)
    # Standardize region names
    .extend(region_clean=lambda x: x.region.strip().upper())
    # Flag outliers
    .extend(is_outlier=lambda x: x.amount > q.mean('amount') + 3 * q.std('amount')))

# Review flagged records
outliers = clean.filter(lambda x: x.is_outlier)
```

### Export and Reporting

```python
# Generate report
summary = q.groupby(
    lambda x: x.category,
    items=lambda g: len(g),
    revenue=lambda g: sum(r.price * r.qty for r in g)
).sort('revenue')

summary.dump('category_summary.csv')

# Multiple exports from same source
high_value = q.filter(lambda x: x.amount > 1000)
high_value.dump('high_value_customers.csv')

low_value = q.filter(lambda x: x.amount <= 1000)
low_value.dump('low_value_customers.csv')
```

### Interactive Analysis Session

```python
$ bend sales_data.csv

# Initial exploration
q.count()                    # How many records?
q.unique('product_category') # What categories exist?
q.mean('price')             # Average price?

# Find top sellers
top = (q
    .extend(revenue=lambda x: x.price * x.units_sold)
    .sort('revenue')
    .head(20))

# Analyze by category
for category in q.unique('product_category'):
    subset = q.filter(lambda x: x.product_category == category)
    print(f"{category}: {subset.sum('units_sold')} units, ${subset.sum('revenue'):.2f}")

# Save analysis
q = q.extend(revenue=lambda x: x.price * x.units_sold)
q = reload()  # If data updated, refresh everything
```

### Change History Inspection

```python
# Build complex pipeline
q2 = (q
    .extend(margin=lambda x: (x.price - x.cost) / x.price)
    .filter(lambda x: x.margin > 0.2)
    .extend(profit=lambda x: x.margin * x.revenue)
    .sort('profit')
    .head(50))

# Inspect what was done
for change_type, _ in q2._changes:
    print(change_type)
# Output: extend, filter, extend, sort, head

# Replay on fresh data
fresh_data = load_csv('updated_data.csv')
q_fresh = Q(fresh_data, source_path='updated_data.csv')
# All operations from q2._changes can be applied to q_fresh
```

### Tips

**Method chaining is your friend:**
```python
result = (q
    .extend(calc1=...)
    .filter(...)
    .extend(calc2=...)
    .sort(...)
    .head(...))
```

**Lambda functions have access to all columns:**
```python
q.extend(new_col=lambda x: x.col1 + x.col2 * x.col3)
```

**Use aggregations for quick insights:**
```python
print(f"Total: ${q.sum('revenue'):,.2f}")
print(f"Range: ${q.min('price')} - ${q.max('price')}")
```

**Hidden columns are still usable:**
```python
q.hide_cols('internal_cost').extend(profit=lambda x: x.price - x.internal_cost)
```

**Reload when external data changes:**
```python
q = q.extend(calculated=...).filter(...)
# File updated by another process
q = reload()  # Reloads file, re-applies all transformations
```

## Version

**1.0.0** - Complete refactoring with change history tracking

