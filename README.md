# Bend

Bend is essentially pandas with training wheels + change tracking + CLI convenience.

Load any CSV and start analyzing with a functional query interface.

## Installation

```bash
pip install -e .
```

## Quick Start

Launch an interactive REPL with your CSV data:

```bash
bend data.csv
```

Or start with no data and load your own:

```bash
bend
# Then in the REPL:
df = load_csv('data.csv')
q = Q(df)
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

# Check available columns and row count
q.columns            # ['customer', 'amount', 'age', 'region']
q.cols               # Alias for q.columns
q.rows               # 100
len(q)               # 100 (same as q.rows)

# Quick dataset summary
print(f"{q.rows} rows × {len(q.columns)} columns")

# Iterate through data
for row in q:
    print(row.customer, row.amount)

# Quick statistics
q.sum('amount')      # 15750
q.mean('age')        # 34.5
q.count()            # 100 rows
q.nunique('region')  # 4 unique regions
```

### Type Conversion

CSV files load columns as strings by default. Use `dtype` to convert columns on load:

```python
# In the REPL or in Python code
from bend.core import load_csv, Q

# Convert columns to specific types
df = load_csv('sales.csv', dtype={'age': int, 'price': float, 'active': bool})
q = Q(df)

# Or when using the CLI with --skip-rows
bend sales.csv --skip-rows 2  # Skip first 2 rows

# Type conversion in the REPL
df = load_csv('data.csv', dtype={'year': int, 'revenue': float})
q = Q(df)
```

### Adding Computed Columns

```python
# Calculate total from price and quantity
q2 = q.assign(total=lambda x: x.price * x.qty)

# Add multiple columns at once (from original columns)
q3 = q.assign(
    total=lambda x: x.price * x.qty,
    discount=lambda x: x.price * 0.1
)

# Chain assignments to reference previous assignments
q4 = (q
    .assign(revenue=lambda x: x.price * x.units)
    .assign(profit=lambda x: x.revenue - x.cost)
    .assign(margin=lambda x: x.profit / x.revenue))
```

### Filtering Data

```python
# High-value customers
q.filter(lambda x: x.purchase_amount > 1000)

# Multiple conditions
q.filter(lambda x: x.age > 25 and x.region == 'West')

# Computed column filters
q.assign(total=lambda x: x.price * x.qty).filter(lambda x: x.total > 100)

# Safe filtering (exceptions treated as False)
q.filter(lambda x: int(x.year) >= 2020)  # Non-numeric years excluded
```

### Sorting and Limiting

```python
# Top 10 by revenue
q.assign(revenue=lambda x: x.price * x.units).sort('revenue').head(10)

# Sort ascending
q.sort('age', ascending=True)

# Sort by multiple columns
q.sort('region', 'sales')

# Bottom performers
q.sort('performance', ascending=True).head(5)
```

### Mapping Data to New Structure

```python
# Combine first and last name
q.map(lambda x: {
    'full_name': f"{x.first} {x.last}",
    'email': x.email
})

# Create summary view
q.map(lambda x: {
    'customer': f"{x.first} {x.last}",
    'total_spent': x.purchases * x.avg_price,
    'status': 'VIP' if x.purchases > 10 else 'Regular'
})

# Extract date components
q.map(lambda x: {
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
q2 = q.assign(commission=lambda x: x.sales * 0.15)

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

### Column Operations

**Display-only (hide/unhide):**
```python
# Hide columns from display (data still there, still usable!)
q.hide('id', 'internal_code', 'audit_timestamp')

# Unhide specific columns
q.unhide('id')

# Unhide everything
q.unhide()

# Hidden columns still work in calculations
q.hide('cost').assign(profit=lambda x: x.revenue - x.cost)
```

**Structural changes (drop/select):**
```python
# Actually remove columns from data
q.drop('id', 'temp_field')  # Removed columns can't be used later

# Keep only specified columns (drops all others)
q.select('name', 'email', 'status')

# Dropped columns can't be used in calculations
q.drop('cost').assign(profit=lambda x: x.revenue - x.cost)  # Error!
```

### Data Quality

**Remove duplicates:**
```python
# Remove completely duplicate rows
q.distinct()

# Keep first occurrence per customer
q.distinct('customer_id')

# Unique by email+phone combination
q.distinct('email', 'phone')

# Common pattern: dedupe before analysis
result = (q
    .distinct('order_id')  # Ensure unique orders
    .assign(revenue=lambda x: x.price * x.qty)
    .groupby(lambda x: x.region, total=lambda g: sum(r.revenue for r in g))
)
```

**Rename columns:**
```python
# Fix bad column names from CSV
q.rename(customerID='customer_id', amt='amount')

# Single rename
q.rename(old_name='new_name')

# Chain with other operations
q.rename(price='unit_price').assign(total=lambda x: x.unit_price * x.qty)
```

### Multi-DataFrame Operations

**Concatenate (stack rows):**
```python
# Combine monthly sales data
q_jan = Q(df_january)
q_feb = Q(df_february)
q_combined = q_jan.concat(q_feb)  # All rows from both

# Self-concatenation (duplicate rows)
q_double = q.concat(q)

# Performance mode for large datasets (non-reproducible)
q_combined = q1.concat(huge_q, deep_copy=False)
```

**Merge (join on keys):**
```python
# Basic merge
customers = Q(customers_df)
orders = Q(orders_df)
q = customers.merge(orders, on='customer_id', how='left')

# Merge with column conflict resolution
q1 = Q(pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "status": ["active", "inactive"]}))
q2 = Q(pd.DataFrame({"id": [1, 2], "status": ["pending", "complete"]}))

# Both have 'status' - must resolve!
q3 = q1.merge(q2, on='id', resolve={'status': lambda left, right: left})

# Self-merge (employee-manager relationship)
employees = Q(emp_df)
managers = employees.merge(
    employees, 
    on='employee_id',
    resolve={
        'name': lambda emp, mgr: emp,
        'manager_id': lambda e, m: e
    }
)

# Multiple keys
q.merge(other, on=['year', 'month', 'region'])
```

**Join (simpler merge without conflicts):**
```python
# If no column conflicts exist, use join()
customers.join(orders, on='customer_id', how='left')

# Inner join (only matching rows)
q1.join(q2, on='id')

# Outer join (all rows from both)
q1.join(q2, on='id', how='outer')
```

**Set operations:**
```python
# Union (unique rows from both)
q1 = Q(pd.DataFrame({"x": [1, 2, 3]}))
q2 = Q(pd.DataFrame({"x": [2, 3, 4]}))
q3 = q1.union(q2)  # [1, 2, 3, 4]

# Intersect (rows in both)
q3 = q1.intersect(q2)  # [2, 3]

# Difference (rows in q1 but not q2)
q3 = q1.difference(q2)  # [1]
```

**Reproducibility tracking:**
```python
# All operations track reproducibility
q1 = Q(df1)
q2 = Q(df2)
q3 = q1.concat(q2)
print(q3.reproducible)  # True (if both are reproducible)

# Non-deterministic operations break reproducibility
q2_sample = q2.sample(10)  # No random_state
print(q2_sample.reproducible)  # False

q4 = q1.concat(q2_sample)
print(q4.reproducible)  # False (propagated)

# To maintain reproducibility, use random_state
q2_sample = q2.sample(10, random_state=42)
print(q2_sample.reproducible)  # True
```

**Deep reload (recursive):**
```python
# Save sources
q1 = Q(load_csv('sales.csv'), source_path='sales.csv')
q2 = Q(load_csv('customers.csv'), source_path='customers.csv')

# Merge
q3 = q1.merge(q2, on='customer_id')

# Update both CSV files, then reload
q4 = q3.reload()  # Recursively reloads both sources + re-applies merge!
```

### Real-World Pipeline Example

```python
# Sales analysis pipeline
result = (q
    # Add calculated fields (chain to reference previous extensions)
    .assign(total=lambda x: x.price * x.quantity)
    .assign(discount=lambda x: x.total * x.discount_pct)
    .assign(final=lambda x: x.total - x.discount)
    # Filter for this year
    .filter(lambda x: x.date.startswith('2024'))
    # Only successful transactions
    .filter(lambda x: x.status == 'completed')
    # Hide internal fields from display
    .hide('internal_id', 'processor_code')
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
    .assign(a=lambda x: x.x * 2)
    .filter(lambda x: x.a > 10)
    .assign(b=lambda x: x.a + 5)
    .filter(lambda x: x.b < 100)
    .assign(c=lambda x: x.b * 3))

# Check change history
len(q2._changes)  # 5 operations

# Flatten to improve performance
q3 = q2.rebase()  # Makes current state the base, clears history
len(q3._changes)  # 0

# Continue building on flattened state
q4 = q3.assign(d=lambda x: x.c / 2)
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
q.assign(
    purchase_freq=lambda x: x.orders / x.months_active,
    avg_order=lambda x: x.total_spent / x.orders,
    lifetime_months=lambda x: 24,  # 2 year projection
    clv=lambda x: x.purchase_freq * x.avg_order * x.lifetime_months
)

# Percentile ranking
sales = q.assign(
    total_sales=lambda x: x.price * x.units
).to_df()
sales_sorted = sorted([r.total_sales for r in rows(sales)])

q.assign(
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
    .assign(region_clean=lambda x: x.region.strip().upper())
    # Flag outliers
    .assign(is_outlier=lambda x: x.amount > q.mean('amount') + 3 * q.std('amount')))

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
    .assign(revenue=lambda x: x.price * x.units_sold)
    .sort('revenue')
    .head(20))

# Analyze by category
for category in q.unique('product_category'):
    subset = q.filter(lambda x: x.product_category == category)
    print(f"{category}: {subset.sum('units_sold')} units, ${subset.sum('revenue'):.2f}")

# Save analysis
q = q.assign(revenue=lambda x: x.price * x.units_sold)
q = reload()  # If data updated, refresh everything
```

### Change History Inspection

```python
# Build complex pipeline
q2 = (q
    .assign(margin=lambda x: (x.price - x.cost) / x.price)
    .filter(lambda x: x.margin > 0.2)
    .assign(profit=lambda x: x.margin * x.revenue)
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
    .assign(calc1=...)
    .filter(...)
    .assign(calc2=...)
    .sort(...)
    .head(...))
```

**Lambda functions have access to all columns:**
```python
q.assign(new_col=lambda x: x.col1 + x.col2 * x.col3)
```

**Use aggregations for quick insights:**
```python
print(f"Total: ${q.sum('revenue'):,.2f}")
print(f"Range: ${q.min('price')} - ${q.max('price')}")
```

**Hidden columns are still usable:**
```python
q.hide('internal_cost').assign(profit=lambda x: x.price - x.internal_cost)
```

**Reload when external data changes:**
```python
q = q.assign(calculated=...).filter(...)
# File updated by another process
q = reload()  # Reloads file, re-applies all transformations
```

## Version

**2.0.0** - V2 - Multi-DataFrame Operations
- **Breaking:** Renamed `extend()` to `assign()` to align with pandas convention
- **Breaking:** Renamed `transform()` to `map()` for clearer functional programming semantics
- Added `concat()` for vertical stacking with optional deep copy
- Added `merge()` with explicit conflict resolution and reproducibility tracking
- Added `join()` as convenience wrapper for merge without conflicts
- Added set operations: `union()`, `intersect()`, `difference()`
- Added `reproducible` property to track pipeline determinism
- Changed `sample()` to be non-idempotent by default (use `random_state` for reproducibility)
- Made `reload()` deep/recursive to handle multi-Q operations
- Added `deep_copy` parameter to all multi-Q operations (default True)

**1.0.0** - V1 - Initial dataset operations. 

