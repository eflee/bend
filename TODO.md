# Bend TODO - Feature Additions

## High Priority - Core Missing Functionality

### 1. Joining/Merging Data
```python
q.join(other_q, on='customer_id', how='left')
q.merge(other_q, left_on='id', right_on='user_id')
```
**Why**: Real-world analysis almost always involves multiple datasets. Currently you'd need to drop to pandas.

**Challenge**: How to track changes from two sources? Need to design change history for multi-source operations.

### 2. Deduplication
```python
q.distinct()  # drop all duplicates
q.distinct('customer_id')  # keep first occurrence per customer
```
**Why**: Data quality issue in every CSV. Very common operation.

**Implementation**: Straightforward - track as change type, apply via pandas drop_duplicates.

### 3. Tail + Sample
```python
q.tail(10)  # complement to head()
q.sample(100)  # random sample of n rows
q.sample(0.1)  # 10% sample
```
**Why**: Head exists but not tail. Sampling is crucial for large dataset exploration.

**Implementation**: Easy - mirror head() implementation, add sample with tracking.

### 4. Select/Drop Columns
```python
q.select('name', 'email', 'age')  # keep only these
q.drop('internal_id', 'temp_field')  # remove these
```
**Why**: More explicit than `hide_cols()`. Current `transform()` can do this but it's awkward.

**Implementation**: Track as change, apply column selection in _apply_changes.

### 5. Rename Columns
```python
q.rename(old_name='new_name', customer_id='cust_id')
```
**Why**: CSV columns often have bad names. Currently you'd need to transform everything.

**Implementation**: Track rename mapping, apply to DataFrame.columns.

## Medium Priority - Data Quality & Analysis

### 6. Missing Data Handling
```python
q.dropna()  # remove rows with any nulls
q.dropna('email')  # remove rows where email is null
q.fillna(0)  # fill all nulls with 0
q.fillna({'age': 0, 'city': 'Unknown'})  # column-specific fills
```
**Why**: Every dataset has missing data. Currently filters can check but can't fix.

**Implementation**: Track as changes, use pandas fillna/dropna methods.

### 7. Window Functions
```python
q.extend(
    cumsum_sales=lambda x: ...,  # needs window context
    rolling_avg=lambda x: ...     # needs previous rows
)
# Or maybe:
q.window('sales', cumsum=True, col_name='cumsum_sales')
```
**Why**: Running totals, moving averages are common. Currently requires manual iteration.

**Challenge**: Breaks Row isolation pattern - rows need context of other rows.

### 8. Value Replacement
```python
q.replace({'region': {'CA': 'California', 'NY': 'New York'}})
q.replace_values('status', {'old': 'new'})
```
**Why**: Data cleaning. Currently needs `extend()` with conditional logic.

**Implementation**: Track replacement mappings, apply via pandas replace.

### 9. Pivot Operations
```python
q.pivot(index='date', columns='category', values='sales')
q.melt(id_vars='date', value_vars=['sales', 'revenue'])  # unpivot
```
**Why**: Reshaping is common but `transform()` is too manual for this.

**Challenge**: Changes column structure dramatically, hard to replay if source columns change.

### 10. Concatenation
```python
q.concat(other_q)  # vertical stack
Q.concat(q1, q2, q3)  # stack multiple
```
**Why**: Combining datasets from same source (e.g., monthly files).

**Challenge**: Multi-source change tracking. Similar to join issue.

## Lower Priority - Convenience Features

### 11. String Operations Helper
```python
# Currently works:
q.extend(domain=lambda x: x.email.split('@')[1])

# Could add helper:
q.str_extract('email', r'@(.+)$', 'domain')
```
**Why**: `extend()` + lambdas already handle this, but helpers could be nice.

**Decision**: Probably not needed - lambdas are flexible enough.

### 12. Date Parsing Helper
```python
q.parse_dates('order_date', format='%Y-%m-%d')
q.extend(year=lambda x: parse_date(x.date).year)
```
**Why**: CSV dates are always strings. Currently manual parsing in lambdas.

**Decision**: Can be done with extend() easily. Low priority.

### 13. Binning/Categorization
```python
q.bin('age', bins=[0, 18, 35, 50, 100], labels=['child', 'young', 'middle', 'senior'])
```
**Why**: Can be done with `extend()` + conditionals but verbose.

**Implementation**: Could be useful helper, track as change type.

### 14. Type Conversion
```python
q.cast({'age': int, 'price': float, 'active': bool})
```
**Why**: CSV columns are all strings initially. pandas handles this on load though.

**Decision**: Load-time option might be better than post-load conversion.

## Implementation Roadmap

### Phase 1: Quick Wins (Easy + High Value)
- [ ] `tail(n)` - Mirror of head()
- [ ] `sample(n)` - Random sampling
- [ ] `distinct()` - Deduplication
- [ ] `select(*cols)` - Column selection
- [ ] `drop(*cols)` - Column removal
- [ ] `rename(**mapping)` - Column renaming

### Phase 2: Data Quality
- [ ] `fillna(value)` - Fill missing values
- [ ] `dropna(subset)` - Remove rows with nulls
- [ ] `replace(mapping)` - Value replacement

### Phase 3: Complex Operations (Need Design)
- [ ] `join(other, on, how)` - Multi-source tracking design needed
- [ ] `merge(other, left_on, right_on, how)` - Same as join
- [ ] `concat(other)` - Vertical stacking
- [ ] Window functions - Row isolation vs context trade-off

### Phase 4: Nice to Have
- [ ] `pivot()` - Reshaping
- [ ] `melt()` - Unpivoting  
- [ ] `bin()` - Categorization helper

## Design Principles to Maintain

**Must preserve:**
- ✅ Immutable operations (all return new Q)
- ✅ Change tracking (can be replayed)
- ✅ Simple, readable API
- ✅ Works with the Row namedtuple pattern
- ✅ Functional programming paradigm

**Challenges to solve:**
- How to handle multi-source operations (join, concat)?
- How to handle operations that need row context (window functions)?
- When to make an operation "terminal" (like groupby) vs tracked?

## Testing Requirements

Each new feature needs:
- [ ] Unit tests in test_core.py
- [ ] README examples
- [ ] Test in test_readme_examples.py
- [ ] Edge case tests in TestEdgeCases
- [ ] Docstring with examples

## Notes

- Consider adding `q.tail()` and `q.sample()` in same PR as they're trivial
- `distinct()` is straightforward and very useful - good first addition
- Join/merge need careful design for change tracking from multiple sources
- Window functions may require rethinking the Row-by-row lambda pattern

