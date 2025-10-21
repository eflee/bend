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

### 13. Binning/Categorization
```python
q.bin('age', bins=[0, 18, 35, 50, 100], labels=['child', 'young', 'middle', 'senior'])
```
**Why**: Can be done with `extend()` + conditionals but verbose.

**Implementation**: Could be useful helper, track as change type.

## Implementation Roadmap

### Phase 0: Infrastructure (Completed)
- [x] `memory_usage()` - Memory breakdown reporting

### Phase 1: Quick Wins (Easy + High Value)
- [x] `tail(n)` - Mirror of head() (COMPLETED)
- [x] `sample(n)` or `sample(frac)` - Random sampling with reproducible default (COMPLETED)
- [x] `dtype` parameter for `load_csv()` - Type conversion at load time (COMPLETED)
- [ ] `distinct()` or `distinct(*cols)` - Deduplication
- [x] `select(*cols)` - Column selection (COMPLETED)
- [x] `drop(*cols)` - Column removal (COMPLETED)
- [ ] `rename(**mapping)` - Column renaming

### Phase 2: Multi-Q Operations (Architecture Validated)
Now feasible with reference-based approach:
- [ ] `concat(other)` - Vertical stacking (simplest, implement first)
- [ ] `merge(other, left_on, right_on, how)` - Join with explicit keys
- [ ] `join(other, on, how)` - Convenience wrapper around merge
- [ ] Set operations: `union(other)`, `intersect(other)`, `difference(other)`

**Implementation notes:**
- Store other Q by reference in change history
- `memory_usage()` will account for referenced Qs
- Use `rebase()` to drop references and flatten

### Phase 3: Data Quality
- [ ] `fillna(value)` or `fillna(mapping)` - Fill missing values
- [ ] `dropna()` or `dropna(*cols)` - Remove rows with nulls
- [ ] `replace(mapping)` - Value replacement

### Phase 4: Complex Operations (Design Required)
- [ ] Window functions - Row isolation vs context trade-off
- [ ] `pivot(index, columns, values)` - Reshaping
- [ ] `melt(id_vars, value_vars)` - Unpivoting

### Phase 5: Nice to Have
- [ ] `bin(col, bins, labels)` - Categorization helper
- [ ] `q.save()` / `Q.load()` - Serialization with dill

## Design Principles to Maintain

**Must preserve (P0 Requirements):**
- ✅ Immutable operations (all return new Q)
- ✅ Change tracking and replay capability (`refresh()`, `reload()`)
- ✅ Q as the core object - all operations return Q
- ✅ Idempotency - same operations always produce same results
- ✅ Simple, readable API
- ✅ Works with the Row namedtuple pattern
- ✅ Functional programming paradigm

**Architecture for Multi-Q Operations (Merge/Join/Concat):**

Multi-Q operations store the other Q **by reference** (not deep copy) in the change history:

```python
# Change history structure:
[
    ("extend", {"total": lambda x: x.price * x.qty}),
    ("merge", {
        "other": <reference_to_other_Q>,  # Just a reference
        "on": "customer_id",
        "how": "left"
    }),
    ("filter", lambda x: x.total > 100)
]
```

**Key principles:**
1. **No deep copies**: Store Q references directly - rely on Python's immutability contract
2. **User responsibility**: If user mutates a Q after merging, behavior is undefined (gentleman's agreement)
3. **Memory management**: Use `rebase()` to flatten history and drop Q references
4. **Replay works**: `refresh()` and `reload()` replay merge using stored Q reference
5. **Memory visibility**: `memory_usage()` reports memory including referenced Qs

**Benefits:**
- ✅ Replay preserved through Q references
- ✅ Idempotent - same Q references produce same results
- ✅ Memory efficient - no deep copies
- ✅ Explicit memory management via `rebase()`
- ✅ Clean, intuitive API

**Challenges to solve:**
- How to handle operations that need row context (window functions)?
- When to make an operation "terminal" (like groupby) vs tracked?
- Pickling/serialization of Q objects with lambdas (see below)

## Serialization & Pickling

**Current Status:**
- ✅ Simple Q objects (no operations) are pickleable with standard `pickle`
- ❌ Q objects with tracked changes containing lambdas are NOT pickleable
- Root cause: Python's `pickle` module cannot serialize lambda functions

**Use Cases Requiring Serialization:**
- Saving analysis pipelines to disk for later replay
- Multiprocessing (passing Q between processes)
- Caching intermediate results
- Sharing pipelines between team members

**Solutions to Implement:**

### Phase 1: Document Limitation
- [ ] Document in README that Q with tracked changes is not pickleable
- [ ] Recommend `rebase()` before pickling if replay isn't needed
- [ ] Add example showing workaround

### Phase 2: Add dill Support
- [ ] Add `dill` as optional dependency: `pip install bend[serialization]`
- [ ] Implement pickle support using dill when available
- [ ] Add `q.save(filename)` and `Q.load(filename)` convenience methods
- [ ] Test serialization roundtrip with all operation types

```python
# Implementation sketch:
def save(self, filename: str) -> None:
    """Save Q object to disk (requires dill for Q with lambdas)."""
    try:
        import dill
        with open(filename, 'wb') as f:
            dill.dump(self, f)
    except ImportError:
        if self._changes:
            raise ImportError(
                "Saving Q with tracked changes requires 'dill'. "
                "Install with: pip install bend[serialization]"
            )
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

@classmethod
def load(cls, filename: str) -> 'Q':
    """Load Q object from disk."""
    try:
        import dill
        with open(filename, 'rb') as f:
            return dill.load(f)
    except ImportError:
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)
```

### Phase 3: Alternative Approaches (Future)
- [ ] Investigate `cloudpickle` as alternative to dill
- [ ] Consider storing change history as serializable format (AST, not lambdas)
- [ ] Evaluate trade-offs of each approach

**Priority:** Medium - useful but not required for core REPL functionality

## Memory Management

**Implemented:**
- [x] `memory_usage()` method - reports memory breakdown
  - Current DataFrame memory
  - Base DataFrame memory  
  - Referenced Q objects in change history
  - Total memory usage in bytes and MB

**Usage:**
```python
q2 = q.extend(total=lambda x: x.price * x.qty)
usage = q2.memory_usage()
print(f"Total: {usage['total_mb']} MB")
print(f"Changes: {usage['changes']} tracked operations")

# After merge with large Q
q3 = q2.merge(large_q, on='id')
usage = q3.memory_usage()  # Includes large_q's memory
print(f"Memory with merge: {usage['total_mb']} MB")

# Flatten to reduce memory
q4 = q3.rebase()  # Drops reference to large_q
usage = q4.memory_usage()  # Should be smaller
```

**Future enhancements:**
- [ ] Add warning when memory exceeds threshold
- [ ] Add `memory_report()` with human-readable breakdown
- [ ] Track memory delta between operations

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

