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
**Why**: More explicit than `hide()`. Current `map()` can do this but it's awkward.

**Implementation**: Track as change, apply column selection in _apply_changes.

### 5. Rename Columns
```python
q.rename(old_name='new_name', customer_id='cust_id')
```
**Why**: CSV columns often have bad names. Currently you'd need to map everything.

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
q.assign(
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
**Why**: Data cleaning. Currently needs `assign()` with conditional logic.

**Implementation**: Track replacement mappings, apply via pandas replace.

### 9. Pivot Operations
```python
q.pivot(index='date', columns='category', values='sales')
q.melt(id_vars='date', value_vars=['sales', 'revenue'])  # unpivot
```
**Why**: Reshaping is common but `map()` is too manual for this.

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
**Why**: Can be done with `assign()` + conditionals but verbose.

**Implementation**: Could be useful helper, track as change type.

## Implementation Roadmap

### Phase 0: Infrastructure (Completed)
- [x] `memory_usage()` - Memory breakdown reporting

### Phase 1: Quick Wins (Easy + High Value) - ✅ COMPLETED
- [x] `tail(n)` - Mirror of head() (COMPLETED)
- [x] `sample(n, frac, random_state)` - Random sampling (COMPLETED)
  - **Note:** `sample()` is **non-idempotent by default** (`random_state=None`)
  - Users must explicitly pass `random_state` for reproducibility
  - `q.deterministic` flag tracks this
- [x] `dtype` parameter for `load_csv()` - Type conversion at load time (COMPLETED)
- [x] `distinct()` or `distinct(*cols)` - Deduplication (COMPLETED)
- [x] `select(*cols)` - Column selection (COMPLETED)
- [x] `drop(*cols)` - Column removal (COMPLETED)
- [x] `rename(**mapping)` - Column renaming (COMPLETED)

### Phase 2: Multi-Q Operations - ✅ COMPLETED
Implemented with **deep copy by default** approach:
- [x] `deterministic` property - Track pipeline determinism (COMPLETED)
- [x] `concat(other, deep_copy=True)` - Vertical stacking (COMPLETED)
- [x] `merge(other, on, how, resolve, deep_copy=True)` - Join with explicit conflict resolution (COMPLETED)
- [x] `join(other, on, how, deep_copy=True)` - Convenience wrapper around merge (COMPLETED)
- [x] Set operations: `union(other, deep_copy=True)`, `intersect(other, deep_copy=True)`, `difference(other, deep_copy=True)` (COMPLETED)
- [x] Deep `reload()` - Recursive reload of entire Q tree from disk (COMPLETED)

**Implementation notes:**
- Store **deep copy** of other Q by default (`deep_copy=True`) for full reproducibility
- Optional `deep_copy=False` for performance (marks result as non-deterministic)
- `q.deterministic` flag propagates through operations
- Column conflicts require explicit `resolve` parameter with lambdas
- `reload()` is **deep/recursive** (reloads entire tree from disk)
- Use `rebase()` to drop deep copies and flatten history

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
- ✅ Change tracking and replay capability (`replay()`, `reload()`)
- ✅ Q as the core object - all operations return Q
- ✅ Idempotency* - same operations produce same results when `deterministic=True`
- ✅ Simple, readable API
- ✅ Works with the Row namedtuple pattern
- ✅ Functional programming paradigm

*Note: `sample()` is **non-idempotent by default** (`random_state=None`). Users must pass explicit `random_state` for reproducibility. The `q.deterministic` flag tracks this.

**Architecture for Multi-Q Operations (Merge/Join/Concat):**

Multi-Q operations store **deep copies** of other Q objects by default for full reproducibility:

```python
# Change history structure:
[
    ("assign", {"total": lambda x: x.price * x.qty}),
    ("merge", {
        "other": <deep_copy_of_other_Q>,  # Full deep copy by default
        "on": "customer_id",
        "how": "left",
        "resolve": {"status": lambda a, b: a},
        "deep_copy": True
    }),
    ("filter", lambda x: x.total > 100)
]
```

**Key principles:**
1. **Deep copy by default**: Store `copy.deepcopy(other)` for full reproducibility
2. **Optional reference mode**: Use `deep_copy=False` for performance (marks as non-deterministic)
3. **Reproducibility tracking**: `q.deterministic` property propagates through all operations
4. **Explicit conflict resolution**: Column conflicts require `resolve` parameter with lambdas
5. **Tree-based history**: Change history forms a tree; `replay()` and `reload()` are recursive
6. **Memory management**: Use `rebase()` to flatten history and drop deep copies
7. **Self-reference protection**: Self-joins deep copy self to avoid circular references

**Benefits:**
- ✅ Replay preserved through deep copies (fully deterministic)
- ✅ User freedom - continue using Q objects after merge without side effects
- ✅ Idempotent - same operations produce same results (if deterministic=True)
- ✅ Clear contract via `deterministic` flag
- ✅ Performance option via `deep_copy=False`
- ✅ Explicit memory management via `rebase()`

**Trade-offs:**
- ⚠️ Higher memory usage (mitigated by `rebase()` and `deep_copy=False` option)
- ⚠️ Column conflicts require explicit resolution (more verbose but safer)

**Remaining challenges:**
- How to handle operations that need row context (window functions)?
- When to make an operation "terminal" (like groupby) vs tracked?
- Pickling/serialization of Q objects with lambdas (see Serialization section below)
- Performance optimization for large deep copies (see `deep_copy=False` option)

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
q2 = q.assign(total=lambda x: x.price * x.qty)
usage = q2.memory_usage()
print(f"Total: {usage['total_mb']} MB")
print(f"Changes: {usage['changes']} tracked operations")

# After merge with large Q
q3 = q2.merge(large_q, on='id')
usage = q3.memory_usage()  # Includes large_q's memory
print(f"Memory with merge: {usage['total_mb']} MB")

# Flatten to reduce memory
q4 = q3.rebase()  # Drops deep copy of large_q
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

