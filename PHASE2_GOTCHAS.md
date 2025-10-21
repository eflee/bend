# Phase 2 Implementation Gotchas & Q Contract Challenges

This document identifies potential issues when implementing Phase 2 (Multi-Q Operations) while maintaining the Q contract guarantees.

## Executive Summary

Phase 2 introduces **Q-to-Q references** in operations like `merge()`, `join()`, and `concat()`. This creates new challenges around:
- **Mutation detection** - Can't enforce immutability of referenced Qs
- **Idempotency** - Referenced Q state might change between operations  
- **Memory management** - Reference chains can keep large objects alive
- **Column conflicts** - Merging Qs with same column names
- **Replay semantics** - What happens when referenced Q has a source_path?

## Design Decisions (Finalized)

The following design decisions have been made for Phase 2:

1. **Deep Copy on Multi-Q Operations** - Use `copy.deepcopy(other)` to store full Q with history
2. **Explicit Conflict Resolution** - Require user-provided `resolve` parameter for column conflicts
3. **Tree-Based History** - Change history becomes a tree, `refresh()` does recursive traversal
4. **Reproducibility Tracking** - Add `q.reproducible` flag that propagates through operations
5. **Follow Pandas Conventions** - Use pandas behavior for `union()`, `merge()`, etc.
6. **Row-by-Row Conflict Resolution** - Use `apply()` for conflict lambda application
7. **Correctness Over Performance** - Prioritize reproducibility and correctness

---

## Critical Gotchas (Must Address)

### 1. Replay with Mutated Q References ⚠️⚠️⚠️

**The Problem:**
```python
q1 = Q(df1)
q2 = Q(df2)

q3 = q1.merge(q2, on='id')  # Stores reference to q2 in q3._changes

# User violates the gentleman's agreement:
q2._df = modified_dataframe  # Direct mutation (bad!)

q3.refresh()  # Uses q2 reference - what state is it in?
# Result: Undefined behavior, breaks idempotency
```

**Why It Matters:**
- Q contract promises idempotency: same operations → same results
- But we can't prevent users from mutating private members
- `refresh()` might produce different results than the original operation

**Solutions to Consider:**

**Option A: Trust the user (current plan)**
- Document clearly: "Don't mutate Q objects after using them in multi-Q operations"
- Add warning in docstrings and README
- Accept that violations lead to undefined behavior

**Option B: Add mutation detection**
```python
class Q:
    def __init__(self, ...):
        self._version = uuid.uuid4()  # Unique version ID
        
    def merge(self, other, ...):
        # Store version at merge time
        new_changes = self._changes + [("merge", {
            "other": other,
            "other_version": other._version,  # Store version
            ...
        })]
        
    def refresh(self):
        for change_type, data in self._changes:
            if change_type == "merge":
                if data["other"]._version != data["other_version"]:
                    raise ValueError(
                        "Referenced Q has been mutated since merge. "
                        "This violates the Q contract. Use rebase() to flatten."
                    )
```
- **Pros**: Catches violations, fails fast with clear error
- **Cons**: Adds complexity, performance overhead, doesn't prevent mutation

**Option C: Deep copy on merge**
```python
def merge(self, other, ...):
    # Store a deep copy instead of reference
    other_copy = Q(other._df.copy(), ...)
    new_changes = self._changes + [("merge", {"other": other_copy, ...})]
```
- **Pros**: Fully safe, no mutation issues
- **Cons**: Violates "no deep copies" principle, high memory cost, expensive

**Recommendation:** **Option A** with excellent documentation. Add Option B (mutation detection) as a Phase 2.5 enhancement if users report issues.

---

### 2. Column Name Conflicts in Merge/Join ⚠️⚠️

**The Problem:**
```python
q1 = Q(pd.DataFrame({'id': [1, 2], 'status': ['active', 'inactive']}))
q2 = Q(pd.DataFrame({'id': [1, 2], 'status': ['pending', 'complete']}))

q3 = q1.merge(q2, on='id')
# Pandas creates: id, status_x, status_y

# User tries:
q3.extend(flag=lambda x: x.status)  # AttributeError! No 'status' attribute
```

**Why It Matters:**
- Namedtuples with `status_x` breaks the intuitive row access pattern
- Users expect `x.status` to work
- Pandas default suffixes `_x` and `_y` are not self-documenting

**Solutions to Consider:**

**Option A: Require explicit suffixes**
```python
# Force users to specify
q3 = q1.merge(q2, on='id', suffixes=('_self', '_other'))
# or
q3 = q1.merge(q2, on='id', suffixes=('_orders', '_customers'))
```
- **Pros**: No ambiguity, clear intent, self-documenting
- **Cons**: More verbose, users might forget

**Option B: Use better defaults**
```python
# Default to _self and _other instead of _x and _y
def merge(self, other, ..., suffixes=('_self', '_other')):
    ...
```
- **Pros**: Better than pandas defaults, still works if ignored
- **Cons**: Still not as clear as explicit naming

**Option C: Detect and warn**
```python
def merge(self, other, on, ...):
    conflicts = set(self.columns) & set(other.columns) - set(on)
    if conflicts and suffixes is None:
        raise ValueError(
            f"Column conflicts detected: {conflicts}. "
            f"Specify suffixes parameter: merge(other, on='{on}', suffixes=('_a', '_b'))"
        )
```
- **Pros**: Forces users to think about conflicts
- **Cons**: Breaks for simple cases where conflicts are intended

**Recommendation:** **Option B + Option C**: Use `('_self', '_other')` as defaults, but **require** explicit suffixes if conflicts exist. This balances usability with clarity.

```python
def merge(self, other, on, how='inner', suffixes=None):
    conflicts = set(self.columns) & set(other.columns) - {on} if isinstance(on, str) else set(on)
    
    if conflicts and suffixes is None:
        raise ValueError(
            f"Column name conflicts detected: {sorted(conflicts)}. "
            f"Specify suffixes to disambiguate: "
            f"merge(other, on='{on}', suffixes=('_left', '_right'))"
        )
    
    suffixes = suffixes or ('_self', '_other')
    # ... rest of implementation
```

---

### 3. Idempotency with Referenced Q Source Paths ⚠️⚠️

**The Problem:**
```python
q1 = Q(df1)
q2 = Q(df2, source_path='data2.csv')

q3 = q1.merge(q2, on='id')  # Stores reference to q2

# Later, data2.csv is updated by external process
q3.refresh()  # Uses stale q2 reference (old data!)
q3.reload()   # Reloads q1's source, but what about q2?
```

**Why It Matters:**
- Q contract promises idempotency
- External data changes break this guarantee
- `refresh()` uses stale reference
- `reload()` behavior is ambiguous for multi-Q operations

**Solutions to Consider:**

**Option A: Shallow reload (reload only primary Q)**
```python
def reload(self):
    """Reload only this Q's source, use current state of referenced Qs."""
    if not self._source_path:
        raise ValueError("No source path specified")
    
    fresh_base = load_csv(self._source_path, skip_rows=self._skip_rows)
    # Referenced Qs in _changes keep their current state
    return Q(fresh_base, source_path=self._source_path, skip_rows=self._skip_rows)
```
- **Pros**: Simple, predictable, no cascading reloads
- **Cons**: Breaks idempotency if referenced Q sources change

**Option B: Deep reload (reload all referenced Qs)**
```python
def reload(self):
    """Reload this Q and all referenced Qs from their sources."""
    if not self._source_path:
        raise ValueError("No source path specified")
    
    fresh_base = load_csv(self._source_path, skip_rows=self._skip_rows)
    
    # Deep copy changes, reloading referenced Qs
    fresh_changes = []
    for change_type, data in self._changes:
        if change_type in ("merge", "join", "concat"):
            other_q = data["other"]
            if other_q._source_path:
                # Reload the referenced Q
                reloaded_other = other_q.reload()
                data = {**data, "other": reloaded_other}
        fresh_changes.append((change_type, data))
    
    new_df = self._apply_changes(fresh_base, fresh_changes)
    return Q(new_df, source_path=self._source_path, ..., changes=fresh_changes)
```
- **Pros**: Maintains idempotency, all sources refreshed
- **Cons**: Complex, expensive, might reload same Q multiple times

**Option C: Freeze references (snapshot at merge time)**
```python
def merge(self, other, on, ...):
    # Store snapshot of other's _df at merge time
    snapshot = other._df.copy()  # Deep copy!
    new_changes = self._changes + [("merge", {
        "other_snapshot": snapshot,
        "on": on,
        ...
    })]
```
- **Pros**: Perfect idempotency, no stale data issues
- **Cons**: Violates "no deep copies" principle, high memory cost

**Recommendation:** **Option A (shallow reload)** with clear documentation:

```python
def reload(self):
    """Reload this Q's source data and re-apply all changes.
    
    NOTE: For multi-Q operations (merge/join/concat), this reloads ONLY
    this Q's source. Referenced Q objects maintain their current state.
    
    If you need to reload all data sources:
        q1_fresh = q1.reload()
        q2_fresh = q2.reload()
        q3_fresh = q1_fresh.merge(q2_fresh, on='id')
    
    Returns:
        New Q with reloaded base data and all changes re-applied
    """
```

This is simple, predictable, and doesn't violate the "no deep copies" principle. Users who need fully fresh data can reload each Q explicitly.

---

### 4. Set Operations Require Deterministic Ordering ⚠️⚠️

**The Problem:**
```python
q1 = Q(pd.DataFrame({'id': [3, 1, 2]}))
q2 = Q(pd.DataFrame({'id': [2, 4, 3]}))

q_union = q1.union(q2)
# What order? [3,1,2,4]? [1,2,3,4]? [3,1,2,2,4,3]?

# Run again:
q_union2 = q1.union(q2)
# Same order? (idempotency!)
```

**Why It Matters:**
- Q contract promises idempotency
- SQL set operations don't guarantee order
- Pandas operations might produce non-deterministic order
- Row position matters for `head()`, `tail()`, iteration

**Solutions to Consider:**

**Option A: Preserve insertion order from first Q**
```python
def union(self, other):
    """Union: all rows from self, then unique rows from other."""
    # Keep all from self, add new from other
    combined = pd.concat([self._df, other._df], ignore_index=True)
    result = combined.drop_duplicates(keep='first')  # Keep first occurrence
    # Order: self's rows first, then other's unique rows
```
- **Pros**: Intuitive, preserves order, deterministic
- **Cons**: Not true SQL UNION (which has no order)

**Option B: Always sort to ensure determinism**
```python
def union(self, other):
    """Union: all unique rows, sorted for determinism."""
    combined = pd.concat([self._df, other._df], ignore_index=True)
    result = combined.drop_duplicates()
    result = result.sort_values(by=result.columns.tolist()).reset_index(drop=True)
    # Order: always sorted by all columns
```
- **Pros**: Guaranteed deterministic, matches SQL UNION behavior
- **Cons**: Changes row order, might be surprising, expensive

**Option C: Document as non-deterministic**
```python
def union(self, other):
    """Union: all unique rows. Order is not guaranteed.
    
    Use .sort() after union if you need deterministic ordering.
    """
```
- **Pros**: Honest, simple, performant
- **Cons**: Violates idempotency guarantee

**Recommendation:** **Option A** - Preserve insertion order from first Q. This maintains idempotency, is intuitive, and users can explicitly sort if they want different ordering.

```python
def union(self, other) -> 'Q':
    """Union: all unique rows from self and other.
    
    Preserves order: self's rows first (in original order), 
    then unique rows from other (in original order).
    Duplicate rows are removed (keeping first occurrence).
    
    This ensures idempotent behavior while maintaining intuitive ordering.
    
    Example:
        >>> q1 = Q(pd.DataFrame({'x': [1, 2, 3]}))
        >>> q2 = Q(pd.DataFrame({'x': [2, 3, 4]}))
        >>> q1.union(q2)  # x: [1, 2, 3, 4]
    """
```

---

## Medium Gotchas (Should Address)

### 5. Memory Explosion with Concat Chains ⚠️

**The Problem:**
```python
q1 = Q(df1)  # 10MB
q2 = q1.concat(Q(df2))  # References q1 + has df2 → 20MB total
q3 = q2.concat(Q(df3))  # References q2 (which refs q1) → 30MB total
q4 = q3.concat(Q(df4))  # References q3 (which refs q2, q1) → 40MB total

# Each Q keeps entire chain alive!
# Plus: base_df for each Q is stored separately
```

**Why It Matters:**
- Long concat chains grow memory linearly
- Each Q in chain maintains its own base_df
- References prevent garbage collection
- Users might not realize they need to `rebase()`

**Solutions:**

**Option A: Document and trust users**
```python
def concat(self, other) -> 'Q':
    """Concatenate rows from other Q to this Q.
    
    NOTE: This stores a reference to the other Q. Long concat chains
    can accumulate memory. Use rebase() to flatten:
    
        q2 = q1.concat(other1).concat(other2).concat(other3)
        q2 = q2.rebase()  # Flatten to single base, clear references
    """
```

**Option B: Auto-rebase after N operations**
```python
def concat(self, other) -> 'Q':
    new_changes = self._changes + [("concat", {"other": other})]
    new_df = self._apply_changes(self._base_df, new_changes)
    result = self._copy_with(df=new_df, changes=new_changes)
    
    # Auto-flatten if chain gets too long
    if len(result._changes) > 10:  # Configurable threshold
        result = result.rebase()
    
    return result
```
- **Pros**: Automatic memory management
- **Cons**: Loses replay history, might be unexpected

**Option C: Optimize concat specifically**
```python
def concat(self, other) -> 'Q':
    """Concatenate - special case: no need to store reference."""
    # Instead of storing reference, just concat the DataFrames
    combined = pd.concat([self._df, other._df], ignore_index=True)
    # Don't track as change - this is a new base state
    return Q(combined)
```
- **Pros**: No memory chain, simple
- **Cons**: Breaks replay, violates change tracking contract

**Recommendation:** **Option A** - Document clearly and provide `memory_usage()` to help users monitor. This maintains the Q contract while educating users about memory implications.

---

### 6. Self-References in Merge ⚠️

**The Problem:**
```python
# Self-join: employees to managers
q = Q(pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'manager_id': [None, 1, 1]
}))

q2 = q.merge(q, left_on='manager_id', right_on='emp_id')
# Does this work? Circular reference?
```

**Why It Matters:**
- Self-joins are legitimate use case
- Storing "reference to self" might cause issues
- Need to test carefully

**Solution:**
- Should work naturally with reference approach
- Store reference to `self` in change history
- On replay, "self" resolves correctly
- **Action:** Write comprehensive tests for self-joins

---

### 7. Column Tracking After Rename in Multi-Q ⚠️

**The Problem:**
```python
q1 = Q(df1).rename(user_id='id')  # Now has 'id' not 'user_id'
q2 = Q(df2)  # Has 'id' column

q3 = q1.merge(q2, on='id')  # Does this work?
```

**The Answer:**
- **It works!** Because rename is in q1's change history
- `q1._df` has 'id' column after rename
- Merge sees the renamed columns

**Why It's Confusing:**
- User might think "q1 originally had 'user_id', will merge fail?"
- But Q contract says operations return new Q with transformed state
- **Action:** Document this clearly with examples

---

## Minor Gotchas (Can Defer)

### 8. Type Preservation in Concat ⚠️

```python
q1 = Q(pd.DataFrame({'id': [1, 2, 3]}))      # int64
q2 = Q(pd.DataFrame({'id': ['4', '5', '6']})) # object (string)

q3 = q1.concat(q2)
# Result: 'id' column is object dtype
# Subsequent operations: q3.filter(lambda x: x.id > 2)  # Fails! '4' > 2 ?
```

**Solution:** Document that concat follows pandas behavior. Users should ensure compatible types or use `dtype` parameter when loading.

---

### 9. Replay Performance with Large Merges ⚠️

```python
q1 = Q(large_df)   # 1M rows
q2 = Q(large_df2)  # 1M rows

q3 = q1.merge(q2, on='id').filter(lambda x: x.status == 'active')

q3.refresh()  # Re-executes 1M x 1M merge, then filters
# Why not cache the merge result?
```

**Trade-off:**
- Correctness vs performance
- Caching breaks replay guarantees
- **Solution:** Encourage `rebase()` after expensive operations

---

### 10. Merge on Multiple Columns

```python
# Need to support:
q.merge(other, on=['col1', 'col2'])  # Composite key
q.merge(other, left_on=['a', 'b'], right_on=['x', 'y'])  # Different names
```

**Solution:** Support both `on` (single column or list) and `left_on`/`right_on` pairs.

---

## Implementation Recommendations

### Phase 2 Implementation Order

Implement in this order to validate architecture early and minimize risk:

1. ✅ **`concat(other)`** - Simplest, tests reference architecture
2. ✅ **Self-concat test** - Validates circular reference handling  
3. ✅ **`merge(other, left_on, right_on, how, suffixes)`** - Core functionality
4. ✅ **Column conflict detection** - Validate suffixes requirement
5. ✅ **Decide and document reload() behavior** - Shallow reload with clear docs
6. ✅ **`join(other, on, how)`** - Wrapper around merge
7. ✅ **Set operations** - `union()`, `intersect()`, `difference()` with deterministic ordering

### Required for Phase 2.0 Release

**Must implement:**
- [ ] Explicit column conflict handling (require suffixes)
- [ ] Deterministic ordering for set operations
- [ ] Comprehensive documentation of Q reference contract
- [ ] Self-join tests

**Should implement:**
- [ ] Clear reload() behavior documentation
- [ ] Memory usage warnings in concat docs
- [ ] Examples of all gotchas in README

**Can defer to Phase 2.1:**
- [ ] Mutation detection (version IDs)
- [ ] Auto-rebase heuristics
- [ ] Deep reload option

---

## Documentation Requirements

### Must Add to README

**Multi-Q Operations Contract:**
```markdown
### Working with Multiple Q Objects

When you merge, join, or concatenate Q objects, Bend stores references
to the other Q objects (not copies). This enables replay but requires
following these rules:

**The Rules:**
1. Don't mutate Q objects after using them in multi-Q operations
2. `reload()` only reloads the primary Q's source, not referenced Qs
3. Column name conflicts require explicit `suffixes` parameter
4. Long concat chains accumulate memory - use `rebase()` to flatten

**Example:**
```python
q1 = Q(df1)
q2 = Q(df2)

# This stores a reference to q2
q3 = q1.merge(q2, on='id', suffixes=('_orders', '_customers'))

# DON'T do this:
q2._df = something  # ❌ Breaks q3.refresh()

# DO this if you need to free memory:
q3 = q3.rebase()  # ✅ Flattens, drops q2 reference
```

**Must Add to Docstrings:**
- Every multi-Q method needs WARNING section about references
- Every multi-Q method needs MEMORY note about rebase()
- `reload()` needs clear explanation of shallow behavior

---

## Testing Requirements

### Critical Test Cases

**Concat:**
- [ ] Basic concat
- [ ] Self-concat (concat with itself)
- [ ] Concat chain (A.concat(B).concat(C))
- [ ] Concat with refresh/reload
- [ ] Concat with different column types

**Merge:**
- [ ] Basic merge on single column
- [ ] Merge on multiple columns
- [ ] Merge with different column names (left_on/right_on)
- [ ] All join types: inner, left, right, outer
- [ ] Self-merge (employee-manager)
- [ ] Column conflicts with suffixes
- [ ] Column conflicts without suffixes (should error)
- [ ] Merge then filter then refresh
- [ ] Merge after rename

**Set Operations:**
- [ ] Union preserves order
- [ ] Union with duplicates
- [ ] Intersect preserves order
- [ ] Difference preserves order
- [ ] Chaining set operations

**Edge Cases:**
- [ ] Merge with empty Q
- [ ] Concat with empty Q
- [ ] Merge then rebase then refresh
- [ ] Multiple Q references in same pipeline

---

## Open Questions

1. **Should we support pandas merge parameters?**
   - `indicator=True` to add _merge column?
   - `validate='one_to_one'` for data validation?
   
2. **Should union() deduplicate by default?**
   - SQL UNION does, UNION ALL doesn't
   - Maybe provide both: `union()` and `union_all()`?

3. **How to handle merge with Q that has hidden columns?**
   - Should hidden columns participate in merge?
   - Current answer: Yes, hiding is display-only

4. **Memory threshold for warnings?**
   - When should `memory_usage()` warn about large references?
   - Maybe: warn if referenced Qs total > 100MB?

---

## Conclusion

Phase 2 is feasible with the reference-based architecture, but requires:
- **Clear documentation** of the Q reference contract
- **Explicit handling** of column conflicts
- **Well-defined semantics** for reload() and set operations
- **Comprehensive testing** of edge cases

The biggest risks are around **idempotency guarantees** when referenced Q objects change. The solution is clear documentation and trusting users to follow the contract (gentleman's agreement), with optional mutation detection in a future release.

**Overall assessment:** Phase 2 can proceed with careful attention to these gotchas. The reference architecture is sound.

