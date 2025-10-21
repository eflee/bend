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

1. **Deep Copy by Default with Optional Reference Mode** - Use `copy.deepcopy(other)` by default, optional `deep_copy=False` for performance
2. **Explicit Conflict Resolution** - Require user-provided `resolve` parameter for column conflicts
3. **Tree-Based History** - Change history becomes a tree, `refresh()` and `reload()` do recursive traversal
4. **Reproducibility Tracking** - Add `q.reproducible` flag that propagates through operations
5. **Follow Pandas Conventions** - Use pandas behavior for `union()`, `merge()`, etc.
6. **Row-by-Row Conflict Resolution** - Use `apply()` for conflict lambda application
7. **Correctness Over Performance** - Prioritize reproducibility and correctness by default, with performance options

---

## Critical Gotchas (Must Address)

### 1. Replay with Mutated Q References ⚠️⚠️⚠️

**The Problem:**
```python
q1 = Q(df1)
q2 = Q(df2)

q3 = q1.merge(q2, on='id')  # What happens to q2?

# Can user continue using q2?
q2_filtered = q2.filter(lambda x: x.status == 'active')
```

**Original Concern:**
- If we store references, user mutations break replay
- Requires "gentleman's agreement" not to mutate referenced Qs

**FINAL DECISION: Deep Copy by Default, Optional Reference Mode**

```python
import copy

def merge(self, other, on, how='inner', resolve=None, deep_copy=True):
    """Merge two Q objects.
    
    Args:
        deep_copy: If True (default), stores a deep copy of other Q for full
                   reproducibility. If False, stores reference and marks result
                   as non-reproducible.
    """
    if deep_copy:
        # Store FULL deep copy of other Q including its history
        other_copy = copy.deepcopy(other)
        new_reproducible = self._reproducible and other._reproducible
    else:
        # Store reference - faster but breaks reproducibility guarantee
        other_copy = other  # Just a reference
        new_reproducible = False  # Can't guarantee reproducibility!
    
    new_changes = self._changes + [("merge", {
        "other": other_copy,
        "on": on,
        "how": how,
        "resolve": resolve,
        "deep_copy": deep_copy
    })]
    
    return self._copy_with(..., reproducible=new_reproducible)
```

**Why This Design:**
- ✅ **Safe by default**: `deep_copy=True` gives full reproducibility
- ✅ **User control**: Advanced users can opt into `deep_copy=False` for performance
- ✅ **Clear contract**: `deep_copy=False` immediately sets `reproducible=False`
- ✅ **No surprises**: Reproducibility flag tells users exactly what they have
- ⚠️ **Power user option**: `deep_copy=False` for large datasets where replay isn't needed

**Use Cases:**

```python
# Default: Safe, reproducible (recommended)
q3 = q1.merge(q2, on='id', resolve={'status': lambda a, b: a})
print(q3.reproducible)  # True (if q1 and q2 are reproducible)

# Advanced: Fast, non-reproducible (for large datasets)
q3 = q1.merge(huge_q, on='id', resolve={...}, deep_copy=False)
print(q3.reproducible)  # False (marked as non-reproducible)
q3.refresh()  # Works, but might produce different results if huge_q changes
```

**Trade-offs:**
- Default (`deep_copy=True`): Higher memory, full safety
- Optional (`deep_copy=False`): Lower memory, manual responsibility
- Reproducibility flag makes the trade-off explicit

---

### 2. Column Name Conflicts in Merge/Join ⚠️⚠️

**The Problem:**
```python
q1 = Q(pd.DataFrame({'id': [1, 2], 'status': ['active', 'inactive']}))
q2 = Q(pd.DataFrame({'id': [1, 2], 'status': ['pending', 'complete']}))

q3 = q1.merge(q2, on='id')
# What happens to the 'status' column?
```

**FINAL DECISION: Explicit Resolution Required**

```python
def merge(self, other, on, how='inner', resolve=None, deep_copy=True):
    """Merge two Q objects.
    
    Args:
        resolve: Dict mapping conflicting column names to resolution lambdas.
                 Lambda signature: lambda left_val, right_val: result_val
        deep_copy: If True (default), stores a deep copy for reproducibility.
                   If False, stores reference and marks as non-reproducible.
                 
    Raises:
        ValueError: If column conflicts exist without complete resolution
    """
    # 1. Detect conflicts
    on_cols = {on} if isinstance(on, str) else set(on)
    conflicts = set(self.columns) & set(other.columns) - on_cols
    
    # 2. Require resolution for ALL conflicts
    if conflicts:
        if not resolve or not all(c in resolve for c in conflicts):
            raise ValueError(
                f"Column conflicts detected: {sorted(conflicts)}. "
                f"Must provide resolution for ALL conflicts.\n"
                f"Example: merge(other, on='id', resolve={{\n"
                + "\n".join(f"    '{c}': lambda left, right: left," 
                           for c in sorted(conflicts))
                + "\n})"
            )
```

**Resolution Examples:**
```python
# Take left value
q3 = q1.merge(q2, on='id', resolve={'status': lambda a, b: a})

# Take right value  
q3 = q1.merge(q2, on='id', resolve={'status': lambda a, b: b})

# Custom logic: coalesce (prefer non-null)
q3 = q1.merge(q2, on='id', resolve={
    'status': lambda a, b: a if pd.notna(a) else b
})

# Multiple conflicts
q3 = q1.merge(q2, on='id', resolve={
    'status': lambda a, b: a,
    'priority': lambda a, b: max(a, b),
    'updated_at': lambda a, b: max(a, b)
})

# If you want BOTH columns, rename first:
q1_renamed = q1.rename(status='order_status')
q2_renamed = q2.rename(status='customer_status')
q3 = q1_renamed.merge(q2_renamed, on='id')  # No conflict now
```

**Implementation: Row-by-Row Application**

```python
# 3. Merge with internal suffixes
merged = pd.merge(
    self._df, 
    other._df, 
    on=on, 
    how=how, 
    suffixes=('_BEND_L', '_BEND_R')
)

# 4. Apply resolution lambdas row-by-row
for col, resolve_fn in (resolve or {}).items():
    left_col = f"{col}_BEND_L"
    right_col = f"{col}_BEND_R"
    
    if left_col in merged.columns:
        # Use pandas apply() with axis=1 for row-by-row processing
        merged[col] = merged.apply(
            lambda row: resolve_fn(row[left_col], row[right_col]), 
            axis=1
        )
        merged = merged.drop(columns=[left_col, right_col])

# 5. Preserve column ordering from left Q
# (Implementation details below)
```

**Why This Approach:**
- ✅ **Explicit**: Forces user to think about conflicts
- ✅ **No Silent Data Loss**: No automatic suffixes that confuse users
- ✅ **Flexible**: Lambda can implement any resolution logic
- ✅ **Consistent with Q**: Same row-by-row lambda pattern as `extend()`
- ✅ **Trackable**: Resolution logic stored in change history for replay

**Trade-offs:**
- ⚠️ **More Verbose**: User must write resolution lambdas
- ⚠️ **Performance**: Row-by-row `apply()` is ~10ms for 10K rows (acceptable)
- ✅ **Correctness Over Convenience**: Explicit is better than implicit

---

### 2a. Implementation Detail: Column Ordering

**Challenge:** Where should resolved columns appear?

```python
df1 = pd.DataFrame({'id': [1], 'a': [10], 'status': ['x'], 'b': [100]})
df2 = pd.DataFrame({'id': [1], 'c': [30], 'status': ['z'], 'd': [300]})

# After merge + resolution
# Option A: status at end ['id', 'a', 'b', 'c', 'd', 'status']
# Option B: status in original position ['id', 'a', 'status', 'b', 'c', 'd']
```

**Decision: Preserve Left Q Column Order**

Resolved columns appear in their **original position from the left Q**:

```python
# After applying resolution, reorder columns
left_cols_order = list(self._df.columns)
right_cols = [c for c in other._df.columns 
              if c not in on_cols and c not in conflicts]

# Build final column order
final_cols = []
for col in left_cols_order:
    if col in merged.columns:
        final_cols.append(col)
        
# Add right Q columns at end
for col in right_cols:
    if col in merged.columns and col not in final_cols:
        final_cols.append(col)

merged = merged[final_cols]
```

**Why:**
- ✅ **Intuitive**: Left Q is primary, maintains its structure
- ✅ **Predictable**: Users know where to find columns
- ✅ **Consistent**: Matches "left join" semantics

---

### 2b. Implementation Detail: Null Handling

**Lambdas receive null values as-is. User's responsibility:**

```python
# Good - handles nulls
resolve = {'status': lambda a, b: a if pd.notna(a) else b}

# Bad - might fail on null
resolve = {'status': lambda a, b: a.upper()}  # AttributeError if a is None
```

**Documentation must clearly state:**
> Resolution lambdas may receive `None` (null) values.  
> Use `pd.notna()` to check for nulls if needed.

---

### 3. History as a Tree - Recursive Refresh ⚠️⚠️

**The Problem:**
```python
q1 = Q(df1, source_path='data1.csv')
q2 = Q(df2, source_path='data2.csv')

q3 = q1.merge(q2, on='id')  # q3 now has q2 deep-copied in its history

# Later, data2.csv is updated externally
q3.refresh()  # What happens to q2's data?
q3.reload()   # What gets reloaded?
```

**FINAL DECISION: Tree-Based History with Recursive Refresh**

With deep copies, the change history becomes a **tree** of Q objects:

```
q3's history tree:
    q3
    ├─ self: base_df1 + [extend, filter]
    └─ merge operation stores:
        └─ other: q2 (deep copy)
            ├─ base_df2
            └─ [extend, sort]
```

**Refresh Implementation (Recursive):**

```python
def refresh(self):
    """Re-apply all changes from base, recursively refreshing referenced Qs."""
    
    # Start with fresh base
    result = self._base_df.copy()
    
    # Apply each change in order
    for change_type, data in self._changes:
        if change_type in ("merge", "join", "concat", "union"):
            # Multi-Q operation - recursively refresh the other Q
            other_q = data["other"]
            other_refreshed = other_q.refresh()  # RECURSIVE CALL
            
            # Apply the multi-Q operation with refreshed other
            result = self._apply_merge(result, other_refreshed._df, data)
        else:
            # Regular single-Q operation
            result = self._apply_single_operation(result, change_type, data)
    
    return Q(result, base_df=self._base_df, changes=self._changes, ...)
```

**Reload Implementation (Deep - Recursively Reloads Tree):**

```python
def reload(self):
    """Reload this Q's source data and re-apply all changes.
    
    This recursively reloads ALL source files in the tree:
    - Reloads this Q's source from disk
    - Recursively reloads referenced Qs' sources from their disks
    - Re-applies all changes in the tree
    
    This enables workflows where source files are updated and you want
    to replay the entire transformation pipeline on fresh data.
    
    Example:
        q1 = Q(load_csv('orders.csv'))
        q2 = Q(load_csv('customers.csv'))
        q3 = q1.merge(q2, on='customer_id', resolve={...})
        
        # Later, both CSV files are updated
        q3_fresh = q3.reload()  # Reloads orders.csv AND customers.csv
    """
    if not self._source_path:
        raise ValueError("No source path specified for reload")
    
    # Reload this Q's base from disk
    fresh_base = load_csv(self._source_path, skip_rows=self._skip_rows)
    
    # Apply changes, recursively reloading referenced Qs
    result = fresh_base.copy()
    for change_type, data in self._changes:
        if change_type in ("merge", "join", "concat", "union"):
            # Multi-Q operation - recursively RELOAD the other Q
            other_q = data["other"]
            if other_q._source_path:
                # Reload from source (recursive)
                other_fresh = other_q.reload()
            else:
                # No source, just refresh from base
                other_fresh = other_q.refresh()
            
            # Apply operation with reloaded other
            result = self._apply_merge(result, other_fresh._df, data)
        else:
            # Regular operation
            result = self._apply_single_operation(result, change_type, data)
    
    return Q(
        result,
        source_path=self._source_path,
        skip_rows=self._skip_rows,
        base_df=fresh_base,
        changes=self._changes.copy(),
        ...
    )
```

**Why Deep Reload:**
- ✅ **Consistent with refresh()**: Both traverse the entire tree
- ✅ **Real-world workflows**: Source files are updated, need fresh data everywhere
- ✅ **Full reproducibility**: Entire tree rebuilt from disk
- ✅ **Distinction from refresh()**:
  - `refresh()`: Re-applies changes to **in-memory** base data
  - `reload()`: Re-loads **from disk**, then re-applies changes

**Use Case:**
```python
# Initial pipeline
q1 = Q(load_csv('orders.csv'))
q2 = Q(load_csv('customers.csv'))  
q3 = q1.merge(q2, on='customer_id', resolve={'status': lambda a, b: a})

# External process updates both CSV files
# ... (orders.csv and customers.csv modified on disk) ...

# Reload entire tree from updated files
q3_updated = q3.reload()  
# ✅ Loads fresh orders.csv
# ✅ Loads fresh customers.csv (from q2's source_path)
# ✅ Re-applies merge with fresh data
```

---

**Memory Implications with Optional Deep Copy:**
```python
# Scenario 1: Default (deep_copy=True)
q1 = Q(df1)  # 10MB
q2 = Q(df2)  # 10MB  
q3 = q1.merge(q2, on='id')  # deep_copy=True by default

# q3 stores:
# - q3._base_df: 10MB (copy of df1)
# - q3._df: 15MB (merged result)
# - q3._changes["merge"]["other"]: full q2 with base_df2 (10MB)
# Total: ~35MB for this tree

# Scenario 2: Reference mode (deep_copy=False)
q1 = Q(df1)  # 10MB
q2 = Q(df2)  # 10MB
q3 = q1.merge(q2, on='id', deep_copy=False)  # Just reference

# q3 stores:
# - q3._base_df: 10MB
# - q3._df: 15MB
# - q3._changes["merge"]["other"]: reference to q2 (no additional memory)
# Total: ~25MB (saves 10MB)
# BUT: q3.reproducible = False

# After rebase (works for both):
q3_flat = q3.rebase()
# - q3_flat._base_df: 15MB (current state becomes base)
# - q3_flat._changes: [] (empty)
# Total: 15MB
```

**Trade-off:** Default safety (deep copy) with opt-in performance (reference mode).

---

### 4. Reproducibility Tracking ⚠️

**The Problem:**
```python
q1 = Q(df).sample(10)  # Is this reproducible?
q2 = Q(df2)
q3 = q1.merge(q2, on='id')

q3.refresh()  # Will produce different results if sample was non-deterministic
```

**FINAL DECISION: Add `reproducible` Public Property + Non-Idempotent Defaults**

```python
class Q:
    def __init__(self, ...):
        self._reproducible = True  # Tracks if operations are deterministic
        
    @property
    def reproducible(self) -> bool:
        """Check if this Q's operations are reproducible.
        
        Returns False if any operation in the history (including referenced Qs)
        contains non-deterministic behavior.
        
        Example:
            >>> q = Q(df).sample(5, random_state=42)  # Explicit seed
            >>> q.reproducible
            True
            >>> q2 = Q(df).sample(5)  # Default: non-deterministic
            >>> q2.reproducible
            False
        """
        return self._reproducible
```

**Change to sample() Default Behavior:**

```python
def sample(self, n=None, frac=None, random_state=None):
    """Random sample of rows.
    
    Args:
        random_state: Random seed. If None (default), sampling is non-deterministic.
                      Pass an integer to make sampling reproducible.
    
    **Idempotent**: No (by default)
                    Yes (if random_state is provided)
    """
    # Non-deterministic by default!
    new_reproducible = self._reproducible and (random_state is not None)
    
    # If random_state is None, uses different random samples each time
    new_changes = self._changes + [("sample", {
        "n": n,
        "frac": frac,
        "random_state": random_state
    })]
    
    return self._copy_with(..., reproducible=new_reproducible)
```

**Rationale for Non-Idempotent Default:**
- ✅ **Natural behavior**: Random sampling should be random by default
- ✅ **Explicit opt-in**: Users who need reproducibility pass `random_state=42`
- ✅ **Clear signal**: `q.reproducible` flag tells users immediately
- ✅ **Matches user expectations**: "Give me a random sample" → actually random
- ✅ **Power when needed**: Pass seed for reproducibility in tests/demos

**Examples:**
```python
# Non-reproducible (default)
q_random = Q(df).sample(100)
print(q_random.reproducible)  # False
q_random.refresh()  # Different sample each time!

# Reproducible (explicit seed)
q_fixed = Q(df).sample(100, random_state=42)
print(q_fixed.reproducible)  # True
q_fixed.refresh()  # Same sample every time

# In multi-Q operations
q1 = Q(df1).sample(50)  # Non-reproducible
q2 = Q(df2)
q3 = q1.merge(q2, on='id', resolve={...})
print(q3.reproducible)  # False (contaminated by q1)
```

**Propagation Rules ("Like Gonorrhea"):**

1. **Starts True**: New Q objects are reproducible by default
2. **Non-deterministic operations set to False**:
   - `sample(random_state=None)` ← Default behavior!
   - `merge(..., deep_copy=False)` ← Reference mode
   - Any future time-dependent operations
3. **Once False, Always False**: Propagates through all subsequent operations
4. **Multi-Q operations**: If ANY referenced Q is non-reproducible, result is non-reproducible

```python
def merge(self, other, ..., deep_copy=True):
    if deep_copy:
        new_reproducible = self._reproducible and other._reproducible
    else:
        new_reproducible = False  # Reference mode breaks reproducibility
    return self._copy_with(..., reproducible=new_reproducible)
```

**User Impact:**

```python
q = Q(df).filter(...).extend(...)
if not q.reproducible:
    print("Warning: This pipeline contains non-deterministic operations")
    print("refresh() and reload() may produce different results")
else:
    print("Safe to refresh() - results will be identical")
```

**No Blocking:** 
- `refresh()` and `reload()` work **regardless** of `reproducible` flag
- The flag is **informational only**
- Users decide whether to proceed based on flag value
- No warnings, no exceptions - just data

**Documentation Strategy:**

Every method docstring includes reproducibility status:
```python
def sample(self, n, random_state=None):
    """...
    
    **Idempotent**: No (by default, random_state=None)
                    Yes (if random_state is provided)
    """
    
def merge(self, other, ..., deep_copy=True):
    """...
    
    **Idempotent**: Yes (if deep_copy=True and both Q objects are reproducible)
                    No (if deep_copy=False)
    """
```

---

### 5. Set Operations - Deterministic Ordering ⚠️⚠️

**The Problem:**
```python
q1 = Q(pd.DataFrame({'id': [3, 1, 2]}))
q2 = Q(pd.DataFrame({'id': [2, 4, 3]}))

q_union = q1.union(q2)
# What order? [3,1,2,4]? [1,2,3,4]? [3,1,2,2,4,3]?
```

**FINAL DECISION: Follow Pandas with Row-Level Deduplication**

```python
def union(self, other) -> 'Q':
    """Union: all unique rows from self and other.
    
    Removes duplicate rows by comparing ALL columns.
    Preserves order: self's rows first (in original order), 
    then unique rows from other (in original order).
    
    NOTE: This is row-level deduplication (entire row must match).
    If you want to keep all rows including duplicates, use concat() instead.
    
    **Idempotent**: Yes
    """
    combined = pd.concat([self._df, other._df], ignore_index=True)
    result = combined.drop_duplicates(keep='first')
    # Order: preserved from self, then unique from other
    # Deterministic and idempotent
```

**Why This Approach:**
- ✅ **Deterministic**: Same inputs always produce same order
- ✅ **Idempotent**: Repeatable results
- ✅ **Intuitive**: Self's rows first, then other's new rows
- ✅ **Pandas-consistent**: Uses standard pandas behavior
- ✅ **Documented**: Clear explanation of row-level deduplication

**Alternative if Deduplication Not Wanted:**
```python
# If user wants all rows (no deduplication)
q3 = q1.concat(q2)  # Keeps all rows including duplicates
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

**FINAL DECISION: Deep Copy Self**

When merging Q with itself, we store a **deep copy of self** in the change history, not a reference to self:

```python
def merge(self, other, ..., deep_copy=True):
    # Even if other is self, deep copy it!
    if deep_copy:
        if other is self:
            # Self-join: deep copy self to avoid circular reference
            other_copy = copy.deepcopy(self)
        else:
            other_copy = copy.deepcopy(other)
    else:
        # Reference mode: even for self-joins
        other_copy = other
    
    new_changes = self._changes + [("merge", {
        "other": other_copy,  # Deep copy, even if it was self
        ...
    })]
```

**Why Deep Copy Self:**
- ✅ **Avoids circular references**: Self-referencing would create infinite recursion
- ✅ **Consistent behavior**: Same logic for `merge(other)` and `merge(self)`
- ✅ **Refresh works**: Recursive refresh doesn't infinitely loop
- ✅ **Clear semantics**: "At merge time, I captured self's state"

**Memory Implication:**
```python
q = Q(df)  # 10MB
q2 = q.merge(q, left_on='manager_id', right_on='emp_id')

# q2 stores:
# - q2._base_df: 10MB (q's base)
# - q2._df: 12MB (joined result)
# - q2._changes["merge"]["other"]: deep copy of q (10MB + history)
# Total: ~32MB (self is deep-copied)
```

**Self-Join is Safe:**
```python
# Employee-manager self-join
employees = Q(load_csv('employees.csv'))
with_managers = employees.merge(
    employees,
    left_on='manager_id',
    right_on='emp_id',
    resolve={'name': lambda emp_name, mgr_name: emp_name}  # Keep employee name
)

# Works perfectly - employees is deep-copied at merge time
with_managers.refresh()  # ✅ Reproduces the join

# User can continue using employees
filtered = employees.filter(lambda x: x.department == 'Engineering')  # ✅ Safe
```

**Test Requirements:**
- [x] Self-merge with all join types (inner, left, right, outer)
- [x] Self-merge with refresh/reload
- [x] Self-merge memory usage
- [x] Chained self-merges (q.merge(q).merge(q))

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

1. ✅ **`concat(other, deep_copy=True)`** - Simplest, tests deep copy architecture
2. ✅ **Self-concat test** - Validates circular reference handling  
3. ✅ **`merge(other, on, how, resolve, deep_copy=True)`** - Core functionality with conflict resolution
4. ✅ **Column conflict detection** - Validate resolve parameter requirement
5. ✅ **Deep reload() implementation** - Recursive reload parallel to refresh()
6. ✅ **Self-merge test** - Deep copy of self to avoid circular references
7. ✅ **`join(other, on, how, deep_copy=True)`** - Wrapper around merge
8. ✅ **Set operations** - `union()`, `intersect()`, `difference()` with deterministic ordering

### Required for Phase 2.0 Release

**Must implement:**
- [ ] Explicit column conflict handling (require resolve parameter)
- [ ] Deterministic ordering for set operations
- [ ] Comprehensive documentation of Q deep copy contract
- [ ] Self-join tests with deep copy of self
- [ ] Deep reload() implementation (recursive)

**Should implement:**
- [ ] Deep reload() behavior documentation (recursive through tree)
- [ ] Memory usage warnings in concat docs
- [ ] Examples of all gotchas in README
- [ ] `deep_copy` parameter in all multi-Q operations

**Can defer to Phase 2.1:**
- [ ] Mutation detection (version IDs) - not needed with deep copy default
- [ ] Auto-rebase heuristics
- [ ] Performance optimizations

---

## Documentation Requirements

### Must Add to README

**Multi-Q Operations Contract:**
```markdown
### Working with Multiple Q Objects

When you merge, join, or concatenate Q objects, Bend stores **deep copies**
of the other Q objects (including their full history). This enables full
reproducibility but requires understanding these behaviors:

**The Contract:**
1. User freedom: Continue using Q objects after multi-Q operations (deep copy protects you)
2. `reload()` recursively reloads entire tree from disk (this Q AND all referenced Qs' sources)
3. Column conflicts require explicit `resolve` parameter with lambdas
4. Long operation chains accumulate memory - use `rebase()` or `deep_copy=False` to manage
5. Check `q.reproducible` to verify if pipeline is deterministic

**Example:**
```python
q1 = Q(df1)
q2 = Q(df2)

# Merge with conflict resolution
q3 = q1.merge(q2, on='id', resolve={'status': lambda left, right: left})

# q2 is deep-copied into q3's history
# You can continue using q2 without affecting q3
q2_filtered = q2.filter(...)  # ✅ Safe

# Check reproducibility
if not q3.reproducible:
    print("Pipeline contains non-deterministic operations")

# Free memory when done
q3 = q3.rebase()  # ✅ Flattens history, drops deep copies
```

**Must Add to Docstrings:**
- Every multi-Q method documents deep copy behavior and `deep_copy` parameter
- Every multi-Q method includes `resolve` parameter explanation
- Every multi-Q method documents `reproducible` flag impact
- `reload()` clearly explains deep/recursive behavior (reloads entire tree)
- All methods include **Idempotent: Yes/No** section

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
- [ ] Self-merge (employee-manager) with deep copy of self
- [ ] Column conflicts with resolve parameter
- [ ] Column conflicts without resolve (should error)
- [ ] Merge with deep_copy=True (default)
- [ ] Merge with deep_copy=False (sets reproducible=False)
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


---

## Summary of Final Decisions

### Core Architectural Choices

1. **Deep Copy by Default, Optional Reference Mode**
   - Store `copy.deepcopy(other)` by default (`deep_copy=True`)
   - Optional `deep_copy=False` for performance (marks as non-reproducible)
   - Self-joins also use deep copy to avoid circular references
   - Enables user freedom with clear trade-offs

2. **Explicit Conflict Resolution**
   - Require `resolve` parameter with lambdas
   - Signature: `lambda left_val, right_val: result_val`
   - Error if conflicts exist without resolution
   - Row-by-row application using `pandas.apply()`

3. **Tree-Based History with Recursive Refresh AND Reload**
   - Change history forms a tree of Q objects
   - `refresh()` recursively refreshes entire tree (in-memory)
   - `reload()` recursively reloads entire tree (from disk + refresh)
   - Fully reproducible if all Qs are reproducible

4. **Reproducibility Tracking**
   - Public `q.reproducible` property
   - Propagates through all operations ("like gonorrhea")
   - `sample()` is **non-idempotent by default** (random_state=None)
   - `merge(..., deep_copy=False)` sets reproducible=False
   - Informational only (doesn't block operations)
   - Documented in every method's **Idempotent** section

5. **Pandas-Consistent Behavior**
   - `union()` uses `drop_duplicates()` (row-level, all columns)
   - Preserves insertion order (self first, then other's unique rows)
   - Deterministic and idempotent

### Implementation Priorities

**Phase 2.0 Release Requirements:**
- ✅ Deep copy implementation with optional reference mode
- ✅ Conflict resolution with `resolve` parameter
- ✅ Reproducibility flag (`q.reproducible`)
- ✅ Change `sample()` default to `random_state=None` (non-idempotent)
- ✅ Deep reload (recursive, parallel to refresh)
- ✅ Self-merge with deep copy of self
- ✅ Column ordering preservation (left Q's order)
- ✅ Comprehensive documentation
- ✅ All critical test cases

**Can Defer to Phase 2.1:**
- Mutation detection (version IDs) - not needed with deep copy default
- Auto-rebase heuristics
- Performance optimizations

### Trade-offs Accepted

| Decision | Cost | Benefit |
|----------|------|---------|
| Deep copy (default) | Higher memory | User freedom, full reproducibility |
| Reference mode (opt-in) | Manual responsibility | Lower memory for large datasets |
| Explicit resolve | More verbose | Zero ambiguity, tracked in history |
| Row-by-row apply | ~10ms/10K rows | Consistency with Q philosophy |
| Deep reload | Potentially slow | Full tree refresh from disk |
| Reproducibility flag | Minor overhead | User awareness, informed decisions |
| Non-idempotent sample default | Surprising for tests | Natural random behavior |

All trade-offs prioritize **correctness and reproducibility by default** with **opt-in performance modes**.

### Success Criteria

Phase 2 is successful if:
1. ✅ All P0 requirements maintained (immutability, replay, idempotency*)
2. ✅ Users can merge/join/concat Q objects safely
3. ✅ Full tree history is reproducible via `refresh()` and `reload()`
4. ✅ Memory can be managed via `rebase()` or `deep_copy=False`
5. ✅ `q.reproducible` flag accurately reflects pipeline state
6. ✅ Self-joins work without circular references
7. ✅ Comprehensive documentation explains all behaviors

*Idempotency maintained for reproducible pipelines; non-reproducible pipelines clearly marked via flag.

---

## Next Steps

1. **Update `sample()` to be non-idempotent by default** (`random_state=None`)
2. **Add `reproducible` property to Q** (Phase 1.5)
3. Implement `concat()` with optional deep copy (validate architecture)
4. Implement `merge()` with:
   - Conflict resolution via `resolve` parameter
   - Optional `deep_copy` parameter
   - Self-merge handling (deep copy self)
   - Column ordering preservation
5. Implement deep `reload()` (recursive like `refresh()`)
6. Implement `union()`, `intersect()`, `difference()`
7. Comprehensive testing of all gotchas
8. Documentation updates (README, docstrings)
9. Phase 2.0 release

**Overall assessment:** Phase 2 design is sound, addresses all gotchas, provides user control via `deep_copy` parameter, and maintains Q contract with explicit trade-offs favoring correctness by default with performance options.

