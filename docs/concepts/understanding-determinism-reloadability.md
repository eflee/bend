# Understanding Replayability, Reloadability, and Determinism

Bend tracks three important properties that help you understand what guarantees you have about your data pipeline:

1. **Replayability** - Can operations be re-executed?
2. **Reloadability** - Can the pipeline be reloaded from source files?
3. **Determinism** - Will operations produce the same results?

This guide explains each concept, how they differ, and their practical implications.

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Replayability](#replayability)
- [Reloadability](#reloadability)
- [Determinism](#determinism)
- [The Relationship Between the Three](#the-relationship-between-the-three)
- [Practical Implications](#practical-implications)
- [Operations Reference](#operations-reference)

---

## Quick Reference

| Concept | Question | Flag/Method | Can Break? |
|---------|----------|-------------|------------|
| **Replayability** | Can I re-execute operations? | Always true | No |
| **Reloadability** | Can I reload from source? | `q.reloadable` | Yes |
| **Determinism** | Same input → same output? | `q.deterministic` | Yes |

```python
# Check your Q's properties
q.reloadable      # Can this be reloaded from source?
q.deterministic   # Are operations deterministic?

# Methods
q.replay()        # Re-execute operations (always works)
q.reload()        # Reload from source + replay (requires reloadable=True)
```

---

## Replayability

### What It Means

**Replayability** is the ability to re-execute the change history against the current base DataFrame.

```python
q = Q(df)
q2 = q.assign(total=lambda x: x.price * x.qty)
q3 = q2.filter(lambda x: x.total > 100)

# Can always replay - re-applies changes to base_df
q4 = q3.replay()  # ✓ Always works
```

### Key Points

- **Always possible** - Every Q is replayable
- **No external dependencies** - Uses in-memory `_base_df` and `_changes`
- **May produce different results** - If operations are non-deterministic
- **Method**: `q.replay()`

### When to Use `replay()`

```python
# Reset after manual DataFrame manipulation
q._df  # Don't do this! But if you did...
q.replay()  # Recomputes from base + changes

# Verify change history
q2 = q.assign(x=lambda r: r.y * 2)
q3 = q2.replay()  # Confirms operations work correctly

# After modifying base (advanced)
q._base_df = new_df  # Advanced use case
q.replay()  # Re-applies all changes to new base
```

### What Can't Break Replayability

**Nothing.** Replayability is always available because it only depends on in-memory data.

---

## Reloadability

### What It Means

**Reloadability** is the ability to reload the entire pipeline from source files on disk.

```python
# Reloadable Q
q = Q(load_csv('data.csv'), source_path='data.csv')
q.reloadable  # True

# File is updated on disk
# ...

# Reload from disk + replay all changes
q2 = q.reload()  # ✓ Reloads from source
```

### Key Points

- **Requires source file** - Must have `source_path`
- **Includes entire tree** - Recursively reloads multi-Q operations
- **History must be intact** - Cleared by `rebase()`
- **Flag**: `q.reloadable` (read-only property)
- **Method**: `q.reload(allow_partial_reload=False)`

### What Breaks Reloadability

#### 1. No Source Path

```python
df = pd.DataFrame({"x": [1, 2, 3]})
q = Q(df)  # No source_path provided

q.reloadable  # False - can't reload without source
q.reload()    # ✗ ValueError: no source path
```

**Fix**: Always provide `source_path` if you need reloadability:

```python
q = Q(load_csv('data.csv'), source_path='data.csv')
q.reloadable  # True ✓
```

#### 2. Rebase Clears History

```python
q = Q(load_csv('data.csv'), source_path='data.csv')
q.reloadable  # True

q2 = q.filter(lambda x: x.age > 18).rebase()
q2.reloadable  # False - history cleared!
q2.reload()    # ✗ ValueError: not reloadable
```

**Why**: `rebase()` flattens history, making current state the new base. Without change history, `reload()` can't reconstruct the pipeline.

**When this is OK**: After complex multi-Q operations when you want to drop deep copies:

```python
q1 = load_and_process_large_dataset()
q2 = q.concat(q1).filter(...).rebase()  # Drops deep copy of q1
# Now just use q2 going forward (no reload needed)
```

#### 3. Multi-Q with Non-Reloadable Qs

```python
csv1 = 'users.csv'
csv2 = 'orders.csv'

q1 = Q(load_csv(csv1), source_path=csv1)
q2 = Q(load_csv(csv2), source_path=csv2)

# Rebase q2 (makes it non-reloadable)
q2_rebased = q2.rebase()

# Concat includes non-reloadable Q
q3 = q1.concat(q2_rebased)
q3.reloadable  # False - q2_rebased can't reload!

q3.reload()  # ✗ ValueError: Q in concat operation is not reloadable
```

**Fix with Partial Reload**:

```python
# Allow partial reload - uses current state for non-reloadable Qs
q4 = q3.reload(allow_partial_reload=True)  # ✓ Works
# Reloads q1 from disk, uses q2_rebased's current state
```

#### 4. Reference Mode (`deep_copy=False`)

```python
q1 = Q(load_csv('file1.csv'), source_path='file1.csv')
q2 = Q(load_csv('file2.csv'), source_path='file2.csv')

q3 = q1.concat(q2, deep_copy=False)
q3.reloadable  # False - references break guarantees
```

**Why**: References don't capture point-in-time state, making reload unpredictable.

### Partial Reload

When part of your pipeline is non-reloadable:

```python
q_rebased = q_some.rebase()  # Not reloadable
q_final = q_other.concat(q_rebased)

# Strict mode (default) - fails
q_final.reload()  # ✗ Error

# Partial mode - uses current state for non-reloadable parts
q_final.reload(allow_partial_reload=True)  # ✓ Works
# Reloads what it can, uses current state for the rest
```

---

## Determinism

### What It Means

**Determinism** is the guarantee that operations produce the same results given the same input.

```python
# Deterministic operations
q = Q(df)
q2 = q.filter(lambda x: x.age > 18)
q3 = q2.assign(decade=lambda x: (x.age // 10) * 10)
q3.deterministic  # True - always same result

# Non-deterministic operations
q4 = q.sample(100)  # Random sampling, no seed
q4.deterministic  # False - different results each time
```

### Key Points

- **Flag**: `q.deterministic` (read-only property)
- **Propagates** - Like "gonorrhea" (one non-deterministic Q taints the whole chain)
- **Independent of reloadability** - Can be deterministic but not reloadable
- **Affects reliability** - Non-deterministic pipelines may behave differently on replay

### What Breaks Determinism

#### 1. Random Sampling Without Seed

```python
q = Q(df)
q2 = q.sample(100)  # No random_state
q2.deterministic  # False

# Different results each time
q3 = q2.replay()  # Different 100 rows
q4 = q2.replay()  # Different 100 rows again
```

**Fix**: Provide a random seed:

```python
q2 = q.sample(100, random_state=42)
q2.deterministic  # True ✓
```

#### 2. Reference Mode (`deep_copy=False`)

```python
q1 = Q(df1)
q2 = Q(df2)

q3 = q1.concat(q2, deep_copy=False)
q3.deterministic  # False - reference breaks guarantees
```

**Why**: Without deep copying, changes to `q2` after the concat can affect `q3`.

#### 3. Propagation Through Multi-Q Operations

```python
q1 = Q(df1)
q2 = Q(df2)

# q2 becomes non-deterministic
q2_sample = q2.sample(10)  # No seed
q2_sample.deterministic  # False

# Propagates through operations
q3 = q1.concat(q2_sample)
q3.deterministic  # False - tainted by q2_sample

q4 = q3.merge(other_q, on='id')
q4.deterministic  # False - still tainted
```

### Determinism vs Reloadability

**Key insight**: These are independent!

```python
# Deterministic but NOT reloadable
df = pd.DataFrame({"x": [1, 2, 3]})
q = Q(df)  # No source_path
q2 = q.assign(y=lambda x: x.x * 2)
q2.deterministic  # True - operations are deterministic
q2.reloadable     # False - no source path

# Reloadable but NOT deterministic
q = Q(load_csv('data.csv'), source_path='data.csv')
q2 = q.sample(10)  # No seed
q2.deterministic  # False - random sampling
q2.reloadable     # True - has source path
```

### Rebase Resets Determinism

```python
q = Q(df)
q2 = q.sample(100)  # No seed
q2.deterministic  # False

q3 = q2.rebase()  # Clears history
q3.deterministic  # True - empty history is deterministic!
```

**Why**: After rebase, there's no change history. An empty history is deterministic by definition.

---

## The Relationship Between the Three

### Matrix of Possibilities

| Replayable | Reloadable | Deterministic | Example | Notes |
|------------|------------|---------------|---------|-------|
| ✓ | ✓ | ✓ | `Q(load_csv('f.csv'), source_path='f.csv').filter(...)` | Ideal state |
| ✓ | ✓ | ✗ | `Q(load_csv('f.csv'), source_path='f.csv').sample(10)` | Reloads, but random |
| ✓ | ✗ | ✓ | `Q(df).filter(...)` | Can replay, no source |
| ✓ | ✗ | ✗ | `Q(df).sample(10)` | Can replay, but random |
| ✓ | ✗ | ✓ | `q.rebase()` after any deterministic ops | Post-rebase state |
| ✓ | ✗ | ✗ | `q.rebase()` after non-deterministic ops | Post-rebase, was random |

**Note**: Replayability is ALWAYS true - it's not listed as a variable.

### Conceptual Diagram

```
┌─────────────────────────────────────────┐
│            Your Q Object                │
├─────────────────────────────────────────┤
│  Replayable: ✓ (always)                │
│  Reloadable: ? (depends on source)     │
│  Deterministic: ? (depends on ops)     │
└─────────────────────────────────────────┘
           │
           ├─ replay()  → Always works, re-executes changes
           │
           └─ reload()  → Requires reloadable=True
                         May give different results if not deterministic
```

---

## Practical Implications

### Scenario 1: Development & Debugging

**Goal**: Quick iteration, don't care about reload.

```python
# Load once, work with in-memory
df = pd.DataFrame({"x": range(1000)})
q = Q(df)

# Iterate rapidly
q2 = q.filter(...).assign(...).filter(...)
q3 = q2.replay()  # Quick verification

# No need for source_path or determinism
```

**Properties**:
- ✓ Replayable
- ✗ Not reloadable (no source)
- ✓ Deterministic (if you avoid `sample()` without seed)

### Scenario 2: Production Pipeline

**Goal**: Reproducible results, reload from updated data.

```python
# Always provide source_path
q = Q(load_csv('data.csv'), source_path='data.csv')

# Use deterministic operations
q2 = q.filter(lambda x: x.age > 18)
q3 = q2.assign(decade=lambda x: (x.age // 10) * 10)

# If sampling, use seed
q4 = q3.sample(100, random_state=42)

# Properties: All true!
q4.reloadable     # True ✓
q4.deterministic  # True ✓
```

**Benefits**:
- Can reload when source data updates
- Results are reproducible
- Can debug by replaying history

### Scenario 3: Large Multi-Q Pipelines

**Goal**: Combine multiple datasets, optimize memory.

```python
# Load multiple sources
q1 = Q(load_csv('users.csv'), source_path='users.csv')
q2 = Q(load_csv('orders.csv'), source_path='orders.csv')

# Merge with deep copy (default)
q3 = q1.merge(q2, on='user_id')
q3.reloadable  # True ✓

# Do heavy processing
q4 = q3.filter(...).assign(...).filter(...)

# Optimize memory: rebase to drop deep copies
q5 = q4.rebase()
q5.reloadable     # False - but that's OK
q5.deterministic  # True - empty history
```

**Trade-off**:
- Lose reloadability
- Gain memory efficiency
- Keep determinism (empty history)
- Can still work with q5 going forward

### Scenario 4: Handling Non-Deterministic Operations

**Goal**: Use sampling but understand implications.

```python
q = Q(load_csv('data.csv'), source_path='data.csv')

# Sample without seed (non-deterministic)
q2 = q.sample(1000)
q2.deterministic  # False ✗

# Can still reload, but results vary
q3 = q2.reload()  # Different 1000 rows each time

# If you need reproducibility, use seed
q4 = q.sample(1000, random_state=42)
q4.deterministic  # True ✓
q5 = q4.reload()  # Same 1000 rows every time
```

---

## Operations Reference

### Complete Method Reference Tables

#### Table 1: Flag Propagation for All Methods

This table shows how each method affects the `deterministic` and `reloadable` flags.

| Method | Deterministic Flag | Reloadable Flag | Notes |
|--------|-------------------|-----------------|-------|
| **Data Manipulation** |
| `assign()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| `filter(lambda ...)` | Inherited from parent | Inherited from parent | Lambda filters preserve parent flags |
| `filter(lambda ..., inverse=True)` | Inherited from parent | Inherited from parent | Inverse lambda preserves parent flags |
| `filter(other_q, on=...)` | Inherited from LEFT (self) | Inherited from LEFT (self) | Semi-join: ignores right Q flags |
| `filter(other_q, on=..., inverse=True)` | Inherited from LEFT (self) | Inherited from LEFT (self) | Anti-join: ignores right Q flags |
| `map()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| **Row Operations** |
| `sort()` | Inherited from parent | Inherited from parent | Deterministic sort (stable) |
| `head()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| `tail()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| `sample(random_state=None)` | **Sets to False** | Inherited from parent | Non-deterministic by default |
| `sample(random_state=42)` | Inherited from parent | Inherited from parent | Deterministic with seed |
| `distinct()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| **Column Operations** |
| `drop()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| `select()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| `rename()` | Inherited from parent | Inherited from parent | Always preserves both flags |
| `hide()` | Inherited from parent | Inherited from parent | Display-only, no flag change |
| `unhide()` | Inherited from parent | Inherited from parent | Display-only, no flag change |
| **Multi-Q Operations** |
| `concat(other, deep_copy=True)` | `self AND other` | `self AND other` | Both must be True for True result |
| `concat(other, deep_copy=False)` | **Sets to False** | **Sets to False** | Reference mode breaks guarantees |
| `merge(other, deep_copy=True)` | `self AND other` | `self AND other` | Both must be True for True result |
| `merge(other, deep_copy=False)` | **Sets to False** | **Sets to False** | Reference mode breaks guarantees |
| `join(other, deep_copy=True)` | `self AND other` | `self AND other` | Both must be True for True result |
| `join(other, deep_copy=False)` | **Sets to False** | **Sets to False** | Reference mode breaks guarantees |
| `union(other, deep_copy=True)` | `self AND other` | `self AND other` | Both must be True for True result |
| `union(other, deep_copy=False)` | **Sets to False** | **Sets to False** | Reference mode breaks guarantees |
| `intersect(other, deep_copy=True)` | `self AND other` | `self AND other` | Both must be True for True result |
| `intersect(other, deep_copy=False)` | **Sets to False** | **Sets to False** | Reference mode breaks guarantees |
| `difference(other, deep_copy=True)` | `self AND other` | `self AND other` | Both must be True for True result |
| `difference(other, deep_copy=False)` | **Sets to False** | **Sets to False** | Reference mode breaks guarantees |
| **Lifecycle** |
| `replay()` | Unchanged | Unchanged | Just re-executes operations |
| `reload()` | Unchanged | Unchanged | Reloads from disk + replays |
| `rebase()` | **Sets to True** | **Sets to False** | Flattens history (deterministic but not reloadable) |
| **Aggregations** |
| `groupby()` | N/A (resets) | **Sets to False** | Terminal operation, creates new Q |
| `sum()`, `mean()`, etc. | N/A (scalar) | N/A (scalar) | Returns scalar, not Q |

**Legend:**
- ✅ **Inherited from parent** - Takes the flag value from the source Q
- ✅ **Inherited from LEFT (self)** - For filtering, only considers the left Q
- ⚠️ **Sets to False** - Operation explicitly sets the flag to False
- ✅ **Sets to True** - Operation explicitly sets the flag to True
- ✅ **`self AND other`** - Result is True only if BOTH Qs have True
- ❌ **N/A** - Not applicable (returns scalar or resets Q)

#### Table 2: Determinism of Each Method

This table shows whether each method's **operation itself** is deterministic.

| Method | Deterministic? | Conditions |
|--------|---------------|------------|
| **Data Manipulation** |
| `assign()` | ✅ Yes | Deterministic if lambda is deterministic |
| `filter()` | ✅ Yes | Deterministic if predicate is deterministic |
| `map()` | ✅ Yes | Deterministic if lambda is deterministic |
| **Row Operations** |
| `sort()` | ✅ Yes | Stable sort (preserves order of equal elements) |
| `head()` | ✅ Yes | Always returns first N rows in order |
| `tail()` | ✅ Yes | Always returns last N rows in order |
| `sample(n, random_state=None)` | ❌ No | Random without seed |
| `sample(n, random_state=42)` | ✅ Yes | Deterministic with seed |
| `distinct()` | ✅ Yes | Keeps first occurrence of each unique row |
| **Column Operations** |
| `drop()` | ✅ Yes | Always drops same columns |
| `select()` | ✅ Yes | Always keeps same columns |
| `rename()` | ✅ Yes | Always renames consistently |
| `hide()` | ✅ Yes | Display-only operation |
| `unhide()` | ✅ Yes | Display-only operation |
| **Multi-Q Operations** |
| `concat(other)` | ✅ Yes | Concatenates in order |
| `merge(other, on, how='inner')` | ✅ Yes | Deterministic merge with explicit `on` |
| `merge(other, on, how='left')` | ✅ Yes | Deterministic with explicit join type |
| `join(other, on)` | ✅ Yes | Wrapper around merge (inner join) |
| `union(other)` | ✅ Yes | De-duplicates with deterministic ordering |
| `intersect(other)` | ✅ Yes | Uses explicit column matching + stable de-duplication |
| `difference(other)` | ✅ Yes | Uses explicit column matching + stable de-duplication |
| **Lifecycle** |
| `replay()` | ✅ Yes | Re-executes operations (same as original) |
| `reload()` | ✅ Yes | Reloads from disk (file content determines result) |
| `rebase()` | ✅ Yes | Flattens to current state |
| **Aggregations** |
| `groupby()` | ⚠️ Partial | Deterministic if group keys and aggregations are |
| `sum()`, `mean()`, etc. | ✅ Yes | Deterministic aggregation functions |

**Key Points:**
- ✅ **Yes** = Operation always produces same output for same input
- ❌ **No** = Operation may produce different output for same input
- ⚠️ **Partial** = Depends on lambda/function determinism

### Always Safe (Preserve All Properties)

These operations preserve both `reloadable` and `deterministic`:

- `assign()` - Add computed columns
- `filter()` - Filter rows (all variants)
- `map()` - Transform rows
- `sort()` - Sort by columns
- `head()` / `tail()` - Take first/last N rows
- `drop()` / `select()` - Column operations
- `distinct()` - Remove duplicates
- `rename()` - Rename columns
- `hide()` / `unhide()` - Hide columns from display

```python
q = Q(load_csv('f.csv'), source_path='f.csv')
q2 = q.filter(...).assign(...).sort(...)
# q2.reloadable ✓, q2.deterministic ✓
```

### May Break Determinism

#### `sample()` without seed

```python
q2 = q.sample(100)  # No random_state
# deterministic: False ✗
# reloadable: unchanged
```

**Fix**: Provide `random_state`:

```python
q2 = q.sample(100, random_state=42)
# deterministic: True ✓
```

#### Multi-Q with `deep_copy=False`

```python
q3 = q1.concat(q2, deep_copy=False)
# deterministic: False ✗
# reloadable: False ✗
```

**Fix**: Use default `deep_copy=True`:

```python
q3 = q1.concat(q2)  # deep_copy=True (default)
# Both flags preserved based on q1 and q2
```

### Breaks Reloadability

#### `rebase()`

```python
q2 = q.filter(...).rebase()
# deterministic: True ✓ (empty history)
# reloadable: False ✗ (history cleared)
```

**When to use**: After heavy operations to save memory.

#### Multi-Q with non-reloadable Q

```python
q1 = Q(load_csv('f.csv'), source_path='f.csv')
q2 = Q(df)  # No source_path

q3 = q1.concat(q2)
# reloadable: False ✗ (q2 not reloadable)
```

### Multi-Q Flag Propagation

All multi-Q operations propagate flags:

- `concat()` - Vertical stacking
- `merge()` - Join with conflict resolution
- `join()` - Simple join (no conflicts)
- `union()` - Concat + distinct
- `intersect()` - Rows in both
- `difference()` - Rows in left but not right

**Rule**: Result is deterministic/reloadable only if ALL inputs are:

```python
q1.deterministic=True,  q1.reloadable=True
q2.deterministic=True,  q2.reloadable=True
q3 = q1.merge(q2, on='id')
q3.deterministic=True ✓, q3.reloadable=True ✓

# But if q2 is non-deterministic:
q2_sample = q2.sample(10)  # Non-deterministic
q4 = q1.merge(q2_sample, on='id')
q4.deterministic=False ✗  # Tainted by q2_sample
q4.reloadable=True ✓      # Still reloadable
```

---

## Best Practices

### 1. Always Provide `source_path` in Production

```python
# Development: OK to skip
q = Q(df)

# Production: Always include
q = Q(load_csv('data.csv'), source_path='data.csv')
```

### 2. Use Seeds for Reproducible Sampling

```python
# Don't
q2 = q.sample(100)  # Non-deterministic

# Do
q2 = q.sample(100, random_state=42)  # Deterministic
```

### 3. Check Flags Before Critical Operations

```python
if not q.deterministic:
    print("Warning: Pipeline contains non-deterministic operations")
    print("Results may vary on reload/replay")

if not q.reloadable:
    print("Warning: Cannot reload from source")
    print("Save results to disk if needed")
```

### 4. Document When You Break Guarantees

```python
# Intentionally non-deterministic for variety
q_sample = q.sample(frac=0.1)  # No seed
# NOTE: Results will vary - that's the point!

# Intentionally rebase to save memory
q_final = q.concat(large_q).filter(...).rebase()
# NOTE: Not reloadable after this point - save if needed
q_final.dump('snapshot.csv')
```

### 5. Use `rebase()` Strategically

```python
# Good: After heavy multi-Q operations
q1 = load_large_dataset_1()
q2 = load_large_dataset_2()
q3 = q1.merge(q2, on='id').filter(...).rebase()
# Drops deep copies, saves memory

# Bad: Too early in pipeline
q2 = q.filter(...).rebase()  # Loses history too soon
# Can't reload anymore - premature optimization
```

### 6. Handle Partial Reload Gracefully

```python
def safe_reload(q, allow_partial=True):
    """Reload with sensible defaults."""
    try:
        return q.reload(allow_partial_reload=allow_partial)
    except ValueError as e:
        if "not reloadable" in str(e):
            print("Warning: Using current state (not reloadable)")
            return q
        raise
```

---

## Summary

- **Replayability**: Always available, re-executes operations on base data
- **Reloadability**: Requires source files and intact history, can break via `rebase()` or no source
- **Determinism**: Same input → same output, can break via random operations or references

All three are independent but complementary:
- You can be deterministic without being reloadable (no source file)
- You can be reloadable without being deterministic (random operations)
- You can always replay regardless of the other two

Check your Q's properties with `q.reloadable` and `q.deterministic` to understand what guarantees you have!

