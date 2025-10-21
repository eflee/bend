# to_df()

Export Q to a pandas DataFrame.

## Signature

```python
q.to_df() -> pd.DataFrame
```

## Returns

A **deep copy** of the underlying DataFrame.

## Basic Usage

```python
df = q.to_df()
# df is a pandas DataFrame
# Modifications to df don't affect q
```

## Use Cases

### 1. Use Pandas Directly
```python
df = q.to_df()
result = df.groupby('category').agg({'sales': ['sum', 'mean', 'std']})
```

### 2. Integration with Other Libraries
```python
df = q.to_df()
# Use with matplotlib, seaborn, scikit-learn, etc.
```

### 3. Advanced Pandas Features
```python
df = q.to_df()
pivot = df.pivot_table(index='date', columns='category', values='sales')
```

## Deep Copy

`to_df()` returns a **copy**, not a reference:

```python
df = q.to_df()
df['new_col'] = 123  # Modify DataFrame
# q is unchanged!
```

## See Also

- [`dump()`](dump.md) - Export to CSV
- [`show()`](show.md) - Print preview
