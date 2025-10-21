# Output

Displaying and exporting data.

## Methods

### [show()](show.md)
Print preview to console.

```python
q.show(50)  # Print 50 rows
```

### [to_df()](to_df.md)
Export to pandas DataFrame.

```python
df = q.to_df()
# Use pandas directly
```

### [dump()](dump.md)
Export to CSV file.

```python
q.dump('output.csv')
```

## Quick Reference

```python
# Display
q.show()            # Print 20 rows (default)
print(q)            # Same as show()

# Export
df = q.to_df()      # Get pandas DataFrame
q.dump('out.csv')   # Write CSV file

# Chaining
q.filter(...).show().assign(...)  # Show intermediate result
```
