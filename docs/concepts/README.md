# Concepts

This section covers fundamental concepts in Bend that help you understand how the tool works and what guarantees it provides.

## Essential Reading

**[Understanding Determinism & Reloadability](understanding-determinism-reloadability.md)** - Comprehensive guide to understanding:

- **Replayability** - Re-executing operations from memory
- **Reloadability** - Reloading from source files
- **Determinism** - Guaranteeing same results from same input

Learn what each concept means, how they differ, what operations can break them, and their practical implications for your data pipelines.

## Quick Reference

```python
# Check your Q's properties
q.deterministic  # Are operations deterministic?
q.reloadable     # Can this be reloaded from source?

# Methods
q.replay()       # Re-execute operations (always works)
q.reload()       # Reload from source + replay (requires reloadable=True)
```

## Related Documentation

- [Lifecycle Operations](../lifecycle/) - `reload()`, `replay()`, `rebase()` methods
- [Properties](../properties/) - `deterministic` and other read-only attributes
- [Row Operations](../row-operations/sample.md) - `sample()` and determinism
- [Multi-Q Operations](../multi-q-operations/) - How determinism and reloadability propagate

