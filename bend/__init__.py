"""Bend: A precise CSV analysis tool with an intuitive query interface."""

from .core import Q, _load_csv_to_dataframe

__version__ = "1.0.0"


def load_csv(path, skip_rows=0, dtype=None):
    """Load a CSV file into a Q object.

    This is the primary way to load data in Bend. It returns a Q object
    that's ready to use, with full reload capability.

    Args:
        path: Path to CSV file or Google Sheets URL
        skip_rows: Number of rows to skip before header (default: 0)
        dtype: Optional dict of column types (e.g., {'age': int, 'price': float})

    Returns:
        Q object with the loaded data

    Examples:
        >>> # Load a CSV
        >>> q = load_csv('data.csv')
        >>>
        >>> # Load with type conversion
        >>> q = load_csv('data.csv', dtype={'age': int, 'price': float})
        >>>
        >>> # Load Google Sheets
        >>> q = load_csv('https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0')
        >>>
        >>> # Skip metadata rows
        >>> q = load_csv('data.csv', skip_rows=3)
    """
    df = _load_csv_to_dataframe(path, skip_rows=skip_rows, dtype=dtype)
    return Q(df, source_path=path, skip_rows=skip_rows)


def dump_csv(q, path):
    """Dump a Q object to a CSV file.

    Args:
        q: Q object to export
        path: Path where CSV will be written

    Returns:
        The original Q object (for chaining)

    Examples:
        >>> q = load_csv('input.csv')
        >>> result = q.filter(lambda x: x.age > 18)
        >>> dump_csv(result, 'output.csv')
        >>>
        >>> # Can chain
        >>> dump_csv(q.filter(lambda x: x.active), 'active_users.csv')
    """
    q.to_df().to_csv(path, index=False)
    return q


# Expose core classes and helpers
__all__ = ["Q", "load_csv", "dump_csv"]
