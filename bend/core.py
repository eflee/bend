"""Core functionality for CSV loading and querying - New architecture."""

import re
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Any
import pandas as pd


def _gsheets_csv(url: str) -> str:
    """Convert a Google Sheets URL to a CSV export URL.
    
    Args:
        url: Either a regular URL or a Google Sheets URL
        
    Returns:
        CSV export URL if input was a Google Sheets URL, otherwise the original URL
    """
    m = re.search(r"/spreadsheets/d/([^/]+)/", url)
    if not m:
        return url
    sheet_id = m.group(1)
    gid = "0"
    g = re.search(r"(?:[#?&]gid=)(\d+)", url)
    if g:
        gid = g.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def load_csv(path_or_url: str, skip_rows: int = 0, dtype: dict = None) -> pd.DataFrame:
    """Load a CSV file from a local path, URL, or Google Sheets URL.
    
    Args:
        path_or_url: Path to a local CSV file, a URL, or a Google Sheets URL
        skip_rows: Number of rows to skip at the beginning of the file (default: 0)
        dtype: Dictionary of column names to data types for type conversion (default: None)
               Example: {'age': int, 'price': float, 'active': bool}
        
    Returns:
        A pandas DataFrame containing the CSV data
        
    Example:
        >>> load_csv('data.csv', dtype={'age': int, 'price': float})
    """
    u = _gsheets_csv(path_or_url)
    kwargs = {}
    if skip_rows > 0:
        kwargs['skiprows'] = skip_rows
    if dtype:
        kwargs['dtype'] = dtype
    return pd.read_csv(u, **kwargs)


def rows(df: pd.DataFrame):
    """Convert a DataFrame into an iterable of Row namedtuples with dot-accessible columns.
    
    Column names are sanitized to be valid Python identifiers by replacing
    non-alphanumeric characters with underscores.
    
    Args:
        df: A pandas DataFrame
        
    Yields:
        Row namedtuples where each column is accessible as an attribute
        
    Example:
        >>> for row in rows(df):
        ...     print(row.column_name)
    """
    safe_cols = [re.sub(r'[^0-9a-zA-Z_]', '_', c) for c in df.columns]
    df2 = df.copy()
    df2.columns = safe_cols
    return df2.itertuples(index=False, name="Row")


class Q:
    """A query interface with tracked change history and replay capabilities.
    
    Q maintains a base DataFrame and a list of changes (extensions, transformations,
    filtrations). The current state can always be reconstructed from base + changes.
    
    Args:
        df: A pandas DataFrame to wrap as the base state
        source_path: Optional path to source CSV file for reload functionality
        skip_rows: Number of rows to skip when loading from source
        
    Example:
        >>> q = Q(df, source_path='data.csv')
        >>> q2 = q.extend(total=lambda x: x.price * x.qty)
        >>> q3 = q2.filter(lambda x: x.total > 100)
        >>> q3.refresh()  # Re-apply changes to base data
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        source_path: str = None,
        skip_rows: int = 0,
        base_df: pd.DataFrame = None,
        changes: List[Tuple[str, Any]] = None,
        hidden_cols: set = None
    ):
        """Initialize a Q object with a DataFrame.
        
        Args:
            df: Current state DataFrame
            source_path: Optional path to source CSV file
            skip_rows: Number of rows to skip when loading from source
            base_df: Base DataFrame (if None, df is used as base)
            changes: List of tracked changes
            hidden_cols: Set of column names to hide from display
        """
        self._base_df = base_df if base_df is not None else df.copy()
        self._changes: List[Tuple[str, Any]] = changes or []
        self._source_path = source_path
        self._skip_rows = skip_rows
        self._hidden_cols = hidden_cols or set()
        self._df = df
    
    def _apply_changes(self, base: pd.DataFrame, changes: List[Tuple[str, Any]] = None) -> pd.DataFrame:
        """Apply tracked changes to a base DataFrame.
        
        Args:
            base: The base DataFrame to apply changes to
            changes: List of changes to apply (if None, uses self._changes)
            
        Returns:
            DataFrame with all changes applied
        """
        result = base.copy()
        changes_to_apply = changes if changes is not None else self._changes
        
        for change_type, change_data in changes_to_apply:
            if change_type == "extend":
                # Add new columns
                add = {}
                for col_name, fn in change_data.items():
                    add[col_name] = [fn(r) for r in rows(result)]
                result = pd.concat([result.reset_index(drop=True), pd.DataFrame(add)], axis=1)
            
            elif change_type == "transform":
                # Transform rows to new structure
                fn = change_data
                recs = []
                for r in rows(result):
                    out = fn(r)
                    if isinstance(out, dict):
                        recs.append(out)
                    elif isinstance(out, tuple):
                        recs.append({f"c{i}": v for i, v in enumerate(out)})
                    else:
                        recs.append({"value": out})
                result = pd.DataFrame.from_records(recs)
            
            elif change_type == "filter":
                # Filter rows
                fn = change_data
                mask = []
                for r in rows(result):
                    try:
                        mask.append(bool(fn(r)))
                    except Exception:
                        mask.append(False)
                result = result[mask]
            
            elif change_type == "sort":
                # Sort by columns
                cols, ascending = change_data
                result = result.sort_values(list(cols) or result.columns.tolist(), ascending=ascending)
            
            elif change_type == "head":
                # Take first n rows
                n = change_data
                result = result.head(n)
            
            elif change_type == "tail":
                # Take last n rows
                n = change_data
                result = result.tail(n)
            
            elif change_type == "sample":
                # Random sample of rows
                params = change_data
                result = result.sample(
                    n=params["n"],
                    frac=params["frac"],
                    random_state=params["random_state"]
                )
            
            elif change_type == "drop":
                # Drop specified columns
                cols_to_drop = change_data
                # Only drop columns that exist
                cols_to_drop = [c for c in cols_to_drop if c in result.columns]
                if cols_to_drop:
                    result = result.drop(columns=cols_to_drop)
            
            elif change_type == "select":
                # Select (keep) only specified columns
                cols_to_keep = change_data
                # Only keep columns that exist
                cols_to_keep = [c for c in cols_to_keep if c in result.columns]
                result = result[cols_to_keep]
            
            elif change_type == "distinct":
                # Remove duplicate rows
                subset = change_data  # None means all columns, or list of specific columns
                result = result.drop_duplicates(subset=subset, keep='first')
            
            elif change_type == "rename":
                # Rename columns
                mapping = change_data
                # Only rename columns that exist
                valid_mapping = {old: new for old, new in mapping.items() if old in result.columns}
                if valid_mapping:
                    result = result.rename(columns=valid_mapping)
        
        return result
    
    def _copy_with(self, **kwargs) -> 'Q':
        """Create a new Q instance with updated attributes.
        
        Args:
            **kwargs: Attributes to update
            
        Returns:
            New Q instance
        """
        params = {
            'df': self._df,
            'source_path': self._source_path,
            'skip_rows': self._skip_rows,
            'base_df': self._base_df,
            'changes': self._changes.copy(),
            'hidden_cols': self._hidden_cols.copy()
        }
        params.update(kwargs)
        return Q(**params)
    
    def _display_df(self) -> pd.DataFrame:
        """Get DataFrame with hidden columns excluded for display."""
        if self._hidden_cols:
            visible_cols = [c for c in self._df.columns if c not in self._hidden_cols]
            return self._df[visible_cols]
        return self._df
    
    def __repr__(self) -> str:
        """Return a string representation for the REPL."""
        return self._display_df().head(20).to_string(index=False)
    
    def __str__(self) -> str:
        """Return a string representation for print()."""
        return self._display_df().head(20).to_string(index=False)
    
    def __iter__(self):
        """Make Q iterable - iterates over rows as namedtuples."""
        return iter(rows(self._df))
    
    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._df)
    
    @property
    def columns(self) -> list:
        """Return list of column names in the DataFrame.
        
        Returns:
            List of column names (includes hidden columns)
            
        Example:
            >>> q.columns
            ['name', 'age', 'salary']
            >>> # Use in lambdas:
            >>> q.extend(total=lambda x: x.price * x.qty)
        """
        return self._df.columns.tolist()
    
    @property
    def cols(self) -> list:
        """Alias for columns. Return list of column names.
        
        Returns:
            List of column names (includes hidden columns)
        """
        return self.columns
    
    @property
    def rows(self) -> int:
        """Return the number of rows in the DataFrame.
        
        Returns:
            Number of rows (same as len(q))
            
        Example:
            >>> q.rows
            1523
            >>> print(f"{q.rows} rows × {len(q.columns)} columns")
            1523 rows × 4 columns
        """
        return len(self._df)
    
    def extend(self, **newcols) -> 'Q':
        """Add new columns to the DataFrame based on existing columns.
        
        Args:
            **newcols: Keyword arguments where keys are new column names and
                      values are functions that take a Row and return the new value
                      
        Returns:
            A new Q object with the additional columns
            
        Example:
            >>> q.extend(total=lambda x: x.price * x.qty, tax=lambda x: x.total * 0.08)
        """
        new_changes = self._changes + [("extend", newcols)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def transform(self, fn: Callable) -> 'Q':
        """Transform each row using a function, creating new columns.
        
        Args:
            fn: A function that takes a Row and returns:
                - A dict: keys become column names
                - A tuple: creates columns named c0, c1, etc.
                - Any other value: creates a single 'value' column
                
        Returns:
            A new Q object with the transformed data
            
        Example:
            >>> q.transform(lambda x: {'name': x.first + ' ' + x.last, 'age': x.age})
        """
        new_changes = self._changes + [("transform", fn)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def filter(self, fn: Callable) -> 'Q':
        """Filter rows based on a predicate function.
        
        Args:
            fn: A function that takes a Row and returns a boolean
            
        Returns:
            A new Q object containing only rows where fn(row) is True
            
        Example:
            >>> q.filter(lambda x: x.region == 'CA')
        """
        new_changes = self._changes + [("filter", fn)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def groupby(self, keyfn: Callable, **aggs) -> 'Q':
        """Group rows by a key function and compute aggregations.
        
        Note: This is a terminal operation that resets the change history.
        
        Args:
            keyfn: A function that takes a Row and returns a grouping key
            **aggs: Keyword arguments where keys are result column names and
                   values are functions that take a list of Rows and return an aggregated value
                   
        Returns:
            A new Q object with one row per group (change history reset)
            
        Example:
            >>> q.groupby(lambda x: x.category, 
            ...           total_sales=lambda g: sum(r.amount for r in g),
            ...           count=lambda g: len(g))
        """
        buckets = defaultdict(list)
        for r in rows(self._df):
            buckets[keyfn(r)].append(r)
        out = []
        for k, grp in buckets.items():
            rec = {"key": k}
            for name, fn in aggs.items():
                rec[name] = fn(grp)
            out.append(rec)
        
        # groupby is terminal - resets to new base
        new_df = pd.DataFrame(out)
        return Q(new_df)
    
    def head(self, n: int = 5) -> 'Q':
        """Return the first n rows.
        
        Args:
            n: Number of rows to return (default: 5)
            
        Returns:
            A new Q object containing the first n rows
        """
        new_changes = self._changes + [("head", n)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def tail(self, n: int = 5) -> 'Q':
        """Return the last n rows.
        
        Args:
            n: Number of rows to return (default: 5)
            
        Returns:
            A new Q object containing the last n rows
            
        Example:
            >>> q.tail(10)  # Last 10 rows
            >>> q.sort('date').tail(20)  # Most recent 20 after sorting
        """
        new_changes = self._changes + [("tail", n)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def sample(self, n: int = None, frac: float = None, random_state: int = 42) -> 'Q':
        """Return a random sample of rows.
        
        By default uses random_state=42 for reproducible samples that honor
        the idempotency requirement. Set random_state=None for truly random
        sampling (note: breaks reproducibility on refresh/reload).
        
        Args:
            n: Number of rows to sample (mutually exclusive with frac)
            frac: Fraction of rows to sample (0.0 to 1.0, mutually exclusive with n)
            random_state: Random seed for reproducibility (default: 42, use None for random)
            
        Returns:
            A new Q object containing the sampled rows
            
        Raises:
            ValueError: If neither or both n and frac are specified
            
        Example:
            >>> q.sample(100)  # 100 random rows (reproducible)
            >>> q.sample(frac=0.1)  # 10% random sample (reproducible)
            >>> q.sample(50, random_state=None)  # 50 rows (truly random)
            >>> q.sample(1000, random_state=123)  # Custom seed
        """
        if n is None and frac is None:
            raise ValueError("Must specify either n or frac")
        if n is not None and frac is not None:
            raise ValueError("Cannot specify both n and frac")
        
        sample_params = {"n": n, "frac": frac, "random_state": random_state}
        new_changes = self._changes + [("sample", sample_params)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def sort(self, *cols, ascending: bool = True) -> 'Q':
        """Sort the DataFrame by one or more columns.
        
        Args:
            *cols: Column names to sort by. If empty, sorts by all columns.
            ascending: Whether to sort in ascending order (default: True)
            
        Returns:
            A new Q object with sorted rows
            
        Example:
            >>> q.sort('price', ascending=False)  # Highest first
            >>> q.sort('age')  # Lowest first (default)
        """
        new_changes = self._changes + [("sort", (cols, ascending))]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def drop(self, *cols) -> 'Q':
        """Remove specified columns from the DataFrame.
        
        This is a structural change that actually removes columns from the data.
        Different from hide() which only affects display.
        
        Args:
            *cols: Column names to remove
            
        Returns:
            A new Q object without the specified columns
            
        Example:
            >>> q.drop('id', 'internal_field')  # Actually removes columns
            >>> q.drop('temp').extend(...)  # Removed columns can't be used
        """
        new_changes = self._changes + [("drop", list(cols))]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def select(self, *cols) -> 'Q':
        """Keep only specified columns, removing all others.
        
        This is equivalent to dropping all columns except the ones specified.
        This is a structural change that actually removes columns from the data.
        
        Args:
            *cols: Column names to keep
            
        Returns:
            A new Q object with only the specified columns
            
        Example:
            >>> q.select('name', 'email', 'age')  # Keep only these columns
        """
        new_changes = self._changes + [("select", list(cols))]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def distinct(self, *cols) -> 'Q':
        """Remove duplicate rows, optionally considering only specific columns.
        
        When columns are specified, keeps the first occurrence of each unique
        combination of values in those columns. When no columns are specified,
        removes rows that are completely identical across all columns.
        
        Args:
            *cols: Optional column names to consider for uniqueness.
                   If empty, considers all columns.
            
        Returns:
            A new Q object with duplicates removed
            
        Examples:
            >>> q.distinct()  # Remove completely duplicate rows
            >>> q.distinct('customer_id')  # Keep first occurrence per customer
            >>> q.distinct('email', 'phone')  # Unique by email+phone combination
        """
        subset = list(cols) if cols else None
        new_changes = self._changes + [("distinct", subset)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def rename(self, **mapping) -> 'Q':
        """Rename columns using keyword arguments.
        
        Args:
            **mapping: Keyword arguments where key is old name and value is new name
            
        Returns:
            A new Q object with renamed columns
            
        Examples:
            >>> q.rename(customer_id='cust_id')
            >>> q.rename(old_name='new_name', another='better_name')
        """
        new_changes = self._changes + [("rename", mapping)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def show(self, n: int = 20) -> 'Q':
        """Print the first n rows of the DataFrame (respects hidden columns).
        
        Args:
            n: Number of rows to display (default: 20)
            
        Returns:
            Self for method chaining
        """
        print(self._display_df().head(n).to_string(index=False))
        return self
    
    def to_df(self) -> pd.DataFrame:
        """Return a copy of the underlying pandas DataFrame.
        
        Returns a deep copy to preserve Q's immutability. External modifications
        to the returned DataFrame will not affect this Q object.
        
        Returns:
            A deep copy of the current state DataFrame
            
        Example:
            >>> df_copy = q.to_df()
            >>> df_copy['new_col'] = 1  # Does not affect q
            >>> q.to_df().groupby('category').sum()  # Use pandas directly
        """
        return self._df.copy()
    
    def dump(self, filename: str) -> 'Q':
        """Write the DataFrame to a CSV file (writes all columns, including hidden ones).
        
        Args:
            filename: Path to the output CSV file
            
        Returns:
            Self for method chaining
        """
        self._df.to_csv(filename, index=False)
        print(f"Wrote {len(self._df)} rows to {filename}")
        return self
    
    def hide(self, *cols) -> 'Q':
        """Hide columns from display when printing.
        
        This is a DISPLAY-ONLY operation. Hidden columns:
        - Are NOT removed from the DataFrame
        - Still work in all operations (extend, filter, etc.)
        - Are included in dump() output
        - Are only hidden in print/repr/show()
        
        Args:
            *cols: Column names to hide from display
            
        Returns:
            A new Q object with the specified columns hidden from display
            
        Example:
            >>> q.hide('id', 'internal_field')  # Hide from display
            >>> q.hide('cost').extend(profit=lambda x: x.revenue - x.cost)  # Still works!
        """
        new_hidden = self._hidden_cols | set(cols)
        return self._copy_with(hidden_cols=new_hidden)
    
    def unhide(self, *cols) -> 'Q':
        """Unhide columns for display when printing.
        
        This is a DISPLAY-ONLY operation that reverses hide().
        
        Args:
            *cols: If provided, unhide these specific columns.
                   If empty, unhide all columns.
            
        Returns:
            A new Q object with the specified visibility settings
            
        Example:
            >>> q.unhide()  # Unhide all columns
            >>> q.unhide('id')  # Unhide just the id column
        """
        if cols:
            # Unhide specific columns
            new_hidden = self._hidden_cols - set(cols)
            return self._copy_with(hidden_cols=new_hidden)
        else:
            # Unhide all columns
            return self._copy_with(hidden_cols=set())
    
    def reload(self) -> 'Q':
        """Reload data from the source CSV file and re-apply all tracked changes.
        
        Validates that all original columns still exist in the reloaded data.
        New columns and rows are allowed.
        
        Returns:
            A new Q object with reloaded base data and re-applied changes
            
        Raises:
            ValueError: If no source path was provided or if required columns are missing
            
        Example:
            >>> q2 = q.extend(total=lambda x: x.price * x.qty)
            >>> q3 = q2.reload()  # Reloads from source and re-applies total column
        """
        if not self._source_path:
            raise ValueError("Cannot reload: no source path available")
        
        # Reload the data from source
        new_base = load_csv(self._source_path, skip_rows=self._skip_rows)
        
        # Validate that all original base columns still exist
        original_cols = set(self._base_df.columns)
        new_cols = set(new_base.columns)
        missing_cols = original_cols - new_cols
        
        if missing_cols:
            raise ValueError(
                f"Cannot reload: required columns missing from source: {', '.join(sorted(missing_cols))}"
            )
        
        # Re-apply all changes to new base
        new_df = self._apply_changes(new_base)
        
        return self._copy_with(df=new_df, base_df=new_base)
    
    def refresh(self) -> 'Q':
        """Re-apply all tracked changes to the in-memory base DataFrame.
        
        This recomputes the current state from base + changes without reloading from disk.
        Useful for resetting any manual DataFrame manipulations.
        
        Returns:
            A new Q object with changes re-applied to base
            
        Example:
            >>> q2 = q.extend(total=lambda x: x.price * x.qty)
            >>> q3 = q2.refresh()  # Re-applies the extension
        """
        new_df = self._apply_changes(self._base_df)
        return self._copy_with(df=new_df)
    
    def rebase(self) -> 'Q':
        """Flatten the change history by making the current state the new base.
        
        The current DataFrame becomes the new base, and the change list is cleared.
        This is useful for performance when you have a long change history.
        
        Returns:
            A new Q object with current state as base and empty change list
            
        Example:
            >>> q2 = q.extend(a=...).filter(...).extend(b=...).filter(...)
            >>> q3 = q2.rebase()  # Flattens: current state becomes new base
        """
        return Q(
            df=self._df.copy(),
            source_path=self._source_path,
            skip_rows=self._skip_rows,
            base_df=self._df.copy(),
            changes=[],
            hidden_cols=self._hidden_cols.copy()
        )
    
    # Aggregation methods (informational, don't modify state)
    
    def sum(self, col: str) -> float:
        """Compute sum of a column.
        
        Args:
            col: Column name
            
        Returns:
            Sum of the column
        """
        return self._df[col].sum()
    
    def mean(self, col: str) -> float:
        """Compute mean of a column.
        
        Args:
            col: Column name
            
        Returns:
            Mean of the column
        """
        return self._df[col].mean()
    
    def median(self, col: str) -> float:
        """Compute median of a column.
        
        Args:
            col: Column name
            
        Returns:
            Median of the column
        """
        return self._df[col].median()
    
    def min(self, col: str):
        """Compute minimum of a column.
        
        Args:
            col: Column name
            
        Returns:
            Minimum value in the column
        """
        return self._df[col].min()
    
    def max(self, col: str):
        """Compute maximum of a column.
        
        Args:
            col: Column name
            
        Returns:
            Maximum value in the column
        """
        return self._df[col].max()
    
    def count(self, col: str = None) -> int:
        """Count non-null values in a column, or total rows if no column specified.
        
        Args:
            col: Optional column name
            
        Returns:
            Count of non-null values or total rows
        """
        if col:
            return self._df[col].count()
        return len(self._df)
    
    def std(self, col: str) -> float:
        """Compute standard deviation of a column.
        
        Args:
            col: Column name
            
        Returns:
            Standard deviation of the column
        """
        return self._df[col].std()
    
    def var(self, col: str) -> float:
        """Compute variance of a column.
        
        Args:
            col: Column name
            
        Returns:
            Variance of the column
        """
        return self._df[col].var()
    
    def unique(self, col: str) -> list:
        """Get unique values in a column.
        
        Args:
            col: Column name
            
        Returns:
            List of unique values
        """
        return self._df[col].unique().tolist()
    
    def nunique(self, col: str) -> int:
        """Count unique values in a column.
        
        Args:
            col: Column name
            
        Returns:
            Number of unique values
        """
        return self._df[col].nunique()
    
    def memory_usage(self, deep: bool = True) -> dict:
        """Get memory usage breakdown for this Q object.
        
        Args:
            deep: If True, introspect data for accurate memory usage (default: True)
            
        Returns:
            Dictionary with memory usage in bytes for each component:
            - 'current_df': Memory used by current DataFrame
            - 'base_df': Memory used by base DataFrame
            - 'changes': Number of tracked changes
            - 'total': Total estimated memory usage
            - 'total_mb': Total memory usage in megabytes
            
        Example:
            >>> q2 = q.extend(total=lambda x: x.price * x.qty).filter(lambda x: x.total > 100)
            >>> usage = q2.memory_usage()
            >>> print(f"Using {usage['total_mb']:.2f} MB")
        """
        current_mem = self._df.memory_usage(deep=deep).sum()
        base_mem = self._base_df.memory_usage(deep=deep).sum()
        
        # Estimate changes overhead (rough approximation)
        # Each change stores function objects, dicts, etc.
        # For multi-Q operations, this could include entire Q objects
        changes_mem = 0
        for change_type, change_data in self._changes:
            if change_type in ("merge", "concat", "join"):
                # If change_data contains a Q object, count its memory
                if isinstance(change_data, dict) and "other" in change_data:
                    other_q = change_data["other"]
                    if isinstance(other_q, Q):
                        changes_mem += other_q.memory_usage(deep=deep)["total"]
        
        total = current_mem + base_mem + changes_mem
        
        return {
            "current_df": int(current_mem),
            "base_df": int(base_mem),
            "changes": len(self._changes),
            "changes_memory": int(changes_mem),
            "total": int(total),
            "total_mb": round(total / (1024 * 1024), 2)
        }

