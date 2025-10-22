"""Core functionality for CSV loading and querying - New architecture."""

import copy
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


def _load_csv_to_dataframe(path_or_url: str, skip_rows: int = 0, dtype: dict = None) -> pd.DataFrame:
    """Internal helper: Load a CSV file to a pandas DataFrame.
    
    This is an internal function. Users should use bend.load_csv() which returns a Q.
    
    Args:
        path_or_url: Path to a local CSV file, a URL, or a Google Sheets URL
        skip_rows: Number of rows to skip at the beginning of the file (default: 0)
        dtype: Dictionary of column names to data types for type conversion (default: None)
               Example: {'age': int, 'price': float, 'active': bool}
        
    Returns:
        A pandas DataFrame containing the CSV data
    """
    u = _gsheets_csv(path_or_url)
    kwargs = {}
    if skip_rows > 0:
        kwargs['skiprows'] = skip_rows
    if dtype:
        kwargs['dtype'] = dtype
    return pd.read_csv(u, **kwargs)


class Q:
    """A query interface with tracked change history and replay capabilities.
    
    Q maintains a base DataFrame and a list of changes (assignments, mappings,
    filtrations). The current state can always be reconstructed from base + changes.
    
    Args:
        df: A pandas DataFrame to wrap as the base state
        source_path: Optional path to source CSV file for reload functionality
        skip_rows: Number of rows to skip when loading from source
        
    Example:
        >>> q = Q(df, source_path='data.csv')
        >>> q2 = q.assign(total=lambda x: x.price * x.qty)
        >>> q3 = q2.filter(lambda x: x.total > 100)
        >>> q3.replay()  # Re-apply changes to base data
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        source_path: str = None,
        skip_rows: int = 0,
        base_df: pd.DataFrame = None,
        changes: List[Tuple[str, Any]] = None,
        hidden_cols: set = None,
        deterministic: bool = True,
        reloadable: bool = None
    ):
        """Initialize a Q object with a DataFrame.
        
        Args:
            df: Current state DataFrame
            source_path: Optional path to source CSV file
            skip_rows: Number of rows to skip when loading from source
            base_df: Base DataFrame (if None, df is used as base)
            changes: List of tracked changes
            hidden_cols: Set of column names to hide from display
            deterministic: Whether this Q's history contains only deterministic operations
            reloadable: Whether this Q can be reloaded from source (if None, inferred from source_path)
        """
        self._base_df = base_df if base_df is not None else df.copy()
        self._changes: List[Tuple[str, Any]] = changes or []
        self._source_path = source_path
        self._skip_rows = skip_rows
        self._hidden_cols = hidden_cols or set()
        self._df = df
        self._deterministic = deterministic
        self._reloadable = reloadable if reloadable is not None else (source_path is not None)
    
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
            if change_type == "assign":
                # Add new columns
                add = {}
                for col_name, fn in change_data.items():
                    add[col_name] = [fn(r) for r in result.itertuples(index=False, name='Row')]
                result = pd.concat([result.reset_index(drop=True), pd.DataFrame(add)], axis=1)
            
            elif change_type == "map":
                # Transform rows to new structure
                fn = change_data
                recs = []
                for r in result.itertuples(index=False, name='Row'):
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
                for row in result.itertuples(index=False, name='Row'):
                    try:
                        mask.append(bool(fn(row)))
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
            
            elif change_type == "concat":
                # Concatenate with another Q
                other_q = change_data["other"]
                # Get the other Q's current DataFrame by applying its changes
                other_df = other_q._apply_changes(other_q._base_df, other_q._changes)
                result = pd.concat([result, other_df], axis=0, ignore_index=True)
            
            elif change_type == "merge":
                # Merge with another Q
                other_q = change_data["other"]
                on = change_data["on"]
                how = change_data["how"]
                resolve = change_data.get("resolve", {})
                
                # Get the other Q's current DataFrame
                other_df = other_q._apply_changes(other_q._base_df, other_q._changes)
                
                # Detect and resolve column conflicts
                if isinstance(on, str):
                    on_cols = {on}
                else:
                    on_cols = set(on)
                
                conflicts = set(result.columns) & set(other_df.columns) - on_cols
                
                if conflicts and not resolve:
                    raise ValueError(
                        f"Column conflicts detected: {sorted(conflicts)}. "
                        f"Use resolve parameter to specify how to handle them. "
                        f"Example: resolve={{'col': lambda left, right: left}}"
                    )
                
                if resolve:
                    missing_resolutions = conflicts - set(resolve.keys())
                    if missing_resolutions:
                        raise ValueError(
                            f"Missing resolution for columns: {sorted(missing_resolutions)}"
                        )
                
                # Perform the merge
                merged = pd.merge(result, other_df, on=on, how=how, suffixes=('_LEFT', '_RIGHT'))
                
                # Apply conflict resolution if needed
                if resolve:
                    for col_name, resolve_fn in resolve.items():
                        left_col = f"{col_name}_LEFT"
                        right_col = f"{col_name}_RIGHT"
                        
                        if left_col in merged.columns and right_col in merged.columns:
                            # Apply resolution lambda row-by-row
                            def apply_resolve(row):
                                left_val = row[left_col]
                                right_val = row[right_col]
                                return resolve_fn(left_val, right_val)
                            
                            merged[col_name] = merged.apply(apply_resolve, axis=1)
                            # Drop the suffixed columns
                            merged = merged.drop(columns=[left_col, right_col])
                
                result = merged
            
            elif change_type == "fillna":
                # Fill null values
                value = change_data
                result = result.fillna(value)
            
            elif change_type == "replace":
                # Replace values
                if 'value' in change_data:
                    # Scalar replacement: replace(old, new)
                    result = result.replace(change_data['to_replace'], change_data['value'])
                else:
                    # Dict replacement: replace({...})
                    result = result.replace(change_data['to_replace'])
        
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
            'hidden_cols': self._hidden_cols.copy(),
            'deterministic': self._deterministic,
            'reloadable': self._reloadable
        }
        params.update(kwargs)
        return Q(**params)
    
    def _semi_join(self, other: 'Q', on) -> 'Q':
        """Keep rows from self where key exists in other (semi-join).
        
        Internal method used by filter() with Q parameter.
        """
        # Normalize 'on' to list
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        
        # Validate columns exist in self
        missing_in_self = [col for col in on_cols if col not in self._df.columns]
        if missing_in_self:
            raise ValueError(
                f"Columns not found in left Q: {missing_in_self}\n"
                f"Available columns: {list(self._df.columns)}"
            )
        
        # Validate columns exist in other
        missing_in_other = [col for col in on_cols if col not in other._df.columns]
        if missing_in_other:
            raise ValueError(
                f"Columns not found in right Q: {missing_in_other}\n"
                f"Available columns: {list(other._df.columns)}"
            )
        
        # Get unique keys from other
        other_keys = other._df[on_cols].drop_duplicates()
        
        # Inner merge gives us rows where keys match, then keep only self's columns
        result_df = pd.merge(self._df, other_keys, on=on_cols, how='inner')
        result_df = result_df[self._df.columns]  # Keep only original columns
        result_df = result_df.reset_index(drop=True)
        
        return self._copy_with(df=result_df)
    
    def _anti_join(self, other: 'Q', on) -> 'Q':
        """Keep rows from self where key does NOT exist in other (anti-join).
        
        Internal method used by filter() with Q parameter and inverse=True.
        """
        # Normalize 'on' to list
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        
        # Validate columns exist in both
        missing_in_self = [col for col in on_cols if col not in self._df.columns]
        if missing_in_self:
            raise ValueError(
                f"Columns not found in left Q: {missing_in_self}\n"
                f"Available columns: {list(self._df.columns)}"
            )
        
        missing_in_other = [col for col in on_cols if col not in other._df.columns]
        if missing_in_other:
            raise ValueError(
                f"Columns not found in right Q: {missing_in_other}\n"
                f"Available columns: {list(other._df.columns)}"
            )
        
        # Get unique keys from other
        other_keys = other._df[on_cols].drop_duplicates()
        
        # Left merge with indicator to find non-matches
        merged = pd.merge(self._df, other_keys, on=on_cols, how='left', indicator=True)
        result_df = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
        result_df = result_df[self._df.columns]  # Keep only original columns
        result_df = result_df.reset_index(drop=True)
        
        return self._copy_with(df=result_df)
    
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
        return self._df.itertuples(index=False, name='Row')
    
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
            >>> q.assign(total=lambda x: x.price * x.qty)
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
    
    @property
    def deterministic(self) -> bool:
        """Return whether this Q's history contains only deterministic operations.
        
        A Q is deterministic if all operations in its history will produce the same
        results when replayed. Non-deterministic operations (like sample() without
        a random_state, or merge() with deep_copy=False) will set this flag to False.
        
        Returns:
            True if all operations are deterministic,
            False if any operation is non-deterministic
            
        Example:
            >>> q = Q(df)
            >>> q.deterministic
            True
            >>> q2 = q.sample(100)  # No random_state - non-deterministic
            >>> q2.deterministic
            False
            >>> q3 = q.sample(100, random_state=42)  # Deterministic
            >>> q3.deterministic
            True
        """
        return self._deterministic
    
    @property
    def reloadable(self) -> bool:
        """Return whether this Q can be reloaded from source file(s).
        
        A Q is reloadable if it (and all Qs in its change tree) were created from
        source files and have not been rebased. After a rebase(), the change history
        is cleared and the Q can no longer be reloaded from source.
        
        Returns:
            True if this Q can be reloaded from source,
            False if it cannot (no source path, or rebased, or contains non-reloadable Qs)
            
        Example:
            >>> q = Q(load_csv('data.csv'), source_path='data.csv')
            >>> q.reloadable
            True
            >>> q2 = q.filter(lambda x: x.age > 18)
            >>> q2.reloadable
            True
            >>> q3 = q2.rebase()  # Clears history
            >>> q3.reloadable
            False
            >>> q4 = Q(df)  # No source path
            >>> q4.reloadable
            False
        """
        return self._reloadable
    
    def assign(self, **newcols) -> 'Q':
        """Add new columns to the DataFrame based on existing columns.
        
        Aligns with pandas DataFrame.assign() for familiarity and consistency.
        
        Args:
            **newcols: Keyword arguments where keys are new column names and
                      values are functions that take a Row and return the new value
                      
        Returns:
            A new Q object with the additional columns
            
        Example:
            >>> q.assign(total=lambda x: x.price * x.qty, tax=lambda x: x.total * 0.08)
            
        Idempotent: Yes
        """
        new_changes = self._changes + [("assign", newcols)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def map(self, fn: Callable) -> 'Q':
        """Map each row to a new structure, replacing all columns.
        
        This is a complete restructuring operation - it processes each row through
        your function and builds an entirely new DataFrame from the results.
        For adding columns while keeping existing ones, use assign() instead.
        
        Args:
            fn: A function that takes a Row and returns:
                - A dict: keys become column names
                - A tuple: creates columns named c0, c1, etc.
                - Any other value: creates a single 'value' column
                
        Returns:
            A new Q object with the transformed data
            
        Example:
            >>> q.map(lambda x: {'name': x.first + ' ' + x.last, 'age': x.age})
            >>> q.map(lambda x: (x.year, x.month, x.day))  # Creates c0, c1, c2
            
        Idempotent: Yes
        """
        new_changes = self._changes + [("map", fn)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def filter(self, fn_or_q, on=None, *, inverse=False) -> 'Q':
        """Filter rows by condition or by existence in another Q.
        
        Args:
            fn_or_q: Either:
                - Callable: Lambda function for traditional filtering
                - Q object: Check if keys exist in this Q (semi/anti-join)
            on: Required if fn_or_q is a Q. Column(s) to match on (string or list).
            inverse: If True, inverts the filter logic:
                - With lambda: keeps rows where fn returns False
                - With Q: keeps rows NOT in other (anti-join)
        
        Returns:
            Filtered Q object
            
        Examples:
            >>> # Traditional filter with lambda
            >>> q.filter(lambda x: x.age > 18)
            
            >>> # Inverse lambda filter (keep rows where condition is False)
            >>> q.filter(lambda x: x.age > 18, inverse=True)  # Keep age <= 18
            
            >>> # Semi-join: keep customers who have orders
            >>> customers.filter(orders, on='customer_id')
            
            >>> # Anti-join: keep customers who DON'T have orders
            >>> customers.filter(orders, on='customer_id', inverse=True)
            
            >>> # Multi-column matching
            >>> q.filter(other, on=['first_name', 'last_name'])
            
        Deterministic: Yes
        """
        if isinstance(fn_or_q, Q):
            # Q-based filtering (semi/anti-join)
            other = fn_or_q
            if on is None:
                raise ValueError(
                    "Must specify 'on' parameter when filtering by Q.\n"
                    "Example: q.filter(other_q, on='id')"
                )
            
            if inverse:
                return self._anti_join(other, on)
            else:
                return self._semi_join(other, on)
        
        elif callable(fn_or_q):
            # Traditional lambda filtering
            if on is not None:
                raise ValueError("'on' parameter only applies when filtering by Q")
            
            fn = fn_or_q
            
            if inverse:
                # Wrap function to invert logic
                inverted_fn = lambda row: not fn(row)
                new_changes = self._changes + [("filter", inverted_fn)]
            else:
                new_changes = self._changes + [("filter", fn)]
            
            new_df = self._apply_changes(self._base_df, new_changes)
            return self._copy_with(df=new_df, changes=new_changes)
        
        else:
            raise TypeError(
                f"Expected Callable or Q, got {type(fn_or_q).__name__}.\n"
                f"Example: q.filter(lambda x: x.age > 18) or q.filter(other_q, on='id')"
            )
    
    def dropna(self, *cols, how='any') -> 'Q':
        """Remove rows with null values (wrapper around filter).
        
        Args:
            *cols: Optional column names to check. If not specified, checks all columns.
            how: Either 'any' (default) or 'all':
                - 'any': Drop row if ANY specified column has null
                - 'all': Drop row if ALL specified columns are null
        
        Returns:
            A new Q object with rows containing nulls removed
            
        Examples:
            >>> q.dropna()  # Remove rows with any null
            >>> q.dropna('email')  # Remove rows where email is null
            >>> q.dropna('email', 'phone')  # Remove rows where either is null
            >>> q.dropna('email', 'phone', how='all')  # Remove only if both null
            
        Deterministic: Yes (wrapper around filter)
        """
        if how not in ('any', 'all'):
            raise ValueError(f"how must be 'any' or 'all', got '{how}'")
        
        # If no columns specified, check all columns
        check_cols = cols if cols else tuple(self.columns)
        
        # Build the filter lambda
        if how == 'any':
            # Drop if ANY column is null (keep if ALL are not null)
            def filter_fn(row):
                return all(pd.notna(getattr(row, col)) for col in check_cols)
        else:  # how == 'all'
            # Drop if ALL columns are null (keep if ANY is not null)
            def filter_fn(row):
                return any(pd.notna(getattr(row, col)) for col in check_cols)
        
        # Use existing filter method
        return self.filter(filter_fn)
    
    def fillna(self, value) -> 'Q':
        """Fill null values with a specified value or mapping.
        
        Args:
            value: Either:
                - A scalar value to fill all nulls
                - A dict mapping column names to fill values
        
        Returns:
            A new Q object with nulls filled
            
        Examples:
            >>> q.fillna(0)  # Fill all nulls with 0
            >>> q.fillna({'age': 0, 'city': 'Unknown'})  # Column-specific fills
            
        Deterministic: Yes
        """
        new_changes = self._changes + [("fillna", value)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def replace(self, to_replace, value=None) -> 'Q':
        """Replace values in the dataset.
        
        Args:
            to_replace: Either:
                - A scalar value to replace across all columns
                - A dict for column-specific replacements: {'col': {'old': 'new'}}
                - A dict for value mapping across all columns: {'old': 'new'}
            value: Replacement value (only if to_replace is scalar)
        
        Returns:
            A new Q object with values replaced
            
        Examples:
            >>> # Replace value across all columns
            >>> q.replace(0, np.nan)
            
            >>> # Column-specific replacements
            >>> q.replace({'region': {'CA': 'California', 'NY': 'New York'}})
            
            >>> # Replace across all columns
            >>> q.replace({'old_val': 'new_val'})
            
        Deterministic: Yes
        """
        if value is not None:
            # Scalar replacement: replace(old, new)
            replace_data = {'to_replace': to_replace, 'value': value}
        else:
            # Dict replacement: replace({...})
            replace_data = {'to_replace': to_replace}
        
        new_changes = self._changes + [("replace", replace_data)]
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
        for r in self._df.itertuples(index=False, name='Row'):
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
            
        Example:
            >>> q.head(10)  # First 10 rows
            >>> q.sort('date').head(20)  # Earliest 20 after sorting
            
        Idempotent: Yes
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
            
        Idempotent: Yes
        """
        new_changes = self._changes + [("tail", n)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def sample(self, n: int = None, frac: float = None, random_state: int = None) -> 'Q':
        """Return a random sample of rows.
        
        By default sampling is **non-deterministic** (random_state=None), which means
        each call will return different results. To make sampling deterministic, pass
        an explicit random_state value. Non-deterministic sampling marks the Q as
        non-deterministic (q.deterministic will be False).
        
        Args:
            n: Number of rows to sample (mutually exclusive with frac)
            frac: Fraction of rows to sample (0.0 to 1.0, mutually exclusive with n)
            random_state: Random seed for reproducibility (default: None for random,
                         pass an int like 42 for deterministic sampling)
            
        Returns:
            A new Q object containing the sampled rows
            
        Raises:
            ValueError: If neither or both n and frac are specified
            
        Example:
            >>> q.sample(100)  # 100 random rows (non-deterministic, different each time)
            >>> q.sample(frac=0.1)  # 10% random sample (non-deterministic)
            >>> q.sample(50, random_state=42)  # 50 rows (deterministic with seed)
            >>> q.sample(1000, random_state=123)  # Custom seed (deterministic)
            
        Deterministic: Only if random_state is specified (not None)
        """
        if n is None and frac is None:
            raise ValueError("Must specify either n or frac")
        if n is not None and frac is not None:
            raise ValueError("Cannot specify both n and frac")
        
        sample_params = {"n": n, "frac": frac, "random_state": random_state}
        new_changes = self._changes + [("sample", sample_params)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        # Mark as non-deterministic if random_state is None
        new_deterministic = self._deterministic and (random_state is not None)
        
        return self._copy_with(df=new_df, changes=new_changes, deterministic=new_deterministic)
    
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
            
        Idempotent: Yes
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
            >>> q.drop('temp').assign(...)  # Removed columns can't be used
            
        Idempotent: Yes
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
            
        Idempotent: Yes
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
            
        Idempotent: Yes
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
            
        Idempotent: Yes
        """
        new_changes = self._changes + [("rename", mapping)]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        return self._copy_with(df=new_df, changes=new_changes)
    
    def concat(self, other: 'Q', deep_copy: bool = True) -> 'Q':
        """Concatenate another Q vertically (stack rows).
        
        Combines rows from both Q objects. Column names must match (or be subset/superset).
        Missing columns are filled with NaN. By default, stores a deep copy of the other Q
        for full reproducibility. Use deep_copy=False for better performance with large
        datasets (marks result as non-deterministic and non-reloadable).
        
        Args:
            other: Another Q object to concatenate
            deep_copy: If True (default), stores a deep copy for full reproducibility.
                      If False, stores reference and marks as non-deterministic/non-reloadable.
                      
        Returns:
            A new Q object with rows from both Q objects
            
        Examples:
            >>> q1 = Q(df1)  # Jan data
            >>> q2 = Q(df2)  # Feb data
            >>> q_combined = q1.concat(q2)  # All rows from both
            >>> 
            >>> # Self-concatenation (duplicates rows)
            >>> q_double = q.concat(q)
            >>>
            >>> # Performance mode for large datasets
            >>> q_combined = q1.concat(huge_q, deep_copy=False)  # Faster but non-deterministic
            
        Deterministic: Yes (if both Q objects are deterministic and deep_copy=True)
        """
        # Handle self-reference: deep copy self to avoid circular reference
        if other is self:
            other_copy = copy.deepcopy(self)
        elif deep_copy:
            # Store FULL deep copy of other Q including its history
            other_copy = copy.deepcopy(other)
        else:
            # Store reference - faster but breaks determinism/reloadability guarantee
            other_copy = other
        
        new_changes = self._changes + [("concat", {"other": other_copy, "deep_copy": deep_copy})]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        # Propagate deterministic and reloadable flags
        if deep_copy:
            new_deterministic = self._deterministic and other._deterministic
            new_reloadable = self._reloadable and other._reloadable
        else:
            new_deterministic = False  # Can't guarantee determinism with references
            new_reloadable = False  # Can't reload with references
        
        return self._copy_with(df=new_df, changes=new_changes, 
                              deterministic=new_deterministic, reloadable=new_reloadable)
    
    def merge(self, other: 'Q', on, how: str = 'inner', resolve: dict = None, deep_copy: bool = True) -> 'Q':
        """Merge with another Q object based on key columns.
        
        Similar to pandas merge/join but requires explicit handling of column conflicts.
        By default, stores a deep copy of the other Q for full reproducibility. Use
        deep_copy=False for better performance with large datasets (marks as non-deterministic/non-reloadable).
        
        Args:
            other: Another Q object to merge with
            on: Column name(s) to merge on. Can be a string for single column or list for multiple.
            how: Type of merge ('inner', 'left', 'right', 'outer'). Default: 'inner'
            resolve: Dict mapping conflicting column names to resolution lambdas.
                    Lambda signature: lambda left_val, right_val: result_val
                    Required if column conflicts exist (excluding merge keys).
            deep_copy: If True (default), stores a deep copy for full reproducibility.
                      If False, stores reference and marks as non-deterministic/non-reloadable.
                      
        Returns:
            A new Q object with merged data
            
        Raises:
            ValueError: If column conflicts exist without complete resolution
            
        Examples:
            >>> # Basic merge on single column
            >>> customers = Q(customers_df)
            >>> orders = Q(orders_df)
            >>> q = customers.merge(orders, on='customer_id', how='left')
            >>> 
            >>> # Merge with conflict resolution
            >>> q1 = Q(pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "status": ["active", "inactive"]}))
            >>> q2 = Q(pd.DataFrame({"id": [1, 2], "status": ["pending", "complete"]}))
            >>> # Both have 'status' column - must resolve!
            >>> q3 = q1.merge(q2, on='id', resolve={'status': lambda left, right: left})
            >>>
            >>> # Self-merge (employee-manager relationship)
            >>> employees = Q(emp_df)
            >>> # Self-references use deep copy to avoid circular references
            >>> q = employees.merge(employees, on='manager_id')
            >>>
            >>> # Performance mode for large datasets
            >>> q = small.merge(huge_q, on='id', deep_copy=False)  # Faster but non-deterministic
            
        Deterministic: Yes (if both Q objects are deterministic and deep_copy=True)
        """
        # Handle self-reference: deep copy self to avoid circular reference
        if other is self:
            other_copy = copy.deepcopy(self)
        elif deep_copy:
            # Store FULL deep copy of other Q including its history
            other_copy = copy.deepcopy(other)
        else:
            # Store reference - faster but breaks reproducibility guarantee
            other_copy = other
        
        # Normalize on parameter
        if isinstance(on, str):
            on_param = on
        else:
            on_param = list(on)
        
        new_changes = self._changes + [("merge", {
            "other": other_copy,
            "on": on_param,
            "how": how,
            "resolve": resolve or {},
            "deep_copy": deep_copy
        })]
        new_df = self._apply_changes(self._base_df, new_changes)
        
        # Propagate deterministic and reloadable flags
        if deep_copy:
            new_deterministic = self._deterministic and other._deterministic
            new_reloadable = self._reloadable and other._reloadable
        else:
            new_deterministic = False  # Can't guarantee determinism with references
            new_reloadable = False  # Can't reload with references
        
        return self._copy_with(df=new_df, changes=new_changes, 
                              deterministic=new_deterministic, reloadable=new_reloadable)
    
    def join(self, other: 'Q', on, how: str = 'inner', deep_copy: bool = True) -> 'Q':
        """Join with another Q object (convenience wrapper around merge).
        
        Simpler interface for merges without column conflicts. If conflicts exist,
        use merge() with explicit resolve parameter instead.
        
        Args:
            other: Another Q object to join with
            on: Column name(s) to join on. Can be a string for single column or list for multiple.
            how: Type of join ('inner', 'left', 'right', 'outer'). Default: 'inner'
            deep_copy: If True (default), stores a deep copy for full reproducibility.
                      If False, stores reference and marks as non-deterministic/non-reloadable.
                      
        Returns:
            A new Q object with joined data
            
        Raises:
            ValueError: If column conflicts exist (use merge() with resolve parameter instead)
            
        Examples:
            >>> customers = Q(customers_df)
            >>> orders = Q(orders_df)
            >>> q = customers.join(orders, on='customer_id', how='left')
            
        Deterministic: Yes (if both Q objects are deterministic and deep_copy=True)
        """
        return self.merge(other, on=on, how=how, resolve=None, deep_copy=deep_copy)
    
    def union(self, other: 'Q', deep_copy: bool = True) -> 'Q':
        """Union with another Q (concat + distinct to remove duplicates).
        
        Combines rows from both Q objects and removes duplicates. Requires both Q objects
        to have identical columns (pure set operation).
        
        Args:
            other: Another Q object to union with (must have identical columns)
            deep_copy: If True (default), stores a deep copy for full reproducibility.
                      If False, stores reference and marks as non-deterministic/non-reloadable.
                      
        Returns:
            A new Q object with unique rows from both Q objects
            
        Raises:
            ValueError: If column sets don't match exactly
            
        Examples:
            >>> q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
            >>> q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))
            >>> q3 = q1.union(q2)  # [1, 2, 3, 4]
            
            >>> # Align schemas first if columns differ
            >>> q1 = q1.select('id', 'name')
            >>> q2 = q2.select('id', 'name')
            >>> q3 = q1.union(q2)
            
        Deterministic: Yes (if both Q objects are deterministic and deep_copy=True)
        """
        # Validate matching schemas (pure set operation requirement)
        if set(self._df.columns) != set(other._df.columns):
            raise ValueError(
                f"Cannot perform union - column mismatch.\n"
                f"Left columns: {sorted(self._df.columns)}\n"
                f"Right columns: {sorted(other._df.columns)}\n"
                f"Hint: Use .select() to align schemas first, or use .merge() for different schemas."
            )
        
        return self.concat(other, deep_copy=deep_copy).distinct()
    
    def intersect(self, other: 'Q', deep_copy: bool = True) -> 'Q':
        """Intersect with another Q (rows that appear in both).
        
        Returns only rows that appear in both Q objects. Requires both Q objects
        to have identical columns (pure set operation).
        
        Args:
            other: Another Q object to intersect with (must have identical columns)
            deep_copy: If True (default), stores a deep copy for full reproducibility.
                      If False, stores reference and marks as non-deterministic/non-reloadable.
                      
        Returns:
            A new Q object with rows common to both Q objects
            
        Raises:
            ValueError: If column sets don't match exactly
            
        Examples:
            >>> q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
            >>> q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))
            >>> q3 = q1.intersect(q2)  # [2, 3]
            
            >>> # Align schemas first if columns differ
            >>> q1 = q1.select('id', 'name')
            >>> q2 = q2.select('id', 'name')
            >>> q3 = q1.intersect(q2)
            
        Deterministic: Yes (if both Q objects are deterministic and deep_copy=True)
        """
        # Validate matching schemas (pure set operation requirement)
        if set(self._df.columns) != set(other._df.columns):
            raise ValueError(
                f"Cannot perform intersect - column mismatch.\n"
                f"Left columns: {sorted(self._df.columns)}\n"
                f"Right columns: {sorted(other._df.columns)}\n"
                f"Hint: Use .select() to align schemas first."
            )
        
        # Get the other Q's DataFrame
        if other is self:
            other_copy = copy.deepcopy(self)
        elif deep_copy:
            other_copy = copy.deepcopy(other)
        else:
            other_copy = other
        
        other_df = other_copy._apply_changes(other_copy._base_df, other_copy._changes)
        
        # Perform intersection using pandas merge on all columns
        common_cols = list(self._df.columns)
        result_df = pd.merge(self._df, other_df, on=common_cols, how='inner')
        result_df = result_df.drop_duplicates(keep='first').reset_index(drop=True)
        
        # Determine deterministic and reloadable flags
        if deep_copy:
            new_deterministic = self._deterministic and other._deterministic
            new_reloadable = self._reloadable and other._reloadable
        else:
            new_deterministic = False
            new_reloadable = False
        
        return Q(result_df, deterministic=new_deterministic, reloadable=new_reloadable)
    
    def difference(self, other: 'Q', deep_copy: bool = True) -> 'Q':
        """Difference from another Q (rows in self but not in other).
        
        Returns rows that appear in self but not in other. Requires both Q objects
        to have identical columns (pure set operation).
        
        Args:
            other: Another Q object to subtract (must have identical columns)
            deep_copy: If True (default), stores a deep copy for full reproducibility.
                      If False, stores reference and marks as non-deterministic/non-reloadable.
                      
        Returns:
            A new Q object with rows in self but not in other
            
        Raises:
            ValueError: If column sets don't match exactly
            
        Examples:
            >>> q1 = Q(pd.DataFrame({'a': [1, 2, 3]}))
            >>> q2 = Q(pd.DataFrame({'a': [2, 3, 4]}))
            >>> q3 = q1.difference(q2)  # [1]
            
            >>> # Align schemas first if columns differ
            >>> q1 = q1.select('id', 'name')
            >>> q2 = q2.select('id', 'name')
            >>> q3 = q1.difference(q2)
            
        Deterministic: Yes (if both Q objects are deterministic and deep_copy=True)
        """
        # Validate matching schemas (pure set operation requirement)
        if set(self._df.columns) != set(other._df.columns):
            raise ValueError(
                f"Cannot perform difference - column mismatch.\n"
                f"Left columns: {sorted(self._df.columns)}\n"
                f"Right columns: {sorted(other._df.columns)}\n"
                f"Hint: Use .select() to align schemas first."
            )
        
        # Get the other Q's DataFrame
        if other is self:
            # Self-difference is empty
            return Q(pd.DataFrame(columns=self._df.columns), 
                    deterministic=self._deterministic, reloadable=self._reloadable)
        elif deep_copy:
            other_copy = copy.deepcopy(other)
        else:
            other_copy = other
        
        other_df = other_copy._apply_changes(other_copy._base_df, other_copy._changes)
        
        # Perform difference using pandas merge with indicator on all columns
        common_cols = list(self._df.columns)
        merged = pd.merge(self._df, other_df, on=common_cols, how='outer', indicator=True)
        result_df = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
        result_df = result_df.drop_duplicates(keep='first').reset_index(drop=True)
        
        # Determine deterministic and reloadable flags
        if deep_copy:
            new_deterministic = self._deterministic and other._deterministic
            new_reloadable = self._reloadable and other._reloadable
        else:
            new_deterministic = False
            new_reloadable = False
        
        return Q(result_df, deterministic=new_deterministic, reloadable=new_reloadable)
    
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
            >>> q.hide('cost').assign(profit=lambda x: x.revenue - x.cost)  # Still works!
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
    
    def reload(self, allow_partial_reload: bool = False) -> 'Q':
        """Reload data from the source CSV file and recursively reload all referenced Qs.
        
        This is a DEEP/RECURSIVE operation that reloads the entire Q tree from disk:
        - Reloads this Q's source file
        - Recursively reloads any Q objects stored in the change history (from concat, merge, etc.)
        - Re-applies all changes to the newly reloaded data
        
        This enables full reproducibility after source files have been updated.
        Validates that all original columns still exist in the reloaded data.
        New columns and rows are allowed.
        
        Args:
            allow_partial_reload: If False (default), raises error if any Q in the change tree
                                 is not reloadable. If True, non-reloadable Qs use their current
                                 base DataFrame (partial reload).
        
        Returns:
            A new Q object with reloaded base data and re-applied changes
            
        Raises:
            ValueError: If no source path was provided, if required columns are missing,
                       or if allow_partial_reload=False and any Q in the tree is not reloadable
            
        Example:
            >>> q2 = q.assign(total=lambda x: x.price * x.qty)
            >>> q3 = q2.concat(other_q)
            >>> # Both source CSVs are updated
            >>> q4 = q3.reload()  # Reloads both sources and re-applies all operations
            >>>
            >>> # After rebase, some Qs may not be reloadable
            >>> q_rebased = q2.rebase()
            >>> q5 = q.concat(q_rebased)
            >>> q5.reload()  # Error: q_rebased is not reloadable
            >>> q5.reload(allow_partial_reload=True)  # OK: uses q_rebased's current state
        """
        if not self._source_path:
            if not allow_partial_reload:
                raise ValueError("Cannot reload: no source path available. Use allow_partial_reload=True to skip.")
            else:
                # No source, just use current state
                return self
        
        if not self._reloadable and not allow_partial_reload:
            raise ValueError(
                "Cannot reload: this Q is not reloadable (may have been rebased or contain non-reloadable Qs). "
                "Use allow_partial_reload=True to reload what's possible."
            )
        
        # First, recursively reload any Q objects in the change history
        new_changes = []
        for change_type, change_data in self._changes:
            if change_type in ("concat", "merge"):
                # Reload the referenced Q
                other_q = change_data["other"]
                if other_q._reloadable and other_q._source_path:
                    # Fully reloadable
                    reloaded_other = other_q.reload(allow_partial_reload=allow_partial_reload)
                elif other_q._source_path and allow_partial_reload:
                    # Has source but not fully reloadable - try anyway
                    try:
                        reloaded_other = other_q.reload(allow_partial_reload=True)
                    except ValueError:
                        # Can't reload, use current state
                        reloaded_other = other_q
                elif not other_q._source_path and allow_partial_reload:
                    # No source path, use current state
                    reloaded_other = other_q
                else:
                    # Not reloadable and strict mode
                    raise ValueError(
                        f"Cannot reload: Q in {change_type} operation at change #{len(new_changes)+1} "
                        f"is not reloadable. Use allow_partial_reload=True to skip non-reloadable Qs."
                    )
                
                # Reconstruct the change with reloaded Q
                new_data = change_data.copy()
                new_data["other"] = reloaded_other
                new_changes.append((change_type, new_data))
            else:
                # Non-multi-Q change, keep as-is
                new_changes.append((change_type, change_data))
        
        # Reload this Q's data from source
        new_base = _load_csv_to_dataframe(self._source_path, skip_rows=self._skip_rows)
        
        # Validate that all original base columns still exist
        original_cols = set(self._base_df.columns)
        new_cols = set(new_base.columns)
        missing_cols = original_cols - new_cols
        
        if missing_cols:
            raise ValueError(
                f"Cannot reload: required columns missing from source: {', '.join(sorted(missing_cols))}"
            )
        
        # Re-apply all changes (now with reloaded Q references) to new base
        new_df = self._apply_changes(new_base, new_changes)
        
        return self._copy_with(df=new_df, base_df=new_base, changes=new_changes)
    
    def replay(self) -> 'Q':
        """Re-apply all tracked changes to the in-memory base DataFrame.
        
        This recomputes the current state from base + changes without reloading from disk.
        Useful for resetting any manual DataFrame manipulations or verifying change history.
        
        Returns:
            A new Q object with changes re-applied to base
            
        Example:
            >>> q2 = q.assign(total=lambda x: x.price * x.qty)
            >>> q3 = q2.replay()  # Re-applies the assignment
        """
        new_df = self._apply_changes(self._base_df)
        return self._copy_with(df=new_df)
    
    def rebase(self) -> 'Q':
        """Flatten the change history by making the current state the new base.
        
        The current DataFrame becomes the new base, and the change list is cleared.
        This is useful for performance when you have a long change history or to drop
        deep copies of other Q objects stored in the history.
        
        After rebasing, the Q is marked as deterministic (empty history is deterministic)
        but non-reloadable (change history needed for reload is lost).
        
        Returns:
            A new Q object with current state as base and empty change list
            
        Example:
            >>> q2 = q.assign(a=...).filter(...).assign(b=...).filter(...)
            >>> q3 = q2.rebase()  # Flattens: current state becomes new base
            >>> q3.deterministic  # True (empty history)
            >>> q3.reloadable  # False (history is gone)
            >>> # After concat with large Q:
            >>> q4 = q.concat(large_q).filter(...).rebase()  # Drops deep copy of large_q
        """
        return Q(
            df=self._df.copy(),
            source_path=self._source_path,
            skip_rows=self._skip_rows,
            base_df=self._df.copy(),
            changes=[],
            hidden_cols=self._hidden_cols.copy(),
            deterministic=True,  # Empty history is deterministic
            reloadable=False  # Can't reload without change history
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
            >>> q2 = q.assign(total=lambda x: x.price * x.qty).filter(lambda x: x.total > 100)
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

