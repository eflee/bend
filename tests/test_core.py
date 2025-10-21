"""Tests for bend.core module with new architecture."""

import pytest
import pandas as pd
import tempfile
import os
from bend.core import Q, load_csv, rows, _gsheets_csv


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_gsheets_csv_regular_url(self):
        """Regular URLs should pass through unchanged."""
        url = "https://example.com/data.csv"
        assert _gsheets_csv(url) == url

    def test_gsheets_csv_conversion(self):
        """Google Sheets URL should be converted."""
        url = "https://docs.google.com/spreadsheets/d/abc123/edit#gid=456"
        expected = "https://docs.google.com/spreadsheets/d/abc123/export?format=csv&gid=456"
        assert _gsheets_csv(url) == expected

    def test_load_csv_basic(self, tmp_path):
        """Should load a basic CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,25\nBob,30\n")
        
        df = load_csv(str(csv_file))
        assert len(df) == 2
        assert list(df.columns) == ["name", "age"]

    def test_load_csv_with_skip_rows(self, tmp_path):
        """Should skip specified rows."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("# Comment\n# Header\nname,age\nAlice,25\n")
        
        df = load_csv(str(csv_file), skip_rows=2)
        assert len(df) == 1
        assert list(df.columns) == ["name", "age"]

    def test_rows_function(self):
        """Should convert DataFrame to iterable of Row namedtuples."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        row_list = list(rows(df))
        
        assert len(row_list) == 2
        assert row_list[0].name == "Alice"
        assert row_list[0].age == 25


class TestQBasics:
    """Tests for Q basic functionality."""

    def test_init(self):
        """Should initialize Q with a DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        assert len(q) == 3
        assert list(q.to_df().columns) == ["x"]

    def test_repr_and_str(self):
        """Should have string representation."""
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        q = Q(df)
        result = str(q)
        assert "Alice" in result
        assert "25" in result

    def test_iterable(self):
        """Q should be iterable."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        values = [row.x for row in q]
        assert values == [1, 2, 3]

    def test_len(self):
        """Should return number of rows."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)
        assert len(q) == 10

    def test_columns_property(self):
        """Should return list of column names."""
        df = pd.DataFrame({"name": ["Alice"], "age": [25], "city": ["NYC"]})
        q = Q(df)
        assert q.columns == ["name", "age", "city"]

    def test_cols_alias(self):
        """cols should be alias for columns."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        q = Q(df)
        assert q.cols == q.columns

    def test_columns_includes_hidden(self):
        """columns should include hidden columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        q = Q(df).hide("b")
        assert q.columns == ["a", "b", "c"]

    def test_rows_property(self):
        """Should return number of rows."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        assert q.rows == 100

    def test_rows_equals_len(self):
        """rows property should equal len()."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        q = Q(df)
        assert q.rows == len(q)

    def test_rows_updates_after_filter(self):
        """rows should update after filtering."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)
        q2 = q.filter(lambda x: x.x < 5)
        assert q2.rows == 5


class TestQExtend:
    """Tests for Q.extend() method."""

    def test_extend_single_column(self):
        """Should add a single computed column."""
        df = pd.DataFrame({"price": [10, 20], "qty": [2, 3]})
        q = Q(df)
        q2 = q.extend(total=lambda x: x.price * x.qty)
        
        assert "total" in q2.to_df().columns
        assert list(q2.to_df()["total"]) == [20, 60]
        assert "price" in q2.to_df().columns  # Original preserved

    def test_extend_multiple_columns(self):
        """Should add multiple computed columns."""
        df = pd.DataFrame({"x": [1, 2]})
        q = Q(df)
        q2 = q.extend(double=lambda x: x.x * 2, triple=lambda x: x.x * 3)
        
        assert "double" in q2.to_df().columns
        assert "triple" in q2.to_df().columns
        assert list(q2.to_df()["double"]) == [2, 4]
        assert list(q2.to_df()["triple"]) == [3, 6]

    def test_extend_chained(self):
        """Should allow chained extends."""
        df = pd.DataFrame({"price": [10, 20], "qty": [2, 3]})
        q = Q(df)
        q2 = q.extend(total=lambda x: x.price * x.qty)
        q3 = q2.extend(tax=lambda x: x.total * 0.1)
        
        assert "total" in q3.to_df().columns
        assert "tax" in q3.to_df().columns
        assert list(q3.to_df()["tax"]) == [2.0, 6.0]

    def test_extend_tracks_changes(self):
        """Should track extends in change history."""
        df = pd.DataFrame({"x": [1]})
        q = Q(df)
        q2 = q.extend(y=lambda x: x.x * 2)
        
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "extend"


class TestQTransform:
    """Tests for Q.transform() method."""

    def test_transform_dict(self):
        """Should transform rows with dict output."""
        df = pd.DataFrame({"first": ["Alice", "Bob"], "last": ["Smith", "Jones"]})
        q = Q(df)
        q2 = q.transform(lambda x: {"full_name": f"{x.first} {x.last}"})
        
        assert list(q2.to_df().columns) == ["full_name"]
        assert list(q2.to_df()["full_name"]) == ["Alice Smith", "Bob Jones"]

    def test_transform_tuple(self):
        """Should transform rows with tuple output."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        q = Q(df)
        q2 = q.transform(lambda x: (x.x + x.y, x.x * x.y))
        
        assert list(q2.to_df().columns) == ["c0", "c1"]
        assert list(q2.to_df()["c0"]) == [4, 6]

    def test_transform_tracks_changes(self):
        """Should track transform in change history."""
        df = pd.DataFrame({"x": [1]})
        q = Q(df)
        q2 = q.transform(lambda x: {"y": x.x * 2})
        
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "transform"


class TestQFilter:
    """Tests for Q.filter() method."""

    def test_filter_basic(self):
        """Should filter rows."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        q2 = q.filter(lambda x: x.x > 2)
        
        assert len(q2) == 3
        assert list(q2.to_df()["x"]) == [3, 4, 5]

    def test_filter_preserves_columns(self):
        """Should preserve all columns."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})
        q = Q(df)
        q2 = q.filter(lambda x: x.age >= 30)
        
        assert list(q2.to_df().columns) == ["name", "age"]
        assert len(q2) == 2

    def test_filter_exception_handling(self):
        """Should treat exceptions as False."""
        df = pd.DataFrame({"value": ["10", "abc", "20"]})
        q = Q(df)
        q2 = q.filter(lambda x: int(x.value) > 15)
        
        assert len(q2) == 1
        assert q2.to_df().iloc[0]["value"] == "20"

    def test_filter_tracks_changes(self):
        """Should track filter in change history."""
        df = pd.DataFrame({"x": [1, 2]})
        q = Q(df)
        q2 = q.filter(lambda x: x.x > 1)
        
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "filter"


class TestQSortHead:
    """Tests for Q.sort() and Q.head() methods."""

    def test_sort_ascending(self):
        """Should sort in ascending order."""
        df = pd.DataFrame({"x": [3, 1, 2]})
        q = Q(df)
        q2 = q.sort("x", ascending=True)
        
        assert list(q2.to_df()["x"]) == [1, 2, 3]

    def test_sort_descending(self):
        """Should sort in descending order (default)."""
        df = pd.DataFrame({"x": [3, 1, 2]})
        q = Q(df)
        q2 = q.sort("x")
        
        assert list(q2.to_df()["x"]) == [3, 2, 1]

    def test_head(self):
        """Should return first n rows."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)
        q2 = q.head(3)
        
        assert len(q2) == 3
        assert list(q2.to_df()["x"]) == [0, 1, 2]

    def test_sort_head_chain(self):
        """Should chain sort and head."""
        df = pd.DataFrame({"x": [3, 1, 4, 1, 5]})
        q = Q(df)
        q2 = q.sort("x", ascending=True).head(3)
        
        assert list(q2.to_df()["x"]) == [1, 1, 3]


class TestQRefreshReload:
    """Tests for Q.refresh() and Q.reload() methods."""

    def test_refresh(self):
        """Should re-apply changes to base."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.extend(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
        
        assert len(q2) == 2
        
        # Refresh should give same result
        q3 = q2.refresh()
        assert len(q3) == 2
        assert list(q3.to_df()["y"]) == [4, 6]

    def test_reload_from_file(self, tmp_path):
        """Should reload from file and re-apply changes."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("price,qty\n10,2\n20,3\n")
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.extend(total=lambda x: x.price * x.qty)
        
        # Modify file
        csv_file.write_text("price,qty\n15,4\n25,5\n")
        
        # Reload
        q3 = q2.reload()
        assert len(q3) == 2
        assert list(q3.to_df()["total"]) == [60, 125]

    def test_reload_validates_columns(self, tmp_path):
        """Should raise error if required columns missing."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("price,qty\n10,2\n")
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        
        # Remove a column
        csv_file.write_text("price\n10\n")
        
        with pytest.raises(ValueError, match="required columns missing"):
            q.reload()

    def test_reload_without_source_raises(self):
        """Should raise error if no source path."""
        df = pd.DataFrame({"x": [1, 2]})
        q = Q(df)
        
        with pytest.raises(ValueError, match="Cannot reload"):
            q.reload()


class TestQRebase:
    """Tests for Q.rebase() method."""

    def test_rebase_clears_changes(self):
        """Should clear change history."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.extend(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
        
        assert len(q2._changes) == 2
        
        q3 = q2.rebase()
        assert len(q3._changes) == 0

    def test_rebase_preserves_state(self):
        """Should preserve current DataFrame state."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.extend(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
        q3 = q2.rebase()
        
        assert len(q3) == len(q2)
        assert list(q3.to_df().columns) == list(q2.to_df().columns)
        assert list(q3.to_df()["y"]) == list(q2.to_df()["y"])


class TestQAggregations:
    """Tests for Q aggregation methods."""

    def test_sum(self):
        """Should compute sum."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        assert q.sum("x") == 15

    def test_mean(self):
        """Should compute mean."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        assert q.mean("x") == 3.0

    def test_median(self):
        """Should compute median."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        assert q.median("x") == 3.0

    def test_min_max(self):
        """Should compute min and max."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        assert q.min("x") == 1
        assert q.max("x") == 5

    def test_count(self):
        """Should count rows or non-null values."""
        df = pd.DataFrame({"x": [1, 2, None, 4, 5]})
        q = Q(df)
        assert q.count() == 5
        assert q.count("x") == 4

    def test_unique_nunique(self):
        """Should get unique values."""
        df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
        q = Q(df)
        assert q.nunique("x") == 3
        assert set(q.unique("x")) == {1, 2, 3}


class TestQDropSelect:
    """Tests for Q.drop() and Q.select() methods."""

    def test_drop_single_column(self):
        """Should remove a single column."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        q = Q(df)
        q2 = q.drop("b")
        
        assert "a" in q2.to_df().columns
        assert "c" in q2.to_df().columns
        assert "b" not in q2.to_df().columns
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "drop"

    def test_drop_multiple_columns(self):
        """Should remove multiple columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        q = Q(df)
        q2 = q.drop("b", "d")
        
        assert list(q2.to_df().columns) == ["a", "c"]

    def test_drop_nonexistent_column(self):
        """Should handle dropping nonexistent columns gracefully."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        q = Q(df)
        q2 = q.drop("a", "nonexistent")
        
        assert "b" in q2.to_df().columns
        assert "a" not in q2.to_df().columns

    def test_drop_prevents_column_use(self):
        """Dropped columns cannot be used in subsequent operations."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        q = Q(df)
        q2 = q.drop("b")
        
        # Trying to use dropped column should fail
        try:
            q3 = q2.extend(c=lambda x: x.b * 2)
            # If we get here, it should fail when we access the data
            list(q3)
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # Expected

    def test_select_keeps_specified(self):
        """Should keep only specified columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        q = Q(df)
        q2 = q.select("a", "c")
        
        assert list(q2.to_df().columns) == ["a", "c"]
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "select"

    def test_select_single_column(self):
        """Should work with single column."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        q = Q(df)
        q2 = q.select("a")
        
        assert list(q2.to_df().columns) == ["a"]

    def test_select_nonexistent_column(self):
        """Should handle selecting nonexistent columns gracefully."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        q = Q(df)
        q2 = q.select("a", "nonexistent")
        
        assert list(q2.to_df().columns) == ["a"]

    def test_select_is_inverse_of_drop(self):
        """Select('a', 'b') should be equivalent to dropping everything except a and b."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        q = Q(df)
        
        selected = q.select("a", "b")
        dropped = q.drop("c", "d")
        
        assert list(selected.to_df().columns) == list(dropped.to_df().columns)

    def test_drop_and_select_with_replay(self):
        """Drop and select should work with refresh/reload."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        q = Q(df)
        q2 = q.drop("b").extend(total=lambda x: x.a + x.c)
        
        assert "total" in q2.to_df().columns
        assert "b" not in q2.to_df().columns
        
        # Refresh should re-apply drop and extend
        q3 = q2.refresh()
        assert "total" in q3.to_df().columns
        assert "b" not in q3.to_df().columns


class TestQHideUnhide:
    """Tests for Q.hide() and Q.unhide() methods."""

    def test_hide(self):
        """Should hide columns from display only."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        q = Q(df)
        q2 = q.hide("b")
        
        # Column exists in data
        assert "b" in q2.to_df().columns
        
        # But not in display
        display = str(q2)
        assert "a" in display
        assert "c" in display
        assert "b" not in display

    def test_unhide_all(self):
        """Should unhide all columns."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        q = Q(df).hide("b")
        q2 = q.unhide()
        
        display = str(q2)
        assert "a" in display
        assert "b" in display

    def test_unhide_specific(self):
        """Should unhide specific columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        q = Q(df).hide("b", "c")
        q2 = q.unhide("b")
        
        display = str(q2)
        assert "a" in display
        assert "b" in display
        assert "c" not in display  # Still hidden


class TestQIntegration:
    """Integration tests for complex scenarios."""

    def test_complex_pipeline(self):
        """Should handle complex chained operations."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "age": [25, 30, 35, 28],
            "salary": [50000, 60000, 70000, 55000]
        })
        q = Q(df)
        
        result = (q
                  .extend(monthly=lambda x: x.salary / 12)
                  .filter(lambda x: x.age >= 28)
                  .sort("monthly", ascending=False)
                  .head(2))
        
        assert len(result) == 2
        assert result.to_df().iloc[0]["name"] == "Charlie"

    def test_change_history_preserved(self):
        """Should preserve complete change history."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        
        q2 = (q
              .extend(y=lambda x: x.x * 2)
              .filter(lambda x: x.y > 4)
              .extend(z=lambda x: x.y + 10)
              .sort("z")
              .head(2))
        
        assert len(q2._changes) == 5
        change_types = [c[0] for c in q2._changes]
        assert change_types == ["extend", "filter", "extend", "sort", "head"]


class TestQMemoryUsage:
    """Tests for Q.memory_usage() method."""

    def test_memory_usage_simple(self):
        """Should report memory usage for simple Q."""
        df = pd.DataFrame({"x": range(100), "y": ["test"] * 100})
        q = Q(df)
        usage = q.memory_usage()
        
        assert "current_df" in usage
        assert "base_df" in usage
        assert "changes" in usage
        assert "total" in usage
        assert "total_mb" in usage
        assert usage["changes"] == 0
        assert usage["current_df"] > 0
        assert usage["base_df"] > 0

    def test_memory_usage_with_operations(self):
        """Should track memory with operations."""
        df = pd.DataFrame({"x": range(1000)})
        q = Q(df)
        # Filter significantly reduces rows
        q2 = q.filter(lambda x: x.x < 10)
        
        usage = q2.memory_usage()
        assert usage["changes"] == 1
        assert usage["current_df"] > 0
        assert usage["base_df"] > 0
        # Current should be smaller than base (heavily filtered)
        assert usage["current_df"] < usage["base_df"]

    def test_memory_usage_after_rebase(self):
        """Should show reduced memory after rebase."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.filter(lambda x: x.x < 50)  # Keep only half
        
        usage_before = q2.memory_usage()
        assert usage_before["changes"] == 1
        
        q3 = q2.rebase()
        usage_after = q3.memory_usage()
        assert usage_after["changes"] == 0
        # After rebase, current and base should be same size
        assert usage_after["current_df"] == usage_after["base_df"]
