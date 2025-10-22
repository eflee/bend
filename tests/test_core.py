"""Tests for bend.core module with new architecture."""

import numpy as np
import pandas as pd
import pytest

from bend import Q
from bend.core import _gsheets_csv, _load_csv_to_dataframe


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

        df = _load_csv_to_dataframe(str(csv_file))
        assert len(df) == 2
        assert list(df.columns) == ["name", "age"]

    def test_load_csv_with_skip_rows(self, tmp_path):
        """Should skip specified rows."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("# Comment\n# Header\nname,age\nAlice,25\n")

        df = _load_csv_to_dataframe(str(csv_file), skip_rows=2)
        assert len(df) == 1
        assert list(df.columns) == ["name", "age"]

    def test_load_csv_with_dtype(self, tmp_path):
        """Should convert column types on load."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,price\nAlice,30,19.99\nBob,25,29.99\n")

        df = _load_csv_to_dataframe(str(csv_file), dtype={"age": int, "price": float})
        assert df["age"].dtype == "int64"
        assert df["price"].dtype == "float64"


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


class TestQAssign:
    """Tests for Q.assign() method."""

    def test_assign_single_column(self):
        """Should add a single computed column."""
        df = pd.DataFrame({"price": [10, 20], "qty": [2, 3]})
        q = Q(df)
        q2 = q.assign(total=lambda x: x.price * x.qty)

        assert "total" in q2.to_df().columns
        assert list(q2.to_df()["total"]) == [20, 60]
        assert "price" in q2.to_df().columns  # Original preserved

    def test_assign_multiple_columns(self):
        """Should add multiple computed columns."""
        df = pd.DataFrame({"x": [1, 2]})
        q = Q(df)
        q2 = q.assign(double=lambda x: x.x * 2, triple=lambda x: x.x * 3)

        assert "double" in q2.to_df().columns
        assert "triple" in q2.to_df().columns
        assert list(q2.to_df()["double"]) == [2, 4]
        assert list(q2.to_df()["triple"]) == [3, 6]

    def test_assign_chained(self):
        """Should allow chained extends."""
        df = pd.DataFrame({"price": [10, 20], "qty": [2, 3]})
        q = Q(df)
        q2 = q.assign(total=lambda x: x.price * x.qty)
        q3 = q2.assign(tax=lambda x: x.total * 0.1)

        assert "total" in q3.to_df().columns
        assert "tax" in q3.to_df().columns
        assert list(q3.to_df()["tax"]) == [2.0, 6.0]

    def test_assign_tracks_changes(self):
        """Should track extends in change history."""
        df = pd.DataFrame({"x": [1]})
        q = Q(df)
        q2 = q.assign(y=lambda x: x.x * 2)

        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "assign"


class TestQMap:
    """Tests for Q.map() method."""

    def test_map_dict(self):
        """Should transform rows with dict output."""
        df = pd.DataFrame({"first": ["Alice", "Bob"], "last": ["Smith", "Jones"]})
        q = Q(df)
        q2 = q.map(lambda x: {"full_name": f"{x.first} {x.last}"})

        assert list(q2.to_df().columns) == ["full_name"]
        assert list(q2.to_df()["full_name"]) == ["Alice Smith", "Bob Jones"]

    def test_map_tuple(self):
        """Should transform rows with tuple output."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        q = Q(df)
        q2 = q.map(lambda x: (x.x + x.y, x.x * x.y))

        assert list(q2.to_df().columns) == ["c0", "c1"]
        assert list(q2.to_df()["c0"]) == [4, 6]

    def test_map_tracks_changes(self):
        """Should track transform in change history."""
        df = pd.DataFrame({"x": [1]})
        q = Q(df)
        q2 = q.map(lambda x: {"y": x.x * 2})

        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "map"


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

    def test_filter_inverse_lambda(self):
        """Should invert lambda filter when inverse=True."""
        df = pd.DataFrame({"age": [25, 15, 30, 12]})
        q = Q(df)

        # Normal filter: age >= 18
        adults = q.filter(lambda x: x.age >= 18)
        assert list(adults.to_df()["age"]) == [25, 30]

        # Inverse filter: age < 18
        minors = q.filter(lambda x: x.age >= 18, inverse=True)
        assert list(minors.to_df()["age"]) == [15, 12]

    def test_filter_semi_join_single_column(self):
        """Should keep rows where key exists in other Q (semi-join)."""
        customers = Q(
            pd.DataFrame({"customer_id": [1, 2, 3, 4], "name": ["Alice", "Bob", "Carol", "Dave"]})
        )
        orders = Q(pd.DataFrame({"order_id": [101, 102, 103], "customer_id": [1, 3, 1]}))

        result = customers.filter(orders, on="customer_id")

        assert len(result) == 2
        assert set(result.to_df()["customer_id"]) == {1, 3}
        assert set(result.to_df()["name"]) == {"Alice", "Carol"}
        assert list(result.to_df().columns) == ["customer_id", "name"]

    def test_filter_semi_join_multiple_columns(self):
        """Should support multi-column keys in semi-join."""
        people = Q(
            pd.DataFrame(
                {
                    "first": ["Alice", "Bob", "Carol"],
                    "last": ["Smith", "Jones", "Smith"],
                    "age": [25, 30, 35],
                }
            )
        )
        verified = Q(pd.DataFrame({"first": ["Alice", "Carol"], "last": ["Smith", "Smith"]}))

        result = people.filter(verified, on=["first", "last"])

        assert len(result) == 2
        assert set(result.to_df()["age"]) == {25, 35}

    def test_filter_anti_join_single_column(self):
        """Should keep rows where key NOT in other Q (anti-join)."""
        customers = Q(
            pd.DataFrame({"customer_id": [1, 2, 3, 4], "name": ["Alice", "Bob", "Carol", "Dave"]})
        )
        orders = Q(pd.DataFrame({"order_id": [101, 102], "customer_id": [1, 3]}))

        result = customers.filter(orders, on="customer_id", inverse=True)

        assert len(result) == 2
        assert set(result.to_df()["customer_id"]) == {2, 4}
        assert set(result.to_df()["name"]) == {"Bob", "Dave"}

    def test_filter_anti_join_multiple_columns(self):
        """Should support multi-column keys in anti-join."""
        people = Q(
            pd.DataFrame(
                {
                    "first": ["Alice", "Bob", "Carol"],
                    "last": ["Smith", "Jones", "Smith"],
                    "age": [25, 30, 35],
                }
            )
        )
        excluded = Q(pd.DataFrame({"first": ["Alice"], "last": ["Smith"]}))

        result = people.filter(excluded, on=["first", "last"], inverse=True)

        assert len(result) == 2
        assert set(result.to_df()["first"]) == {"Bob", "Carol"}

    def test_filter_q_without_on_raises(self):
        """Should raise ValueError if 'on' not specified with Q."""
        q1 = Q(pd.DataFrame({"id": [1, 2]}))
        q2 = Q(pd.DataFrame({"id": [1]}))

        with pytest.raises(ValueError, match="Must specify 'on' parameter"):
            q1.filter(q2)

    def test_filter_lambda_with_on_raises(self):
        """Should raise ValueError if 'on' specified with lambda."""
        q = Q(pd.DataFrame({"x": [1, 2]}))

        with pytest.raises(ValueError, match="'on' parameter only applies"):
            q.filter(lambda x: x.x > 1, on="x")

    def test_filter_invalid_type_raises(self):
        """Should raise TypeError for invalid fn_or_q type."""
        q = Q(pd.DataFrame({"x": [1, 2]}))

        with pytest.raises(TypeError, match="Expected Callable or Q"):
            q.filter("not a callable or Q")

    def test_filter_missing_column_in_self_raises(self):
        """Should raise ValueError if 'on' column missing in self."""
        q1 = Q(pd.DataFrame({"id": [1, 2]}))
        q2 = Q(pd.DataFrame({"other_id": [1]}))

        with pytest.raises(ValueError, match="Columns not found in left Q: \\['other_id'\\]"):
            q1.filter(q2, on="other_id")

    def test_filter_missing_column_in_other_raises(self):
        """Should raise ValueError if 'on' column missing in other."""
        q1 = Q(pd.DataFrame({"id": [1, 2]}))
        q2 = Q(pd.DataFrame({"other_id": [1]}))

        with pytest.raises(ValueError, match="Columns not found in right Q: \\['id'\\]"):
            q1.filter(q2, on="id")

    def test_filter_q_preserves_only_left_columns(self):
        """Semi/anti-join should only return left Q's columns."""
        left = Q(pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}))
        right = Q(pd.DataFrame({"id": [1, 3], "value": [100, 200]}))

        result = left.filter(right, on="id")

        assert list(result.to_df().columns) == ["id", "name"]
        assert "value" not in result.to_df().columns

    def test_filter_q_deduplicates_right_keys(self):
        """Should handle duplicate keys in right Q correctly."""
        left = Q(pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}))
        right = Q(pd.DataFrame({"id": [1, 1, 1, 3]}))  # Duplicates

        result = left.filter(right, on="id")

        # Should match rows with id=1 and id=3 (no duplicates from join)
        assert len(result) == 2
        assert set(result.to_df()["id"]) == {1, 3}


class TestQFilterFlags:
    """Tests for deterministic and reloadable flag propagation in filter operations."""

    def test_filter_lambda_preserves_deterministic_true(self):
        """Lambda filter on deterministic Q should remain deterministic."""
        q = Q(pd.DataFrame({"x": [1, 2, 3]}))
        assert q.deterministic is True

        result = q.filter(lambda x: x.x > 1)
        assert result.deterministic is True

    def test_filter_lambda_preserves_deterministic_false(self):
        """Lambda filter on non-deterministic Q should remain non-deterministic."""
        q = Q(pd.DataFrame({"x": range(10)}))
        q_non_det = q.sample(n=5)  # Non-deterministic
        assert q_non_det.deterministic is False

        result = q_non_det.filter(lambda x: x.x > 2)
        assert result.deterministic is False

    def test_filter_inverse_lambda_preserves_flags(self):
        """Inverse lambda filter should preserve both flags."""
        q = Q(pd.DataFrame({"x": [1, 2, 3]}))
        assert q.deterministic is True
        assert q.reloadable is False

        result = q.filter(lambda x: x.x > 1, inverse=True)
        assert result.deterministic is True
        assert result.reloadable is False

    def test_filter_lambda_preserves_reloadable_true(self, tmp_path):
        """Lambda filter on reloadable Q should remain reloadable."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,a\n2,b\n3,c\n")

        q = Q(_load_csv_to_dataframe(str(csv_file)), source_path=str(csv_file))
        assert q.reloadable is True

        result = q.filter(lambda x: x.x > 1)
        assert result.reloadable is True

    def test_filter_lambda_preserves_reloadable_false(self):
        """Lambda filter on non-reloadable Q should remain non-reloadable."""
        q = Q(pd.DataFrame({"x": [1, 2, 3]}))
        assert q.reloadable is False

        result = q.filter(lambda x: x.x > 1)
        assert result.reloadable is False

    def test_semi_join_preserves_left_deterministic_true(self):
        """Semi-join should preserve deterministic flag from left Q."""
        left = Q(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}))
        right = Q(pd.DataFrame({"id": [1, 3]}))

        assert left.deterministic is True
        result = left.filter(right, on="id")
        assert result.deterministic is True

    def test_semi_join_preserves_left_deterministic_false(self):
        """Semi-join from non-deterministic left should remain non-deterministic."""
        left = Q(pd.DataFrame({"id": range(10), "val": range(10)}))
        left_non_det = left.sample(n=5)
        right = Q(pd.DataFrame({"id": [1, 2, 3]}))

        assert left_non_det.deterministic is False
        result = left_non_det.filter(right, on="id")
        assert result.deterministic is False

    def test_semi_join_preserves_left_reloadable_true(self, tmp_path):
        """Semi-join should preserve reloadable flag from left Q."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n3,Carol\n")

        left = Q(_load_csv_to_dataframe(str(csv_file)), source_path=str(csv_file))
        right = Q(pd.DataFrame({"id": [1, 3]}))

        assert left.reloadable is True
        assert right.reloadable is False

        result = left.filter(right, on="id")
        assert result.reloadable is True  # Should inherit from left, not right

    def test_semi_join_preserves_left_reloadable_false(self, tmp_path):
        """Semi-join from non-reloadable left should remain non-reloadable."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")

        left = Q(pd.DataFrame({"id": [1, 2, 3]}))
        right = Q(_load_csv_to_dataframe(str(csv_file)), source_path=str(csv_file))

        assert left.reloadable is False
        assert right.reloadable is True

        result = left.filter(right, on="id")
        assert result.reloadable is False  # Should inherit from left, not right

    def test_anti_join_preserves_left_deterministic_true(self):
        """Anti-join should preserve deterministic flag from left Q."""
        left = Q(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}))
        right = Q(pd.DataFrame({"id": [1]}))

        assert left.deterministic is True
        result = left.filter(right, on="id", inverse=True)
        assert result.deterministic is True

    def test_anti_join_preserves_left_deterministic_false(self):
        """Anti-join from non-deterministic left should remain non-deterministic."""
        left = Q(pd.DataFrame({"id": range(10), "val": range(10)}))
        left_non_det = left.sample(n=5)
        right = Q(pd.DataFrame({"id": [1, 2, 3]}))

        assert left_non_det.deterministic is False
        result = left_non_det.filter(right, on="id", inverse=True)
        assert result.deterministic is False

    def test_anti_join_preserves_left_reloadable_true(self, tmp_path):
        """Anti-join should preserve reloadable flag from left Q."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n3,Carol\n")

        left = Q(_load_csv_to_dataframe(str(csv_file)), source_path=str(csv_file))
        right = Q(pd.DataFrame({"id": [1]}))

        assert left.reloadable is True
        assert right.reloadable is False

        result = left.filter(right, on="id", inverse=True)
        assert result.reloadable is True  # Should inherit from left, not right

    def test_anti_join_preserves_left_reloadable_false(self, tmp_path):
        """Anti-join from non-reloadable left should remain non-reloadable."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")

        left = Q(pd.DataFrame({"id": [1, 2, 3]}))
        right = Q(_load_csv_to_dataframe(str(csv_file)), source_path=str(csv_file))

        assert left.reloadable is False
        assert right.reloadable is True

        result = left.filter(right, on="id", inverse=True)
        assert result.reloadable is False  # Should inherit from left, not right

    def test_filter_chain_preserves_flags(self, tmp_path):
        """Chained filter operations should preserve flags correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,age,name\n1,25,Alice\n2,15,Bob\n3,30,Carol\n")

        q = Q(_load_csv_to_dataframe(str(csv_file)), source_path=str(csv_file))
        other = Q(pd.DataFrame({"id": [1, 3]}))

        # Chain: lambda -> semi-join -> lambda -> anti-join
        result = (
            q.filter(lambda x: x.age >= 18)  # deterministic=T, reloadable=T
            .filter(other, on="id")  # deterministic=T, reloadable=T
            .filter(lambda x: x.id < 10)  # deterministic=T, reloadable=T
        )

        assert result.deterministic is True
        assert result.reloadable is True

    def test_filter_on_rebased_q(self):
        """Filter after rebase should have correct flags."""
        q = Q(pd.DataFrame({"id": [1, 2, 3]}))
        filtered = q.filter(lambda x: x.id > 1)
        rebased = filtered.rebase()

        assert rebased.deterministic is True  # Empty history is deterministic
        assert rebased.reloadable is False  # History lost

        # Filter the rebased Q
        result = rebased.filter(lambda x: x.id < 3)
        assert result.deterministic is True
        assert result.reloadable is False


class TestQSortHeadTailSample:
    """Tests for Q.sort(), Q.head(), Q.tail(), and Q.sample() methods."""

    def test_sort_ascending_default(self):
        """Should sort in ascending order by default."""
        df = pd.DataFrame({"x": [3, 1, 2]})
        q = Q(df)
        q2 = q.sort("x")

        assert list(q2.to_df()["x"]) == [1, 2, 3]

    def test_sort_descending(self):
        """Should sort in descending order when specified."""
        df = pd.DataFrame({"x": [3, 1, 2]})
        q = Q(df)
        q2 = q.sort("x", ascending=False)

        assert list(q2.to_df()["x"]) == [3, 2, 1]

    def test_head(self):
        """Should return first n rows."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)
        q2 = q.head(3)

        assert len(q2) == 3
        assert list(q2.to_df()["x"]) == [0, 1, 2]

    def test_tail(self):
        """Should return last n rows."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)
        q2 = q.tail(3)

        assert len(q2) == 3
        assert list(q2.to_df()["x"]) == [7, 8, 9]

    def test_tail_default(self):
        """tail() with no args should return last 5 rows."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)
        q2 = q.tail()

        assert len(q2) == 5
        assert list(q2.to_df()["x"]) == [5, 6, 7, 8, 9]

    def test_sort_head_chain(self):
        """Should chain sort and head."""
        df = pd.DataFrame({"x": [3, 1, 4, 1, 5]})
        q = Q(df)
        q2 = q.sort("x", ascending=True).head(3)

        assert list(q2.to_df()["x"]) == [1, 1, 3]

    def test_sort_tail_chain(self):
        """Should chain sort and tail."""
        df = pd.DataFrame({"x": [3, 1, 4, 1, 5]})
        q = Q(df)
        q2 = q.sort("x", ascending=False).tail(2)

        # Descending sort: [5, 4, 3, 1, 1], tail(2) = [1, 1]
        assert list(q2.to_df()["x"]) == [1, 1]

    def test_sample_with_n(self):
        """Should sample n rows."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(10)

        assert len(q2) == 10
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "sample"

    def test_sample_with_frac(self):
        """Should sample fraction of rows."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(frac=0.1)

        assert len(q2) == 10

    def test_sample_reproducible_default(self):
        """Sample with explicit random_state should be reproducible."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)

        q1 = q.sample(10, random_state=42)
        q2 = q.sample(10, random_state=42)

        # Same seed = same sample
        assert list(q1.to_df()["x"]) == list(q2.to_df()["x"])
        assert q1.deterministic  # Should be reproducible with seed

    def test_sample_refresh_reproducible(self):
        """Sample with random_state should give same results on refresh."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(10, random_state=42)
        q3 = q2.replay()

        # Should get same results with explicit seed
        assert list(q2.to_df()["x"]) == list(q3.to_df()["x"])
        assert q2.deterministic  # Should be reproducible
        assert q3.deterministic

    def test_sample_non_deterministic_default(self):
        """Sample without random_state should be non-deterministic."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)

        q1 = q.sample(10)  # No random_state
        assert not q1.deterministic  # Should be marked non-reproducible

    def test_sample_different_seeds(self):
        """Different random_state should give different samples."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)

        q1 = q.sample(10, random_state=42)
        q2 = q.sample(10, random_state=123)

        # Different seeds = (probably) different samples
        assert list(q1.to_df()["x"]) != list(q2.to_df()["x"])

    def test_sample_requires_n_or_frac(self):
        """Sample should require either n or frac."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)

        with pytest.raises(ValueError, match="Must specify either n or frac"):
            q.sample()

    def test_sample_not_both_n_and_frac(self):
        """Sample should not allow both n and frac."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)

        with pytest.raises(ValueError, match="Cannot specify both"):
            q.sample(n=5, frac=0.5)

    def test_sample_n_larger_than_data(self):
        """Sample with n > dataset should work (pandas behavior)."""
        df = pd.DataFrame({"x": range(10)})
        q = Q(df)

        # pandas allows this with replace=False (default) - returns all rows
        # This might raise, depends on pandas version
        try:
            q2 = q.sample(100)
            # If it doesn't raise, it should return all rows
            assert len(q2) == 10
        except ValueError:
            # pandas raised ValueError, which is fine
            pass


class TestQReplayReload:
    """Tests for Q.replay() and Q.reload() methods."""

    def test_refresh(self):
        """Should re-apply changes to base."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)

        assert len(q2) == 2

        # Replay should give same result
        q3 = q2.replay()
        assert len(q3) == 2
        assert list(q3.to_df()["y"]) == [4, 6]

    def test_reload_from_file(self, tmp_path):
        """Should reload from file and re-apply changes."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("price,qty\n10,2\n20,3\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.assign(total=lambda x: x.price * x.qty)

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

        df = _load_csv_to_dataframe(str(csv_file))
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
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)

        assert len(q2._changes) == 2

        q3 = q2.rebase()
        assert len(q3._changes) == 0

    def test_rebase_preserves_state(self):
        """Should preserve current DataFrame state."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
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

    def test_std(self):
        """Should compute standard deviation."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        result = q.std("x")
        expected = pd.Series([1, 2, 3, 4, 5]).std()
        assert abs(result - expected) < 0.01

    def test_var(self):
        """Should compute variance."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)
        result = q.var("x")
        expected = pd.Series([1, 2, 3, 4, 5]).var()
        assert abs(result - expected) < 0.01


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
        with pytest.raises(AttributeError):
            q3 = q2.assign(c=lambda x: x.b * 2)
            # Force evaluation
            list(q3)

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
        q2 = q.drop("b").assign(total=lambda x: x.a + x.c)

        assert "total" in q2.to_df().columns
        assert "b" not in q2.to_df().columns

        # Replay should re-apply drop and extend
        q3 = q2.replay()
        assert "total" in q3.to_df().columns
        assert "b" not in q3.to_df().columns


class TestQDistinct:
    """Tests for Q.distinct() method."""

    def test_distinct_all_columns(self):
        """Should remove completely duplicate rows."""
        df = pd.DataFrame({"a": [1, 2, 1, 3], "b": [10, 20, 10, 30]})
        q = Q(df)
        q2 = q.distinct()

        assert len(q2) == 3
        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "distinct"

    def test_distinct_single_column(self):
        """Should deduplicate based on single column."""
        df = pd.DataFrame({"customer": ["Alice", "Bob", "Alice", "Charlie"], "order": [1, 2, 3, 4]})
        q = Q(df)
        q2 = q.distinct("customer")

        # Should keep first occurrence of each customer
        assert len(q2) == 3
        customers = list(q2.to_df()["customer"])
        assert customers == ["Alice", "Bob", "Charlie"]
        orders = list(q2.to_df()["order"])
        assert orders == [1, 2, 4]

    def test_distinct_multiple_columns(self):
        """Should deduplicate based on combination of columns."""
        df = pd.DataFrame(
            {
                "email": ["a@x.com", "b@x.com", "a@x.com", "a@y.com"],
                "phone": ["111", "222", "111", "111"],
            }
        )
        q = Q(df)
        q2 = q.distinct("email", "phone")

        # Unique by email+phone combination
        assert len(q2) == 3

    def test_distinct_preserves_order(self):
        """Should keep first occurrence (stable order)."""
        df = pd.DataFrame({"x": [3, 1, 3, 2, 1]})
        q = Q(df)
        q2 = q.distinct("x")

        # Should keep order: 3 (first), 1 (first), 2
        assert list(q2.to_df()["x"]) == [3, 1, 2]

    def test_distinct_with_other_operations(self):
        """Should chain with other operations."""
        df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 3], "y": [10, 20, 20, 30, 30, 30]})
        q = Q(df)
        q2 = q.filter(lambda row: row.x > 1).distinct("x")

        assert len(q2) == 2  # x=2 and x=3

    def test_distinct_with_refresh(self):
        """Should work with refresh/replay."""
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})
        q = Q(df)
        q2 = q.distinct("x")
        q3 = q2.replay()

        assert len(q3) == 3
        assert list(q3.to_df()["x"]) == [1, 2, 3]

    def test_distinct_no_duplicates(self):
        """Should handle case with no duplicates."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.distinct()

        assert len(q2) == 3

    def test_distinct_all_duplicates(self):
        """Should handle case where all rows are duplicates."""
        df = pd.DataFrame({"x": [1, 1, 1, 1]})
        q = Q(df)
        q2 = q.distinct()

        assert len(q2) == 1


class TestQRename:
    """Tests for Q.rename() method."""

    def test_rename_single_column(self):
        """Should rename a single column."""
        df = pd.DataFrame({"old_name": [1, 2, 3]})
        q = Q(df)
        q2 = q.rename(old_name="new_name")

        assert "new_name" in q2.to_df().columns
        assert "old_name" not in q2.to_df().columns
        assert list(q2.to_df()["new_name"]) == [1, 2, 3]

    def test_rename_multiple_columns(self):
        """Should rename multiple columns at once."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        q = Q(df)
        q2 = q.rename(a="x", c="z")

        assert list(q2.to_df().columns) == ["x", "b", "z"]

    def test_rename_nonexistent_column(self):
        """Should gracefully handle renaming nonexistent columns."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        q = Q(df)
        q2 = q.rename(nonexistent="new_name", a="x")

        # Should rename 'a' but ignore 'nonexistent'
        assert list(q2.to_df().columns) == ["x", "b"]

    def test_rename_tracks_changes(self):
        """Should track rename in change history."""
        df = pd.DataFrame({"old": [1]})
        q = Q(df)
        q2 = q.rename(old="new")

        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "rename"

    def test_rename_with_refresh(self):
        """Should work with refresh/replay."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        q = Q(df)
        q2 = q.rename(a="x")
        q3 = q2.replay()

        assert "x" in q3.to_df().columns
        assert "a" not in q3.to_df().columns

    def test_rename_chains_with_operations(self):
        """Should chain with other operations."""
        df = pd.DataFrame({"old_name": [1, 2, 3, 4]})
        q = Q(df)
        q2 = q.rename(old_name="value").filter(lambda x: x.value > 2)

        assert list(q2.to_df().columns) == ["value"]
        assert list(q2.to_df()["value"]) == [3, 4]

    def test_rename_before_extend(self):
        """Should be able to extend using renamed columns."""
        df = pd.DataFrame({"price": [10, 20], "qty": [2, 3]})
        q = Q(df)
        q2 = q.rename(price="unit_price").assign(total=lambda x: x.unit_price * x.qty)

        assert "total" in q2.to_df().columns
        assert list(q2.to_df()["total"]) == [20, 60]

    def test_rename_preserves_data(self):
        """Should only change column names, not data."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        q = Q(df)
        q2 = q.rename(a="x", b="y")

        assert list(q2.to_df()["x"]) == [1, 2, 3]
        assert list(q2.to_df()["y"]) == [4, 5, 6]


class TestQConcat:
    """Tests for Q.concat() method."""

    def test_concat_basic(self):
        """Should concatenate two Q objects vertically."""
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2)

        assert len(q3) == 4
        assert list(q3.to_df()["x"]) == [1, 2, 5, 6]
        assert list(q3.to_df()["y"]) == [3, 4, 7, 8]

    def test_concat_self_reference(self):
        """Should handle self-concatenation (duplicates rows)."""
        df = pd.DataFrame({"x": [1, 2]})
        q = Q(df)
        q2 = q.concat(q)

        assert len(q2) == 4
        assert list(q2.to_df()["x"]) == [1, 2, 1, 2]

    def test_concat_deep_copy_default(self):
        """Should deep copy by default for reproducibility."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2)

        # Should be reproducible with deep copy
        assert q3.deterministic

        # Verify it's a deep copy by checking we can modify original
        # and it doesn't affect the concat result
        _q4 = q2.assign(y=lambda x: x.x * 2)  # noqa: F841 - intentionally unused
        q5 = q3.replay()

        # q3 should still have original q2 data (no y column)
        assert "y" not in q5.to_df().columns

    def test_concat_with_deep_copy_false(self):
        """Should use reference mode when deep_copy=False."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2, deep_copy=False)

        # Should be marked as non-reproducible
        assert not q3.deterministic

    def test_concat_propagates_reproducibility(self):
        """Should propagate reproducibility flag from both Qs."""
        df1 = pd.DataFrame({"x": range(100)})
        df2 = pd.DataFrame({"x": range(100, 200)})
        q1 = Q(df1)
        q2 = Q(df2)

        # Both reproducible
        q3 = q1.concat(q2)
        assert q3.deterministic

        # One non-reproducible
        q2_sample = q2.sample(10)  # No random_state
        assert not q2_sample.deterministic
        q4 = q1.concat(q2_sample)
        assert not q4.deterministic

    def test_concat_tracks_changes(self):
        """Should track concat in change history."""
        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"x": [2]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2)

        assert len(q3._changes) == 1
        assert q3._changes[0][0] == "concat"

    def test_concat_with_refresh(self):
        """Should work with refresh/replay."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2)
        q4 = q3.replay()

        assert len(q4) == 4
        assert list(q4.to_df()["x"]) == [1, 2, 3, 4]

    def test_concat_chains_with_operations(self):
        """Should chain with other operations."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [4, 5, 6]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2).filter(lambda r: r.x > 2)

        assert len(q3) == 4  # 3, 4, 5, 6
        assert list(q3.to_df()["x"]) == [3, 4, 5, 6]

    def test_concat_with_different_columns(self):
        """Should handle different column sets (fills with NaN)."""
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"x": [5, 6], "z": [7, 8]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2)

        assert len(q3) == 4
        assert set(q3.to_df().columns) == {"x", "y", "z"}
        # First two rows should have y, last two should have NaN for y
        assert pd.isna(q3.to_df()["y"].iloc[2])
        assert pd.isna(q3.to_df()["z"].iloc[0])

    def test_concat_with_operations_on_both(self):
        """Should concat Q objects that have their own operations."""
        df1 = pd.DataFrame({"x": [1, 2, 3, 4]})
        df2 = pd.DataFrame({"x": [10, 20, 30, 40]})
        q1 = Q(df1).filter(lambda r: r.x > 2)  # [3, 4]
        q2 = Q(df2).filter(lambda r: r.x < 30)  # [10, 20]
        q3 = q1.concat(q2)

        assert len(q3) == 4
        assert list(q3.to_df()["x"]) == [3, 4, 10, 20]

    def test_concat_rebase_drops_deep_copy(self):
        """Should drop deep copy reference when rebase() is called."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2)

        # q3 has deep copy in history
        assert len(q3._changes) == 1

        q4 = q3.rebase()
        # After rebase, history should be empty
        assert len(q4._changes) == 0
        # But data should be preserved
        assert len(q4) == 4


class TestQMerge:
    """Tests for Q.merge() method."""

    def test_merge_basic_inner(self):
        """Should merge two Q objects on a common column."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        df2 = pd.DataFrame({"id": [1, 2, 4], "age": [25, 30, 35]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id")

        assert len(q3) == 2  # Only id 1 and 2 exist in both
        assert set(q3.to_df().columns) == {"id", "name", "age"}

    def test_merge_left_join(self):
        """Should perform left join."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        df2 = pd.DataFrame({"id": [1, 2], "age": [25, 30]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id", how="left")

        assert len(q3) == 3  # All from left
        assert list(q3.to_df()["name"]) == ["Alice", "Bob", "Charlie"]

    def test_merge_with_conflict_resolution(self):
        """Should resolve column conflicts with lambda."""
        df1 = pd.DataFrame(
            {"id": [1, 2], "name": ["Alice", "Bob"], "status": ["active", "inactive"]}
        )
        df2 = pd.DataFrame({"id": [1, 2], "status": ["pending", "complete"]})
        q1 = Q(df1)
        q2 = Q(df2)

        # Resolve by taking left value
        q3 = q1.merge(q2, on="id", resolve={"status": lambda left, right: left})

        assert list(q3.to_df()["status"]) == ["active", "inactive"]

    def test_merge_conflict_requires_resolution(self):
        """Should raise error if conflicts exist without resolution."""
        df1 = pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]})
        df2 = pd.DataFrame({"id": [1, 2], "status": ["pending", "complete"]})
        q1 = Q(df1)
        q2 = Q(df2)

        with pytest.raises(ValueError, match="Column conflicts detected"):
            q1.merge(q2, on="id")

    def test_merge_self_reference(self):
        """Should handle self-merge (employee-manager)."""
        df = pd.DataFrame(
            {"emp_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "manager_id": [None, 1, 1]}
        )
        q = Q(df)

        # Self-merge to get manager names (with conflict resolution)
        q2 = q.merge(
            q, on="emp_id", resolve={"name": lambda emp, mgr: emp, "manager_id": lambda e, m: e}
        )

        # Should work without circular reference issues
        assert len(q2) == 3

    def test_merge_deep_copy_default(self):
        """Should deep copy by default for reproducibility."""
        df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df2 = pd.DataFrame({"id": [1, 2], "age": [25, 30]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id")

        # Should be reproducible with deep copy
        assert q3.deterministic

        # Verify it's a deep copy
        _q4 = q2.assign(salary=lambda x: 50000)  # noqa: F841 - intentionally unused
        q5 = q3.replay()

        # q3 should still have original q2 data (no salary column)
        assert "salary" not in q5.to_df().columns

    def test_merge_with_deep_copy_false(self):
        """Should use reference mode when deep_copy=False."""
        df1 = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
        df2 = pd.DataFrame({"id": [1, 2], "y": [30, 40]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id", deep_copy=False)

        # Should be marked as non-reproducible
        assert not q3.deterministic

    def test_merge_propagates_reproducibility(self):
        """Should propagate reproducibility flag from both Qs."""
        df1 = pd.DataFrame({"id": range(100), "x": range(100)})
        df2 = pd.DataFrame({"id": range(100), "y": range(100, 200)})
        q1 = Q(df1)
        q2 = Q(df2)

        # Both reproducible
        q3 = q1.merge(q2, on="id")
        assert q3.deterministic

        # One non-reproducible
        q2_sample = q2.sample(10)  # No random_state
        assert not q2_sample.deterministic
        q4 = q1.merge(q2_sample, on="id")
        assert not q4.deterministic

    def test_merge_multiple_keys(self):
        """Should merge on multiple columns."""
        df1 = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 1], "x": [10, 20, 30]})
        df2 = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 1], "y": [100, 200, 300]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on=["a", "b"])

        assert len(q3) == 3
        assert set(q3.to_df().columns) == {"a", "b", "x", "y"}

    def test_merge_tracks_changes(self):
        """Should track merge in change history."""
        df1 = pd.DataFrame({"id": [1], "x": [10]})
        df2 = pd.DataFrame({"id": [1], "y": [20]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id")

        assert len(q3._changes) == 1
        assert q3._changes[0][0] == "merge"

    def test_merge_with_refresh(self):
        """Should work with refresh/replay."""
        df1 = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
        df2 = pd.DataFrame({"id": [1, 2], "y": [30, 40]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id")
        q4 = q3.replay()

        assert len(q4) == 2
        assert set(q4.to_df().columns) == {"id", "x", "y"}

    def test_merge_chains_with_operations(self):
        """Should chain with other operations."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
        df2 = pd.DataFrame({"id": [1, 2, 3], "y": [100, 200, 300]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.merge(q2, on="id").filter(lambda r: r.x > 15)

        assert len(q3) == 2  # Only id 2 and 3
        assert list(q3.to_df()["id"]) == [2, 3]

    def test_merge_conflict_resolution_handles_none(self):
        """Should handle None values in conflict resolution."""
        df1 = pd.DataFrame({"id": [1, 2], "val": ["a", None]})
        df2 = pd.DataFrame({"id": [1, 2], "val": ["b", "c"]})
        q1 = Q(df1)
        q2 = Q(df2)

        # Take right if left is None
        q3 = q1.merge(
            q2, on="id", resolve={"val": lambda left, right: right if pd.isna(left) else left}
        )

        assert list(q3.to_df()["val"]) == ["a", "c"]


class TestQJoin:
    """Tests for Q.join() method (wrapper around merge)."""

    def test_join_basic(self):
        """Should join two Q objects without conflicts."""
        df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df2 = pd.DataFrame({"id": [1, 2], "age": [25, 30]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.join(q2, on="id")

        assert len(q3) == 2
        assert set(q3.to_df().columns) == {"id", "name", "age"}

    def test_join_raises_on_conflict(self):
        """Should raise error if conflicts exist (use merge instead)."""
        df1 = pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]})
        df2 = pd.DataFrame({"id": [1, 2], "status": ["pending", "complete"]})
        q1 = Q(df1)
        q2 = Q(df2)

        with pytest.raises(ValueError, match="Column conflicts detected"):
            q1.join(q2, on="id")


class TestQSetOperations:
    """Tests for Q union(), intersect(), difference() methods."""

    def test_union_basic(self):
        """Should union two Q objects (removes duplicates)."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [2, 3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.union(q2)

        assert len(q3) == 4  # [1, 2, 3, 4]
        assert set(q3.to_df()["x"]) == {1, 2, 3, 4}

    def test_union_with_duplicates(self):
        """Should remove duplicates from union."""
        df1 = pd.DataFrame({"x": [1, 1, 2, 2]})
        df2 = pd.DataFrame({"x": [2, 2, 3, 3]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.union(q2)

        assert set(q3.to_df()["x"]) == {1, 2, 3}

    def test_intersect_basic(self):
        """Should find rows in both Q objects."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [2, 3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.intersect(q2)

        assert set(q3.to_df()["x"]) == {2, 3}

    def test_intersect_no_common(self):
        """Should return empty if no common rows."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.intersect(q2)

        assert len(q3) == 0

    def test_intersect_self(self):
        """Should intersect with self."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.intersect(q)

        assert len(q2) == 3
        assert set(q2.to_df()["x"]) == {1, 2, 3}

    def test_difference_basic(self):
        """Should find rows in self but not other."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [2, 3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.difference(q2)

        assert set(q3.to_df()["x"]) == {1}

    def test_difference_self(self):
        """Should return empty for self-difference."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.difference(q)

        assert len(q2) == 0

    def test_difference_none_remaining(self):
        """Should return empty if all rows removed."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [1, 2, 3]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.difference(q2)

        assert len(q3) == 0

    def test_set_ops_propagate_reproducibility(self):
        """Should propagate reproducibility through set operations."""
        df1 = pd.DataFrame({"x": range(100)})
        df2 = pd.DataFrame({"x": range(50, 150)})
        q1 = Q(df1)
        q2 = Q(df2)

        # Union
        q3 = q1.union(q2)
        assert q3.deterministic

        # With non-reproducible Q
        q2_sample = q2.sample(10)
        q4 = q1.union(q2_sample)
        assert not q4.deterministic


class TestQShowDump:
    """Tests for Q.show() and Q.dump() methods."""

    def test_show_returns_self(self):
        """Should print and return self for chaining."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.show()

        assert q2 is q  # Same object

    def test_dump_creates_file(self, tmp_path):
        """Should write DataFrame to CSV file."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        q = Q(df)

        output_file = tmp_path / "output.csv"
        q2 = q.dump(str(output_file))

        # Check file exists and has content
        assert output_file.exists()
        loaded = pd.read_csv(str(output_file))
        assert len(loaded) == 3
        assert list(loaded.columns) == ["x", "y"]

        # Check returns self
        assert q2 is q


class TestQGroupby:
    """Tests for Q.groupby() method."""

    def test_groupby_simple(self):
        """Should group and aggregate data."""
        df = pd.DataFrame({"category": ["A", "A", "B", "B"], "value": [10, 20, 30, 40]})
        q = Q(df)
        result = q.groupby(lambda x: x.category, total=lambda g: sum(r.value for r in g))

        assert len(result) == 2
        result_df = result.to_df()
        a_row = result_df[result_df["key"] == "A"].iloc[0]
        b_row = result_df[result_df["key"] == "B"].iloc[0]
        assert a_row["total"] == 30
        assert b_row["total"] == 70

    def test_groupby_multiple_aggregations(self):
        """Should support multiple aggregation functions."""
        df = pd.DataFrame({"category": ["A", "A", "B"], "value": [10, 20, 30]})
        q = Q(df)
        result = q.groupby(
            lambda x: x.category,
            count=lambda g: len(g),
            total=lambda g: sum(r.value for r in g),
            avg=lambda g: sum(r.value for r in g) / len(g),
        )

        result_df = result.to_df()
        assert "count" in result_df.columns
        assert "total" in result_df.columns
        assert "avg" in result_df.columns

    def test_groupby_resets_history(self):
        """Groupby should create a new Q with no history."""
        df = pd.DataFrame({"cat": ["A", "B"], "val": [1, 2]})
        q = Q(df)
        q2 = q.assign(double=lambda x: x.val * 2)
        result = q2.groupby(lambda x: x.cat, total=lambda g: sum(r.val for r in g))

        # Should be a new Q with fresh history
        assert len(result._changes) == 0


class TestQReloadRecursive:
    """Tests for deep/recursive reload functionality."""

    def test_reload_with_concat(self, tmp_path):
        """Should recursively reload both Q objects in concat."""
        # Create two CSV files
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n")
        csv2.write_text("x\n3\n4\n")

        # Load and concat
        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))
        q3 = q1.concat(q2)

        assert len(q3) == 4

        # Update both files
        csv1.write_text("x\n10\n20\n30\n")
        csv2.write_text("x\n40\n50\n")

        # Reload should reload both
        q4 = q3.reload()
        assert len(q4) == 5
        assert set(q4.to_df()["x"]) == {10, 20, 30, 40, 50}

    def test_reload_preserves_reproducibility(self, tmp_path):
        """Should preserve reproducibility flag on reload."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.sample(1, random_state=42)  # Reproducible

        # Update file
        csv_file.write_text("x\n10\n20\n")

        q3 = q2.reload()
        assert q3.deterministic  # Should still be reproducible


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
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Diana"],
                "age": [25, 30, 35, 28],
                "salary": [50000, 60000, 70000, 55000],
            }
        )
        q = Q(df)

        result = (
            q.assign(monthly=lambda x: x.salary / 12)
            .filter(lambda x: x.age >= 28)
            .sort("monthly", ascending=False)
            .head(2)
        )

        assert len(result) == 2
        assert result.to_df().iloc[0]["name"] == "Charlie"

    def test_change_history_preserved(self):
        """Should preserve complete change history."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        q = Q(df)

        q2 = (
            q.assign(y=lambda x: x.x * 2)
            .filter(lambda x: x.y > 4)
            .assign(z=lambda x: x.y + 10)
            .sort("z")
            .head(2)
        )

        assert len(q2._changes) == 5
        change_types = [c[0] for c in q2._changes]
        assert change_types == ["assign", "filter", "assign", "sort", "head"]


class TestQMemoryUsageBasic:
    """Tests for Q.memory_usage() method - basic tests."""

    def test_memory_usage_simple(self):
        """Should report memory for simple Q."""
        df = pd.DataFrame({"x": [1, 2, 3] * 100})
        q = Q(df)

        usage = q.memory_usage()
        assert "base_df" in usage
        assert "current_df" in usage
        assert "changes" in usage
        assert usage["changes"] == 0

    def test_memory_usage_with_operations(self):
        """Should report memory after operations."""
        df = pd.DataFrame({"x": range(1000), "y": range(1000)})
        q = Q(df)
        q2 = q.filter(lambda r: r.x < 100)  # Reduces rows

        usage = q2.memory_usage()
        assert usage["changes"] == 1

    def test_memory_usage_after_rebase(self):
        """Memory should reflect flattened state after rebase."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df).assign(y=lambda r: r.x * 2).assign(z=lambda r: r.x * 3)
        q_rebased = q.rebase()

        usage = q_rebased.memory_usage()
        assert usage["changes"] == 0  # History cleared


class TestQEdgeCases:
    """Tests for edge cases and error handling."""

    def test_map_scalar_output(self):
        """Should handle transform that returns scalar values."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.map(lambda r: r.x * 2)  # Returns scalar, not dict/tuple

        result_df = q2.to_df()
        assert "value" in result_df.columns
        assert list(result_df["value"]) == [2, 4, 6]

    def test_merge_partial_conflict_resolution(self):
        """Should error if not all conflicts are resolved."""
        df1 = pd.DataFrame({"id": [1], "status": ["a"], "flag": ["x"]})
        df2 = pd.DataFrame({"id": [1], "status": ["b"], "flag": ["y"]})
        q1 = Q(df1)
        q2 = Q(df2)

        # Only resolve one of two conflicts
        with pytest.raises(ValueError, match="Missing resolution"):
            q1.merge(q2, on="id", resolve={"status": lambda left, right: left})

    def test_reload_with_merge(self, tmp_path):
        """Should handle reload with merge operation."""
        csv1 = tmp_path / "f1.csv"
        csv2 = tmp_path / "f2.csv"
        csv1.write_text("id,name\n1,Alice\n")
        csv2.write_text("id,age\n1,30\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))
        q3 = q1.merge(q2, on="id")

        # Update files
        csv1.write_text("id,name\n1,Bob\n")
        csv2.write_text("id,age\n1,25\n")

        # Reload should work
        q4 = q3.reload()
        assert q4.to_df()["name"].iloc[0] == "Bob"
        assert q4.to_df()["age"].iloc[0] == 25


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


"""Tests for Phase 3: Data Quality methods (dropna, fillna, replace)."""


class TestDropna:
    """Tests for Q.dropna() method."""

    def test_dropna_all_columns_default(self):
        """Should drop rows with any null by default."""
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8], "c": [9, 10, 11, 12]})
        q = Q(df)
        q2 = q.dropna()

        assert len(q2) == 2  # Only rows 0 and 3 have no nulls
        assert list(q2.to_df()["a"]) == [1.0, 4.0]

    def test_dropna_single_column(self):
        """Should drop rows where specified column is null."""
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8], "c": [9, 10, 11, 12]})
        q = Q(df)
        q2 = q.dropna("a")

        assert len(q2) == 3  # Only row 2 dropped
        assert list(q2.to_df()["c"]) == [9.0, 10.0, 12.0]

    def test_dropna_multiple_columns_any(self):
        """Should drop rows where ANY specified column is null."""
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8], "c": [9, 10, 11, 12]})
        q = Q(df)
        q2 = q.dropna("a", "b")

        assert len(q2) == 2  # Rows 0 and 3 have neither a nor b null
        assert list(q2.to_df()["c"]) == [9.0, 12.0]

    def test_dropna_multiple_columns_all(self):
        """Should drop rows where ALL specified columns are null."""
        df = pd.DataFrame({"a": [1, None, None, 4], "b": [5, None, None, 8], "c": [9, 10, 11, 12]})
        q = Q(df)
        q2 = q.dropna("a", "b", how="all")

        assert len(q2) == 2  # Rows 1 and 2 have BOTH a and b null
        assert list(q2.to_df()["c"]) == [9.0, 12.0]

    def test_dropna_no_nulls(self):
        """Should return unchanged Q if no nulls."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        q = Q(df)
        q2 = q.dropna()

        assert len(q2) == 3
        assert list(q2.to_df()["a"]) == [1, 2, 3]

    def test_dropna_all_nulls(self):
        """Should return empty Q if all rows have nulls."""
        df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
        q = Q(df)
        q2 = q.dropna()

        assert len(q2) == 0

    def test_dropna_invalid_how(self):
        """Should raise ValueError for invalid 'how' parameter."""
        df = pd.DataFrame({"a": [1, None, 3]})
        q = Q(df)

        with pytest.raises(ValueError, match="how must be 'any' or 'all'"):
            q.dropna(how="invalid")

    def test_dropna_is_wrapper_around_filter(self):
        """dropna should delegate to filter (check immutability)."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        q = Q(df)
        q2 = q.dropna("a")

        # Original should be unchanged
        assert len(q) == 3
        # New Q should have nulls removed
        assert len(q2) == 2
        # Should have filter in change history
        assert any(change[0] == "filter" for change in q2._changes)


class TestFillna:
    """Tests for Q.fillna() method."""

    def test_fillna_scalar_all_columns(self):
        """Should fill all nulls with scalar value."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 5, 6]})
        q = Q(df)
        q2 = q.fillna(0)

        result = q2.to_df()
        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert result["b"].tolist() == [0.0, 5.0, 6.0]

    def test_fillna_mapping(self):
        """Should fill nulls with column-specific values."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 5, 6], "c": [7, None, 9]})
        q = Q(df)
        q2 = q.fillna({"a": 0, "c": 99})

        result = q2.to_df()
        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert pd.isna(result["b"].iloc[0])  # b not filled
        assert result["c"].tolist() == [7.0, 99.0, 9.0]

    def test_fillna_string_columns(self):
        """Should fill nulls in string columns."""
        df = pd.DataFrame({"name": ["Alice", None, "Charlie"], "city": [None, "NYC", "LA"]})
        q = Q(df)
        q2 = q.fillna({"name": "Unknown", "city": "Unknown"})

        result = q2.to_df()
        assert result["name"].tolist() == ["Alice", "Unknown", "Charlie"]
        assert result["city"].tolist() == ["Unknown", "NYC", "LA"]

    def test_fillna_no_nulls(self):
        """Should work fine on data with no nulls."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        q = Q(df)
        q2 = q.fillna(0)

        result = q2.to_df()
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == [4, 5, 6]

    def test_fillna_immutability(self):
        """Original Q should remain unchanged."""
        df = pd.DataFrame({"a": [1, None, 3]})
        q = Q(df)
        q2 = q.fillna(0)

        # Original should still have null
        assert pd.isna(q.to_df()["a"].iloc[1])
        # New Q should have fill
        assert q2.to_df()["a"].iloc[1] == 0.0

    def test_fillna_in_change_history(self):
        """fillna should be tracked in change history."""
        df = pd.DataFrame({"a": [1, None, 3]})
        q = Q(df)
        q2 = q.fillna(0)

        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "fillna"
        assert q2._changes[0][1] == 0

    def test_fillna_chaining(self):
        """Should chain with other operations."""
        df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": [10, 20, 30, 40, 50]})
        q = Q(df)
        q2 = q.fillna(0).filter(lambda x: x.a > 0).assign(c=lambda x: x.a + x.b)

        result = q2.to_df()
        assert len(result) == 3  # Filter keeps a > 0
        assert result["c"].tolist() == [11.0, 33.0, 55.0]


class TestReplace:
    """Tests for Q.replace() method."""

    def test_replace_scalar_all_columns(self):
        """Should replace value across all columns."""
        df = pd.DataFrame({"a": [1, 0, 3], "b": [0, 5, 6]})
        q = Q(df)
        q2 = q.replace(0, -1)

        result = q2.to_df()
        assert result["a"].tolist() == [1, -1, 3]
        assert result["b"].tolist() == [-1, 5, 6]

    def test_replace_column_specific_mapping(self):
        """Should replace values in specific columns."""
        df = pd.DataFrame(
            {
                "region": ["CA", "NY", "CA", "TX"],
                "status": ["active", "inactive", "active", "inactive"],
            }
        )
        q = Q(df)
        q2 = q.replace(
            {
                "region": {"CA": "California", "NY": "New York"},
                "status": {"active": "Active", "inactive": "Inactive"},
            }
        )

        result = q2.to_df()
        assert result["region"].tolist() == ["California", "New York", "California", "TX"]
        assert result["status"].tolist() == ["Active", "Inactive", "Active", "Inactive"]

    def test_replace_value_mapping_all_columns(self):
        """Should replace values across all columns with mapping."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
        q = Q(df)
        q2 = q.replace({2: 99, 3: 88})

        result = q2.to_df()
        assert result["a"].tolist() == [1, 99, 88]
        assert result["b"].tolist() == [99, 88, 4]

    def test_replace_no_matches(self):
        """Should work fine when no values match."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        q = Q(df)
        q2 = q.replace(999, -1)

        result = q2.to_df()
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == [4, 5, 6]

    def test_replace_with_nan(self):
        """Should replace values with NaN."""
        df = pd.DataFrame({"a": [1, 0, 3], "b": [0, 5, 6]})
        q = Q(df)
        q2 = q.replace(0, np.nan)

        result = q2.to_df()
        assert result["a"].iloc[0] == 1
        assert pd.isna(result["a"].iloc[1])
        assert pd.isna(result["b"].iloc[0])

    def test_replace_nan_with_value(self):
        """Should replace NaN with value."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 5, 6]})
        q = Q(df)
        q2 = q.replace(np.nan, 0)

        result = q2.to_df()
        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert result["b"].tolist() == [0.0, 5.0, 6.0]

    def test_replace_immutability(self):
        """Original Q should remain unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        q = Q(df)
        q2 = q.replace(2, 99)

        # Original unchanged
        assert q.to_df()["a"].tolist() == [1, 2, 3]
        # New Q has replacement
        assert q2.to_df()["a"].tolist() == [1, 99, 3]

    def test_replace_in_change_history(self):
        """replace should be tracked in change history."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        q = Q(df)
        q2 = q.replace(2, 99)

        assert len(q2._changes) == 1
        assert q2._changes[0][0] == "replace"

    def test_replace_chaining(self):
        """Should chain with other operations."""
        df = pd.DataFrame({"status": ["old", "old", "new", "old"], "value": [10, 20, 30, 40]})
        q = Q(df)
        q2 = (
            q.replace({"status": {"old": "legacy"}})
            .filter(lambda x: x.status == "legacy")
            .assign(doubled=lambda x: x.value * 2)
        )

        result = q2.to_df()
        assert len(result) == 3
        assert result["doubled"].tolist() == [20, 40, 80]


class TestPhase3Integration:
    """Integration tests for Phase 3 methods."""

    def test_all_three_methods_together(self):
        """Test using dropna, fillna, and replace in sequence."""
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Charlie", "Dave"],
                "region": ["CA", "NY", None, "CA"],
                "score": [85, 90, None, 75],
            }
        )
        q = Q(df)

        # Drop rows with null name, fill other nulls, replace CA
        q2 = (
            q.dropna("name")
            .fillna({"region": "Unknown", "score": 0})
            .replace({"region": {"CA": "California"}})
        )

        result = q2.to_df()
        assert len(result) == 3
        assert result["name"].tolist() == ["Alice", "Charlie", "Dave"]
        assert result["region"].tolist() == ["California", "Unknown", "California"]
        assert result["score"].tolist() == [85.0, 0.0, 75.0]

    def test_phase3_with_phase1_operations(self):
        """Test Phase 3 with Phase 1 operations."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "status": ["old", "new", None, "old", "new"],
                "value": [10, None, 30, 40, 50],
            }
        )
        q = Q(df)

        q2 = (
            q.fillna({"status": "unknown", "value": 0})
            .replace({"status": {"old": "legacy"}})
            .filter(lambda x: x.value > 0)
            .distinct("status")
            .sort("value")
        )

        result = q2.to_df()
        assert len(result) == 3
        # After filling nulls, replacing, filtering, distinct, and sorting by value
        # we get: legacy(10), unknown(30), new(50)
        assert result["status"].tolist() == ["legacy", "unknown", "new"]

    def test_replay_with_phase3_operations(self):
        """Test that Phase 3 operations replay correctly."""
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        q = Q(df)
        q2 = q.fillna({"a": 0, "b": "z"}).replace({"b": {"x": "X"}})

        # Replay should produce same result
        q3 = q2.replay()

        assert q2.to_df().equals(q3.to_df())

    def test_deterministic_flag_preserved(self):
        """Phase 3 operations should preserve deterministic flag."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        q = Q(df)

        assert q.deterministic is True

        q2 = q.dropna("a")
        assert q2.deterministic is True

        q3 = q2.fillna(0)
        assert q3.deterministic is True

        q4 = q3.replace(0, -1)
        assert q4.deterministic is True

    def test_reloadable_flag_preserved(self):
        """Phase 3 operations should preserve reloadable flag."""
        df = pd.DataFrame({"a": [1, None, 3]})
        q = Q(df, source_path="test.csv")

        assert q.reloadable is True

        q2 = q.fillna(0).dropna().replace(0, 1)
        assert q2.reloadable is True


"""Comprehensive tests for deterministic and reloadable flags."""


class TestDeterministicFlag:
    """Tests for the deterministic flag functionality."""

    def test_deterministic_true_by_default(self):
        """Q should be deterministic by default."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        assert q.deterministic

    def test_deterministic_after_operations(self):
        """Deterministic operations should preserve flag."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
        assert q2.deterministic

    def test_sample_without_seed_not_deterministic(self):
        """Sample without seed should mark as non-deterministic."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(10)  # No random_state
        assert not q2.deterministic

    def test_sample_with_seed_deterministic(self):
        """Sample with seed should remain deterministic."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(10, random_state=42)
        assert q2.deterministic

    def test_deterministic_propagates_through_concat(self):
        """Concat should propagate deterministic flag."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)

        # Both deterministic
        q3 = q1.concat(q2)
        assert q3.deterministic

        # One non-deterministic
        q2_sample = q2.sample(1)
        q4 = q1.concat(q2_sample)
        assert not q4.deterministic

    def test_deterministic_false_with_deep_copy_false(self):
        """deep_copy=False should set deterministic to False."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        q1 = Q(df1)
        q2 = Q(df2)
        q3 = q1.concat(q2, deep_copy=False)
        assert not q3.deterministic

    def test_rebase_sets_deterministic_true(self):
        """Rebase should set deterministic to True (empty history)."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(10)  # Non-deterministic
        assert not q2.deterministic

        q3 = q2.rebase()
        assert q3.deterministic  # Empty history is deterministic


class TestReloadableFlag:
    """Tests for the reloadable flag functionality."""

    def test_reloadable_with_source_path(self, tmp_path):
        """Q with source_path should be reloadable."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        assert q.reloadable

    def test_not_reloadable_without_source_path(self):
        """Q without source_path should not be reloadable."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        assert not q.reloadable

    def test_reloadable_after_operations(self, tmp_path):
        """Operations should preserve reloadable flag."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
        assert q2.reloadable

    def test_rebase_sets_reloadable_false(self, tmp_path):
        """Rebase should set reloadable to False."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        assert q.reloadable

        q2 = q.filter(lambda x: x.x > 1).rebase()
        assert not q2.reloadable

    def test_reloadable_propagates_through_concat(self, tmp_path):
        """Concat should propagate reloadable flag."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n")
        csv2.write_text("x\n2\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        # Both reloadable
        q3 = q1.concat(q2)
        assert q3.reloadable

        # One not reloadable
        q2_rebased = q2.rebase()
        q4 = q1.concat(q2_rebased)
        assert not q4.reloadable

    def test_reloadable_false_with_deep_copy_false(self, tmp_path):
        """deep_copy=False should set reloadable to False."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n")
        csv2.write_text("x\n2\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))
        q3 = q1.concat(q2, deep_copy=False)
        assert not q3.reloadable


class TestReloadWithPartialReload:
    """Tests for reload() with allow_partial_reload parameter."""

    def test_reload_fails_without_allow_partial_after_rebase(self, tmp_path):
        """Reload should fail if tree contains non-reloadable Q."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n")
        csv2.write_text("x\n3\n4\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        # Rebase q2, making it non-reloadable
        q2_rebased = q2.rebase()
        q3 = q1.concat(q2_rebased)

        # Should fail without allow_partial_reload
        with pytest.raises(ValueError, match="not reloadable"):
            q3.reload()

    def test_reload_succeeds_with_allow_partial(self, tmp_path):
        """Reload with allow_partial_reload should use current state for non-reloadable Qs."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n")
        csv2.write_text("x\n3\n4\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        # Rebase q2
        q2_rebased = q2.rebase()
        q3 = q1.concat(q2_rebased)

        # Update csv1
        csv1.write_text("x\n10\n20\n")

        # Should succeed with allow_partial_reload
        q4 = q3.reload(allow_partial_reload=True)
        assert len(q4) == 4
        # First two rows from updated csv1, last two from q2_rebased's current state
        assert 10 in q4.to_df()["x"].values
        assert 20 in q4.to_df()["x"].values

    def test_reload_without_source_path_fails(self):
        """Reload should fail if Q has no source path."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)

        with pytest.raises(ValueError, match="no source path"):
            q.reload()

    def test_reload_without_source_succeeds_with_allow_partial(self):
        """Reload with allow_partial_reload should return self if no source."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)

        q2 = q.reload(allow_partial_reload=True)
        assert q2 is q  # Same object


class TestRebaseFlags:
    """Tests for rebase() flag behavior."""

    def test_rebase_deterministic_true(self, tmp_path):
        """Rebase should always set deterministic=True."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n" + "\n".join(str(i) for i in range(100)))

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))

        # Make non-deterministic
        q2 = q.sample(10)
        assert not q2.deterministic

        # Rebase should set to True
        q3 = q2.rebase()
        assert q3.deterministic

    def test_rebase_reloadable_false(self, tmp_path):
        """Rebase should always set reloadable=False."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        assert q.reloadable

        q2 = q.filter(lambda x: x.x > 1).rebase()
        assert not q2.reloadable

    def test_rebase_keeps_source_path(self, tmp_path):
        """Rebase should keep source_path but Q is not reloadable."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.filter(lambda x: x.x > 1).rebase()

        # Still has source_path
        assert q2._source_path == str(csv_file)
        # But not reloadable
        assert not q2.reloadable
        # Reload should fail
        with pytest.raises(ValueError, match="not reloadable"):
            q2.reload()


class TestFlagPropagationMultiQ:
    """Tests for flag propagation through multi-Q operations."""

    def test_merge_propagates_both_flags(self, tmp_path):
        """Merge should propagate both deterministic and reloadable."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("id,x\n1,10\n2,20\n")
        csv2.write_text("id,y\n1,100\n2,200\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        # Both deterministic and reloadable
        q3 = q1.merge(q2, on="id")
        assert q3.deterministic
        assert q3.reloadable

        # One non-deterministic
        q2_sample = q2.sample(1)
        q4 = q1.merge(q2_sample, on="id")
        assert not q4.deterministic
        assert q4.reloadable  # Still reloadable

        # One non-reloadable
        q2_rebased = q2.rebase()
        q5 = q1.merge(q2_rebased, on="id")
        assert q5.deterministic  # Both were deterministic before rebase
        assert not q5.reloadable

    def test_union_propagates_flags(self, tmp_path):
        """Union should propagate flags through concat."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n")
        csv2.write_text("x\n2\n3\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        q3 = q1.union(q2)
        assert q3.deterministic
        assert q3.reloadable

    def test_intersect_propagates_flags(self, tmp_path):
        """Intersect should propagate flags."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n3\n")
        csv2.write_text("x\n2\n3\n4\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        q3 = q1.intersect(q2)
        assert q3.deterministic
        assert q3.reloadable

        # With non-deterministic Q
        q2_sample = q2.sample(2)
        q4 = q1.intersect(q2_sample)
        assert not q4.deterministic
        assert q4.reloadable

    def test_difference_propagates_flags(self, tmp_path):
        """Difference should propagate flags."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n3\n")
        csv2.write_text("x\n2\n3\n4\n")

        df1 = _load_csv_to_dataframe(str(csv1))
        df2 = _load_csv_to_dataframe(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))

        q3 = q1.difference(q2)
        assert q3.deterministic
        assert q3.reloadable


class TestReplayVsReload:
    """Tests to verify replay() and reload() work correctly."""

    def test_replay_works_without_source(self):
        """replay() should work without source_path."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)

        q3 = q2.replay()
        assert len(q3) == 2
        assert list(q3.to_df()["y"]) == [4, 6]

    def test_replay_works_on_non_deterministic(self):
        """replay() should work even on non-deterministic Q (may give different results)."""
        df = pd.DataFrame({"x": range(100)})
        q = Q(df)
        q2 = q.sample(10)  # Non-deterministic

        # Should work (but results may vary)
        q3 = q2.replay()
        assert len(q3) == 10
        assert not q3.deterministic

    def test_reload_requires_source(self):
        """reload() requires source_path."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        q = Q(df)

        with pytest.raises(ValueError, match="no source path"):
            q.reload()

    def test_reload_reloads_from_disk(self, tmp_path):
        """reload() should reload from disk."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")

        df = _load_csv_to_dataframe(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.assign(y=lambda x: x.x * 2)

        # Update file
        csv_file.write_text("x\n10\n20\n30\n")

        # Reload gets new data
        q3 = q2.reload()
        assert list(q3.to_df()["x"]) == [10, 20, 30]
        assert list(q3.to_df()["y"]) == [20, 40, 60]
