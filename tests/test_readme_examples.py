"""Test examples from README to ensure they work as described."""

import pytest
import pandas as pd
import tempfile
import os
from bend.core import Q, load_csv, rows


class TestReadmeExamples:
    """Verify examples from README work correctly."""

    def test_basic_exploration(self):
        """Basic data exploration example."""
        df = pd.DataFrame({
            'customer': ['Alice', 'Bob', 'Charlie'],
            'amount': [100, 200, 150],
            'age': [25, 35, 30]
        })
        q = Q(df)
        
        # Iterate through data
        names = [row.customer for row in q]
        assert names == ['Alice', 'Bob', 'Charlie']
        
        # Quick statistics
        assert q.sum('amount') == 450
        assert q.mean('age') == 30.0
        assert q.count() == 3
        assert q.nunique('customer') == 3

    def test_adding_computed_columns(self):
        """Extend examples from README."""
        df = pd.DataFrame({'price': [10, 20], 'qty': [2, 3]})
        q = Q(df)
        
        # Single column
        q2 = q.extend(total=lambda x: x.price * x.qty)
        assert list(q2.df['total']) == [20, 60]
        
        # Multiple columns (referencing original columns)
        q3 = q.extend(
            total=lambda x: x.price * x.qty,
            discount=lambda x: x.price * 0.1
        )
        assert 'total' in q3.df.columns
        assert 'discount' in q3.df.columns
        
        # Chain extensions (can reference previous extensions)
        q4 = (q
            .extend(revenue=lambda x: x.price * x.qty)
            .extend(cost=lambda x: x.revenue * 0.6)
            .extend(profit=lambda x: x.revenue - x.cost))
        assert 'profit' in q4.df.columns
        assert abs(q4.df.iloc[0]['profit'] - 8.0) < 0.01  # 20 * 0.4

    def test_filtering_examples(self):
        """Filter examples from README."""
        df = pd.DataFrame({
            'purchase_amount': [500, 1500, 800, 2000],
            'age': [20, 30, 25, 35],
            'region': ['East', 'West', 'East', 'West']
        })
        q = Q(df)
        
        # High-value customers
        high_value = q.filter(lambda x: x.purchase_amount > 1000)
        assert len(high_value) == 2
        
        # Multiple conditions
        west_adults = q.filter(lambda x: x.age > 25 and x.region == 'West')
        assert len(west_adults) == 2
        
        # Computed column filters
        q2 = q.extend(total=lambda x: x.purchase_amount).filter(lambda x: x.total > 1000)
        assert len(q2) == 2

    def test_sorting_and_limiting(self):
        """Sort and head examples."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'revenue': [100, 400, 200, 300]
        })
        q = Q(df)
        
        # Top by revenue
        top3 = q.sort('revenue').head(3)
        assert len(top3) == 3
        assert top3.df.iloc[0]['revenue'] == 400
        
        # Sort ascending
        asc = q.sort('revenue', ascending=True)
        assert asc.df.iloc[0]['revenue'] == 100

    def test_transform_examples(self):
        """Transform examples from README."""
        df = pd.DataFrame({
            'first': ['Alice', 'Bob'],
            'last': ['Smith', 'Jones'],
            'email': ['alice@example.com', 'bob@example.com']
        })
        q = Q(df)
        
        # Combine names
        q2 = q.transform(lambda x: {
            'full_name': f"{x.first} {x.last}",
            'email': x.email
        })
        assert list(q2.df.columns) == ['full_name', 'email']
        assert q2.df.iloc[0]['full_name'] == 'Alice Smith'

    def test_groupby_examples(self):
        """Groupby examples from README."""
        df = pd.DataFrame({
            'region': ['East', 'West', 'East', 'West'],
            'amount': [100, 200, 150, 250]
        })
        q = Q(df)
        
        # Sales by region
        summary = q.groupby(
            lambda x: x.region,
            total_sales=lambda g: sum(r.amount for r in g),
            count=lambda g: len(g)
        )
        
        assert len(summary.df) == 2
        result_dict = {row.key: row.total_sales for row in summary}
        assert result_dict['East'] == 250
        assert result_dict['West'] == 450

    def test_external_data_updates(self, tmp_path):
        """Reload example from README."""
        csv_file = tmp_path / "daily_sales.csv"
        csv_file.write_text("sales,units\n100,5\n200,10\n")
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.extend(commission=lambda x: x.sales * 0.15)
        
        # Update file
        csv_file.write_text("sales,units\n150,6\n250,12\n")
        
        # Reload and re-apply
        q3 = q2.reload()
        assert len(q3.df) == 2
        assert 'commission' in q3.df.columns
        assert q3.df.iloc[0]['commission'] == 22.5  # 150 * 0.15

    def test_column_visibility(self):
        """Hide/unhide examples from README."""
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'cost': [50, 75],
            'revenue': [100, 150]
        })
        q = Q(df)
        
        # Hide internal columns
        q2 = q.hide('id', 'cost')
        assert 'id' in q2.df.columns  # Still in data
        display = str(q2)
        assert 'name' in display
        assert 'id' not in display
        
        # Unhide all
        q3 = q2.unhide()
        display = str(q3)
        assert 'name' in display
        assert 'id' in display
        assert 'cost' in display
        
        # Hidden columns work in calculations
        q4 = q.hide('cost').extend(profit=lambda x: x.revenue - x.cost)
        assert 'profit' in q4.df.columns
        assert 'cost' not in str(q4)

    def test_real_world_pipeline(self):
        """Complex pipeline from README."""
        df = pd.DataFrame({
            'price': [100, 200, 150, 180],
            'quantity': [2, 3, 1, 4],
            'discount_pct': [0.1, 0.05, 0.15, 0.0],
            'date': ['2024-01-15', '2024-02-20', '2023-12-10', '2024-03-05'],
            'status': ['completed', 'completed', 'pending', 'completed'],
            'internal_id': [1, 2, 3, 4]
        })
        q = Q(df)
        
        result = (q
            .extend(total=lambda x: x.price * x.quantity)
            .extend(discount=lambda x: x.total * x.discount_pct)
            .extend(final=lambda x: x.total - x.discount)
            .filter(lambda x: x.date.startswith('2024'))
            .filter(lambda x: x.status == 'completed')
            .hide('internal_id')
            .sort('final')
            .head(10))
        
        assert len(result) == 3
        assert 'final' in result.df.columns
        assert 'internal_id' not in str(result)

    def test_performance_optimization(self):
        """Rebase example from README."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        q = Q(df)
        
        q2 = (q
            .extend(a=lambda x: x.x * 2)
            .filter(lambda x: x.a > 4)
            .extend(b=lambda x: x.a + 5)
            .filter(lambda x: x.b < 20)
            .extend(c=lambda x: x.b * 2))
        
        assert len(q2._changes) == 5
        
        # Flatten
        q3 = q2.rebase()
        assert len(q3._changes) == 0
        assert len(q3.df) == len(q2.df)
        
        # Continue building
        q4 = q3.extend(d=lambda x: x.c / 2)
        assert len(q4._changes) == 1

    def test_data_quality_checks(self):
        """Quality check examples from README."""
        df = pd.DataFrame({
            'email': ['a@example.com', None, 'b@example.com'],
            'status': ['active', 'active', 'inactive'],
            'age': [25, 30, 35]
        })
        q = Q(df)
        
        # Check for nulls
        assert q.count() == 3
        assert q.count('email') == 2
        
        # Unique values
        assert q.nunique('status') == 2
        assert set(q.unique('status')) == {'active', 'inactive'}
        
        # Ranges
        assert q.min('age') == 25
        assert q.max('age') == 35
        assert q.mean('age') == 30.0

    def test_validation_pipeline(self):
        """Validation example from README."""
        df = pd.DataFrame({
            'email': ['alice@example.com', 'test@test.com', 'bob@example.com'],
            'date': ['2024-01-15', '2024-02-20', '2024-30'],  # Bad date
            'amount': [100, -50, 200],  # Negative
            'region': [' East ', 'west', 'NORTH']
        })
        q = Q(df)
        
        clean = (q
            .filter(lambda x: not x.email.endswith('@test.com'))
            .filter(lambda x: len(x.date.split('-')) == 3)
            .filter(lambda x: x.amount > 0)
            .extend(region_clean=lambda x: x.region.strip().upper()))
        
        assert len(clean) == 1  # Only one row passes all filters
        assert clean.df.iloc[0]['email'] == 'alice@example.com'

    def test_export(self, tmp_path):
        """Export example from README."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A'],
            'price': [10, 20, 15],
            'qty': [2, 3, 4]
        })
        q = Q(df)
        
        summary = q.groupby(
            lambda x: x.category,
            items=lambda g: len(g),
            revenue=lambda g: sum(r.price * r.qty for r in g)
        )
        
        output_file = tmp_path / "summary.csv"
        summary.dump(str(output_file))
        
        assert output_file.exists()
        loaded = pd.read_csv(output_file)
        assert len(loaded) == 2

    def test_change_history_inspection(self):
        """Change history example from README."""
        df = pd.DataFrame({
            'price': [100, 200],
            'cost': [60, 120],
            'revenue': [200, 400]
        })
        q = Q(df)
        
        q2 = (q
            .extend(margin=lambda x: (x.price - x.cost) / x.price)
            .filter(lambda x: x.margin > 0.2)
            .extend(profit=lambda x: x.margin * x.revenue)
            .sort('profit')
            .head(50))
        
        change_types = [c[0] for c in q2._changes]
        assert change_types == ['extend', 'filter', 'extend', 'sort', 'head']


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Should handle empty DataFrames."""
        df = pd.DataFrame()
        q = Q(df)
        assert len(q) == 0
        assert str(q) == "Empty DataFrame\nColumns: []\nIndex: []"

    def test_single_row(self):
        """Should handle single row DataFrames."""
        df = pd.DataFrame({'x': [1]})
        q = Q(df)
        assert len(q) == 1
        q2 = q.extend(y=lambda x: x.x * 2)
        assert q2.df.iloc[0]['y'] == 2

    def test_many_columns(self):
        """Should handle DataFrames with many columns."""
        df = pd.DataFrame({f'col{i}': [i] for i in range(100)})
        q = Q(df)
        assert len(q.df.columns) == 100

    def test_filter_removes_all_rows(self):
        """Should handle filter that removes all rows."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        q = Q(df)
        q2 = q.filter(lambda x: x.x > 10)
        assert len(q2) == 0

    def test_head_larger_than_data(self):
        """Head with n larger than data should return all rows."""
        df = pd.DataFrame({'x': [1, 2]})
        q = Q(df)
        q2 = q.head(100)
        assert len(q2) == 2

    def test_transform_to_scalar(self):
        """Transform can return scalar values."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        q = Q(df)
        q2 = q.transform(lambda x: x.x * 2)
        assert list(q2.df.columns) == ['value']
        assert list(q2.df['value']) == [2, 4, 6]

    def test_extend_chaining_references(self):
        """Extended columns can reference previous extensions via chaining."""
        df = pd.DataFrame({'x': [1, 2]})
        q = Q(df)
        q2 = q.extend(y=lambda x: x.x * 2).extend(z=lambda x: x.y + 10)
        assert list(q2.df['z']) == [12, 14]

    def test_sort_empty_cols(self):
        """Sort with no columns should sort by all columns."""
        df = pd.DataFrame({'a': [3, 1, 2], 'b': [6, 4, 5]})
        q = Q(df)
        q2 = q.sort()
        # Sorts by all columns descending
        assert q2.df.iloc[0]['a'] == 3

    def test_to_df_returns_dataframe(self):
        """to_df should return the underlying DataFrame."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        q = Q(df)
        result = q.to_df()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_show_returns_self(self, capsys):
        """show() should return self for chaining."""
        df = pd.DataFrame({'x': [1, 2]})
        q = Q(df)
        result = q.show(1)
        assert result is q
        captured = capsys.readouterr()
        assert 'x' in captured.out

    def test_aggregations_on_filtered_data(self):
        """Aggregations should work on filtered subsets."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        q = Q(df)
        q2 = q.filter(lambda x: x.x > 2)
        assert q2.sum('x') == 12  # 3+4+5
        assert q2.mean('x') == 4.0
        assert q2.count() == 3

    def test_multiple_filters(self):
        """Multiple filters should be ANDed together."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})
        q = Q(df)
        q2 = q.filter(lambda x: x.x > 2).filter(lambda x: x.y > 2)
        assert len(q2) == 1  # Only row with x=3, y=3

    def test_groupby_resets_history(self):
        """groupby should reset change history (terminal operation)."""
        df = pd.DataFrame({'cat': ['A', 'B'], 'val': [1, 2]})
        q = Q(df)
        q2 = q.extend(doubled=lambda x: x.val * 2)
        assert len(q2._changes) == 1
        
        q3 = q2.groupby(lambda x: x.cat, total=lambda g: sum(r.val for r in g))
        assert len(q3._changes) == 0  # History reset

    def test_refresh_with_no_changes(self):
        """Refresh with no changes should return equivalent object."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        q = Q(df)
        q2 = q.refresh()
        assert len(q2.df) == len(q.df)
        assert list(q2.df['x']) == list(q.df['x'])

    def test_reload_adds_new_columns(self, tmp_path):
        """Reload should allow new columns in source."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,2\n")
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        
        # Add a new column to file
        csv_file.write_text("x,y,z\n1,2,3\n")
        
        q2 = q.reload()
        assert 'z' in q2.df.columns

    def test_std_and_var(self):
        """Test standard deviation and variance."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        q = Q(df)
        std = q.std('x')
        var = q.var('x')
        assert abs(std - 1.58113883) < 0.001
        assert abs(var - 2.5) < 0.001

