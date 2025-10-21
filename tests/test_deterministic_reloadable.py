"""Comprehensive tests for deterministic and reloadable flags."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from bend.core import Q, load_csv


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
        
        df = load_csv(str(csv_file))
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
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.assign(y=lambda x: x.x * 2).filter(lambda x: x.y > 2)
        assert q2.reloadable
    
    def test_rebase_sets_reloadable_false(self, tmp_path):
        """Rebase should set reloadable to False."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")
        
        df = load_csv(str(csv_file))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df = load_csv(str(csv_file))
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
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        assert q.reloadable
        
        q2 = q.filter(lambda x: x.x > 1).rebase()
        assert not q2.reloadable
    
    def test_rebase_keeps_source_path(self, tmp_path):
        """Rebase should keep source_path but Q is not reloadable."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")
        
        df = load_csv(str(csv_file))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
        q1 = Q(df1, source_path=str(csv1))
        q2 = Q(df2, source_path=str(csv2))
        
        # Both deterministic and reloadable
        q3 = q1.merge(q2, on='id')
        assert q3.deterministic
        assert q3.reloadable
        
        # One non-deterministic
        q2_sample = q2.sample(1)
        q4 = q1.merge(q2_sample, on='id')
        assert not q4.deterministic
        assert q4.reloadable  # Still reloadable
        
        # One non-reloadable
        q2_rebased = q2.rebase()
        q5 = q1.merge(q2_rebased, on='id')
        assert q5.deterministic  # Both were deterministic before rebase
        assert not q5.reloadable
    
    def test_union_propagates_flags(self, tmp_path):
        """Union should propagate flags through concat."""
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text("x\n1\n2\n")
        csv2.write_text("x\n2\n3\n")
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df1 = load_csv(str(csv1))
        df2 = load_csv(str(csv2))
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
        
        df = load_csv(str(csv_file))
        q = Q(df, source_path=str(csv_file))
        q2 = q.assign(y=lambda x: x.x * 2)
        
        # Update file
        csv_file.write_text("x\n10\n20\n30\n")
        
        # Reload gets new data
        q3 = q2.reload()
        assert list(q3.to_df()["x"]) == [10, 20, 30]
        assert list(q3.to_df()["y"]) == [20, 40, 60]

