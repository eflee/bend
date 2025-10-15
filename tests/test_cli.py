"""Tests for bend.cli module."""

import sys
from unittest.mock import patch
import pytest


class TestCLI:
    """Tests for CLI main function."""

    def test_no_arguments_exits(self):
        """Should exit with error when no CSV file provided."""
        with patch.object(sys, 'argv', ['bend']):
            with pytest.raises(SystemExit) as exc_info:
                from bend.cli import main
                main()
            assert exc_info.value.code != 0

    def test_help_argument(self):
        """Should display help message."""
        with patch.object(sys, 'argv', ['bend', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                from bend.cli import main
                main()
            assert exc_info.value.code == 0

    @patch('bend.cli.load_csv')
    @patch('IPython.start_ipython')
    def test_with_csv_file_launches_ipython(self, mock_ipython, mock_load):
        """Should load CSV and launch IPython when file provided."""
        import pandas as pd
        
        mock_df = pd.DataFrame({"x": [1, 2, 3]})
        mock_load.return_value = mock_df
        
        with patch.object(sys, 'argv', ['bend', 'test.csv']):
            from bend.cli import main
            main()
        
        mock_load.assert_called_once_with('test.csv', skip_rows=0)
        mock_ipython.assert_called_once()

    @patch('bend.cli.load_csv')
    @patch('IPython.start_ipython')
    def test_with_skip_rows_argument(self, mock_ipython, mock_load):
        """Should pass skip_rows argument to load_csv."""
        import pandas as pd
        
        mock_df = pd.DataFrame({"x": [1, 2, 3]})
        mock_load.return_value = mock_df
        
        with patch.object(sys, 'argv', ['bend', 'test.csv', '--skip-rows', '3']):
            from bend.cli import main
            main()
        
        mock_load.assert_called_once_with('test.csv', skip_rows=3)
        mock_ipython.assert_called_once()
