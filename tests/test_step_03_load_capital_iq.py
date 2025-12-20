"""
Tests for Step 03: Load Capital IQ earnings data.

This module contains smoke tests and basic validation tests for the data
loading functionality in Step 03.
"""

import pytest
from pathlib import Path

from src.config import settings
from src.data.load_data import (
    load_raw_earnings_data,
    load_quarter_file,
    load_all_quarter_files,
)
from src.step_03_load_capital_iq import run_step_03


def test_load_raw_earnings_data():
    """
    Test that RAW_DATA.csv can be loaded successfully.
    
    This test verifies that:
    - The file exists and can be loaded
    - The resulting DataFrame is not empty
    - The DataFrame has columns
    """
    df = load_raw_earnings_data(settings)
    
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should have rows"
    assert len(df.columns) > 0, "DataFrame should have columns"


def test_load_quarter_file():
    """
    Test that individual quarterly files can be loaded.
    
    This test verifies that Quarter_1.csv can be loaded successfully.
    """
    df = load_quarter_file(settings, 1)
    
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should have rows"
    assert len(df.columns) > 0, "DataFrame should have columns"


def test_load_quarter_file_invalid_quarter():
    """
    Test that invalid quarter numbers raise ValueError.
    """
    with pytest.raises(ValueError):
        load_quarter_file(settings, 5)
    
    with pytest.raises(ValueError):
        load_quarter_file(settings, 0)


def test_load_all_quarter_files():
    """
    Test that all quarterly files can be loaded at once.
    
    This test verifies that:
    - All four quarterly files can be loaded
    - Each DataFrame is not empty
    - The function returns exactly 4 DataFrames
    """
    q1, q2, q3, q4 = load_all_quarter_files(settings)
    
    # Check that all DataFrames are returned
    assert q1 is not None, "Q1 DataFrame should not be None"
    assert q2 is not None, "Q2 DataFrame should not be None"
    assert q3 is not None, "Q3 DataFrame should not be None"
    assert q4 is not None, "Q4 DataFrame should not be None"
    
    # Check that all DataFrames have data
    assert len(q1) > 0, "Q1 DataFrame should have rows"
    assert len(q2) > 0, "Q2 DataFrame should have rows"
    assert len(q3) > 0, "Q3 DataFrame should have rows"
    assert len(q4) > 0, "Q4 DataFrame should have rows"


def test_run_step_03_executes():
    """
    Simple smoke test: run Step 03 and check that it creates expected output files.
    
    This test verifies that:
    - run_step_03() executes without raising exceptions
    - The expected output files are created
    - The completion marker exists
    """
    # Run the step
    run_step_03()
    
    # Check that output files were created
    step_results_dir = settings.get_step_results_dir(3)
    
    parquet_file = step_results_dir / "earnings_raw.parquet"
    sample_file = step_results_dir / "earnings_head.csv"
    completion_file = step_results_dir / "step_03_completed.txt"
    
    assert parquet_file.exists(), f"Parquet file should exist at {parquet_file}"
    assert sample_file.exists(), f"Sample CSV should exist at {sample_file}"
    assert completion_file.exists(), f"Completion marker should exist at {completion_file}"
    
    # Verify parquet file is not empty
    assert parquet_file.stat().st_size > 0, "Parquet file should not be empty"
