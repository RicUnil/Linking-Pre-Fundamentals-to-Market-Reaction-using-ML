"""
Tests for Step 04: Load SPY/benchmark data.

This module contains smoke tests and basic validation tests for the benchmark
data loading functionality in Step 04.
"""

import pytest
from pathlib import Path

from src.config import settings
from src.data.load_data import load_benchmark_data
from src.step_04_load_spy_benchmark import run_step_04


def test_load_benchmark_data():
    """
    Test that BENCHMARK.csv can be loaded successfully.
    
    This test verifies that:
    - The file exists and can be loaded
    - The resulting DataFrame is not empty
    - The DataFrame has columns
    """
    df = load_benchmark_data(settings)
    
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should have rows"
    assert len(df.columns) > 0, "DataFrame should have columns"


def test_benchmark_has_data_columns():
    """
    Test that the benchmark data contains columns beyond just the identifier.
    
    The benchmark should contain data columns (which may be numeric or string
    format depending on locale/parsing).
    """
    df = load_benchmark_data(settings)
    
    # Check that we have more than just one column (identifier column)
    assert len(df.columns) > 1, "Benchmark should have multiple columns (identifier + data)"
    
    # Check that at least some columns have non-null values
    non_null_cols = [col for col in df.columns if df[col].notna().any()]
    assert len(non_null_cols) > 0, "Benchmark should have columns with data"


def test_run_step_04_executes():
    """
    Smoke test for Step 04: ensure it runs without raising an exception.
    
    This test verifies that:
    - run_step_04() executes without raising exceptions
    - The expected output files are created
    - The completion marker exists
    
    This assumes that data/BENCHMARK.csv is present in the default data directory.
    """
    # Run the step
    run_step_04()
    
    # Check that output files were created
    step_results_dir = settings.get_step_results_dir(4)
    
    parquet_file = step_results_dir / "benchmark_raw.parquet"
    sample_file = step_results_dir / "benchmark_head.csv"
    completion_file = step_results_dir / "step_04_completed.txt"
    
    assert parquet_file.exists(), f"Parquet file should exist at {parquet_file}"
    assert sample_file.exists(), f"Sample CSV should exist at {sample_file}"
    assert completion_file.exists(), f"Completion marker should exist at {completion_file}"
    
    # Verify parquet file is not empty
    assert parquet_file.stat().st_size > 0, "Parquet file should not be empty"


def test_benchmark_file_missing_raises_error():
    """
    Test that attempting to load a non-existent benchmark file raises FileNotFoundError.
    """
    from src.config import Settings
    from pathlib import Path
    
    # Create a settings object with a non-existent data directory
    fake_settings = Settings()
    fake_settings.DATA_DIR = Path("/nonexistent/path")
    
    with pytest.raises(FileNotFoundError):
        load_benchmark_data(fake_settings)
