"""
Configuration module for the Market Fundamentals Project.

This module defines all project-wide settings including paths, random seeds,
train/test split dates, and utility methods for directory management.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List


@dataclass
class Settings:
    """
    Central configuration class for the Market Fundamentals Project.
    
    Attributes
    ----------
    PROJECT_ROOT : Path
        Root directory of the project.
    DATA_DIR : Path
        Directory containing raw data files.
    RESULTS_DIR : Path
        Directory for storing step outputs and results.
    NOTEBOOKS_DIR : Path
        Directory for Jupyter notebooks.
    TESTS_DIR : Path
        Directory for unit tests.
    REPORT_DIR : Path
        Directory for final reports and visualizations.
    EXPERIMENTS_DIR : Path
        Directory for experimental runs and hyperparameter tuning.
    RANDOM_SEED : int
        Random seed for reproducibility across all steps.
    TRAIN_TEST_SPLIT_DATE : str
        Cutoff date (YYYY-MM-DD) for train/test split.
    SPY_TICKER : str
        Ticker symbol for the benchmark (S&P 500 ETF).
    EARNINGS_TICKER_COLUMN : str
        Column name in earnings dataset containing ticker symbols.
    EARNINGS_DATE_COLUMN : str
        Column name in earnings dataset containing earnings announcement dates.
    MAX_TICKERS_DAILY_DOWNLOAD : int
        Maximum number of tickers to download daily data for (for development).
    YFINANCE_CACHE_DIR : Path
        Directory for caching yfinance downloads.
    LOG_LEVEL : int
        Logging level for the project.
    LOG_FORMAT : str
        Format string for log messages.
    """
    
    # Directory paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    TESTS_DIR: Path = PROJECT_ROOT / "tests"
    REPORT_DIR: Path = PROJECT_ROOT / "report"
    EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
    YFINANCE_CACHE_DIR: Path = DATA_DIR / "cache" / "yfinance"
    
    # Model and data settings
    RANDOM_SEED: int = 42
    TRAIN_TEST_SPLIT_DATE: str = "2020-01-01"
    SPY_TICKER: str = "SPY"
    
    # Earnings data column names (configurable - adjust based on actual data)
    EARNINGS_TICKER_COLUMN: str = "ticker"  # Ticker symbol column
    EARNINGS_DATE_COLUMN: str = "earnings_date"  # Earnings announcement date column
    
    # Data download limits
    MAX_TICKERS_DAILY_DOWNLOAD: int = 450  # Increased to cover all 411 companies with tickers
    
    # Logging settings
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def ensure_directories(self) -> None:
        """
        Create all required project directories if they don't exist.
        
        This method ensures that the complete folder structure is in place
        before any pipeline steps are executed.
        
        Returns
        -------
        None
        """
        directories: List[Path] = [
            self.DATA_DIR,
            self.RESULTS_DIR,
            self.NOTEBOOKS_DIR,
            self.TESTS_DIR,
            self.REPORT_DIR,
            self.EXPERIMENTS_DIR,
            self.YFINANCE_CACHE_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self, step_name: str = "pipeline") -> logging.Logger:
        """
        Configure and return a logger for a specific step.
        
        Parameters
        ----------
        step_name : str, optional
            Name of the pipeline step (default: "pipeline").
            
        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger(step_name)
        logger.setLevel(self.LOG_LEVEL)
        
        # Remove existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.LOG_LEVEL)
        formatter = logging.Formatter(self.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_step_results_dir(self, step_number: int) -> Path:
        """
        Get the results directory for a specific step.
        
        Parameters
        ----------
        step_number : int
            Step number (1-20).
            
        Returns
        -------
        Path
            Path to the step's results directory.
        """
        step_dir = self.RESULTS_DIR / f"step_{step_number:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir


# Global settings instance
settings = Settings()
