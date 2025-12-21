"""
Experiment 01: 10-Day Excess Returns Prediction

This experiment tests whether shorter-term post-earnings returns (10-day)
are more predictable than the main project's 30-day horizon.

Key principles:
- Reuses cleaned features from Step 10 (no modification of main pipeline)
- Computes new 10-day excess return target
- Trains lightweight models: Baseline, Ridge, Random Forest
- Saves outputs to results/experiments/returns_10d/

Usage:
    python src/experiments/experiment_01_returns_10d.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def setup_logging() -> logging.Logger:
    """Set up logging for the experiment."""
    logger = logging.getLogger("experiment_01_returns_10d")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_data(logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    Load cleaned feature matrices and metadata from Step 10.
    
    Returns
    -------
    Dict containing:
        - X_train, X_val, X_test: feature matrices
        - cleaned_train, cleaned_val, cleaned_test: metadata DataFrames
        - scaler: fitted StandardScaler
        - imputer: fitted SimpleImputer
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM STEP 10")
    logger.info("=" * 70)
    
    step_10_dir = settings.get_step_results_dir(10)
    step_11_dir = settings.get_step_results_dir(11)
    
    # Load feature matrices
    logger.info("\nLoading feature matrices...")
    X_train = np.load(step_10_dir / "X_train.npy")
    X_val = np.load(step_10_dir / "X_val.npy")
    X_test = np.load(step_10_dir / "X_test.npy")
    
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}")
    logger.info(f"  X_test:  {X_test.shape}")
    
    # Load metadata (contains ticker, earnings_date, original targets)
    logger.info("\nLoading metadata DataFrames...")
    cleaned_train = pd.read_parquet(step_10_dir / "cleaned_train.parquet")
    cleaned_val = pd.read_parquet(step_10_dir / "cleaned_val.parquet")
    cleaned_test = pd.read_parquet(step_10_dir / "cleaned_test.parquet")
    
    logger.info(f"  cleaned_train: {cleaned_train.shape}")
    logger.info(f"  cleaned_val:   {cleaned_val.shape}")
    logger.info(f"  cleaned_test:  {cleaned_test.shape}")
    
    # Load scaler
    logger.info("\nLoading scaler...")
    scaler = joblib.load(step_10_dir / "scaler.joblib")
    logger.info("  ✓ Scaler loaded")
    
    # Load imputer from Step 11
    logger.info("\nLoading imputer from Step 11...")
    imputer_path = step_11_dir / "feature_imputer.joblib"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        logger.info("  ✓ Imputer loaded")
    else:
        logger.warning("  ⚠ Imputer not found, will handle NaNs manually")
        imputer = None
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "cleaned_train": cleaned_train,
        "cleaned_val": cleaned_val,
        "cleaned_test": cleaned_test,
        "scaler": scaler,
        "imputer": imputer,
    }


def build_10d_target(
    df: pd.DataFrame,
    daily_data_dir: Path,
    logger: logging.Logger
) -> pd.Series:
    """
    Compute 10-day excess return for each earnings event.
    
    excess_return_10d = (stock return day 1→10) - (SPY return day 1→10)
    
    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame with 'ticker' and 'earnings_date' columns
    daily_data_dir : Path
        Directory containing daily price data from Step 05
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.Series
        10-day excess returns aligned with df index
    """
    logger.info("\nComputing 10-day excess returns...")
    
    # Check for consolidated daily data
    consolidated_path = daily_data_dir / "cache" / "all_daily_data.parquet"
    
    if consolidated_path.exists():
        logger.info(f"  Using consolidated daily data: {consolidated_path}")
        daily_df = pd.read_parquet(consolidated_path)
    else:
        logger.info("  Consolidated data not found, loading individual files...")
        daily_files = list(daily_data_dir.glob("*_daily.parquet"))
        if not daily_files:
            raise FileNotFoundError(
                f"No daily data files found in {daily_data_dir}"
            )
        
        # Load each file and add ticker column
        dfs = []
        for f in daily_files:
            df_temp = pd.read_parquet(f)
            # Extract ticker from filename (e.g., "AAPL_daily.parquet" -> "AAPL")
            ticker = f.stem.replace('_daily', '')
            df_temp['ticker'] = ticker
            dfs.append(df_temp)
        
        # Concatenate without ignore_index to preserve date index
        daily_df = pd.concat(dfs, ignore_index=False)
    
    # Find the date column - check multiple possible names and locations
    date_col = None
    
    # First, check if index is datetime (most common case)
    if pd.api.types.is_datetime64_any_dtype(daily_df.index):
        logger.info(f"  Date found in index (type: {type(daily_df.index).__name__})")
        daily_df = daily_df.reset_index()
        # After reset_index, the date column will be the first column or named 'index'
        if daily_df.index.name in ['Date', 'date', 'datetime', 'timestamp']:
            date_col = daily_df.columns[0]  # First column after reset
        else:
            date_col = daily_df.columns[0]  # Default to first column
        logger.info(f"  Using date column: '{date_col}'")
    # Check if date is in a named index
    elif daily_df.index.name in ['Date', 'date', 'datetime', 'timestamp']:
        daily_df = daily_df.reset_index()
        date_col = daily_df.columns[0]
    elif 'Date' in daily_df.index.names:
        daily_df = daily_df.reset_index()
        date_col = 'Date'
    # Check if date is already a column
    elif 'Date' in daily_df.columns:
        date_col = 'Date'
    elif 'date' in daily_df.columns:
        date_col = 'date'
    elif 'datetime' in daily_df.columns:
        date_col = 'datetime'
    elif 'timestamp' in daily_df.columns:
        date_col = 'timestamp'
    else:
        # Print available columns for debugging
        logger.error(f"Available columns: {daily_df.columns.tolist()}")
        logger.error(f"Index name: {daily_df.index.name}")
        logger.error(f"Index type: {type(daily_df.index)}")
        logger.error(f"First few rows:\n{daily_df.head()}")
        raise ValueError(
            f"Cannot find date column in daily data. "
            f"Available columns: {daily_df.columns.tolist()}, "
            f"Index: {daily_df.index.name}"
        )
    
    # Ensure date column is datetime
    daily_df[date_col] = pd.to_datetime(daily_df[date_col])
    
    # Use adjusted close if available, otherwise close
    price_col = 'adj_close' if 'adj_close' in daily_df.columns else 'close'
    logger.info(f"  Using price column: {price_col}")
    
    # Create a pivot table for fast lookups: date x ticker -> price
    price_pivot = daily_df.pivot_table(
        index=date_col,
        columns='ticker',
        values=price_col,
        aggfunc='first'
    )
    
    # Compute 10-day returns for each event
    excess_returns = []
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        earnings_date = pd.to_datetime(row['earnings_date'])
        
        # Define window: day 1 to day 10 after earnings
        start_date = earnings_date + pd.Timedelta(days=1)
        end_date = earnings_date + pd.Timedelta(days=10)
        
        # Get prices
        try:
            # Stock prices
            stock_prices = price_pivot.loc[
                (price_pivot.index >= start_date) & (price_pivot.index <= end_date),
                ticker
            ]
            
            # SPY prices
            spy_prices = price_pivot.loc[
                (price_pivot.index >= start_date) & (price_pivot.index <= end_date),
                settings.SPY_TICKER
            ]
            
            # Need at least 2 prices to compute return
            if len(stock_prices) < 2 or len(spy_prices) < 2:
                excess_returns.append(np.nan)
                continue
            
            # Compute cumulative returns
            stock_return = (stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1
            spy_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1
            
            excess_return = stock_return - spy_return
            excess_returns.append(excess_return)
            
        except (KeyError, IndexError):
            # Ticker or date not found
            excess_returns.append(np.nan)
    
    result = pd.Series(excess_returns, index=df.index, name='excess_return_10d')
    
    # Log statistics
    valid_count = result.notna().sum()
    logger.info(f"  ✓ Computed {valid_count:,} / {len(result):,} valid 10-day returns")
    logger.info(f"  Missing: {result.isna().sum():,}")
    logger.info(f"  Mean: {result.mean():.6f}")
    logger.info(f"  Std:  {result.std():.6f}")
    
    return result


def align_targets(
    data: Dict,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 10-day targets for train/val/test splits and remove NaN rows.
    
    Parameters
    ----------
    data : Dict
        Dictionary containing cleaned DataFrames
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    Tuple of (y_train_10d, y_val_10d, y_test_10d, updated data dict)
    """
    logger.info("=" * 70)
    logger.info("BUILDING 10-DAY TARGETS")
    logger.info("=" * 70)
    
    daily_data_dir = settings.get_step_results_dir(5)
    
    # Build targets for each split
    y_train_10d = build_10d_target(data['cleaned_train'], daily_data_dir, logger)
    y_val_10d = build_10d_target(data['cleaned_val'], daily_data_dir, logger)
    y_test_10d = build_10d_target(data['cleaned_test'], daily_data_dir, logger)
    
    # Remove NaN targets
    logger.info("\n" + "=" * 70)
    logger.info("REMOVING NaN TARGETS")
    logger.info("=" * 70)
    
    # Train
    train_valid = y_train_10d.notna()
    logger.info(f"\nTrain: Removing {(~train_valid).sum():,} NaN rows")
    data['X_train'] = data['X_train'][train_valid]
    data['cleaned_train'] = data['cleaned_train'][train_valid]
    y_train_10d = y_train_10d[train_valid].values
    logger.info(f"  New train size: {len(y_train_10d):,}")
    
    # Validation
    val_valid = y_val_10d.notna()
    logger.info(f"\nValidation: Removing {(~val_valid).sum():,} NaN rows")
    data['X_val'] = data['X_val'][val_valid]
    data['cleaned_val'] = data['cleaned_val'][val_valid]
    y_val_10d = y_val_10d[val_valid].values
    logger.info(f"  New val size: {len(y_val_10d):,}")
    
    # Test
    test_valid = y_test_10d.notna()
    logger.info(f"\nTest: Removing {(~test_valid).sum():,} NaN rows")
    data['X_test'] = data['X_test'][test_valid]
    data['cleaned_test'] = data['cleaned_test'][test_valid]
    y_test_10d = y_test_10d[test_valid].values
    logger.info(f"  New test size: {len(y_test_10d):,}")
    
    # Apply imputer if available
    if data['imputer'] is not None:
        logger.info("\n" + "=" * 70)
        logger.info("APPLYING FEATURE IMPUTATION")
        logger.info("=" * 70)
        
        logger.info("\nImputing missing feature values...")
        data['X_train'] = data['imputer'].transform(data['X_train'])
        data['X_val'] = data['imputer'].transform(data['X_val'])
        data['X_test'] = data['imputer'].transform(data['X_test'])
        logger.info("  ✓ Imputation complete")
    else:
        # Fallback: simple median imputation
        logger.info("\n⚠ Using fallback median imputation")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        data['X_train'] = imputer.fit_transform(data['X_train'])
        data['X_val'] = imputer.transform(data['X_val'])
        data['X_test'] = imputer.transform(data['X_test'])
    
    return y_train_10d, y_val_10d, y_test_10d, data


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    logger: logging.Logger
) -> Dict:
    """
    Train baseline, Ridge, and Random Forest models.
    
    Returns
    -------
    Dict of trained models
    """
    logger.info("=" * 70)
    logger.info("TRAINING MODELS")
    logger.info("=" * 70)
    
    models = {}
    
    # 1. Baseline: Mean predictor
    logger.info("\n1. Training Baseline (Mean Predictor)...")
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    models['baseline_mean'] = baseline
    logger.info("  ✓ Baseline trained")
    
    # 2. Ridge Regression
    logger.info("\n2. Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=settings.RANDOM_SEED)
    ridge.fit(X_train, y_train)
    models['ridge'] = ridge
    logger.info("  ✓ Ridge trained")
    
    # 3. Random Forest
    logger.info("\n3. Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    logger.info("  ✓ Random Forest trained")
    
    return models


def evaluate_models(
    models: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger
) -> Dict:
    """
    Evaluate all models on train/val/test sets.
    
    Returns
    -------
    Dict of metrics and predictions
    """
    logger.info("=" * 70)
    logger.info("EVALUATING MODELS")
    logger.info("=" * 70)
    
    metrics = {}
    predictions = {}
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Store predictions
        predictions[f"{model_name}_train"] = y_train_pred
        predictions[f"{model_name}_val"] = y_val_pred
        predictions[f"{model_name}_test"] = y_test_pred
        
        # Compute metrics
        metrics[model_name] = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_r2': r2_score(y_val, y_val_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred),
        }
        
        # Log test metrics
        logger.info(f"  Test MAE:  {metrics[model_name]['test_mae']:.6f}")
        logger.info(f"  Test RMSE: {metrics[model_name]['test_rmse']:.6f}")
        logger.info(f"  Test R²:   {metrics[model_name]['test_r2']:.6f}")
    
    return metrics, predictions


def save_outputs(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    metrics: Dict,
    predictions: Dict,
    cleaned_test: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """
    Save all experiment outputs.
    """
    logger.info("=" * 70)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 70)
    
    output_dir = PROJECT_ROOT / "results" / "experiments" / "returns_10d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save targets
    logger.info("\nSaving targets...")
    np.save(output_dir / "y_train_10d.npy", y_train)
    np.save(output_dir / "y_val_10d.npy", y_val)
    np.save(output_dir / "y_test_10d.npy", y_test)
    logger.info("  ✓ Targets saved")
    
    # Save metrics as JSON
    logger.info("\nSaving metrics (JSON)...")
    import json
    with open(output_dir / "regression_metrics_10d.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("  ✓ Metrics JSON saved")
    
    # Save metrics as CSV
    logger.info("\nSaving metrics (CSV)...")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'model'
    metrics_df.to_csv(output_dir / "regression_metrics_10d.csv")
    logger.info("  ✓ Metrics CSV saved")
    
    # Save predictions
    logger.info("\nSaving predictions...")
    for pred_name, pred_values in predictions.items():
        np.save(output_dir / f"predictions_10d_{pred_name}.npy", pred_values)
    logger.info(f"  ✓ {len(predictions)} prediction files saved")
    
    # Save Ridge residuals for analysis
    logger.info("\nSaving Ridge residuals...")
    ridge_test_pred = predictions['ridge_test']
    residuals_df = cleaned_test.copy()
    residuals_df['y_true'] = y_test
    residuals_df['y_pred'] = ridge_test_pred
    residuals_df['residual'] = y_test - ridge_test_pred
    residuals_df.to_parquet(output_dir / "residuals_10d_ridge.parquet")
    logger.info("  ✓ Residuals saved")
    
    # Save completion marker
    logger.info("\nSaving completion marker...")
    with open(output_dir / "step_experiment_01_completed.txt", 'w') as f:
        f.write("Experiment 01: 10-day excess returns completed successfully.\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Val samples: {len(y_val)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
    logger.info("  ✓ Completion marker saved")
    
    logger.info(f"\n✓ All outputs saved to: {output_dir}")


def main():
    """Main execution function."""
    logger = setup_logging()
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 01: 10-DAY EXCESS RETURNS")
    logger.info("=" * 70)
    logger.info("\nObjective: Test if 10-day returns are more predictable than 30-day")
    logger.info("Strategy: Reuse Step 10 features, compute new 10-day target")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load data
        data = load_data(logger)
        
        # Step 2: Build and align 10-day targets
        y_train_10d, y_val_10d, y_test_10d, data = align_targets(data, logger)
        
        # Step 3: Train models
        models = train_models(
            data['X_train'], y_train_10d,
            data['X_val'], y_val_10d,
            logger
        )
        
        # Step 4: Evaluate models
        metrics, predictions = evaluate_models(
            models,
            data['X_train'], y_train_10d,
            data['X_val'], y_val_10d,
            data['X_test'], y_test_10d,
            logger
        )
        
        # Step 5: Save outputs
        save_outputs(
            y_train_10d, y_val_10d, y_test_10d,
            metrics, predictions,
            data['cleaned_test'],
            logger
        )
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 01 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\nTest Set Results:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  R² = {model_metrics['test_r2']:.6f}")
            logger.info(f"  MAE = {model_metrics['test_mae']:.6f}")
            logger.info(f"  RMSE = {model_metrics['test_rmse']:.6f}")
        
        logger.info("\n" + "=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
