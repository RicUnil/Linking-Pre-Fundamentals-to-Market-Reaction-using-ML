"""
Experiment 03: Day 0 (Announcement Day) Market Reaction

This experiment tests whether fundamentals can predict the IMMEDIATE market reaction
to earnings announcements (Day 0 return).

Research Question: Do fundamentals correlate with announcement day returns?

Usage:
    python3 experiments_03/src/experiment_03_day0_reaction.py
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("experiment_03_day0_reaction")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_data(logger: logging.Logger) -> Dict:
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM STEP 10")
    logger.info("=" * 70)
    
    step_10_dir = settings.get_step_results_dir(10)
    step_11_dir = settings.get_step_results_dir(11)
    
    logger.info("\nLoading feature matrices...")
    X_train = np.load(step_10_dir / "X_train.npy")
    X_val = np.load(step_10_dir / "X_val.npy")
    X_test = np.load(step_10_dir / "X_test.npy")
    
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}")
    logger.info(f"  X_test:  {X_test.shape}")
    
    logger.info("\nLoading metadata DataFrames...")
    cleaned_train = pd.read_parquet(step_10_dir / "cleaned_train.parquet")
    cleaned_val = pd.read_parquet(step_10_dir / "cleaned_val.parquet")
    cleaned_test = pd.read_parquet(step_10_dir / "cleaned_test.parquet")
    
    logger.info(f"  cleaned_train: {cleaned_train.shape}")
    logger.info(f"  cleaned_val:   {cleaned_val.shape}")
    logger.info(f"  cleaned_test:  {cleaned_test.shape}")
    
    logger.info("\nLoading scaler...")
    scaler = joblib.load(step_10_dir / "scaler.joblib")
    logger.info("  ✓ Scaler loaded")
    
    logger.info("\nLoading imputer from Step 11...")
    imputer_path = step_11_dir / "feature_imputer.joblib"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        logger.info("  ✓ Imputer loaded")
    else:
        logger.warning("  ⚠ Imputer not found")
        imputer = None
    
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "cleaned_train": cleaned_train, "cleaned_val": cleaned_val, "cleaned_test": cleaned_test,
        "scaler": scaler, "imputer": imputer
    }


def build_day0_target(df: pd.DataFrame, daily_data_dir: Path, logger: logging.Logger) -> pd.Series:
    logger.info("\nComputing Day 0 (announcement day) excess returns...")
    
    consolidated_path = daily_data_dir / "cache" / "all_daily_data.parquet"
    
    if consolidated_path.exists():
        logger.info(f"  Using consolidated daily data: {consolidated_path}")
        daily_df = pd.read_parquet(consolidated_path)
    else:
        logger.info("  Loading individual files...")
        daily_files = list(daily_data_dir.glob("*_daily.parquet"))
        if not daily_files:
            raise FileNotFoundError(f"No daily data files found in {daily_data_dir}")
        # Load each file and add ticker column
        dfs = []
        for f in daily_files:
            df_temp = pd.read_parquet(f)
            ticker = f.stem.replace('_daily', '')
            df_temp['ticker'] = ticker
            dfs.append(df_temp)
        daily_df = pd.concat(dfs, ignore_index=False)
    
    # Find the date column
    date_col = None
    if pd.api.types.is_datetime64_any_dtype(daily_df.index):
        daily_df = daily_df.reset_index()
        date_col = daily_df.columns[0]
    elif daily_df.index.name in ['Date', 'date', 'datetime', 'timestamp']:
        daily_df = daily_df.reset_index()
        date_col = daily_df.columns[0]
    elif 'Date' in daily_df.columns:
        date_col = 'Date'
    elif 'date' in daily_df.columns:
        date_col = 'date'
    else:
        raise ValueError(f"Cannot find date column. Columns: {daily_df.columns.tolist()}, Index: {daily_df.index.name}")
    
    daily_df[date_col] = pd.to_datetime(daily_df[date_col])
    
    price_col = 'adj_close' if 'adj_close' in daily_df.columns else 'close'
    logger.info(f"  Using price column: {price_col}")
    
    has_open = 'open' in daily_df.columns
    
    if has_open:
        logger.info("  Computing intraday returns (open to close)")
        open_pivot = daily_df.pivot_table(index=date_col, columns='ticker', values='open', aggfunc='first')
    else:
        logger.info("  Computing close-to-close returns")
    
    close_pivot = daily_df.pivot_table(index=date_col, columns='ticker', values=price_col, aggfunc='first')
    
    excess_returns = []
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        earnings_date = pd.to_datetime(row['earnings_date'])
        
        try:
            if has_open:
                stock_open = open_pivot.loc[earnings_date, ticker]
                stock_close = close_pivot.loc[earnings_date, ticker]
                spy_open = open_pivot.loc[earnings_date, settings.SPY_TICKER]
                spy_close = close_pivot.loc[earnings_date, settings.SPY_TICKER]
                
                if pd.isna(stock_open) or pd.isna(stock_close) or pd.isna(spy_open) or pd.isna(spy_close):
                    excess_returns.append(np.nan)
                    continue
                
                stock_return = (stock_close - stock_open) / stock_open
                spy_return = (spy_close - spy_open) / spy_open
            else:
                prev_date = earnings_date - pd.Timedelta(days=1)
                
                stock_prev = close_pivot.loc[prev_date, ticker] if prev_date in close_pivot.index else np.nan
                stock_curr = close_pivot.loc[earnings_date, ticker] if earnings_date in close_pivot.index else np.nan
                spy_prev = close_pivot.loc[prev_date, settings.SPY_TICKER] if prev_date in close_pivot.index else np.nan
                spy_curr = close_pivot.loc[earnings_date, settings.SPY_TICKER] if earnings_date in close_pivot.index else np.nan
                
                if pd.isna(stock_prev) or pd.isna(stock_curr) or pd.isna(spy_prev) or pd.isna(spy_curr):
                    excess_returns.append(np.nan)
                    continue
                
                stock_return = (stock_curr - stock_prev) / stock_prev
                spy_return = (spy_curr - spy_prev) / spy_prev
            
            excess_return = stock_return - spy_return
            excess_returns.append(excess_return)
            
        except (KeyError, IndexError):
            excess_returns.append(np.nan)
    
    result = pd.Series(excess_returns, index=df.index, name='excess_return_day0')
    
    valid_count = result.notna().sum()
    logger.info(f"  ✓ Computed {valid_count:,} / {len(result):,} valid Day 0 returns")
    logger.info(f"  Missing: {result.isna().sum():,}")
    logger.info(f"  Mean: {result.mean():.6f}")
    logger.info(f"  Std:  {result.std():.6f}")
    
    return result


def align_targets(data: Dict, logger: logging.Logger) -> Tuple:
    logger.info("=" * 70)
    logger.info("BUILDING DAY 0 TARGETS")
    logger.info("=" * 70)
    
    daily_data_dir = settings.get_step_results_dir(5)
    
    y_train_day0 = build_day0_target(data['cleaned_train'], daily_data_dir, logger)
    y_val_day0 = build_day0_target(data['cleaned_val'], daily_data_dir, logger)
    y_test_day0 = build_day0_target(data['cleaned_test'], daily_data_dir, logger)
    
    logger.info("\n" + "=" * 70)
    logger.info("REMOVING NaN TARGETS")
    logger.info("=" * 70)
    
    train_valid = y_train_day0.notna()
    logger.info(f"\nTrain: Removing {(~train_valid).sum():,} NaN rows")
    data['X_train'] = data['X_train'][train_valid]
    data['cleaned_train'] = data['cleaned_train'][train_valid]
    y_train_day0 = y_train_day0[train_valid].values
    logger.info(f"  New train size: {len(y_train_day0):,}")
    
    val_valid = y_val_day0.notna()
    logger.info(f"\nValidation: Removing {(~val_valid).sum():,} NaN rows")
    data['X_val'] = data['X_val'][val_valid]
    data['cleaned_val'] = data['cleaned_val'][val_valid]
    y_val_day0 = y_val_day0[val_valid].values
    logger.info(f"  New val size: {len(y_val_day0):,}")
    
    test_valid = y_test_day0.notna()
    logger.info(f"\nTest: Removing {(~test_valid).sum():,} NaN rows")
    data['X_test'] = data['X_test'][test_valid]
    data['cleaned_test'] = data['cleaned_test'][test_valid]
    y_test_day0 = y_test_day0[test_valid].values
    logger.info(f"  New test size: {len(y_test_day0):,}")
    
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
        logger.info("\n⚠ Using fallback median imputation")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        data['X_train'] = imputer.fit_transform(data['X_train'])
        data['X_val'] = imputer.transform(data['X_val'])
        data['X_test'] = imputer.transform(data['X_test'])
    
    return y_train_day0, y_val_day0, y_test_day0, data


def train_models(X_train, y_train, X_val, y_val, logger) -> Dict:
    logger.info("=" * 70)
    logger.info("TRAINING MODELS")
    logger.info("=" * 70)
    
    models = {}
    
    logger.info("\n1. Training Baseline (Mean Predictor)...")
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    models['baseline_mean'] = baseline
    logger.info("  ✓ Baseline trained")
    
    logger.info("\n2. Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=settings.RANDOM_SEED)
    ridge.fit(X_train, y_train)
    models['ridge'] = ridge
    logger.info("  ✓ Ridge trained")
    
    logger.info("\n3. Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, random_state=settings.RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    logger.info("  ✓ Random Forest trained")
    
    return models


def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test, logger):
    logger.info("=" * 70)
    logger.info("EVALUATING MODELS")
    logger.info("=" * 70)
    
    metrics = {}
    predictions = {}
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        predictions[f"{model_name}_train"] = y_train_pred
        predictions[f"{model_name}_val"] = y_val_pred
        predictions[f"{model_name}_test"] = y_test_pred
        
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
        
        logger.info(f"  Test MAE:  {metrics[model_name]['test_mae']:.6f}")
        logger.info(f"  Test RMSE: {metrics[model_name]['test_rmse']:.6f}")
        logger.info(f"  Test R²:   {metrics[model_name]['test_r2']:.6f}")
    
    return metrics, predictions


def save_outputs(y_train, y_val, y_test, metrics, predictions, cleaned_test, logger):
    logger.info("=" * 70)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 70)
    
    output_dir = PROJECT_ROOT / "experiments_03" / "results" / "day0_reaction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nSaving targets...")
    np.save(output_dir / "y_train_day0.npy", y_train)
    np.save(output_dir / "y_val_day0.npy", y_val)
    np.save(output_dir / "y_test_day0.npy", y_test)
    logger.info("  ✓ Targets saved")
    
    logger.info("\nSaving metrics (JSON)...")
    import json
    with open(output_dir / "regression_metrics_day0.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("  ✓ Metrics JSON saved")
    
    logger.info("\nSaving metrics (CSV)...")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'model'
    metrics_df.to_csv(output_dir / "regression_metrics_day0.csv")
    logger.info("  ✓ Metrics CSV saved")
    
    logger.info("\nSaving predictions...")
    for pred_name, pred_values in predictions.items():
        np.save(output_dir / f"predictions_day0_{pred_name}.npy", pred_values)
    logger.info(f"  ✓ {len(predictions)} prediction files saved")
    
    logger.info("\nSaving Ridge residuals...")
    ridge_test_pred = predictions['ridge_test']
    residuals_df = cleaned_test.copy()
    residuals_df['y_true'] = y_test
    residuals_df['y_pred'] = ridge_test_pred
    residuals_df['residual'] = y_test - ridge_test_pred
    residuals_df.to_parquet(output_dir / "residuals_day0_ridge.parquet")
    logger.info("  ✓ Residuals saved")
    
    logger.info("\nSaving completion marker...")
    with open(output_dir / "step_experiment_03_completed.txt", 'w') as f:
        f.write("Experiment 03: Day 0 announcement reaction completed successfully.\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Val samples: {len(y_val)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
    logger.info("  ✓ Completion marker saved")
    
    logger.info(f"\n✓ All outputs saved to: {output_dir}")


def main():
    logger = setup_logging()
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 03: DAY 0 (ANNOUNCEMENT DAY) MARKET REACTION")
    logger.info("=" * 70)
    logger.info("\nObjective: Test if fundamentals predict IMMEDIATE market reaction")
    logger.info("Strategy: Reuse Step 10 features, compute Day 0 excess return")
    logger.info("Hypothesis: Unlike post-announcement returns, Day 0 might be predictable")
    logger.info("=" * 70)
    
    try:
        data = load_data(logger)
        y_train_day0, y_val_day0, y_test_day0, data = align_targets(data, logger)
        models = train_models(data['X_train'], y_train_day0, data['X_val'], y_val_day0, logger)
        metrics, predictions = evaluate_models(
            models, data['X_train'], y_train_day0, data['X_val'], y_val_day0,
            data['X_test'], y_test_day0, logger
        )
        save_outputs(y_train_day0, y_val_day0, y_test_day0, metrics, predictions, data['cleaned_test'], logger)
        
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 03 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\nTest Set Results:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  R² = {model_metrics['test_r2']:.6f}")
            logger.info(f"  MAE = {model_metrics['test_mae']:.6f}")
            logger.info(f"  RMSE = {model_metrics['test_rmse']:.6f}")
        
        best_r2 = max(m['test_r2'] for m in metrics.values())
        logger.info("\n" + "=" * 70)
        logger.info("INTERPRETATION")
        logger.info("=" * 70)
        if best_r2 > 0.05:
            logger.info(f"\n✓ SIGNIFICANT PREDICTABILITY FOUND! (R² = {best_r2:.4f})")
            logger.info("  Fundamentals DO correlate with announcement day reaction!")
        elif best_r2 > 0.01:
            logger.info(f"\n⚠ WEAK PREDICTABILITY (R² = {best_r2:.4f})")
            logger.info("  Some correlation exists but very weak")
        else:
            logger.info(f"\n✗ NO PREDICTABILITY (R² = {best_r2:.4f})")
            logger.info("  Even Day 0 reaction is unpredictable from fundamentals")
        logger.info("\n" + "=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
