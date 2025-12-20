"""
Step 22 — Comprehensive Data Quality Analysis

This step performs a thorough analysis of data quality issues including:
1. Missing data patterns and statistics
2. Systematic outlier detection
3. Data leakage verification
4. Temporal alignment checks
5. Feature distribution analysis

CRITICAL: This step is READ-ONLY. It does NOT modify any data.
It only analyzes and reports on data quality issues.
"""

from typing import NoReturn, Dict, List, Tuple
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import timedelta

from src.config import Settings


def analyze_missing_data(df: pd.DataFrame, feature_cols: List[str], logger: logging.Logger) -> Dict:
    """
    Analyze missing data patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    feature_cols : list
        List of feature column names
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    missing_stats : dict
        Statistics about missing data
    """
    logger.info("\n" + "=" * 70)
    logger.info("MISSING DATA ANALYSIS")
    logger.info("=" * 70)
    
    # Overall missing statistics
    total_cells = len(df) * len(feature_cols)
    missing_cells = df[feature_cols].isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    logger.info(f"\nOverall Missing Data:")
    logger.info(f"  Total cells: {total_cells:,}")
    logger.info(f"  Missing cells: {missing_cells:,}")
    logger.info(f"  Missing percentage: {missing_pct:.2f}%")
    
    # Per-feature missing statistics
    missing_per_feature = df[feature_cols].isnull().sum()
    missing_pct_per_feature = (missing_per_feature / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'feature': feature_cols,
        'missing_count': missing_per_feature.values,
        'missing_pct': missing_pct_per_feature.values
    }).sort_values('missing_pct', ascending=False)
    
    logger.info(f"\nTop 10 Features with Most Missing Data:")
    for idx, row in missing_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['missing_count']:,} ({row['missing_pct']:.2f}%)")
    
    # Complete cases analysis
    complete_cases = df[feature_cols].notna().all(axis=1).sum()
    complete_pct = (complete_cases / len(df)) * 100
    
    logger.info(f"\nComplete Cases:")
    logger.info(f"  Rows with no missing values: {complete_cases:,} ({complete_pct:.2f}%)")
    logger.info(f"  Rows with any missing values: {len(df) - complete_cases:,} ({100 - complete_pct:.2f}%)")
    
    # Missing data patterns
    missing_patterns = df[feature_cols].isnull().sum(axis=1).value_counts().sort_index()
    
    logger.info(f"\nMissing Data Patterns:")
    logger.info(f"  0 missing features: {missing_patterns.get(0, 0):,} rows")
    logger.info(f"  1-5 missing features: {missing_patterns[missing_patterns.index.isin(range(1, 6))].sum():,} rows")
    logger.info(f"  6-10 missing features: {missing_patterns[missing_patterns.index.isin(range(6, 11))].sum():,} rows")
    logger.info(f"  >10 missing features: {missing_patterns[missing_patterns.index > 10].sum():,} rows")
    
    # Test if missing at random (MAR)
    logger.info(f"\nMissing at Random (MAR) Test:")
    if 'target' in df.columns:
        complete_target_mean = df[df[feature_cols].notna().all(axis=1)]['target'].mean()
        incomplete_target_mean = df[~df[feature_cols].notna().all(axis=1)]['target'].mean()
        
        t_stat, p_value = stats.ttest_ind(
            df[df[feature_cols].notna().all(axis=1)]['target'].dropna(),
            df[~df[feature_cols].notna().all(axis=1)]['target'].dropna()
        )
        
        logger.info(f"  Complete cases target mean: {complete_target_mean:.6f}")
        logger.info(f"  Incomplete cases target mean: {incomplete_target_mean:.6f}")
        logger.info(f"  T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            logger.warning(f"  ⚠️ Missing data is NOT random (p < 0.05)")
            logger.warning(f"  This suggests systematic bias in missing data")
        else:
            logger.info(f"  ✓ Missing data appears random (p >= 0.05)")
    
    return {
        'total_cells': int(total_cells),
        'missing_cells': int(missing_cells),
        'missing_pct': float(missing_pct),
        'complete_cases': int(complete_cases),
        'complete_pct': float(complete_pct),
        'missing_per_feature': missing_df.to_dict('records'),
        'missing_patterns': missing_patterns.to_dict()
    }


def analyze_outliers(df: pd.DataFrame, feature_cols: List[str], logger: logging.Logger) -> Dict:
    """
    Perform systematic outlier detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    feature_cols : list
        List of feature column names
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    outlier_stats : dict
        Statistics about outliers
    """
    logger.info("\n" + "=" * 70)
    logger.info("OUTLIER ANALYSIS")
    logger.info("=" * 70)
    
    outlier_stats = {}
    
    # Z-score method (|z| > 3)
    logger.info(f"\nZ-Score Method (|z| > 3):")
    z_scores = np.abs(stats.zscore(df[feature_cols].fillna(df[feature_cols].median()), nan_policy='omit'))
    outliers_z = (z_scores > 3).sum(axis=0)
    
    outlier_features_z = pd.DataFrame({
        'feature': feature_cols,
        'outlier_count': outliers_z,
        'outlier_pct': (outliers_z / len(df)) * 100
    }).sort_values('outlier_count', ascending=False)
    
    logger.info(f"  Total outlier detections: {outliers_z.sum():,}")
    logger.info(f"  Top 5 features with most outliers:")
    for idx, row in outlier_features_z.head(5).iterrows():
        logger.info(f"    {row['feature']}: {row['outlier_count']:,} ({row['outlier_pct']:.2f}%)")
    
    # IQR method
    logger.info(f"\nIQR Method (beyond 1.5 * IQR):")
    outliers_iqr = {}
    for col in feature_cols:
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        else:
            outliers_iqr[col] = 0
    
    outlier_features_iqr = pd.DataFrame({
        'feature': list(outliers_iqr.keys()),
        'outlier_count': list(outliers_iqr.values()),
        'outlier_pct': [(v / len(df)) * 100 for v in outliers_iqr.values()]
    }).sort_values('outlier_count', ascending=False)
    
    logger.info(f"  Total outlier detections: {sum(outliers_iqr.values()):,}")
    logger.info(f"  Top 5 features with most outliers:")
    for idx, row in outlier_features_iqr.head(5).iterrows():
        logger.info(f"    {row['feature']}: {row['outlier_count']:,} ({row['outlier_pct']:.2f}%)")
    
    # Target outliers (if exists)
    if 'target' in df.columns:
        logger.info(f"\nTarget Variable Outliers:")
        target_mean = df['target'].mean()
        target_std = df['target'].std()
        target_outliers = np.abs(df['target'] - target_mean) > 3 * target_std
        
        logger.info(f"  Mean: {target_mean:.6f}")
        logger.info(f"  Std: {target_std:.6f}")
        logger.info(f"  Outliers (|z| > 3): {target_outliers.sum():,} ({(target_outliers.sum() / len(df)) * 100:.2f}%)")
        
        if target_outliers.sum() > 0:
            logger.info(f"  Min outlier: {df.loc[target_outliers, 'target'].min():.6f}")
            logger.info(f"  Max outlier: {df.loc[target_outliers, 'target'].max():.6f}")
    
    # Extreme values check
    logger.info(f"\nExtreme Values Check:")
    extreme_features = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:
            col_max = df[col].max()
            col_min = df[col].min()
            col_range = col_max - col_min
            
            # Check for suspiciously large values
            if col_max > 1e6 or col_min < -1e6:
                extreme_features.append({
                    'feature': col,
                    'min': col_min,
                    'max': col_max,
                    'range': col_range
                })
    
    if extreme_features:
        logger.warning(f"  ⚠️ Found {len(extreme_features)} features with extreme values (>1e6 or <-1e6):")
        for feat in extreme_features[:5]:
            logger.warning(f"    {feat['feature']}: min={feat['min']:.2e}, max={feat['max']:.2e}")
    else:
        logger.info(f"  ✓ No extreme values detected")
    
    return {
        'z_score_outliers': outlier_features_z.to_dict('records'),
        'iqr_outliers': outlier_features_iqr.to_dict('records'),
        'extreme_features': extreme_features
    }


def verify_data_leakage(df: pd.DataFrame, logger: logging.Logger) -> Dict:
    """
    Verify potential data leakage issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    leakage_checks : dict
        Results of data leakage checks
    """
    logger.info("\n" + "=" * 70)
    logger.info("DATA LEAKAGE VERIFICATION")
    logger.info("=" * 70)
    
    leakage_issues = []
    
    # Check 1: Temporal alignment
    logger.info(f"\n1. Temporal Alignment Check:")
    if 'earnings_date' in df.columns and 'target_end_date' in df.columns:
        # Convert to datetime if needed
        earnings_date = pd.to_datetime(df['earnings_date'])
        target_end_date = pd.to_datetime(df['target_end_date'])
        
        # Check if earnings_date < target_end_date (should always be true)
        temporal_violations = (earnings_date >= target_end_date).sum()
        
        logger.info(f"  Checking: earnings_date < target_end_date")
        logger.info(f"  Violations: {temporal_violations:,} ({(temporal_violations / len(df)) * 100:.2f}%)")
        
        if temporal_violations > 0:
            logger.error(f"  ❌ CRITICAL: Found {temporal_violations} temporal violations!")
            logger.error(f"  This indicates potential data leakage!")
            leakage_issues.append({
                'check': 'temporal_alignment',
                'violations': int(temporal_violations),
                'severity': 'CRITICAL'
            })
        else:
            logger.info(f"  ✓ No temporal violations detected")
    else:
        logger.warning(f"  ⚠️ Cannot verify: earnings_date or target_end_date not found")
    
    # Check 2: Future information in features
    logger.info(f"\n2. Future Information Check:")
    future_info_features = []
    
    # Check for features that might contain future information
    suspicious_patterns = ['forward', 'future', 'next', 'ahead', 'post']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in suspicious_patterns):
            future_info_features.append(col)
    
    if future_info_features:
        logger.warning(f"  ⚠️ Found {len(future_info_features)} features with suspicious names:")
        for feat in future_info_features[:10]:
            logger.warning(f"    - {feat}")
        leakage_issues.append({
            'check': 'future_information',
            'features': future_info_features,
            'severity': 'WARNING'
        })
    else:
        logger.info(f"  ✓ No suspicious feature names detected")
    
    # Check 3: Target correlation with features
    logger.info(f"\n3. Suspiciously High Correlations:")
    if 'target' in df.columns:
        feature_cols = [col for col in df.columns if col not in ['target', 'earnings_date', 'ticker', 'company_name']]
        correlations = df[feature_cols + ['target']].corr()['target'].drop('target').abs()
        
        high_corr_features = correlations[correlations > 0.5].sort_values(ascending=False)
        
        if len(high_corr_features) > 0:
            logger.warning(f"  ⚠️ Found {len(high_corr_features)} features with |correlation| > 0.5:")
            for feat, corr in high_corr_features.head(10).items():
                logger.warning(f"    {feat}: {corr:.4f}")
            logger.warning(f"  These might indicate data leakage or very strong predictors")
            leakage_issues.append({
                'check': 'high_correlation',
                'features': high_corr_features.to_dict(),
                'severity': 'WARNING'
            })
        else:
            logger.info(f"  ✓ No suspiciously high correlations (all |r| < 0.5)")
    
    # Check 4: Duplicate rows
    logger.info(f"\n4. Duplicate Rows Check:")
    duplicates = df.duplicated().sum()
    logger.info(f"  Duplicate rows: {duplicates:,} ({(duplicates / len(df)) * 100:.2f}%)")
    
    if duplicates > 0:
        logger.warning(f"  ⚠️ Found {duplicates} duplicate rows")
        leakage_issues.append({
            'check': 'duplicates',
            'count': int(duplicates),
            'severity': 'WARNING'
        })
    else:
        logger.info(f"  ✓ No duplicate rows detected")
    
    # Check 5: Data alignment between fundamentals and earnings dates
    logger.info(f"\n5. Fundamental Data Alignment Check:")
    if 'earnings_date' in df.columns and 'quarter' in df.columns:
        # Check if fundamental data quarter matches earnings quarter
        earnings_date = pd.to_datetime(df['earnings_date'])
        earnings_quarter = earnings_date.dt.quarter
        
        if df['quarter'].dtype == 'object':
            # Extract quarter from string like "Q1 2020"
            fundamental_quarter = df['quarter'].str.extract(r'Q(\d)')[0].astype(float)
        else:
            fundamental_quarter = df['quarter']
        
        quarter_mismatches = (earnings_quarter != fundamental_quarter).sum()
        
        logger.info(f"  Quarter mismatches: {quarter_mismatches:,} ({(quarter_mismatches / len(df)) * 100:.2f}%)")
        
        if quarter_mismatches > len(df) * 0.1:  # More than 10% mismatches
            logger.warning(f"  ⚠️ High number of quarter mismatches detected")
            logger.warning(f"  This might indicate fundamental data is not properly aligned")
            leakage_issues.append({
                'check': 'quarter_alignment',
                'mismatches': int(quarter_mismatches),
                'severity': 'WARNING'
            })
        else:
            logger.info(f"  ✓ Quarter alignment looks reasonable")
    
    # Summary
    logger.info(f"\n" + "=" * 70)
    logger.info(f"DATA LEAKAGE SUMMARY")
    logger.info(f"=" * 70)
    
    critical_issues = [issue for issue in leakage_issues if issue['severity'] == 'CRITICAL']
    warning_issues = [issue for issue in leakage_issues if issue['severity'] == 'WARNING']
    
    logger.info(f"  Critical issues: {len(critical_issues)}")
    logger.info(f"  Warning issues: {len(warning_issues)}")
    
    if len(critical_issues) == 0:
        logger.info(f"\n  ✅ No critical data leakage issues detected")
    else:
        logger.error(f"\n  ❌ CRITICAL DATA LEAKAGE ISSUES FOUND!")
        logger.error(f"  Your results may be invalid due to data leakage!")
    
    return {
        'leakage_issues': leakage_issues,
        'critical_count': len(critical_issues),
        'warning_count': len(warning_issues)
    }


def analyze_feature_distributions(df: pd.DataFrame, feature_cols: List[str], logger: logging.Logger) -> Dict:
    """
    Analyze feature distributions for anomalies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    feature_cols : list
        List of feature column names
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    distribution_stats : dict
        Statistics about feature distributions
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)
    
    distribution_issues = []
    
    # Check for constant or near-constant features
    logger.info(f"\n1. Constant/Near-Constant Features:")
    constant_features = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:
            unique_ratio = df[col].nunique() / df[col].notna().sum()
            if unique_ratio < 0.01:  # Less than 1% unique values
                constant_features.append({
                    'feature': col,
                    'unique_values': int(df[col].nunique()),
                    'unique_ratio': float(unique_ratio)
                })
    
    if constant_features:
        logger.warning(f"  ⚠️ Found {len(constant_features)} near-constant features:")
        for feat in constant_features[:10]:
            logger.warning(f"    {feat['feature']}: {feat['unique_values']} unique values ({feat['unique_ratio']:.2%})")
        distribution_issues.extend(constant_features)
    else:
        logger.info(f"  ✓ No constant features detected")
    
    # Check for highly skewed distributions
    logger.info(f"\n2. Skewness Analysis:")
    skewed_features = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:
            skewness = df[col].skew()
            if abs(skewness) > 3:  # Highly skewed
                skewed_features.append({
                    'feature': col,
                    'skewness': float(skewness)
                })
    
    if skewed_features:
        logger.warning(f"  ⚠️ Found {len(skewed_features)} highly skewed features (|skew| > 3):")
        for feat in sorted(skewed_features, key=lambda x: abs(x['skewness']), reverse=True)[:10]:
            logger.warning(f"    {feat['feature']}: skewness = {feat['skewness']:.2f}")
    else:
        logger.info(f"  ✓ No highly skewed features detected")
    
    # Check for infinite values
    logger.info(f"\n3. Infinite Values Check:")
    infinite_features = []
    for col in feature_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            infinite_features.append({
                'feature': col,
                'inf_count': int(inf_count),
                'inf_pct': float((inf_count / len(df)) * 100)
            })
    
    if infinite_features:
        logger.error(f"  ❌ Found {len(infinite_features)} features with infinite values:")
        for feat in infinite_features:
            logger.error(f"    {feat['feature']}: {feat['inf_count']:,} ({feat['inf_pct']:.2f}%)")
    else:
        logger.info(f"  ✓ No infinite values detected")
    
    return {
        'constant_features': constant_features,
        'skewed_features': skewed_features,
        'infinite_features': infinite_features
    }


def create_visualizations(
    df: pd.DataFrame,
    feature_cols: List[str],
    missing_stats: Dict,
    outlier_stats: Dict,
    output_dir: Path
) -> None:
    """
    Create visualizations for data quality analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    feature_cols : list
        Feature columns
    missing_stats : dict
        Missing data statistics
    outlier_stats : dict
        Outlier statistics
    output_dir : Path
        Output directory
    """
    # 1. Missing data heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sample features if too many
    if len(feature_cols) > 30:
        sample_features = feature_cols[:30]
    else:
        sample_features = feature_cols
    
    missing_matrix = df[sample_features].isnull().astype(int)
    sns.heatmap(missing_matrix.T, cbar=True, cmap='YlOrRd', ax=ax)
    ax.set_title('Missing Data Pattern (First 30 Features)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_data_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Missing data bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    missing_df = pd.DataFrame(missing_stats['missing_per_feature'])
    top_missing = missing_df.nlargest(20, 'missing_pct')
    
    ax.barh(range(len(top_missing)), top_missing['missing_pct'])
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing['feature'])
    ax.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Top 20 Features by Missing Data', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_data_by_feature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Outlier detection comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Z-score outliers
    z_outliers = pd.DataFrame(outlier_stats['z_score_outliers'])
    top_z = z_outliers.nlargest(15, 'outlier_count')
    
    axes[0].barh(range(len(top_z)), top_z['outlier_count'])
    axes[0].set_yticks(range(len(top_z)))
    axes[0].set_yticklabels(top_z['feature'])
    axes[0].set_xlabel('Outlier Count', fontsize=12)
    axes[0].set_title('Z-Score Method (|z| > 3)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # IQR outliers
    iqr_outliers = pd.DataFrame(outlier_stats['iqr_outliers'])
    top_iqr = iqr_outliers.nlargest(15, 'outlier_count')
    
    axes[1].barh(range(len(top_iqr)), top_iqr['outlier_count'])
    axes[1].set_yticks(range(len(top_iqr)))
    axes[1].set_yticklabels(top_iqr['feature'])
    axes[1].set_xlabel('Outlier Count', fontsize=12)
    axes[1].set_title('IQR Method (1.5 * IQR)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_detection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Target distribution
    if 'target' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df['target'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Target Value', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Q-Q plot
        stats.probplot(df['target'].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Target)', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_step_22() -> NoReturn:
    """
    Execute Step 22: Comprehensive Data Quality Analysis.
    
    This step performs thorough analysis of data quality issues WITHOUT
    modifying any data. It is purely analytical and diagnostic.
    """
    settings = Settings()
    logger = settings.setup_logging("step_22_data_quality_analysis")
    
    logger.info("=" * 70)
    logger.info("STEP 22: COMPREHENSIVE DATA QUALITY ANALYSIS")
    logger.info("=" * 70)
    logger.info("\nThis step analyzes data quality WITHOUT modifying any data.")
    logger.info("It is purely diagnostic and will generate a detailed report.")
    
    settings.ensure_directories()
    
    # Create output directory
    output_dir = settings.get_step_results_dir(22)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nOutput directory: {output_dir}")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    # Load from Step 10 (preprocessed data)
    step_10_dir = settings.get_step_results_dir(10)
    
    logger.info(f"\nLoading data from Step 10...")
    train_df = pd.read_parquet(step_10_dir / "cleaned_train.parquet")
    val_df = pd.read_parquet(step_10_dir / "cleaned_val.parquet")
    test_df = pd.read_parquet(step_10_dir / "cleaned_test.parquet")
    
    # Combine for comprehensive analysis
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Total columns: {len(df.columns)}")
    logger.info(f"  Train: {len(train_df):,} rows")
    logger.info(f"  Val: {len(val_df):,} rows")
    logger.info(f"  Test: {len(test_df):,} rows")
    
    # Load feature names
    with open(step_10_dir / "dataset_spec.json", 'r') as f:
        spec = json.load(f)
    feature_cols = spec['feature_columns']
    
    logger.info(f"  Features: {len(feature_cols)}")
    
    # ========================================================================
    # Perform Analyses
    # ========================================================================
    
    # 1. Missing Data Analysis
    missing_stats = analyze_missing_data(df, feature_cols, logger)
    
    # 2. Outlier Analysis
    outlier_stats = analyze_outliers(df, feature_cols, logger)
    
    # 3. Data Leakage Verification
    leakage_stats = verify_data_leakage(df, logger)
    
    # 4. Feature Distribution Analysis
    distribution_stats = analyze_feature_distributions(df, feature_cols, logger)
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)
    
    # Save statistics as JSON
    all_stats = {
        'missing_data': missing_stats,
        'outliers': outlier_stats,
        'data_leakage': leakage_stats,
        'distributions': distribution_stats
    }
    
    with open(output_dir / 'data_quality_statistics.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"  ✓ Saved: data_quality_statistics.json")
    
    # ========================================================================
    # Create Visualizations
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    create_visualizations(df, feature_cols, missing_stats, outlier_stats, output_dir)
    logger.info(f"  ✓ Saved: missing_data_heatmap.png")
    logger.info(f"  ✓ Saved: missing_data_by_feature.png")
    logger.info(f"  ✓ Saved: outlier_detection_comparison.png")
    logger.info(f"  ✓ Saved: target_distribution.png")
    
    # ========================================================================
    # Generate Report
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING REPORT")
    logger.info("=" * 70)
    
    report_lines = [
        "# Data Quality Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Samples:** {len(df):,}",
        f"**Total Features:** {len(feature_cols)}",
        "",
        "---",
        "",
        "## 1. Missing Data Analysis",
        "",
        f"### Overall Statistics",
        "",
        f"- **Total cells:** {missing_stats['total_cells']:,}",
        f"- **Missing cells:** {missing_stats['missing_cells']:,} ({missing_stats['missing_pct']:.2f}%)",
        f"- **Complete cases:** {missing_stats['complete_cases']:,} ({missing_stats['complete_pct']:.2f}%)",
        f"- **Incomplete cases:** {len(df) - missing_stats['complete_cases']:,} ({100 - missing_stats['complete_pct']:.2f}%)",
        "",
        "### Key Findings",
        ""
    ]
    
    # Add missing data findings
    if missing_stats['missing_pct'] > 30:
        report_lines.append(f"❌ **CRITICAL:** {missing_stats['missing_pct']:.1f}% of data is missing!")
    elif missing_stats['missing_pct'] > 10:
        report_lines.append(f"⚠️ **WARNING:** {missing_stats['missing_pct']:.1f}% of data is missing")
    else:
        report_lines.append(f"✅ **GOOD:** Only {missing_stats['missing_pct']:.1f}% of data is missing")
    
    report_lines.extend([
        "",
        "### Top 10 Features with Most Missing Data",
        "",
        "| Feature | Missing Count | Missing % |",
        "|---------|--------------|-----------|"
    ])
    
    for feat in missing_stats['missing_per_feature'][:10]:
        report_lines.append(f"| {feat['feature']} | {feat['missing_count']:,} | {feat['missing_pct']:.2f}% |")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Outlier Analysis",
        "",
        "### Z-Score Method (|z| > 3)",
        ""
    ])
    
    z_outliers_df = pd.DataFrame(outlier_stats['z_score_outliers'])
    total_z_outliers = z_outliers_df['outlier_count'].sum()
    report_lines.append(f"**Total outliers detected:** {total_z_outliers:,}")
    report_lines.append("")
    report_lines.append("**Top 5 features:**")
    report_lines.append("")
    
    for feat in outlier_stats['z_score_outliers'][:5]:
        report_lines.append(f"- {feat['feature']}: {feat['outlier_count']:,} ({feat['outlier_pct']:.2f}%)")
    
    report_lines.extend([
        "",
        "### IQR Method (1.5 * IQR)",
        ""
    ])
    
    iqr_outliers_df = pd.DataFrame(outlier_stats['iqr_outliers'])
    total_iqr_outliers = iqr_outliers_df['outlier_count'].sum()
    report_lines.append(f"**Total outliers detected:** {total_iqr_outliers:,}")
    report_lines.append("")
    report_lines.append("**Top 5 features:**")
    report_lines.append("")
    
    for feat in outlier_stats['iqr_outliers'][:5]:
        report_lines.append(f"- {feat['feature']}: {feat['outlier_count']:,} ({feat['outlier_pct']:.2f}%)")
    
    if outlier_stats['extreme_features']:
        report_lines.extend([
            "",
            "### ⚠️ Extreme Values Detected",
            "",
            "The following features contain extreme values (>1e6 or <-1e6):",
            ""
        ])
        for feat in outlier_stats['extreme_features'][:10]:
            report_lines.append(f"- **{feat['feature']}**: min={feat['min']:.2e}, max={feat['max']:.2e}")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Data Leakage Verification",
        ""
    ])
    
    if leakage_stats['critical_count'] == 0:
        report_lines.append("✅ **No critical data leakage issues detected**")
    else:
        report_lines.append(f"❌ **CRITICAL: {leakage_stats['critical_count']} data leakage issue(s) found!**")
    
    report_lines.append("")
    report_lines.append(f"**Summary:**")
    report_lines.append(f"- Critical issues: {leakage_stats['critical_count']}")
    report_lines.append(f"- Warning issues: {leakage_stats['warning_count']}")
    report_lines.append("")
    
    if leakage_stats['leakage_issues']:
        report_lines.append("### Issues Detected:")
        report_lines.append("")
        for issue in leakage_stats['leakage_issues']:
            severity_icon = "❌" if issue['severity'] == 'CRITICAL' else "⚠️"
            report_lines.append(f"{severity_icon} **{issue['check']}** ({issue['severity']})")
            if 'violations' in issue:
                report_lines.append(f"  - Violations: {issue['violations']:,}")
            if 'count' in issue:
                report_lines.append(f"  - Count: {issue['count']:,}")
            report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 4. Feature Distribution Analysis",
        ""
    ])
    
    if distribution_stats['constant_features']:
        report_lines.append(f"⚠️ **Found {len(distribution_stats['constant_features'])} near-constant features**")
        report_lines.append("")
        report_lines.append("These features have very low variance and may not be useful:")
        report_lines.append("")
        for feat in distribution_stats['constant_features'][:10]:
            report_lines.append(f"- {feat['feature']}: {feat['unique_values']} unique values ({feat['unique_ratio']:.2%})")
        report_lines.append("")
    
    if distribution_stats['skewed_features']:
        report_lines.append(f"⚠️ **Found {len(distribution_stats['skewed_features'])} highly skewed features (|skew| > 3)**")
        report_lines.append("")
        for feat in sorted(distribution_stats['skewed_features'], key=lambda x: abs(x['skewness']), reverse=True)[:10]:
            report_lines.append(f"- {feat['feature']}: skewness = {feat['skewness']:.2f}")
        report_lines.append("")
    
    if distribution_stats['infinite_features']:
        report_lines.append(f"❌ **CRITICAL: Found {len(distribution_stats['infinite_features'])} features with infinite values!**")
        report_lines.append("")
        for feat in distribution_stats['infinite_features']:
            report_lines.append(f"- {feat['feature']}: {feat['inf_count']:,} infinite values ({feat['inf_pct']:.2f}%)")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 5. Recommendations",
        "",
        "### High Priority",
        ""
    ])
    
    # Generate recommendations based on findings
    if missing_stats['missing_pct'] > 30:
        report_lines.append("1. **Address high missing data rate:** Consider:")
        report_lines.append("   - Investigating why 30%+ of data is missing")
        report_lines.append("   - Using more sophisticated imputation methods")
        report_lines.append("   - Collecting additional data sources")
        report_lines.append("")
    
    if leakage_stats['critical_count'] > 0:
        report_lines.append("2. **Fix data leakage issues immediately:** Your results may be invalid")
        report_lines.append("")
    
    if distribution_stats['infinite_features']:
        report_lines.append("3. **Remove or fix infinite values:** These will cause model failures")
        report_lines.append("")
    
    report_lines.extend([
        "### Medium Priority",
        ""
    ])
    
    if len(outlier_stats['extreme_features']) > 0:
        report_lines.append("4. **Investigate extreme values:** Verify these are not data errors")
        report_lines.append("")
    
    if distribution_stats['skewed_features']:
        report_lines.append("5. **Consider transforming skewed features:** Log or Box-Cox transformations")
        report_lines.append("")
    
    report_lines.extend([
        "### Low Priority",
        ""
    ])
    
    if distribution_stats['constant_features']:
        report_lines.append("6. **Remove near-constant features:** They provide little information")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 6. Conclusion",
        ""
    ])
    
    # Overall assessment
    critical_issues = []
    if missing_stats['missing_pct'] > 30:
        critical_issues.append("High missing data rate")
    if leakage_stats['critical_count'] > 0:
        critical_issues.append("Data leakage detected")
    if distribution_stats['infinite_features']:
        critical_issues.append("Infinite values present")
    
    if critical_issues:
        report_lines.append("❌ **CRITICAL ISSUES FOUND:**")
        report_lines.append("")
        for issue in critical_issues:
            report_lines.append(f"- {issue}")
        report_lines.append("")
        report_lines.append("**These issues must be addressed before trusting model results.**")
    else:
        report_lines.append("✅ **No critical data quality issues detected.**")
        report_lines.append("")
        report_lines.append("While there are some warnings (missing data, outliers), the data quality")
        report_lines.append("is acceptable for modeling. The main concerns are:")
        report_lines.append("")
        report_lines.append(f"- {missing_stats['missing_pct']:.1f}% missing data (being handled by imputation)")
        report_lines.append(f"- {total_z_outliers:,} outliers detected (may be legitimate extreme values)")
        report_lines.append("")
        report_lines.append("**Your current preprocessing pipeline appears adequate.**")
    
    report_lines.extend([
        "",
        "---",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Step:** 22 - Data Quality Analysis",
        f"**Status:** Analysis complete (no data modified)"
    ])
    
    # Save report
    report_path = output_dir / "DATA_QUALITY_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"  ✓ Saved: DATA_QUALITY_REPORT.md")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 22 COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    
    logger.info("\n✅ Data quality analysis complete!")
    logger.info(f"\n📁 Results saved to: {output_dir}")
    logger.info("\n📊 Key outputs:")
    logger.info("  - data_quality_statistics.json")
    logger.info("  - DATA_QUALITY_REPORT.md")
    logger.info("  - missing_data_heatmap.png")
    logger.info("  - missing_data_by_feature.png")
    logger.info("  - outlier_detection_comparison.png")
    logger.info("  - target_distribution.png")
    
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)
    logger.info(f"\n  Missing data: {missing_stats['missing_pct']:.2f}%")
    logger.info(f"  Complete cases: {missing_stats['complete_pct']:.2f}%")
    logger.info(f"  Outliers (Z-score): {total_z_outliers:,}")
    logger.info(f"  Data leakage issues: {leakage_stats['critical_count']} critical, {leakage_stats['warning_count']} warnings")
    
    if critical_issues:
        logger.error("\n❌ CRITICAL ISSUES DETECTED - Review DATA_QUALITY_REPORT.md")
    else:
        logger.info("\n✅ No critical issues - Data quality is acceptable")
    
    logger.info("\n✅ NO DATA WAS MODIFIED - This was a read-only analysis")


if __name__ == "__main__":
    run_step_22()
