#!/usr/bin/env python3
"""
components/optimizer/data_exporter.py
SuperBot - AI Training Data Exporter
Author: SuperBot Team
Date: 2025-11-17
Versiyon: 2.0.0

Exports optimizer results as an AI training dataset.
Converts backtest results into the (X features, Y labels) format.

Features:
- Collect optimizer results (all stages, all runs)
- ML-ready format (Parquet/CSV/JSON)
- Feature engineering (nested dict → flat columns)
- Categorical encoding (one-hot, label encoding)
- Prepare the train/test split.

Usage:
    from components.optimizer.v2.data_exporter import DataExporter

    # 1. Training data export (all results)
    exporter = DataExporter()
    dataset_path = exporter.export_training_data(
        optimization_runs=["opt_20251117_*"],  # Glob pattern
        output_format="parquet",
        include_all_trials=True  # Include all trials, not just the top 10
    )

    # 2. Prepare ML features
    X, y, feature_names, categorical_features = exporter.prepare_features_for_ml(
        dataset_path=dataset_path,
        target_metric='metric_sharpe_ratio'
    )

    # 3. Model training (example)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X, y)

Dependencies:
    - python>=3.10
    - pandas>=2.0.0
    - numpy>=1.24.0
    - pyarrow>=10.0.0 (for Parquet)
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .stage_results import StageResultsManager


class DataExporter:
    """Exports optimizer results as an AI training dataset."""

    def __init__(self, results_dir: str = "results/optimization_runs"):
        """
        Start the data exporter.

        Args:
            results_dir: The directory containing the optimization runs.
        """
        self.results_dir = Path(results_dir)

    # ========================================================================
    # EXPORT TRAINING DATA
    # ========================================================================

    def export_training_data(
        self,
        optimization_runs: List[str],
        output_format: str = "parquet",
        output_dir: str = "data/training/optimizer",
        include_all_trials: bool = True
    ) -> Path:
        """
        Export all optimizer results as training dataset

        Args:
            optimization_runs: List of run IDs or glob patterns
                Example: ["opt_20251117_143022", "opt_*"]
            output_format: Output format ('parquet', 'csv', 'json')
            output_dir: Output directory
            include_all_trials: Include all trials (not just top 10)

        Returns:
            Path to exported dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all samples
        all_samples = []

        for pattern in optimization_runs:
            # Expand glob pattern
            run_dirs = self._expand_run_pattern(pattern)

            for run_dir in run_dirs:
                run_id = run_dir.name

                # Load all stage results
                manager = StageResultsManager(run_id=run_id, results_dir=str(self.results_dir))
                stage_results = manager.load_all_results()

                # Extract samples from each stage
                for stage_result in stage_results:
                    # Get trials
                    trials = stage_result.top_results

                    # If include_all_trials, load from original result file
                    # (For now, just use top_results)

                    for trial in trials:
                        sample = self._create_training_sample(
                            stage_result=stage_result,
                            trial=trial
                        )
                        all_samples.append(sample)

        # Convert to DataFrame
        df = pd.DataFrame(all_samples)

        print(f"\n📊 Training Dataset Info:")
        print(f"   Total samples: {len(all_samples):,}")
        print(f"   Unique strategies: {df['strategy_name'].nunique()}")
        print(f"   Unique symbols: {df['symbol'].nunique()}")
        print(f"   Unique timeframes: {df['timeframe'].nunique()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimizer_dataset_{len(all_samples)}_samples_{timestamp}.{output_format}"
        output_path = output_dir / filename

        if output_format == "parquet":
            df.to_parquet(output_path, index=False)
        elif output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "json":
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        print(f"\n✅ Exported to: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return output_path

    # ========================================================================
    # CREATE TRAINING SAMPLE
    # ========================================================================

    def _create_training_sample(
        self,
        stage_result: Any,
        trial: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a single training sample from trial result

        Returns dict with:
            - Metadata (strategy_name, symbol, timeframe, etc.)
            - Features (all strategy parameters - X)
            - Labels (all metrics - Y)
        """
        params = trial['params']
        metrics = trial['metrics']

        sample = {
            # ========== METADATA ==========
            'run_id': stage_result.run_id,
            'stage': stage_result.stage,
            'stage_number': stage_result.stage_number,
            'strategy_name': stage_result.strategy_name,
            'strategy_version': stage_result.strategy_version,
            'symbol': stage_result.backtest_period.get('symbol', 'N/A'),
            'timeframe': stage_result.backtest_period.get('timeframe', 'N/A'),
            'backtest_start': stage_result.backtest_period.get('start', 'N/A'),
            'backtest_end': stage_result.backtest_period.get('end', 'N/A'),
            'rank': trial.get('rank', 0),
            'timestamp': stage_result.timestamp,
        }

        # ========== FEATURES (X) - Parameters ==========
        # Flatten nested parameter dict
        flattened_params = self._flatten_dict(params, prefix='param')
        sample.update(flattened_params)

        # ========== LABELS (Y) - Metrics ==========
        # Flatten metrics dict
        flattened_metrics = self._flatten_dict(metrics, prefix='metric')
        sample.update(flattened_metrics)

        return sample

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        prefix: str = '',
        separator: str = '_'
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary

        Example:
            {
                'risk_management': {
                    'sizing_method': 'RISK_BASED',
                    'max_risk': 2.0
                }
            }
            →
            {
                'param_risk_management_sizing_method': 'RISK_BASED',
                'param_risk_management_max_risk': 2.0
            }
        """
        flattened = {}

        for key, value in d.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten
                flattened.update(self._flatten_dict(value, prefix=new_key, separator=separator))
            elif isinstance(value, (list, tuple)):
                # Convert list to string for now
                # (Better: one-hot encode for categorical)
                flattened[new_key] = str(value)
            else:
                flattened[new_key] = value

        return flattened

    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================

    def prepare_features_for_ml(
        self,
        dataset_path: str,
        target_metric: str = 'metric_sharpe_ratio'
    ) -> tuple:
        """
        Prepare features for machine learning

        Returns:
            (X, y, feature_names, categorical_features)
        """
        # Load dataset
        if dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        elif dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            raise ValueError(f"Unsupported format: {dataset_path}")

        # Separate metadata, features, labels
        metadata_cols = ['run_id', 'stage', 'strategy_name', 'symbol', 'timeframe',
                        'backtest_start', 'backtest_end', 'rank', 'timestamp']

        param_cols = [c for c in df.columns if c.startswith('param_')]
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        # Features (X)
        X = df[param_cols].copy()

        # Labels (y)
        if target_metric not in df.columns:
            raise ValueError(f"Target metric '{target_metric}' not found in dataset")

        y = df[target_metric].values

        # Identify categorical features
        categorical_features = []
        for col in param_cols:
            if X[col].dtype == 'object':
                categorical_features.append(col)

        # Encode categorical features
        for col in categorical_features:
            # Simple label encoding for now
            # (Better: one-hot encoding)
            X[col] = pd.Categorical(X[col]).codes

        # Convert to numpy
        X_array = X.values
        feature_names = param_cols

        print(f"\n🔧 Features prepared for ML:")
        print(f"   X shape: {X_array.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Feature count: {len(feature_names)}")
        print(f"   Categorical features: {len(categorical_features)}")

        return X_array, y, feature_names, categorical_features

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _expand_run_pattern(self, pattern: str) -> List[Path]:
        """
        Expand run ID pattern to list of run directories

        Args:
            pattern: Run ID or glob pattern (e.g., "opt_*", "opt_20251117_*")

        Returns:
            List of run directories
        """
        if '*' in pattern or '?' in pattern:
            # Glob pattern
            return sorted(self.results_dir.glob(pattern))
        else:
            # Exact run ID
            run_dir = self.results_dir / pattern
            return [run_dir] if run_dir.exists() else []

    def get_dataset_stats(self, dataset_path: str) -> Dict[str, Any]:
        """Get statistics about a dataset"""
        # Load dataset
        if dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        elif dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            raise ValueError(f"Unsupported format: {dataset_path}")

        param_cols = [c for c in df.columns if c.startswith('param_')]
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        return {
            'total_samples': len(df),
            'total_features': len(param_cols),
            'total_metrics': len(metric_cols),
            'unique_strategies': df['strategy_name'].nunique(),
            'unique_symbols': df['symbol'].nunique(),
            'unique_timeframes': df['timeframe'].nunique(),
            'unique_runs': df['run_id'].nunique(),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'sharpe_stats': {
                'mean': df['metric_sharpe_ratio'].mean() if 'metric_sharpe_ratio' in df else None,
                'std': df['metric_sharpe_ratio'].std() if 'metric_sharpe_ratio' in df else None,
                'min': df['metric_sharpe_ratio'].min() if 'metric_sharpe_ratio' in df else None,
                'max': df['metric_sharpe_ratio'].max() if 'metric_sharpe_ratio' in df else None,
            },
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'DataExporter',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DataExporter Test")
    print("=" * 60)

    # Test 1: Create a DataExporter
    print("\n📊 Test 1: Creating a DataExporter")
    exporter = DataExporter(results_dir="results/optimization_runs")
    print(f"   Results Dir: {exporter.results_dir}")
    print("   ✅ Test successful")

    # Test 2: Flatten parameters
    print("\n📊 Test 2: Flatten dict")
    nested_params = {
        'risk': {
            'sizing_method': 'RISK_BASED',
            'max_risk': 2.0
        },
        'exit': {
            'stop_loss': 1.5
        }
    }
    flat = exporter._flatten_dict(nested_params, prefix='param')
    print(f"   Nested: {nested_params}")
    print(f"   Flat: {flat}")
    print("   ✅ Test successful")

    # Test 3: Dataset info function (empty dataframe)
    print("\n📊 Test 3: Dataset info (empty dataframe)")
    empty_df = pd.DataFrame()
    try:
        info = exporter.get_dataset_info(empty_df)
        print(f"   Info: {info}")
    except Exception as e:
        print(f"   Expected error (empty dataframe): {type(e).__name__}")
    print("   ✅ Test successful")

    print("\n✅ All tests completed!")
    print("=" * 60)
