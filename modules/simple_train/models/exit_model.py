#!/usr/bin/env python3
"""
modules/simple_train/models/exit_model.py
SuperBot - Exit Model
Author: SuperBot Team
Date: 2026-01-15
Versiyon: 1.0.0

Exit Model - Dynamic Exit Parameter Optimization

Learns from rich labels:
- Optimal TP/SL multipliers (max_favorable, max_adverse'den)
- Usage of trailing stop (from exit_reason)
- Usage of break even
- Expected duration

Multi-output regression:
- tp_multiplier: 0.5-3.0 (How many times the base take profit?)
- sl_multiplier: 0.5-2.0 (How many times the base stop loss?)
- use_trailing: 0/1 (binary)
- use_break_even: 0/1 (binary)
- expected_bars: 1-200 (regression)

Usage:
    model = ExitModel(model_type="xgboost")
    model.fit(X_train, y_train_dict)  # y_train_dict: {'tp_mult': [...], 'sl_mult': [...], ...}
    predictions = model.predict(X_test)  # Dict of predictions
"""

from __future__ import annotations

import sys
from pathlib import Path
import pickle
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# PATH SETUP
# =============================================================================

SIMPLE_TRAIN_ROOT = Path(__file__).parent.parent
SUPERBOT_ROOT = SIMPLE_TRAIN_ROOT.parent.parent

if str(SUPERBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERBOT_ROOT))

# =============================================================================
# LOGGER SETUP
# =============================================================================

try:
    from core.logger_engine import get_logger
    logger = get_logger("modules.simple_train.models.exit_model")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.models.exit_model")


# =============================================================================
# EXIT MODEL
# =============================================================================

class ExitModel:
    """
    Exit Model - Dynamic Exit Parameter Optimization.
    
    Multi-output regression model:
    - Learns optimal TP/SL from rich labels (max_favorable, max_adverse)
    - Predicts when to use trailing stop (exit_reason analysis)
    - Estimates trade duration (bars_to_exit)
    
    Architecture: Multi-output XGBoost/LightGBM
    - Separate model for each output
    - Shared input features
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        config: dict | None = None
    ):
        """
        Initialize Exit Model.
        
        Args:
            model_type: Model type (xgboost, lightgbm)
            config: Model config
        """
        self.model_type = model_type
        self.config = config or {}
        
        # Multi-output models
        self.models: Dict[str, Any] = {}
        self.output_names = [
            'tp_multiplier',      # TP multiplier (regression, 0.5-3.0)
            'sl_multiplier',      # SL multiplier (regression, 0.5-2.0)
            'use_trailing',       # Use trailing stop (binary)
            'use_break_even',     # Use break even (binary)
            'expected_bars'       # Expected duration (regression)
        ]
        
        self.is_fitted = False
        self._feature_names = []
        
        logger.info(f"🎯 ExitModel initialized: type={model_type}")
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y_dict: Dict[str, np.ndarray],
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val_dict: Dict[str, np.ndarray] | None = None
    ) -> ExitModel:
        """
        Train multi-output model.
        
        Args:
            X: Training features (N, F)
            y_dict: Dict of targets {'tp_multiplier': array, 'sl_multiplier': array, ...}
            X_val: Validation features (optional)
            y_val_dict: Validation targets (optional)
            
        Returns:
            self (fluent interface)
        """
        # DataFrame -> numpy
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values
        else:
            # Only auto-generate feature names if not already set manually
            if not self._feature_names:
                self._feature_names = [f"f{i}" for i in range(X.shape[1])]
        
        logger.info(f"🎓 Training Exit Model: X.shape={X.shape}")
        
        # Train each output separately
        for output_name in self.output_names:
            if output_name not in y_dict:
                logger.warning(f"⚠️ Missing target: {output_name}, skipping")
                continue
            
            y = y_dict[output_name]
            y_val = y_val_dict.get(output_name) if y_val_dict else None
            
            logger.info(f"   📊 Training {output_name}...")
            
            # Create model for this output
            if self.model_type == "xgboost":
                self.models[output_name] = self._create_xgboost_model(output_name)
            elif self.model_type == "lightgbm":
                self.models[output_name] = self._create_lightgbm_model(output_name)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Fit
            if output_name in ['use_trailing', 'use_break_even']:
                # Binary classification
                self._fit_classifier(self.models[output_name], X, y, X_val, y_val, output_name)
            else:
                # Regression
                self._fit_regressor(self.models[output_name], X, y, X_val, y_val, output_name)
        
        self.is_fitted = True
        logger.info("✅ Exit Model training complete")
        return self
    
    def _create_xgboost_model(self, output_name: str):
        """Create XGBoost model (classifier or regressor)."""
        import xgboost as xgb
        
        if output_name in ['use_trailing', 'use_break_even']:
            # Binary classifier
            return xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            )
        else:
            # Regressor
            return xgb.XGBRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            )
    
    def _create_lightgbm_model(self, output_name: str):
        """Create LightGBM model (classifier or regressor)."""
        import lightgbm as lgb
        
        if output_name in ['use_trailing', 'use_break_even']:
            # Binary classifier
            return lgb.LGBMClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            # Regressor
            return lgb.LGBMRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
    
    def _fit_classifier(self, model, X, y, X_val, y_val, name):
        """Fit binary classifier."""
        # Ensure binary labels
        y = (y > 0.5).astype(int)
        if y_val is not None:
            y_val = (y_val > 0.5).astype(int)
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X, y)
        
        # Log accuracy
        train_acc = model.score(X, y)
        logger.info(f"      {name}: Train Acc={train_acc:.3f}")
    
    def _fit_regressor(self, model, X, y, X_val, y_val, name):
        """Fit regressor."""
        if y_val is not None:
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X, y)
        
        # Log R²
        train_r2 = model.score(X, y)
        logger.info(f"      {name}: Train R²={train_r2:.3f}")
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict exit parameters.
        
        Args:
            X: Features (N, F)
            
        Returns:
            Dict of predictions {
                'tp_multiplier': array,
                'sl_multiplier': array,
                'use_trailing': array (binary),
                'use_break_even': array (binary),
                'expected_bars': array
            }
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        # DataFrame -> numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = {}
        for output_name, model in self.models.items():
            if output_name in ['use_trailing', 'use_break_even']:
                # Binary prediction
                predictions[output_name] = model.predict(X)
            else:
                # Regression prediction
                preds = model.predict(X)
                
                # Clip to valid ranges
                if output_name == 'tp_multiplier':
                    preds = np.clip(preds, 0.5, 3.0)
                elif output_name == 'sl_multiplier':
                    preds = np.clip(preds, 0.5, 2.0)
                elif output_name == 'expected_bars':
                    preds = np.clip(preds, 1, 200)
                
                predictions[output_name] = preds
        
        return predictions
    
    def save(self, path: str):
        """Save model to file."""
        save_data = {
            'model_type': self.model_type,
            'config': self.config,
            'models': self.models,
            'output_names': self.output_names,
            'feature_names': self._feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"💾 Exit Model saved: {path}")
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model_type = save_data['model_type']
        self.config = save_data['config']
        self.models = save_data['models']
        self.output_names = save_data['output_names']
        self._feature_names = save_data['feature_names']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"✅ Exit Model loaded: {path}")


# =============================================================================
# EXIT LABEL GENERATOR
# =============================================================================

class ExitLabelGenerator:
    """
    Generate training labels for Exit Model from rich labels.
    
    Uses Rich Labels:
    - max_favorable: Highest profit seen
    - max_adverse: Deepest loss seen
    - exit_reason: How trade closed
    - bars_to_exit: Trade duration
    - pnl_pct: Final PnL
    
    Generates:
    - tp_multiplier: max_favorable / base_tp (how much TP could have been)
    - sl_multiplier: abs(max_adverse) / base_sl (how much SL was needed)
    - use_trailing: 1 if exit_reason=='TRAILING_STOP', else 0
    - use_break_even: 1 if max_adverse < -1% but pnl_pct > 0 (BE would help)
    - expected_bars: bars_to_exit
    """
    
    def __init__(self, base_tp: float = 6.0, base_sl: float = 3.2):
        """
        Initialize Exit Label Generator.
        
        Args:
            base_tp: Base TP percent (default from strategy)
            base_sl: Base SL percent
        """
        self.base_tp = base_tp
        self.base_sl = base_sl
        logger.info(f"🏷️ ExitLabelGenerator: base_tp={base_tp}%, base_sl={base_sl}%")
    
    def generate(self, rich_labels_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate exit training labels from rich labels.
        
        Args:
            rich_labels_df: DataFrame with rich labels
                Columns: pnl_pct, exit_reason, max_favorable, max_adverse, bars_to_exit
                
        Returns:
            Dict of labels {'tp_multiplier': array, 'sl_multiplier': array, ...}
        """
        n = len(rich_labels_df)
        
        # TP Multiplier: max_favorable / base_tp
        # Example: max_favorable=9% -> 9/6 = 1.5 x (We could have increased the TP by a factor of 1.5)
        tp_mult = (rich_labels_df['max_favorable'] / self.base_tp).clip(0.5, 3.0).values
        
        # SL Multiplier: abs(max_adverse) / base_sl
        # Example: max_adverse=-5% -> 5/3.2 = 1.56 x (SL needs to be widened)
        sl_mult = (np.abs(rich_labels_df['max_adverse']) / self.base_sl).clip(0.5, 2.0).values
        
        # Use Trailing: 1 if exit_reason contains 'TRAILING'
        use_trailing = (rich_labels_df['exit_reason'] == 'TRAILING_STOP').astype(int).values
        
        # Use Break Even: 1 if went negative but closed positive (BE would save)
        went_negative = rich_labels_df['max_adverse'] < -1.0  # Went down >1%
        closed_positive = rich_labels_df['pnl_pct'] > 0
        use_break_even = (went_negative & closed_positive).astype(int).values
        
        # Expected bars
        expected_bars = rich_labels_df['bars_to_exit'].clip(1, 200).values
        
        logger.info(f"✅ Exit labels generated: {n} samples")
        logger.info(f"   TP mult: {tp_mult.mean():.2f} ± {tp_mult.std():.2f}")
        logger.info(f"   SL mult: {sl_mult.mean():.2f} ± {sl_mult.std():.2f}")
        logger.info(f"   Use Trailing: {use_trailing.mean()*100:.1f}%")
        logger.info(f"   Use BE: {use_break_even.mean()*100:.1f}%")
        
        return {
            'tp_multiplier': tp_mult,
            'sl_multiplier': sl_mult,
            'use_trailing': use_trailing,
            'use_break_even': use_break_even,
            'expected_bars': expected_bars
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Exit Model Test")
    print("=" * 60)
    
    # Dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    
    # Dummy rich labels
    rich_labels = pd.DataFrame({
        'pnl_pct': np.random.uniform(-3, 6, n_samples),
        'exit_reason': np.random.choice(['TP', 'SL', 'TRAILING_STOP', 'TIMEOUT'], n_samples),
        'max_favorable': np.random.uniform(0, 12, n_samples),
        'max_adverse': np.random.uniform(-6, 0, n_samples),
        'bars_to_exit': np.random.randint(5, 100, n_samples)
    })
    
    # Generate exit labels
    label_gen = ExitLabelGenerator(base_tp=6.0, base_sl=3.2)
    y_dict = label_gen.generate(rich_labels)
    
    # Train model
    model = ExitModel(model_type="xgboost")
    model.fit(X, y_dict)
    
    # Predict
    predictions = model.predict(X[:5])
    
    print("\n📊 Predictions (first 5):")
    for name, values in predictions.items():
        print(f"   {name}: {values}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
