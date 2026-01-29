#!/usr/bin/env python3
"""
modules/simple_train/scripts/train_exit.py
SuperBot - Exit Model Training Script
Author: SuperBot Team
Date: 2026-01-15
Versiyon: 1.0.0

Exit Model training script.

Learns from rich labels:
- Optimal TP/SL multipliers
- Usage of trailing stop
- Usage of break even
- Expected trade duration

Usage:
    # Train the model (reads from the 'multi' folder)
    python -m modules.simple_train.scripts.train_exit --symbol BTCUSDT --years 2024 --multi
    
    # Save path
    python -m modules.simple_train.scripts.train_exit --multi --output data/ai/checkpoints/simple_train/simple_rsi/exit_model.pkl
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SIMPLE_TRAIN_ROOT = SCRIPT_DIR.parent
SUPERBOT_ROOT = SIMPLE_TRAIN_ROOT.parent.parent

if str(SUPERBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERBOT_ROOT))

# =============================================================================
# LOGGER SETUP
# =============================================================================

from core.logger_engine import get_logger
logger = get_logger("modules.simple_train.scripts.train_exit")

# =============================================================================
# IMPORTS
# =============================================================================

from modules.simple_train.models.exit_model import ExitModel, ExitLabelGenerator


# =============================================================================
# TRAINING
# =============================================================================

def train_exit_model(args):
    """Exit Model training."""
    
    # Config
    config_path = args.config or SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    symbol = args.symbol or config.get("data", {}).get("symbols", ["BTCUSDT"])[0]
    strategy = args.strategy or config.get("data", {}).get("strategy", "simple_rsi")
    
    # Multi folder
    prepared_dir = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train" / symbol / strategy
    
    if args.multi:
        prepared_dir = prepared_dir / "multi"
        if not prepared_dir.exists():
            logger.error(f"❌ Multi folder not found: {prepared_dir}")
            logger.info("   Before: python -m modules.simple_train.scripts.prepare_data --multi")
            return 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎯 Exit Model Training")
    logger.info("=" * 60)
    logger.info(f"📊 Symbol: {symbol}")
    logger.info(f"🎯 Strategy: {strategy}")
    logger.info(f"📁 Data: {prepared_dir}")
    logger.info("=" * 60)
    logger.info("")
    
    # Load multi parquet files
    years = args.years.split(",") if args.years else ["2024"]
    
    train_dfs = []
    val_dfs = []
    
    for year in years:
        train_file = prepared_dir / f"train_5m_{year}.parquet"
        val_file = prepared_dir / f"val_5m_{year}.parquet"
        
        if not train_file.exists():
            logger.warning(f"⚠️ {year}: train file not found")
            continue
        
        train_dfs.append(pd.read_parquet(train_file))
        val_dfs.append(pd.read_parquet(val_file))
        logger.info(f"   ✅ {year}: loaded")
    
    if not train_dfs:
        logger.error("❌ No data found")
        return 1
    
    # Combine
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
    
    logger.info(f"\n📦 Combined: Train={len(train_df)}, Val={len(val_df) if val_df is not None else 0}")
    
    # Rich label columns
    rich_label_cols = ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']
    
    # Check rich labels
    missing = [c for c in rich_label_cols if c not in train_df.columns]
    if missing:
        logger.error(f"❌ Missing rich label columns: {missing}")
        logger.info("   This data was not prepared in multi-mode. Run prepare_data with --multi.")
        return 1
    
    # Separate features and rich labels
    feature_cols = [c for c in train_df.columns if c not in ['label'] + rich_label_cols]
    
    X_train = train_df[feature_cols].values
    rich_labels_train = train_df[rich_label_cols]
    
    X_val = val_df[feature_cols].values if val_df is not None else None
    rich_labels_val = val_df[rich_label_cols] if val_df is not None else None
    
    logger.info(f"✨ Features: {len(feature_cols)} {feature_cols[:5]}...")
    logger.info(f"🏷️ Rich Labels: {rich_label_cols}")
    
    # Generate exit training labels
    logger.info("\n🏷️ Generating exit training labels...")
    
    # Strategy'den base TP/SL al
    exit_config = config.get("exit_model", {}).get("profiles", {}).get("balanced", {})
    base_tp = exit_config.get("tp_percent", 6.0)
    base_sl = exit_config.get("sl_percent", 3.2)
    
    label_gen = ExitLabelGenerator(base_tp=base_tp, base_sl=base_sl)
    
    y_train_dict = label_gen.generate(rich_labels_train)
    y_val_dict = label_gen.generate(rich_labels_val) if rich_labels_val is not None else None
    
    # Train Exit Model
    logger.info("\n🚀 Training Exit Model...")
    
    model_type = args.model_type or config.get("exit_model", {}).get("type", "xgboost")
    model = ExitModel(model_type=model_type)
    
    # Manually set feature names to ensure they are saved with the model
    model._feature_names = feature_cols
    
    model.fit(X_train, y_train_dict, X_val, y_val_dict)
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SUPERBOT_ROOT / "data" / "ai" / "checkpoints" / "simple_train" / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"exit_model_{timestamp}.pkl"
    
    model.save(str(output_path))
    
    # Save metadata
    metadata = {
        "symbol": symbol,
        "strategy": strategy,
        "model_type": model_type,
        "base_tp": base_tp,
        "base_sl": base_sl,
        "feature_names": feature_cols,
        "train_samples": len(X_train),
        "val_samples": len(X_val) if X_val is not None else 0,
        "created_at": datetime.now().isoformat()
    }
    
    metadata_path = output_path.parent / f"exit_metadata_{output_path.stem}.yaml"
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Model Training Completed!")
    logger.info("=" * 60)
    logger.info(f"💾 Model: {output_path}")
    logger.info(f"📋 Metadata: {metadata_path}")
    logger.info("=" * 60)
    
    # Create symlink for latest
    latest_path = output_path.parent / "exit_model.pkl"
    if latest_path.exists():
        latest_path.unlink()
    
    import shutil
    shutil.copy(output_path, latest_path)
    logger.info(f"🔗 Latest: {latest_path}")
    
    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SuperBot Simple Train - Exit Model Training"
    )
    
    parser.add_argument('--symbol', '-s', type=str, default=None,
                        help='Trading symbol (BTCUSDT)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Strategy name (simple_rsi)')
    parser.add_argument('--years', type=str, default="2024",
                        help='Years to train on (2024 or 2024,2025)')
    
    parser.add_argument('--model-type', type=str, default=None,
                        help='Model type (xgboost, lightgbm)')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path')
    
    parser.add_argument('--multi', action='store_true',
                        help='Use multi-output data (rich labels)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    return train_exit_model(args)


if __name__ == "__main__":
    sys.exit(main())
