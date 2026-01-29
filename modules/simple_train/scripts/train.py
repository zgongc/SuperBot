#!/usr/bin/env python3
"""
modules/simple_train/scripts/train.py
SuperBot - Simple Train CLI Script
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Entry model training script.
Reads values from the config, can be overridden with the CLI.

Usage:
    # Train from prepared data (fast - recommended)
    python -m modules.simple_train.scripts.prepare_data
    python -m modules.simple_train.scripts.train --from-prepared

    # Full pipeline (prepare data + train)
    python -m modules.simple_train.scripts.train

    # Override with CLI
    python -m modules.simple_train.scripts.train --from-prepared --symbol ETHUSDT --year 2024
    python -m modules.simple_train.scripts.train --model-type lightgbm
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np

# =============================================================================
# PATH SETUP
# =============================================================================

SUPERBOT_ROOT = Path(__file__).parent.parent.parent.parent
SIMPLE_TRAIN_ROOT = Path(__file__).parent.parent

if str(SUPERBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERBOT_ROOT))

# =============================================================================
# LOGGER SETUP
# =============================================================================

try:
    from core.logger_engine import get_logger
    logger = get_logger("modules.simple_train.scripts.train")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("modules.simple_train.scripts.train")


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_path: str | None = None) -> dict:
    """
    Loads the training.yaml configuration file.

    Args:
        config_path: Path to the configuration file (None=default)

    Returns:
        dict: Config
    """
    if config_path is None:
        config_path = SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"⚠️ Configuration not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def find_available_years(symbol: str, base_tf: str, strategy: str = "simple_rsi", own_backtest: bool = False, multi: bool = False) -> list[str]:
    """Finds the existing years for the prepared data."""
    # New structure: data/ai/prepared/simple_train/BTCUSDT/simple_rsi/
    prepared_dir = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train" / symbol / strategy

    # Check the own-backtest subdirectory.
    if own_backtest:
        own_backtest_dir = prepared_dir / "own-backtest"
        if own_backtest_dir.exists():
            prepared_dir = own_backtest_dir
            
    # Check the multi subdirectory.
    if multi:
        multi_dir = prepared_dir / "multi"
        if multi_dir.exists():
            prepared_dir = multi_dir

    if not prepared_dir.exists():
        # Try the old structure (backward compatibility)
        prepared_dir_old = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train" / symbol
        if prepared_dir_old.exists():
            prepared_dir = prepared_dir_old
        else:
            return []

    years = []
    for meta_file in prepared_dir.glob(f"metadata_{base_tf}_*.yaml"):
        # metadata_5m_2024.yaml -> 2024
        year = meta_file.stem.split("_")[-1]
        years.append(year)

    return sorted(years)


def train_from_prepared(args, config: dict):
    """Trains using prepared parquet files."""
    from modules.simple_train.training import EntryTrainer

    data_config = config.get("data", {})

    # Get the parameters
    symbol = args.symbol or data_config.get("symbols", ["BTCUSDT"])[0]
    timeframe = args.timeframe or data_config.get("timeframe", "5m")
    strategy = args.strategy or data_config.get("strategy", "simple_rsi")

    # MTF: Get all timeframes
    timeframes = [tf.strip() for tf in timeframe.split(",")]
    base_tf = timeframes[0]
    extra_tfs = timeframes[1:]
    
    if extra_tfs:
        logger.info(f"🔄 MTF Mode: Base={base_tf}, Extras={extra_tfs}")

    # File paths (new structure: symbol/strategy/)
    prepared_dir = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train" / symbol / strategy

    # --own-backtest: Use the own-backtest subdirectory.
    if getattr(args, 'own_backtest', False):
        own_backtest_dir = prepared_dir / "own-backtest"
        if own_backtest_dir.exists():
            prepared_dir = own_backtest_dir
            logger.info(f"🎯 Using own-backtest data: {prepared_dir}")
        else:
            logger.error(f"❌ Own-backtest data not found: {own_backtest_dir}")
            logger.info("   Before: python -m modules.simple_train.scripts.prepare_data --own-backtest")
            return 1

    # --multi: use the multi subdirectory (rich labels)
    if getattr(args, 'multi', False):
        multi_dir = prepared_dir / "multi"
        if multi_dir.exists():
            prepared_dir = multi_dir
            logger.info(f"🎯 Using multi-output data: {prepared_dir}")
        else:
            logger.error(f"❌ Multi-output data not found: {multi_dir}")
            logger.info("   Before: python -m modules.simple_train.scripts.prepare_data --multi")
            return 1

    # Backward compatibility - try the old structure
    if not prepared_dir.exists():
        prepared_dir_old = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train" / symbol
        if prepared_dir_old.exists():
            logger.warning(f"⚠️ Using the old folder structure: {prepared_dir_old}")
            prepared_dir = prepared_dir_old
        else:
            logger.error(f"❌ Prepared data not found: {prepared_dir}")
            return 1

    # Determine the years
    own_backtest = getattr(args, 'own_backtest', False)
    multi = getattr(args, 'multi', False)
    available_years = find_available_years(symbol, base_tf, strategy, own_backtest=own_backtest, multi=multi)
    if not available_years:
        logger.error(f"❌ No prepared data: {symbol}/{base_tf}")
        return 1

    # Filtering with the --years parameter
    if args.years:
        if "-" in args.years and "," not in args.years:
            start_y, end_y = args.years.split("-")
            selected_years = [y for y in available_years if start_y <= y <= end_y]
        else:
            selected_years = [y.strip() for y in args.years.split(",")]
            selected_years = [y for y in selected_years if y in available_years]
    else:
        selected_years = available_years

    if not selected_years:
        logger.error(f"❌ The selected years are not available. Available: {available_years}")
        return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("🤖 SuperBot Simple Train - From Prepared Data (MTF)")
    logger.info("=" * 60)
    logger.info(f"📊 Symbol: {symbol}")
    logger.info(f"⏱️ Timeframes: {timeframes}")
    logger.info(f"📅 Years: {selected_years}")
    logger.info("=" * 60)

    # Load and merge data for each year
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    all_feature_names = []  # MTF feature names
    metadata = None

    for year in selected_years:
        # Load Base TF Data
        meta_path = prepared_dir / f"metadata_{base_tf}_{year}.yaml"
        if not meta_path.exists():
            logger.warning(f"   ⚠️ {year}: metadata not found ({base_tf})")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            year_meta = yaml.safe_load(f)

        if metadata is None:
            metadata = year_meta

        files = year_meta.get("files", {})
        
        # Load datasets
        try:
            # Base TF
            df_train_base = pd.read_parquet(prepared_dir / files.get("train", f"train_{base_tf}_{year}.parquet"))
            df_val_base = pd.read_parquet(prepared_dir / files.get("val", f"val_{base_tf}_{year}.parquet"))
            df_test_base = pd.read_parquet(prepared_dir / files.get("test", f"test_{base_tf}_{year}.parquet"))
            
            # Feature names from metadata - MULTI MODE FIX: Filter rich labels
            rich_label_cols = ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']
            base_features_raw = year_meta.get("feature_names", [])
            base_features = [f for f in base_features_raw if f not in rich_label_cols]
            
            # Timestamp column check (index to column)
            for df in [df_train_base, df_val_base, df_test_base]:
                if "timestamp" not in df.columns:
                    # Index'ten timestamp yarat (assume int64 index in prepare_data)
                    df["timestamp"] = df.index
            
            # --- MTF MERGE LOGIC ---
            if extra_tfs:
                # Initialize the base and feature lists
                current_train = df_train_base
                current_val = df_val_base
                current_test = df_test_base
                
                # Create the feature list only for the first year
                if not all_feature_names:
                    all_feature_names.extend(base_features)

                for tf in extra_tfs:
                    # Find extra TF files (e.g., train_15m_2024.parquet)
                    # Note: We are not checking the metadata file, we are relying on the convention.
                    extra_train_path = prepared_dir / f"train_{tf}_{year}.parquet"
                    extra_val_path = prepared_dir / f"val_{tf}_{year}.parquet"
                    extra_test_path = prepared_dir / f"test_{tf}_{year}.parquet"
                    
                    if not extra_train_path.exists():
                        logger.warning(f"   ⚠️ Missing {tf} data for {year}, skipping MTF merge for this TF")
                        continue
                        
                    # Load Extra TF
                    df_train_extra = pd.read_parquet(extra_train_path)
                    df_val_extra = pd.read_parquet(extra_val_path)
                    df_test_extra = pd.read_parquet(extra_test_path)
                    
                    # Prepare extra columns (rename with suffix)
                    suffix = f"_{tf}"
                    
                    # MTF MULTI MODE FIX: Filter rich labels
                    rich_label_cols = ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']
                    excluded_cols = ["label", "timestamp", "open", "high", "low", "close", "volume"] + rich_label_cols
                    extra_features = [c for c in df_train_extra.columns if c not in excluded_cols]
                    
                    # Create rename map
                    rename_map = {c: f"{c}{suffix}" for c in extra_features}
                    
                    # Add features to global list (only once)
                    if len(all_feature_names) == len(base_features): # First loop setup
                         all_feature_names.extend(rename_map.values())
                    elif len(all_feature_names) < len(base_features) + len(rename_map): # Subsequent loops
                         # This logic is imperfect for multiple years but generally safe if structure is constant
                         # Better: Check if already added
                         for new_feat in rename_map.values():
                             if new_feat not in all_feature_names:
                                 all_feature_names.append(new_feat)

                    # Rename and minimal columns
                    for d_extra in [df_train_extra, df_val_extra, df_test_extra]:
                        # Ensure timestamp
                        if "timestamp" not in d_extra.columns:
                            d_extra["timestamp"] = d_extra.index
                        # Rename features
                        d_extra.rename(columns=rename_map, inplace=True)
                    
                    # Select only timestamp + features to merge
                    cols_to_merge = ["timestamp"] + list(rename_map.values())
                    
                    # Merge using merge_asof (backward fill)
                    # Note: Both must be sorted by timestamp
                    current_train = pd.merge_asof(current_train.sort_values("timestamp"), 
                                                  df_train_extra[cols_to_merge].sort_values("timestamp"), 
                                                  on="timestamp", direction="backward")
                                                  
                    current_val = pd.merge_asof(current_val.sort_values("timestamp"), 
                                                df_val_extra[cols_to_merge].sort_values("timestamp"), 
                                                on="timestamp", direction="backward")
                                                
                    current_test = pd.merge_asof(current_test.sort_values("timestamp"), 
                                                 df_test_extra[cols_to_merge].sort_values("timestamp"), 
                                                 on="timestamp", direction="backward")
                
                # Update base DFs with merged result
                df_train_base = current_train
                df_val_base = current_val
                df_test_base = current_test
            
            else:
                # No MTF, just base features
                if not all_feature_names:
                    all_feature_names = base_features

            # Add to yearly list
            train_dfs.append(df_train_base)
            val_dfs.append(df_val_base)
            test_dfs.append(df_test_base)
            
            logger.info(f"   ✅ {year}: loaded (Cols: {len(df_train_base.columns)})")

        except Exception as e:
            logger.error(f"   ❌ Error loading {year}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not train_dfs:
        logger.error("❌ No training data could be loaded")
        return 1

    # Merge
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else None
    
    # Fill NaN from merge (if any early timestamps missing in higher TFs)
    train_df = train_df.ffill()
    train_df.fillna(0, inplace=True) # First rows
    if val_df is not None:
        val_df = val_df.ffill()
        val_df.fillna(0, inplace=True)
    if test_df is not None:
        test_df = test_df.ffill()
        test_df.fillna(0, inplace=True)

    # Separate features and labels
    # all_feature_names now includes MTF features as well
    # MULTI MODE FIX: Separate rich label columns from features.
    rich_label_cols = ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']
    
    # Only take the actual features (excluding rich labels)
    actual_features = [f for f in all_feature_names if f not in rich_label_cols]
    
    X_train = train_df[actual_features].values
    y_train = train_df["label"].values
    
    X_val = val_df[actual_features].values if val_df is not None else None
    y_val = val_df["label"].values if val_df is not None else None
    
    X_test = test_df[actual_features].values if test_df is not None else None
    y_test = test_df["label"].values if test_df is not None else None

    # Model type
    entry_config = config.get("entry_model", {})
    model_type = args.model_type or entry_config.get("type", "xgboost")
    strategy = metadata.get("strategy", "simple_rsi") if metadata else "simple_rsi"

    logger.info("")
    logger.info(f"📦 Combined: Train={len(X_train)}, Val={len(X_val) if X_val is not None else 0}")
    logger.info(f"✨ Features: {len(actual_features)} {actual_features[:5]}...")
    logger.info(f"🎯 Strategy: {strategy}")
    logger.info(f"🔧 Model: {model_type}")
    logger.info("=" * 60)
    logger.info("")

    # Create a trainer
    trainer = EntryTrainer(
        strategy_name=strategy,
        config_path=args.config,
        output_dir=args.output_dir
    )

    # Override the model type
    if args.model_type:
        trainer.config["entry_model"]["type"] = args.model_type
        trainer._model = None

    # Assign the data to the trainer
    trainer.X_train = X_train
    trainer.y_train = y_train
    trainer.X_val = X_val
    trainer.y_val = y_val
    trainer.X_test = X_test
    trainer.y_test = y_test
    trainer.feature_names = actual_features  # Excluding rich labels

    # Class Imbalance Handling
    if args.cih:
        logger.info("⚖️ Class Imbalance Handling enabled")
        trainer.model.enable_class_imbalance(trainer.y_train)

    # Train
    logger.info("🚀 Model training...")
    trainer.train()

    # ========================================================================
    # HARD EXAMPLE MINING (Learning from incorrect predictions)
    # ========================================================================
    if args.hem:
        logger.info("")
        logger.info("🎯 Hard Example Mining is starting...")
        
        # Make a prediction on validation
        y_val_pred = trainer.model.predict(trainer.X_val)
        
        # Find incorrect predictions
        wrong_mask = (y_val_pred != trainer.y_val)
        n_wrong = np.sum(wrong_mask)
        
        if n_wrong > 0:
            logger.info(f"   📊 Found incorrect predictions: {n_wrong}/{len(trainer.y_val)} ({n_wrong/len(trainer.y_val)*100:.1f}%)")
            
            # Remove incorrect examples
            X_hard = trainer.X_val[wrong_mask]
            y_hard = trainer.y_val[wrong_mask]
            
            # Add to the training set with 2 x weight (to teach hard examples)
            X_train_augmented = np.vstack([trainer.X_train, X_hard, X_hard])
            y_train_augmented = np.hstack([trainer.y_train, y_hard, y_hard])
            
            logger.info(f"   🔄 Retraining: {len(X_train_augmented)} samples (original: {len(trainer.X_train)})")
            
            # Update the Trainer.
            trainer.X_train = X_train_augmented
            trainer.y_train = y_train_augmented
            
            # Retrain
            trainer.train()
            logger.info("   ✅ Hard example mining completed")
        else:
            logger.info("   ℹ️  No incorrect predictions in validation, mining skipped")
        
        logger.info("")
    else:
        logger.info("ℹ️ Hard Example Mining is disabled (can be enabled with --hem)")
        logger.info("")

    # Evaluate
    metrics = trainer.evaluate()

    # Save
    model_path = None
    if not args.no_save:
        model_path = trainer.save()

    # Show the results
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Results")
    logger.info("=" * 60)

    test_metrics = metrics.get("test", {})
    logger.info(f"   Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    logger.info(f"   Test F1 Score: {test_metrics.get('f1_score', 0):.4f}")
    logger.info(f"   Test ROC AUC: {test_metrics.get('roc_auc', 0):.4f}")
    logger.info(f"   Test Precision: {test_metrics.get('precision', 0):.4f}")
    logger.info(f"   Test Recall: {test_metrics.get('recall', 0):.4f}")

    if model_path:
        logger.info(f"")
        logger.info(f"💾 Model: {model_path}")

    logger.info("=" * 60)
    return 0





def evaluate(args, config: dict):
    """Run evaluation only."""
    from modules.simple_train.training import EntryTrainer

    data_config = config.get("data", {})

    symbol = args.symbol or data_config.get("symbols", ["BTCUSDT"])[0]
    timeframe = args.timeframe or data_config.get("timeframe", "5m")
    start_date = args.start_date or data_config.get("start_date")
    end_date = args.end_date or data_config.get("end_date")
    strategy = args.strategy or "simple_rsi"

    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 SuperBot Simple Train - Evaluation")
    logger.info("=" * 60)

    trainer = EntryTrainer(strategy_name=strategy)

    if not trainer.load(args.model_path):
        logger.error(f"❌ Model could not be loaded: {args.model_path}")
        return 1

    # Load data
    if not trainer.load_data(symbol, timeframe, start_date, end_date):
        return 1

    # Signals and features
    trainer.generate_signals()
    trainer.prepare_features()
    trainer.generate_labels()
    trainer.split_data()

    # Evaluate
    metrics = trainer.evaluate()

    logger.info("")
    logger.info("📊 Evaluation Results:")
    for split, m in metrics.items():
        logger.info(f"   {split}: acc={m.get('accuracy', 0):.4f}, f1={m.get('f1_score', 0):.4f}")

    return 0


def show_config(config: dict):
    """Displays configuration values."""
    data = config.get("data", {})
    entry = config.get("entry_model", {})

    # MTF check
    timeframe = data.get('timeframe', '5m')
    timeframes = [t.strip() for t in timeframe.split(",")]
    is_mtf = len(timeframes) > 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 Configuration Values (training.yaml)")
    logger.info("=" * 60)
    logger.info(f"   Symbol: {data.get('symbols', ['?'])[0]}")
    logger.info(f"   Strategy: {data.get('strategy', 'simple_rsi')}")
    if is_mtf:
        logger.info(f"   Timeframes: {timeframes} (MTF)")
    else:
        logger.info(f"   Timeframe: {timeframe}")
    logger.info(f"   Start: {data.get('start_date', '?')}")
    logger.info(f"   End: {data.get('end_date', '?')}")
    logger.info(f"   Warmup: {data.get('warmup_bars', '?')} bars")
    logger.info(f"   Model: {entry.get('type', '?')}")
    logger.info(f"   Mode: {entry.get('mode', '?')}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("💡 Overridden via CLI: --symbol ETHUSDT --timeframe 1h")
    logger.info("")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SuperBot Simple Train - Entry Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reads from the configuration, can be overridden by the command-line interface.

Examples:
    # 1. First, prepare the data (all years)
    python -m modules.simple_train.scripts.prepare_data --symbol BTCUSDT

    # 2. Train on prepared data (combines all years)
    python -m modules.simple_train.scripts.train --from-prepared

    # Select specific years
    python -m modules.simple_train.scripts.train --from-prepared --years 2020-2024
    python -m modules.simple_train.scripts.train --from-prepared --years 2022,2023,2024

    # Full pipeline (prepare data + train)
    python -m modules.simple_train.scripts.train

    # Show the configuration
    python -m modules.simple_train.scripts.train --show-config
        """
    )

    # Override arguments (None = take from config)
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default=None,
        help='Trading symbol (default: from config)'
    )

    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        default=None,
        help='Timeframe (default: config\'den)'
    )

    parser.add_argument(
        '--start-date', '--start',
        type=str,
        default=None,
        help='Start date (default: from config)'
    )

    parser.add_argument(
        '--end-date', '--end',
        type=str,
        default=None,
        help='End date (default: from config)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Strategy name (default: simple_rsi)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['xgboost', 'lightgbm', 'lstm'],
        default=None,
        help='Model type (default: from config)'
    )

    # Paths
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config file path (default: configs/training.yaml)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output folder (default: from config)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the model to be loaded (for evaluation)'
    )

    parser.add_argument(
        '--years',
        type=str,
        default=None,
        help='Years: "2020,2021,2022" or "2020-2024" (default: all)'
    )



    # Flags
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show configuration values and exit'
    )

    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluation (no training)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Model\'i kaydetme'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Detailed output'
    )

    parser.add_argument(
        '--hem', '--retrain',
        action='store_true',
        help='Hard Example Mining (retraining with misclassifications)'
    )

    parser.add_argument(
        '--cih',
        action='store_true',
        help='Class Imbalance Handling (balance WIN/LOSE weights if imbalanced)'
    )

    parser.add_argument(
        '--own-backtest',
        action='store_true',
        help='Use own-backtest data (prepared with TradeSimulator)'
    )

    parser.add_argument(
        '--multi',
        action='store_true',
        help='Multi-output model (rich labels: pnl_pct, exit_reason, max_favorable, etc.)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Show config
    if args.show_config:
        show_config(config)
        return 0

    # Eval modu
    if args.eval_only:
        if not args.model_path:
            logger.error("❌ --model-path is required for --eval-only")
            return 1
        return evaluate(args, config)

    # By default, train using pre-existing data.
    return train_from_prepared(args, config)


if __name__ == "__main__":
    sys.exit(main())
