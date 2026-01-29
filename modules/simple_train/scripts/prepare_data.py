#!/usr/bin/env python3
"""
modules/simple_train/scripts/prepare_data.py
SuperBot - Data Preparation Script
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Data preparation script - Feature extraction and label generation.
The prepared data is saved to parquet files and loaded by train.py.

Output structure:
    data/ai/prepared/simple_train/BTCUSDT/
    ├── train_5m_2025.parquet
    ├── val_5m_2025.parquet
    ├── test_5m_2025.parquet
    ├── metadata_5m_2025.yaml
    ├── train_4h_2017.parquet
    └── ...

Usage:
    # Prepare with all values from the config.
    python -m modules.simple_train.scripts.prepare_data

    # Override with CLI
    python -m modules.simple_train.scripts.prepare_data --symbol ETHUSDT
    python -m modules.simple_train.scripts.prepare_data --timeframe 1h

    # Specify the output folder
    python -m modules.simple_train.scripts.prepare_data --output data/prepared/custom
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

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
    logger = get_logger("modules.simple_train.scripts.prepare_data")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("modules.simple_train.scripts.prepare_data")


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_path: str | None = None) -> dict:
    """Loads the training.yaml configuration file."""
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
# DATA PREPARATION
# =============================================================================

def find_available_years(symbol: str, timeframe: str) -> list[int]:
    """
    Finds the available parquet years for the given symbol and timeframe.

    Args:
        symbol: Trading pair (BTCUSDT)
        timeframe: Base timeframe (5m)

    Returns:
        The list of available years is [2017, 2018, ..., 2025]
    """
    parquet_dir = SUPERBOT_ROOT / "data" / "parquets" / symbol
    if not parquet_dir.exists():
        return []

    # Pattern: BTCUSDT_5m_2017.parquet
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(parquet_dir.glob(pattern))

    years = []
    for f in files:
        # BTCUSDT_5m_2017.parquet -> 2017
        try:
            year_str = f.stem.split("_")[-1]
            year = int(year_str)
            years.append(year)
        except (ValueError, IndexError):
            continue

    return sorted(years)


def prepare_data(args, config: dict) -> int:
    """
    Prepares data and saves it to Parquet files.

    Multiple symbol and timeframe support:
    - --symbols BTCUSDT,ETHUSDT: Processes for each symbol
    - --timeframes 1h,4h: Processes for each timeframe
    - --start 2020 or 2020-01-01: Processes years starting from 2020

    Separate files for each symbol/timeframe/year:
    - train_1h_2025.parquet, val_1h_2025.parquet, test_1h_2025.parquet
    - metadata_1h_2025.yaml
    """
    from modules.simple_train.core import DataLoader, FeatureExtractor, Normalizer
    from modules.simple_train.models.entry_model import LabelGenerator

    # Get values from the config
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    exit_config = config.get("exit_model", {}).get("profiles", {}).get("balanced", {})

    # Parse symbols (comma-separated)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = data_config.get("symbols", ["BTCUSDT"])
        if isinstance(symbols, str):
            symbols = [symbols]

    # Parse timeframes (comma-separated)
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    else:
        timeframe_str = data_config.get("timeframe", "5m")
        timeframes = [tf.strip() for tf in timeframe_str.split(",")]

    warmup_bars = data_config.get("warmup_bars", 200)
    strategy_name = args.strategy or data_config.get("strategy", "simple_rsi")

    # TP/SL parameters
    tp_percent = exit_config.get("tp_percent", 6.0)
    sl_percent = exit_config.get("sl_percent", 3.0)
    timeout_bars = model_config.get("lookback_window", 100)

    logger.info("")
    logger.info("=" * 60)
    logger.info("📦 SuperBot Simple Train - Data Preparation")
    logger.info("=" * 60)
    logger.info(f"📊 Symbols: {symbols}")
    logger.info(f"⏱️ Timeframes: {timeframes}")
    logger.info(f"🎯 Strategy: {strategy_name}")
    logger.info(f"📈 TP: %{tp_percent} | SL: %{sl_percent}")
    logger.info("=" * 60)
    logger.info("")

    # Process for each symbol and timeframe
    total_success = 0
    total_tasks = 0

    for symbol in symbols:
        for timeframe in timeframes:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"🔄 Processing: {symbol} / {timeframe}")
            logger.info("=" * 60)

            # Start/End date belirle
            start_date = args.start_date or data_config.get("start_date", "2017-01-01")
            end_date = args.end_date or data_config.get("end_date", "2025-12-31")

            # Normalize the date range (convert to YYYY-MM-DD format)
            if len(start_date) == 4:  # If only the year is provided
                start_date = f"{start_date}-01-01"
            if len(end_date) == 4:
                end_date = f"{end_date}-12-31"

            # Check for the existence of Parquet files
            available_years = find_available_years(symbol, timeframe)
            if not available_years:
                logger.warning(f"⚠️ Parquet file not found: {symbol}/{timeframe}")
                continue

            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
            needed_years = [y for y in available_years if start_year <= y <= end_year]

            if not needed_years:
                logger.warning(f"⚠️ No data found for the specified date range: {symbol}/{timeframe}")
                continue

            logger.info(f"📅 Date Range: {start_date} - {end_date}")
            logger.info(f"📅 Using years: {needed_years}")

            # Process each year separately (creates separate files per year)
            for year in needed_years:
                total_tasks += 1
                year_start = f"{year}-01-01"
                year_end = f"{year}-12-31"
                
                logger.info(f"\n{'='*40}")
                logger.info(f"📅 {symbol} / {timeframe} / {year}")
                logger.info(f"{'='*40}")

                result = prepare_single_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=year_start,
                    end_date=year_end,
                    warmup_bars=warmup_bars,
                    strategy_name=strategy_name,
                    tp_percent=tp_percent,
                    sl_percent=sl_percent,
                    timeout_bars=timeout_bars,
                    training_config=training_config,
                    model_config=model_config,
                    args=args
                )

                if result == 0:
                    total_success += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Total Summary")
    logger.info("=" * 60)
    logger.info(f"   ✅ Success: {total_success}/{total_tasks} files")
    logger.info("=" * 60)

    return 0 if total_success > 0 else 1


def prepare_single_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    warmup_bars: int,
    strategy_name: str,
    tp_percent: float,       # Not used anymore - from strategy
    sl_percent: float,       # Not used anymore - from strategy
    timeout_bars: int,       # Not used anymore - from strategy
    training_config: dict,
    model_config: dict,
    args
) -> int:
    """Prepares data for a single symbol/timeframe/date range."""
    from modules.simple_train.core import DataLoader, FeatureExtractor, Normalizer
    from modules.simple_train.models.entry_model import LabelGenerator
    
    # ==========================================================================
    # 0. STRATEGY LOADING (Single Source of Truth)
    # ==========================================================================
    logger.info("🎯 [0/6] Loading strategy...")
    
    # Create a strategy instance
    import importlib.util
    strategy_path = SUPERBOT_ROOT / "components" / "strategies" / "templates" / f"{strategy_name}.py"
    
    if not strategy_path.exists():
        logger.error(f"❌ Strategy not found: {strategy_path}")
        return 1
    
    spec = importlib.util.spec_from_file_location(f"strategy_{strategy_name}", strategy_path)
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    strategy = strategy_module.Strategy()
    
    # TP/SL/Timeout parametrelerini strategy'den al (SINGLE SOURCE OF TRUTH!)
    tp_percent = strategy.exit_strategy.take_profit_percent
    sl_percent = strategy.exit_strategy.stop_loss_percent
    
    # Timeout: get from position_timeout (convert minutes to bar)
    # position_timeout: 1800 minutes = 30 hours
    # 5m timeframe: 30 hours = 360 bars
    if hasattr(strategy, 'position_management') and strategy.position_management.position_timeout_enabled:
        timeout_minutes = strategy.position_management.position_timeout
        # Convert to bar according to the timeframe
        if timeframe.endswith('m'):
            tf_minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            tf_minutes = int(timeframe[:-1]) * 60
        else:
            tf_minutes = 5  # Default
        timeout_bars = int(timeout_minutes / tf_minutes)
    else:
        # Fallback: from model config or default 100
        timeout_bars = model_config.get("lookback_window", 100)
    
    logger.info(f"   ✅ Strategy: {strategy_name}")
    logger.info(f"   📊 TP: {tp_percent}% | SL: {sl_percent}% | Timeout: {timeout_bars} bars")
    logger.info(f"   🎯 BE: {strategy.exit_strategy.break_even_enabled} ({strategy.exit_strategy.break_even_trigger_profit_percent}%)")
    logger.info(f"   🎯 PE: {strategy.exit_strategy.partial_exit_enabled} {strategy.exit_strategy.partial_exit_levels}")
    logger.info(f"   🎯 TS: {strategy.exit_strategy.trailing_stop_enabled} ({strategy.exit_strategy.trailing_activation_profit_percent}%)")

    # ==========================================================================
    # 1. DATA LOADING
    # ==========================================================================
    logger.info("📂 [1/6] Loading data...")

    data_loader = DataLoader()
    df = data_loader.load(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        warmup_bars=warmup_bars
    )

    if df is None or len(df) == 0:
        logger.warning(f"⚠️ Data could not be loaded: {start_date} - {end_date}")
        return 1

    logger.info(f"   ✅ {len(df)} bar loaded")

    # ==========================================================================
    # 2. FEATURE EXTRACTION
    # ==========================================================================
    logger.info("🔧 [2/6] Feature extraction...")

    feature_extractor = FeatureExtractor(strategy_name=strategy_name)
    features_df = feature_extractor.extract(df)

    # Keep only the feature columns (excluding OHLCV)
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in features_df.columns if c not in ohlcv_cols]
    features_only_df = features_df[feature_cols]

    if features_only_df is None or len(features_only_df) == 0:
        logger.error("❌ Feature extraction failed")
        return 1

    # Get feature names from DataFrame columns
    feature_names = list(features_only_df.columns)
    logger.info(f"   ✅ A total of {len(feature_names)} features were extracted")

    # ==========================================================================
    # 3. SIGNAL GENERATION (using production vectorized_conditions + features)
    # ==========================================================================
    logger.info("📊 [3/6] Creating strategy signals...")

    # features_df contains OHLCV data, while features_only_df contains only features.
    # OHLCV + features are required for signal generation.
    signals = generate_strategy_signals(df, strategy, features_df)
    if signals is None:
        logger.error("❌ Could not create signal")
        return 1

    long_signals_raw = (signals == 1).sum()
    short_signals_raw = (signals == -1).sum()
    total_signals_raw = long_signals_raw + short_signals_raw

    logger.info(f"   📊 Raw signals: {total_signals_raw} (LONG: {long_signals_raw}, SHORT: {short_signals_raw})")

    # ==========================================================================
    # 3.5. FILTER ACTIONABLE SIGNALS (BacktestEngine-based)
    # ==========================================================================
    logger.info("🔍 [3.5/6] Actionable signals filtering (via BacktestEngine)...")
    
    # A timestamp column is required for the BacktestEngine.
    # DataLoader timestamp is used as the index, let's convert it back to a column.
    df_with_timestamp = df.copy()
    if 'timestamp' not in df_with_timestamp.columns and 'open_time' not in df_with_timestamp.columns:
        # Add the index as a timestamp column
        df_with_timestamp['timestamp'] = df_with_timestamp.index
    
    # Filter with BacktestEngine (use the strategy instance)
    signals_filtered = filter_actionable_signals(
        df=df_with_timestamp,
        signals=signals,
        strategy=strategy,  # Already loaded strategy instance
        symbol=symbol,
        timeframe=timeframe
    )

    long_signals = (signals_filtered == 1).sum()
    short_signals = (signals_filtered == -1).sum()
    total_signals = long_signals + short_signals

    filtered_count = total_signals_raw - total_signals
    logger.info(f"   ✅ {total_signals} actionable signals (LONG: {long_signals}, SHORT: {short_signals})")
    logger.info(f"   🚫 {filtered_count} signals filtered (position was open)")
    
    # Use the filtered signals.
    signals = signals_filtered

    # ==========================================================================
    # 4. NORMALIZATION
    # ==========================================================================
    logger.info("📊 [4/6] Normalization...")

    normalizer = Normalizer()
    normalizer.fit(features_only_df, feature_names)
    features_normalized = normalizer.normalize(features_only_df, feature_names)

    logger.info(f"   ✅ Normalization completed")

    # ==========================================================================
    # 5. LABEL GENERATION
    # ==========================================================================
    logger.info("🏷️ [5/6] Label generation...")

    # --multi: Use RichLabelGenerator
    if getattr(args, 'multi', False):
        logger.info("   🎯 RichLabelGenerator (multi-output mode)")
        from modules.simple_train.models.rich_label_generator import RichLabelGenerator

        rich_gen = RichLabelGenerator()
        labels_df = rich_gen.generate(
            df=df,
            signals=signals,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            timeout_bars=timeout_bars,
            strategy=strategy
        )

        # labels_df columns: result, pnl_pct, exit_reason, bars_to_exit, max_favorable, max_adverse, peak_to_exit_ratio
        if len(labels_df.dropna(subset=['result'])) == 0:
            logger.error("❌ No labels could be created")
            return 1

        # Use the 'result' column as labels for binary compatibility
        labels = labels_df['result']

        # Statistics
        valid_labels = labels_df.dropna(subset=['result'])
        profitable = (valid_labels['result'] == 1).sum()
        losing = (valid_labels['result'] == 0).sum()
        win_rate = profitable / len(valid_labels) * 100 if len(valid_labels) > 0 else 0

        logger.info(f"   ✅ {len(valid_labels)} rich labels")
        logger.info(f"      Win: {win_rate:.1f}% | Avg PnL: {valid_labels['pnl_pct'].mean():.2f}%")
        logger.info(f"      Avg max_favorable: {valid_labels['max_favorable'].mean():.2f}%")
        logger.info(f"      Avg max_adverse: {valid_labels['max_adverse'].mean():.2f}%")
    
    # --own-backtest: Use TradeSimulator
    elif getattr(args, 'own_backtest', False):
        logger.info("   🎯 Label generation with TradeSimulator (own-backtest)")
        from modules.simple_train.backtest.trade_simulator import TradeSimulator

        simulator = TradeSimulator(strategy_name=strategy_name)
        results = simulator.simulate(df, signals, max_bars=timeout_bars)

        if len(results) == 0:
            logger.error("❌ No trade could be simulated")
            return 1

        # Convert the results to a label series.
        labels = pd.Series(index=df.index, dtype=float)
        labels[:] = np.nan

        for _, row in results.iterrows():
            signal_idx = row['signal_idx']
            if signal_idx in df.index:
                labels.loc[signal_idx] = row['label']

        signal_labels = labels.dropna()
        profitable = (signal_labels == 1).sum()
        losing = (signal_labels == 0).sum()
        win_rate = profitable / len(signal_labels) * 100

        logger.info(f"   ✅ {len(signal_labels)} label (Win: {win_rate:.1f}%)")
        logger.info(f"   📊 Exit reasons: {results['exit_reason'].value_counts().to_dict()}")
        
        labels_df = None  # No rich labels
    else:
        # Standard: Use LabelGenerator
        labeling_config = model_config.get("labeling", {})
        from modules.simple_train.models.entry_model import LabelGenerator
        label_generator = LabelGenerator(
            method=labeling_config.get("method", "trade_result"),
            config=labeling_config
        )

        labels = label_generator.generate(
            df=df,
            signals=signals,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            timeout_bars=timeout_bars,
            strategy=strategy  # Pass strategy for BE/PE/TS logic
        )

        signal_labels = labels.dropna()
        if len(signal_labels) == 0:
            logger.error("❌ No labels could be created")
            return 1

        profitable = (signal_labels == 1).sum()
        losing = (signal_labels == 0).sum()
        win_rate = profitable / len(signal_labels) * 100

        logger.info(f"   ✅ {len(signal_labels)} label (Win: {win_rate:.1f}%)")
        
        labels_df = None  # No rich labels

    # ==========================================================================
    # 6. SPLIT & SAVE
    # ==========================================================================
    logger.info("💾 [6/6] Split & Save...")

    # Align the indices - features_normalized and labels must have the same index.
    # Find common indices (rows that are both features and labels)
    valid_idx = features_normalized.index.intersection(labels.dropna().index)

    # Filter out rows that do not contain NaN values.
    features_clean = features_normalized.loc[valid_idx].dropna()
    valid_idx = features_clean.index

    X = features_clean.values
    y = labels.loc[valid_idx].values

    if len(X) == 0:
        logger.error("❌ No valid samples found (feature + label)")
        return 1

    logger.info(f"   Valid sample: {len(X)}")

    # Split ratios
    split_config = training_config.get("split", {})
    train_ratio = split_config.get("train_ratio", 0.7)
    val_ratio = split_config.get("val_ratio", 0.15)

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(f"   Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Output directory - structure based on symbol/strategy
    # data/ai/prepared/simple_train/BTCUSDT/simple_rsi/
    # --own-backtest: data/ai/prepared/simple_train/BTCUSDT/simple_rsi/own-backtest/
    # --multi: data/ai/prepared/simple_train/BTCUSDT/simple_rsi/multi/
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train" / symbol / strategy_name
        if getattr(args, 'own_backtest', False):
            output_dir = output_dir / "own-backtest"
        elif getattr(args, 'multi', False):
            output_dir = output_dir / "multi"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create file name
    # If it's a full year: train_1h_2025.parquet
    # If it's a custom range: train_1h_2025_01_05_2025_03_30.parquet
    start_year = start_date[:4]
    end_year = end_date[:4]
    
    # Full year check
    is_full_year = (start_date == f"{start_year}-01-01" and 
                    end_date == f"{end_year}-12-31" and 
                    start_year == end_year)
    
    if is_full_year:
        # Full year - basit format
        date_suffix = start_year
    else:
        # Custom range - detailed format
        start_str = start_date.replace("-", "_")
        end_str = end_date.replace("-", "_")
        date_suffix = f"{start_str}_{end_str}"
    
    tf_str = timeframe

    # Save as parquet files
    # --multi mode: Save additional rich label columns
    if labels_df is not None:
        # Rich labels are available - apply split on the labels_df DataFrame.
        # aligned rich labels with valid_idx
        rich_labels_aligned = labels_df.loc[valid_idx]
        
        # Split
        rich_train = rich_labels_aligned.iloc[:train_end]
        rich_val = rich_labels_aligned.iloc[train_end:val_end]
        rich_test = rich_labels_aligned.iloc[val_end:]
    
    # Train
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['label'] = y_train
    if labels_df is not None:
        # Add rich labels (pnl_pct, exit_reason, etc.)
        for col in ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']:
            if col in rich_train.columns:
                train_df[col] = rich_train[col].values
    train_filename = f"train_{tf_str}_{date_suffix}.parquet"
    train_df.to_parquet(output_dir / train_filename, index=False)

    # Val
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['label'] = y_val
    if labels_df is not None:
        for col in ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']:
            if col in rich_val.columns:
                val_df[col] = rich_val[col].values
    val_filename = f"val_{tf_str}_{date_suffix}.parquet"
    val_df.to_parquet(output_dir / val_filename, index=False)

    # Test
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['label'] = y_test
    if labels_df is not None:
        for col in ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']:
            if col in rich_test.columns:
                test_df[col] = rich_test[col].values
    test_filename = f"test_{tf_str}_{date_suffix}.parquet"
    test_df.to_parquet(output_dir / test_filename, index=False)

    # Metadata
    metadata_filename = f"metadata_{tf_str}_{date_suffix}.yaml"
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "date_range": f"{start_date} - {end_date}",
        "start_date": start_date,
        "end_date": end_date,
        "strategy": strategy_name,
        "tp_percent": tp_percent,
        "sl_percent": sl_percent,
        "feature_names": feature_names,  # Original features (NOT including rich labels)
        "total_signals": int(total_signals),
        "win_rate": float(win_rate),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "files": {
            "train": train_filename,
            "val": val_filename,
            "test": test_filename
        },
        "created_at": datetime.now().isoformat()
    }

    with open(output_dir / metadata_filename, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"   ✅ {date_suffix}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test | Win: {win_rate:.1f}%")

    return 0


def filter_actionable_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    strategy,  # Strategy instance
    symbol: str,
    timeframe: str
) -> pd.Series:
    """
    Filters actionable signals using the BacktestEngine.
    
    Performs a mini backtest and returns only the executed trades.
    Returns the entry times. This allows for features like BE, PE, trailing stop, etc.
    all backtest logic is automatically applied.
    
    Args:
        df: OHLCV DataFrame
        signals: Raw signals (1=LONG, -1=SHORT, 0=HOLD)
        strategy: Strategy instance (for BacktestEngine)
        symbol: Symbol name
        timeframe: Timeframe
    
    Returns:
        pd.Series: Filtered signals (only executed trades)
    """
    import asyncio
    from modules.backtest.backtest_engine import BacktestEngine
    
    # Configure the strategy - for the mini backtest.
    # Date range'i df'den al
    if 'timestamp' in df.columns:
        # Timestamp column varsa (ms cinsinden)
        start_date = pd.to_datetime(df['timestamp'].iloc[0], unit='ms')
        end_date = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')
    elif 'open_time' in df.columns:
        # open_time column varsa (ms cinsinden)
        start_date = pd.to_datetime(df['open_time'].iloc[0], unit='ms')
        end_date = pd.to_datetime(df['open_time'].iloc[-1], unit='ms')
    elif isinstance(df.index[0], pd.Timestamp):
        # Index datetime ise
        start_date = df.index[0]
        end_date = df.index[-1]
    else:
        # Fallback - extract timestamp from index
        # Assume index is RangeIndex, timestamp should be in a separate column.
        logger.error(f"❌ Datetime column not found in DataFrame! Columns: {df.columns.tolist()}")
        return signals  # Return raw signals
    
    # BacktestEngine instance
    logger.info(f"   🔄 Running mini backtest to filter signals...")
    engine = BacktestEngine(logger=logger, debug=False, enable_ai_logging=False)
    
    # Strategy config update (for mini backtest)
    strategy.backtest_start_date = start_date.strftime("%Y-%m-%dT%H:%M")
    strategy.backtest_end_date = end_date.strftime("%Y-%m-%dT%H:%M")
    
    # Async run
    async def run_backtest():
        result = await engine.run(strategy, use_cache=False)
        return result
    
    # Run backtest
    try:
        result = asyncio.run(run_backtest())
    except RuntimeError as e:
        # Already in event loop (Jupyter vs script)
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_backtest())
    
    # Extract entry times from trades
    filtered_signals = pd.Series(0, index=df.index)
    
    for trade in result.trades:
        # Find the trade entry time.
        entry_time = trade.entry_time
        
        # Find the index in the DataFrame that corresponds to this time.
        entry_time_ms = int(entry_time.timestamp() * 1000)
        
        if 'timestamp' in df.columns:
            # timestamp column varsa
            matching_idx = df[df['timestamp'] == entry_time_ms].index
        elif 'open_time' in df.columns:
            # open_time column varsa
            matching_idx = df[df['open_time'] == entry_time_ms].index
        elif isinstance(df.index[0], pd.Timestamp):
            # Index datetime ise
            matching_idx = df[df.index == entry_time].index
        else:
            # Fallback - find from the timestamp column
            logger.warning(f"⚠️ Trade entry time matching failed for {entry_time}")
            continue
        
        if len(matching_idx) > 0:
            idx = matching_idx[0]
            # Trade direction: LONG=1, SHORT=-1
            signal_value = 1 if trade.side.name == 'LONG' else -1
            filtered_signals.loc[idx] = signal_value
    
    logger.info(f"   ✅ Backtest completed: {len(result.trades)} executed trades")
    
    return filtered_signals


def generate_strategy_signals(df: pd.DataFrame, strategy, features_df: pd.DataFrame) -> pd.Series | None:
    """
    Generates a signal using a strategy.
    
    Uses the SAME logic as the backtest engine:
    - Uses already calculated features (no duplicate calculations)
    - Uses the vectorized_conditions module
    - Evaluates entry conditions (crossunder, crossover, etc.)
    
    Args:
        df: OHLCV DataFrame
        strategy: Strategy instance (already loaded)
        features_df: Feature DataFrame
    
    Returns:
        pd.Series: Signals (1=LONG, -1=SHORT, 0=HOLD)
    """
    try:
        # Add features to the DataFrame - features from FeatureExtractor
        data_with_indicators = df.copy()
        
        # Preserve OHLCV, add features.
        for col in features_df.columns:
            data_with_indicators[col] = features_df[col]
        
        # Use the SAME logic as the backtest engine: use vectorized_conditions
        from modules.backtest.vectorized_conditions import build_conditions_mask
        
        warmup = strategy.warmup_period
        
        # LONG signals
        long_conditions = strategy.entry_conditions.get('long', [])
        if long_conditions:
            long_mask = build_conditions_mask(
                long_conditions,
                data_with_indicators,
                warmup,
                logic='AND',
                debug=False
            )
        else:
            long_mask = pd.Series(False, index=data_with_indicators.index)
        
        # SHORT signals
        short_conditions = strategy.entry_conditions.get('short', [])
        if short_conditions:
            short_mask = build_conditions_mask(
                short_conditions,
                data_with_indicators,
                warmup,
                logic='AND',
                debug=False
            )
        else:
            short_mask = pd.Series(False, index=data_with_indicators.index)
        
        # Combine to single series: 1 = LONG, -1 = SHORT, 0 = None
        signals = pd.Series(0, index=df.index)
        signals[long_mask] = 1
        signals[short_mask] = -1
        
        return signals

    except Exception as e:
        logger.error(f"❌ Error creating signal: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def show_config(config: dict):
    """Displays configuration values."""
    data = config.get("data", {})
    exit_config = config.get("exit_model", {}).get("profiles", {}).get("balanced", {})

    # Symbols
    symbols = data.get('symbols', ['BTCUSDT'])
    if isinstance(symbols, str):
        symbols = [symbols]

    # Timeframes
    timeframe_str = data.get('timeframe', '5m')
    timeframes = [tf.strip() for tf in timeframe_str.split(",")]

    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 Configuration Values (training.yaml)")
    logger.info("=" * 60)
    logger.info(f"   Symbols: {symbols}")
    logger.info(f"   Strategy: {data.get('strategy', 'simple_rsi')}")
    logger.info(f"   Timeframes: {timeframes}")
    logger.info(f"   Start: {data.get('start_date', '?')}")
    logger.info(f"   End: {data.get('end_date', '?')}")
    logger.info(f"   Warmup: {data.get('warmup_bars', '?')} bars")
    logger.info(f"   TP: %{exit_config.get('tp_percent', '?')}")
    logger.info(f"   SL: %{exit_config.get('sl_percent', '?')}")
    logger.info("=" * 60)
    logger.info("")


def list_prepared(args):
    """Lists the prepared data folders."""
    base_dir = SUPERBOT_ROOT / "data" / "ai" / "prepared" / "simple_train"

    if not base_dir.exists():
        logger.info("📂 No prepared data found")
        return 0

    # Find symbol folders
    symbol_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not symbol_dirs:
        logger.info("📂 No prepared data found")
        return 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("📂 Prepared Data")
    logger.info("=" * 60)

    for symbol_dir in sorted(symbol_dirs):
        symbol = symbol_dir.name
        logger.info(f"   📁 {symbol}/")

        # There are strategy folders under each symbol
        strategy_dirs = [d for d in symbol_dir.iterdir() if d.is_dir()]
        
        if not strategy_dirs:
            # Old structure - directly from metadata files
            metadata_files = list(symbol_dir.glob("metadata_*.yaml"))
            if metadata_files:
                logger.info(f"      (old structure - strategy folder does not exist)")
                for meta_path in sorted(metadata_files):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = yaml.safe_load(f)
                        tf = meta.get('timeframe', '?')
                        year = meta.get('year', '?')
                        n_train = meta.get('train_samples', 0)
                        win_rate = meta.get('win_rate', 0)
                        logger.info(f"      📊 {tf}_{year}: {n_train} samples | Win={win_rate:.1f}%")
                    except Exception:
                        logger.info(f"      ⚠️ {meta_path.name} (could not be read)")
            continue

        # New structure - strategy based
        for strategy_dir in sorted(strategy_dirs):
            strategy_name = strategy_dir.name
            logger.info(f"      🎯 {strategy_name}/")
            
            # Find the metadata files for this strategy.
            metadata_files = list(strategy_dir.glob("metadata_*.yaml"))
            if not metadata_files:
                logger.info(f"         (no metadata)")
                continue

            for meta_path in sorted(metadata_files):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = yaml.safe_load(f)

                    tf = meta.get('timeframe', '?')
                    date_range = meta.get('date_range', '?')
                    n_train = meta.get('train_samples', 0)
                    win_rate = meta.get('win_rate', 0)

                    logger.info(f"         📊 {tf} {date_range}: {n_train} samples | Win={win_rate:.1f}%")
                except Exception:
                    logger.info(f"         ⚠️ {meta_path.name} (could not be read)")

    logger.info("=" * 60)
    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SuperBot Simple Train - Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reads from the configuration, can be overridden by the CLI.

Examples:
    # Prepare from config
    python -m modules.simple_train.scripts.prepare_data

    # Show the configuration.
    python -m modules.simple_train.scripts.prepare_data --show-config

    # List prepared files
    python -m modules.simple_train.scripts.prepare_data --list

    # Override with CLI (comma-separated)
    python -m modules.simple_train.scripts.prepare_data --symbols BTCUSDT,ETHUSDT
    python -m modules.simple_train.scripts.prepare_data --timeframes 1h,4h
    python -m modules.simple_train.scripts.prepare_data --symbols BTCUSDT,ETHUSDT --timeframes 1h,4h
    python -m modules.simple_train.scripts.prepare_data --start 2020 --end 2024
        """
    )

    # Override arguments
    parser.add_argument('--symbols', '-s', type=str, default=None,
                        help='Trading symbols (comma-separated: BTCUSDT,ETHUSDT)')
    parser.add_argument('--timeframes', '-t', type=str, default=None,
                        help='Timeframes (comma-separated: 1h,4h)')
    parser.add_argument('--start-date', '--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD or YYYY)')
    parser.add_argument('--end-date', '--end', type=str, default=None,
                        help='End date (YYYY-MM-DD or YYYY)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Strategy name (default: from config)')

    # Paths
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the configuration file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output folder path')

    # Flags
    parser.add_argument('--show-config', action='store_true',
                        help='Displays configuration values')
    parser.add_argument('--list', action='store_true',
                        help='List prepared files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Detailed output')
    parser.add_argument('--own-backtest', action='store_true',
                        help='Use your own backtest with TradeSimulator (instead of BacktestEngine)')
    parser.add_argument('--multi', action='store_true',
                        help='Multi-output model (rich labels: pnl_pct, exit_reason, max_favorable, etc.)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)

    if args.show_config:
        show_config(config)
        return 0

    if args.list:
        return list_prepared(args)

    return prepare_data(args, config)


if __name__ == "__main__":
    sys.exit(main())
