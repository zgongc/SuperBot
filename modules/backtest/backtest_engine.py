#!/usr/bin/env python3
"""
modules/backtest/backtest_engine.py
SuperBot - Backtest Engine V3
Yazar: SuperBot Team
Tarih: 2025-11-16
Versiyon: 3.0.0

Vectorized backtest engine using existing components.

CRITICAL: Optimizer-friendly design!
- Strategy object based (not path)
- Data caching support
- Uses existing manager components
- Fast vectorized execution

Features:
- Uses existing managers (ParquetsEngine, IndicatorManager, etc.)
- Multi-timeframe support
- Position sizing works correctly
- Metrics are calculated correctly
- Optimized for the optimizer

Usage:
    from modules.backtest.backtest_engine import BacktestEngine

    engine = BacktestEngine(logger)
    result = await engine.run(strategy, use_cache=True)

Dependencies:
    - python>=3.10
    - pandas>=2.0.0
    - numpy>=1.24.0
    - components/managers/*
    - components/strategies/*
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import time
import warnings
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import pandas as pd
import numpy as np

# Core
from core.logger_engine import get_logger
from core.config_engine import get_config

# Managers
from components.managers.parquets_engine import ParquetsEngine
from components.strategies.risk_manager import RiskManager
from components.strategies.position_manager import PositionManager
from components.indicators.indicator_manager import IndicatorManager
from components.strategies.strategy_executor import StrategyExecutor

# Note: TP/SL indicators (ATR, SwingPoints, FibonacciRetracement) are now used
# via ExitManager, not directly imported here

# Backtest modules
from modules.backtest.backtest_types import (
    BacktestConfig, BacktestResult, Trade, PositionSide, ExitReason
)
from modules.backtest.backtest_config import build_config, get_cache_key
from modules.backtest.backtest_metrics import calculate_metrics

# Strategy enums
from components.strategies.base_strategy import ExitMethod, StopLossMethod

if TYPE_CHECKING:
    from components.strategies.base_strategy import Strategy


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_atr_value_from_row(row: pd.Series) -> Optional[float]:
    """
    Get the ATR value from the row. If 'atr' is not present, search for the 'atr_xx' pattern.

    Args:
        row: DataFrame row (pd.Series)

    Returns:
        ATR value or None

    Logic:
        1. If 'atr' exists, use it directly.
        2. Otherwise, search for patterns like 'atr_14', 'atr_20'.
        3. Return the first one you find
    """
    # First try 'atr'
    if 'atr' in row.index:
        return row.get('atr')

    # Try atr_xx pattern
    import re
    atr_pattern = re.compile(r'^atr_\d+$')

    for col in row.index:
        if atr_pattern.match(col):
            return row.get(col)

    return None


# ============================================================================
# BACKTEST ENGINE V3
# ============================================================================

class BacktestEngine:
    """
    Vectorized backtest engine using existing components

    Design Principles:
    1. Strategy object based (optimizer creates strategy)
    2. Uses existing manager components (no reinventing wheel)
    3. Data caching for fast optimizer runs
    4. Accurate position sizing and metrics
    """

    def __init__(self, logger=None, debug: bool = False, enable_ai_logging: bool = True):
        """
        Initialize BacktestEngine

        Args:
            logger: Logger instance (optional, defaults to core logger)
            debug: Debug mode flag (controls verbose output)
            enable_ai_logging: Enable AI training data collection (default: True)
        """
        self.debug = debug
        self.logger = logger or get_logger("modules.backtest.engine")
        self.config_engine = get_config()

        # Components (initialized once)
        # ParquetsEngine uses config for data path
        self.parquets_engine = ParquetsEngine(config_engine=self.config_engine)
        self.risk_manager = RiskManager(logger=self.logger)
        # NOTE: PositionManager is created per-run with strategy (see _execute_backtest)

        # AI Predictor (lazy loaded when needed)
        self._ai_predictor = None
        self._ai_predictions_cache: Optional[pd.DataFrame] = None
        
        # Exit Model (lazy loaded when needed)
        self._exit_model = None

        # Cache (for optimizer)
        self._cached_data: Optional[Dict[str, pd.DataFrame]] = None
        self._cache_key: Optional[str] = None

        # Counters
        self.trade_counter = 0

        # AI Logging Components
        self.enable_ai_logging = enable_ai_logging
        self.future_logger = None
        self.feature_extractor = None

        if self.enable_ai_logging:
            try:
                from components.ai.future_logger import FutureLogger
                from components.ai.feature_extractor import FeatureExtractor

                self.future_logger = FutureLogger(
                    config={
                        'output_dir': 'data/ai/features',
                        'mode': 'backtest',  # Backtest: single file, all symbols
                        'batch_size': 1000,  # Backtest is usually fast, large batch
                    },
                    logger=self.logger
                )
                self.feature_extractor = FeatureExtractor(logger=self.logger)
                self.logger.info("✅ AI data collection is active (FutureLogger)")
            except ImportError as e:
                self.logger.warning(f"⚠️  AI modules could not be loaded: {e}")
                self.enable_ai_logging = False

        # Log initialization
        self.logger.info("🚀 BacktestEngine V3 started")
        if self.debug:
            self.logger.info("⚠️  DEBUG MODE ENABLED")
            self.logger.info("=" * 60)

    async def run(
        self,
        strategy: Strategy,
        use_cache: bool = True
    ) -> BacktestResult:
        """
        Run backtest for strategy

        Args:
            strategy: Strategy instance (optimizer creates this)
            use_cache: Reuse cached data if config matches

        Returns:
            BacktestResult: Complete backtest result
        """
        start_time = time.time()

        # Override debug mode from strategy (if optimizer set it)
        if hasattr(strategy, 'debug'):
            self.debug = strategy.debug

        self.logger.info("=" * 60)
        self.logger.info("🚀 Backtest is starting...")
        self.logger.info("=" * 60)

        # 1. Build config from strategy
        config = build_config(strategy)
        cache_key = get_cache_key(config)

        self.logger.info(f"📊 Strategy: {config.strategy_name}")
        self.logger.info(f"📈 Symbol: {config.symbols[0]}")
        self.logger.info(f"⏰ Primary TF: {config.primary_timeframe}")
        if len(config.mtf_timeframes) > 1:
            self.logger.info(f"⏰ MTF: {', '.join(config.mtf_timeframes)}")
        self.logger.info(f"📅 Period: {config.start_date.date()} → {config.end_date.date()}")

        # Update FutureLogger config (symbol + timeframe for filename)
        if self.enable_ai_logging and self.future_logger:
            self.future_logger.backtest_symbol = config.symbols[0]
            self.future_logger.backtest_timeframe = config.primary_timeframe

        # 2. Load data (with caching)
        # Cache invalidate check (sets when validator ATR is added)
        cache_invalidated = getattr(strategy, '_cache_invalidated', False)

        if use_cache and cache_key == self._cache_key and self._cached_data and not cache_invalidated:
            self.logger.info("✅ Using cached data")
            mtf_data = self._cached_data
        else:
            if cache_invalidated:
                self.logger.info("🔄 Cache invalidated (indicator added), data is being reloaded...")
            else:
                self.logger.info("📂 Data is loading...")
            mtf_data = await self._load_data(config)
            self._cached_data = mtf_data
            self._cache_key = cache_key
            # Reset flag
            if hasattr(strategy, '_cache_invalidated'):
                strategy._cache_invalidated = False

        # 3. Calculate indicators (MTF)
        self.logger.info("📊 Indicators are being calculated...")
        indicators_mtf = self._calculate_indicators(mtf_data, strategy, config)

        # 4. Generate signals (vectorized with MTF support)
        self.logger.info("🎯 Signals are being generated...")
        long_mask, short_mask = self._generate_signals(
            mtf_data[config.primary_timeframe],
            indicators_mtf,
            strategy,
            config.primary_timeframe
        )

        # 5. Initialize StrategyExecutor for exit logic (with AI predictor for DYNAMIC_AI exits)
        strategy_executor = StrategyExecutor(strategy, logger=self.logger, ai_predictor=self._ai_predictor)

        # 6. Simulate positions
        self.logger.info("💼 Positions are being simulated...")
        trades = self._simulate_positions(
            long_mask,
            short_mask,
            mtf_data[config.primary_timeframe],
            indicators_mtf,
            strategy,
            strategy_executor,
            config
        )

        # 6. Calculate metrics
        self.logger.info("📈 Metrics are being calculated...")
        metrics = calculate_metrics(trades, config)

        # 7. Build equity curve
        equity_curve = self._build_equity_curve(trades, config)

        # 8. Flush AI logger (save remaining data)
        if self.enable_ai_logging and self.future_logger:
            try:
                if self.future_logger.completed_trades:
                    self.future_logger._save_batch()
                    self.logger.info(f"✅ AI data saved: {self.future_logger.total_saved} trade")
            except Exception as e:
                self.logger.debug(f"⚠️  AI data flush error: {e}")

        # Build result
        execution_time = time.time() - start_time
        result = BacktestResult(
            config=config,
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            execution_time_seconds=execution_time,
        )

        if self.debug:
            self.logger.info("=" * 60)
            self.logger.info("✅ Backtest completed!")
            self.logger.info(f"⏱️  Duration: {execution_time:.2f}s")
            self.logger.info(f"💼 Trade Count: {len(trades)}")
            self.logger.info(f"💰 Return: {metrics.total_return_pct:+.2f}%")
            self.logger.info(f"📊 Profit Factor: {metrics.profit_factor:.2f}")
            self.logger.info(f"📈 Win Rate: {metrics.win_rate:.1f}%")
            self.logger.info("=" * 60)

        return result

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    async def _load_data(self, config: BacktestConfig) -> Dict[str, pd.DataFrame]:
        """
        Load multi-timeframe data

        Args:
            config: Backtest config

        Returns:
            Dict[str, pd.DataFrame]: {timeframe: data}
        """
        mtf_data = {}

        for timeframe in config.mtf_timeframes:
            # Load data for this timeframe
            data = await self.parquets_engine.get_historical_data(
                symbol=config.symbols[0],  # Single symbol for now
                timeframe=timeframe,
                start_date=config.start_date.isoformat(),
                end_date=config.end_date.isoformat(),
                warmup_candles=config.warmup_period,
                utc_offset=0
            )

            mtf_data[timeframe] = data

            if self.debug:
                self.logger.info(f"   ✅ {timeframe}: {len(data):,} mum")

        return mtf_data

    # ========================================================================
    # INDICATORS
    # ========================================================================

    def _calculate_indicators(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        strategy: Strategy,
        config: BacktestConfig
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate indicators for all timeframes

        Args:
            mtf_data: Multi-timeframe data
            strategy: Strategy instance
            config: BacktestConfig

        Returns:
            Dict[str, Dict[str, np.ndarray]]: {timeframe: {indicator_name: values}}
        """
        from components.strategies.helpers.strategy_indicator_bridge import create_indicator_manager_from_strategy

        # MTF indicators: {timeframe: {indicator: values}}
        indicators_mtf = {}

        # Get primary timeframe data for resampling
        primary_data = mtf_data[config.primary_timeframe]
        primary_index = primary_data.index

        # Calculate indicators for each timeframe
        for timeframe, tf_data in mtf_data.items():
            # Create IndicatorManager from strategy (using bridge helper)
            # Pass logger only in debug mode for verbose indicator loading logs
            indicator_logger = self.logger if self.debug else None
            indicator_manager = create_indicator_manager_from_strategy(
                strategy=strategy,
                logger=indicator_logger
            )

            # Calculate all indicators using calculate_batch
            raw_indicator_results = {}
            for indicator_name, indicator_instance in indicator_manager.indicators.items():
                try:
                    # Special handling for aisignal - use MTF prediction
                    if indicator_name == 'aisignal' and hasattr(indicator_instance, 'calculate_batch_mtf'):
                        # Only calculate on primary timeframe with all MTF data
                        if timeframe == config.primary_timeframe:
                            result = indicator_instance.calculate_batch_mtf(mtf_data)
                        else:
                            continue  # Skip aisignal on non-primary timeframes
                    # Use calculate_batch for vectorized calculation
                    elif hasattr(indicator_instance, 'calculate_batch'):
                        result = indicator_instance.calculate_batch(tf_data)
                    elif hasattr(indicator_instance, 'calculate'):
                        result = indicator_instance.calculate(tf_data)
                    else:
                        continue

                    # Store result (can be Series or DataFrame)
                    if isinstance(result, pd.Series):
                        # Align to primary timeframe if needed (merge on timestamp)
                        if timeframe != config.primary_timeframe:
                            result = self._align_to_primary(result, primary_index, mtf_data=tf_data, primary_data=primary_data)
                        raw_indicator_results[indicator_name] = result
                    elif isinstance(result, pd.DataFrame):
                        # Multiple outputs (e.g., MACD has macd, signal, histogram)
                        # Convert DataFrame to dict of Series for bridge formatting
                        result_dict = {}
                        for col in result.columns:
                            col_series = result[col]
                            if timeframe != config.primary_timeframe:
                                col_series = self._align_to_primary(col_series, primary_index, mtf_data=tf_data, primary_data=primary_data)
                            result_dict[col] = col_series
                        raw_indicator_results[indicator_name] = result_dict
                    elif isinstance(result, dict):
                        # Dict of Series - align each series
                        result_dict = {}
                        for key, series in result.items():
                            if hasattr(series, 'values'):
                                if timeframe != config.primary_timeframe:
                                    series = self._align_to_primary(series, primary_index, mtf_data=tf_data, primary_data=primary_data)
                                result_dict[key] = series
                            else:
                                result_dict[key] = series
                        raw_indicator_results[indicator_name] = result_dict
                except Exception as e:
                    if self.debug:
                        self.logger.warning(f"   ⚠️  {timeframe}/{indicator_name} could not be calculated: {e}")

            # Prepare OHLCV data for MTF (align if needed)
            ohlcv_data_aligned = tf_data.copy() if timeframe == config.primary_timeframe else None
            if timeframe != config.primary_timeframe:
                # Align OHLCV to primary timeframe
                ohlcv_data_aligned = pd.DataFrame(index=primary_index)
                for ohlcv_col in ['open', 'high', 'low', 'close', 'volume']:
                    if ohlcv_col in tf_data.columns:
                        ohlcv_data_aligned[ohlcv_col] = self._align_to_primary(tf_data[ohlcv_col], primary_index, mtf_data=tf_data, primary_data=primary_data)

            # Apply bridge formatting (smart aliasing)
            from components.strategies.helpers.strategy_indicator_bridge import format_indicator_results_for_strategy
            formatted_results = format_indicator_results_for_strategy(raw_indicator_results, timeframe, ohlcv_data=ohlcv_data_aligned)

            # Convert Series to numpy arrays for backtest engine
            indicators_dict = {}
            for key, value in formatted_results.items():
                if hasattr(value, 'values'):
                    indicators_dict[key] = value.values
                else:
                    indicators_dict[key] = value

            indicators_mtf[timeframe] = indicators_dict

            if self.debug:
                self.logger.info(f"   ✅ {timeframe}: {len(indicators_dict)} indicators calculated")
                self.logger.info(f"      📋 Keys: {list(indicators_dict.keys())}")

        return indicators_mtf

    def _align_to_primary(
        self,
        series: pd.Series,
        primary_index: pd.Index,
        mtf_data: pd.DataFrame = None,
        primary_data: pd.DataFrame = None
    ) -> pd.Series:
        """
        Align MTF indicator to primary timeframe using merge_asof

        For each primary timeframe candle, takes the MOST RECENT value
        from the MTF indicator (backward fill).

        Example:
            5m data:  09:00, 09:05, 09:10, 09:15, 09:20, ...
            15m data: 09:15, 09:30, ...
            Result:   09:15 → use 09:15 from 5m
                      09:30 → use 09:30 from 5m

        Args:
            series: MTF indicator series (with RangeIndex)
            primary_index: Primary timeframe index (RangeIndex)
            mtf_data: Original MTF DataFrame (for timestamp conversion)
            primary_data: Original primary DataFrame (for timestamp conversion)

        Returns:
            Series aligned to primary_index (same length as primary, with RangeIndex)
        """
        # FIX: Convert to DatetimeIndex for merge_asof, then convert back
        # ParquetsEngine returns RangeIndex, but merge_asof needs DatetimeIndex for time-based alignment

        # Get timestamps from DataFrames
        if mtf_data is not None and 'timestamp' in mtf_data.columns:
            mtf_timestamps = pd.to_datetime(mtf_data['timestamp'], unit='ms')
        else:
            # Fallback: use series index as-is (will fail if RangeIndex)
            mtf_timestamps = series.index

        if primary_data is not None and 'timestamp' in primary_data.columns:
            primary_timestamps = pd.to_datetime(primary_data['timestamp'], unit='ms')
        else:
            # Fallback: use primary_index as-is
            primary_timestamps = primary_index

        # Create DataFrames with DatetimeIndex for merge_asof
        mtf_df = pd.DataFrame({'value': series.values}, index=mtf_timestamps)
        primary_df = pd.DataFrame(index=primary_timestamps)

        # Merge: for each primary timestamp, get the MOST RECENT mtf value
        aligned = pd.merge_asof(
            primary_df,
            mtf_df,
            left_index=True,
            right_index=True,
            direction='backward'  # Take most recent past value
        )

        # Return Series with ORIGINAL RangeIndex (not DatetimeIndex)
        return pd.Series(aligned['value'].values, index=primary_index)

    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================

    def _generate_signals(
        self,
        data: pd.DataFrame,
        indicators_mtf: Dict[str, Dict[str, np.ndarray]],
        strategy: Strategy,
        primary_timeframe: str
    ) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry signals (vectorized) with MTF support

        Args:
            data: Price data (primary timeframe)
            indicators_mtf: Multi-timeframe indicators {timeframe: {indicator: values}}
            strategy: Strategy instance
            primary_timeframe: Primary timeframe (e.g., "15m")

        Returns:
            tuple[pd.Series, pd.Series]: (long_mask, short_mask) - Boolean masks
        """
        # Add PRIMARY timeframe indicators to data for vectorized conditions
        data_with_indicators = data.copy()
        primary_indicators = indicators_mtf[primary_timeframe]
        for ind_name, ind_values in primary_indicators.items():
            data_with_indicators[ind_name] = ind_values

        # Use vectorized conditions module WITH MTF support
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
                indicators_mtf=indicators_mtf,  # ← MTF indicators for conditions like ['rsi_9', '>', 25, '5m']
                debug=self.debug
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
                indicators_mtf=indicators_mtf,  # ← MTF indicators
                debug=self.debug
            )
        else:
            short_mask = pd.Series(False, index=data_with_indicators.index)

        if self.debug:
            self.logger.info(f"   ✅ LONG: {long_mask.sum()}, SHORT: {short_mask.sum()}")

        return long_mask, short_mask

    # ========================================================================
    # POSITION SIMULATION
    # ========================================================================

    def _simulate_positions(
        self,
        long_mask: pd.Series,
        short_mask: pd.Series,
        data: pd.DataFrame,
        indicators_mtf: Dict[str, Dict[str, np.ndarray]],
        strategy: Strategy,
        strategy_executor: StrategyExecutor,
        config: BacktestConfig
    ) -> List[Trade]:
        """
        Simulate positions with proper sizing and exit logic (ADAPTED FROM OLD ENGINE)

        Args:
            long_mask: Boolean mask for LONG signals
            short_mask: Boolean mask for SHORT signals
            data: Price data
            indicators: Indicators
            strategy: Strategy instance
            config: Backtest config

        Returns:
            List[Trade]: Completed trades
        """
        # Merge ALL timeframe indicators into single DataFrame (Trading Engine approach)
        data_with_indicators = data.copy()

        # Suppress PerformanceWarning for DataFrame fragmentation
        # (iterative assignment is simpler and works correctly with mixed data types)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

            # Primary timeframe: no suffix
            primary_indicators = indicators_mtf[config.primary_timeframe]
            for ind_name, ind_values in primary_indicators.items():
                data_with_indicators[ind_name] = ind_values

            # MTF timeframes: with suffix (e.g., ema_89_4h)
            for tf, indicators_dict in indicators_mtf.items():
                if tf == config.primary_timeframe:
                    continue  # Already added above

                for ind_name, ind_values in indicators_dict.items():
                    # Add with timeframe suffix
                    col_name = f"{ind_name}_{tf}"
                    data_with_indicators[col_name] = ind_values

        # Convert masks to signal indices (eski engine 992-994)
        long_signals = long_mask[long_mask].index.tolist()
        short_signals = short_mask[short_mask].index.tolist()

        # Combine and sort signals
        all_signals = [(idx, 1) for idx in long_signals] + [(idx, -1) for idx in short_signals]
        all_signals.sort(key=lambda x: x[0])

        # Convert to dict for fast lookup
        signal_dict = {idx: side for idx, side in all_signals}

        trades = []
        positions = []  # Changed to list to support multiple positions (hedging)
        balance = config.initial_balance
        self.trade_counter = 0

        # Create PositionManager for this strategy (handles position rules from template)
        position_manager = PositionManager(strategy, logger=self.logger)

        # Load AI predictor if enabled
        ai_enabled = self._load_ai_predictor(strategy)
        
        # Load Exit Model if enabled
        exit_model_enabled = self._load_exit_model(strategy)

        # Get AI confidence threshold from AIConfig or old-style attribute
        ai_config = getattr(strategy, 'ai_config', None)
        if ai_config is not None:
            ai_confidence_threshold = ai_config.confidence_threshold
            ai_entry_decision = ai_config.entry_decision  # Should it be used in AI entry decision?
        else:
            ai_confidence_threshold = getattr(strategy, 'ai_confidence_threshold', 0.6)
            ai_entry_decision = True  # Old style: always use for entry decision

        ai_filtered_count = 0

        # Pre-compute AI predictions for all signal points (BATCH mode - much faster)
        ai_predictions_map = {}
        ai_rejected_signals = []  # Track rejected signals for accuracy analysis
        if ai_enabled and self._ai_predictor is not None:
            signal_indices = list(signal_dict.keys())
            if signal_indices:
                self.logger.info(f"   🤖 Computing AI predictions for {len(signal_indices)} signals...")
                ai_predictions_map = self._compute_batch_ai_predictions(
                    data_with_indicators,
                    signal_indices,
                    config.symbols[0],
                    config.primary_timeframe,
                    signal_dict  # Pass signal directions
                )
                self.logger.info(f"   🤖 AI predictions ready ({len(ai_predictions_map)} predictions)")

        # Iterate through data (eski engine 1001-1008)
        warmup = strategy.warmup_period
        for i in range(warmup, len(data_with_indicators)):
            row = data_with_indicators.iloc[i]
            current_time = row['open_time']  # Get timestamp from open_time column
            signal = signal_dict.get(i, 0)  # 0 if no signal

            # 1. Check exits (for all open positions)
            for position in positions[:]:  # Copy list to allow removal during iteration
                # Check position timeout using PositionManager
                should_timeout, timeout_reason = position_manager.check_position_timeout(
                    position, current_time
                )
                if should_timeout:
                    trade = self._close_position(
                        position,
                        row['close'],
                        ExitReason.MANUAL,  # Timeout = manual close
                        current_time,
                        config
                    )
                    trades.append(trade)
                    balance += trade.net_pnl_usd
                    positions.remove(position)
                    continue  # Skip to next position

                # Use StrategyExecutor to evaluate exit (handles trailing stop, break-even, SL/TP)
                # Pass data up to current candle for strategy executor
                data_to_pass = data_with_indicators.iloc[:i+1]

                exit_result = strategy_executor.evaluate_exit(
                    symbol=position['symbol'],
                    position=position,
                    data=data_to_pass,
                    current_price=row['close']
                )

                # Check if SL should be updated (break-even or trailing)
                if exit_result.get('updated_sl') and exit_result['updated_sl'] != position.get('sl_price'):
                    old_sl = position.get('sl_price')
                    new_sl = exit_result['updated_sl']
                    position['sl_price'] = new_sl
                    position['stop_loss'] = new_sl  # Update both keys for consistency

                    # Track break-even activation
                    if exit_result.get('break_even_moved'):
                        position['break_even_activated'] = True

                    # Debug output for SL update
                    if self.debug and self.logger:
                        if exit_result.get('break_even_moved'):
                            self.logger.info(f"🔄 Break-even SL moved: ${old_sl:,.2f} → ${new_sl:,.2f}")
                        elif exit_result.get('tp_trailing_activated'):
                            self.logger.info(f"🎯 TP trailing activated: SL = ${new_sl:,.2f} (TP hit, continuing...)")
                        else:
                            self.logger.info(f"🔄 Trailing SL updated: ${old_sl:,.2f} → ${new_sl:,.2f}")

                # Check if position should exit
                if exit_result.get('should_exit'):
                    exit_type = exit_result.get('exit_type', 'UNKNOWN')

                    # Handle PARTIAL exit differently
                    if exit_type == 'PARTIAL':
                        partial_size = exit_result.get('partial_exit_size', 0)
                        partial_level = exit_result.get('partial_exit_level', 1)

                        # Calculate partial quantity (percentage of ORIGINAL position)
                        original_quantity = position.get('original_quantity', position['quantity'])
                        partial_quantity = original_quantity * partial_size

                        # Ensure we don't close more than remaining
                        if partial_quantity > position['quantity']:
                            partial_quantity = position['quantity']

                        # Apply spread to exit
                        spread_cost = config.spread_pct / 100 / 2
                        if position['side'] == PositionSide.LONG:
                            exit_price = row['close'] * (1 - spread_cost)  # Sell at bid
                        else:
                            exit_price = row['close'] * (1 + spread_cost)  # Buy at ask

                        # Close partial position
                        partial_trade = self._close_partial_position(
                            position,
                            partial_quantity,
                            exit_price,
                            ExitReason.TAKE_PROFIT,  # Partial exit is profit taking
                            current_time,
                            config,
                            partial_level
                        )
                        trades.append(partial_trade)
                        balance += partial_trade.net_pnl_usd

                        # Update position quantity and partial exit counter
                        position['quantity'] -= partial_quantity
                        position['completed_partial_exits'] = position.get('completed_partial_exits', 0) + 1

                        # Store original quantity for next partial exits
                        if 'original_quantity' not in position:
                            position['original_quantity'] = original_quantity

                        # Debug output
                        if self.debug and self.logger:
                            remaining_pct = (position['quantity'] / original_quantity) * 100
                            self.logger.info(f"📤 Partial exit {partial_level}: Closed {partial_size*100:.0f}% @ ${exit_price:,.2f}")
                            self.logger.info(f"   → Remaining: {remaining_pct:.0f}% of original position")
                            self.logger.info(f"   → Profit: ${partial_trade.net_pnl_usd:,.2f}")

                        # If all position closed, remove from list
                        if position['quantity'] <= 0:
                            positions.remove(position)

                    else:
                        # Normal full exit (SL, TP, SIGNAL, AI_EXIT)
                        # Map exit types to ExitReason
                        exit_reason_map = {
                            'SL': ExitReason.STOP_LOSS,
                            'TP': ExitReason.TAKE_PROFIT,
                            'EXIT_SIGNAL': ExitReason.SIGNAL,
                            'TRAILING_STOP': ExitReason.STOP_LOSS,  # Trailing is also SL
                            'AI_EXIT': ExitReason.SIGNAL,  # AI exit is treated as signal-based
                        }
                        exit_reason = exit_reason_map.get(exit_type, ExitReason.SIGNAL)

                        # Get exit price (use SL/TP price if hit)
                        exit_price = row['close']
                        if exit_type == 'SL' and position['sl_price']:
                            exit_price = position['sl_price']
                        elif exit_type == 'TP' and position['tp_price']:
                            exit_price = position['tp_price']

                        # Close position
                        trade = self._close_position(
                            position,
                            exit_price,
                            exit_reason,
                            current_time,
                            config
                        )
                        trades.append(trade)
                        balance += trade.net_pnl_usd
                        positions.remove(position)

            # 2. Check entries
            if signal != 0:
                # Market tradeable check (session/time/day filters)
                if not strategy_executor.market_manager.is_market_tradeable(current_time):
                    continue  # Skip signal, market not tradeable

                # Check side_method (LONG, SHORT, BOTH, FLAT)
                from components.strategies.base_strategy import TradingSide
                if strategy.side_method == TradingSide.FLAT:
                    continue  # No trading allowed
                elif strategy.side_method == TradingSide.LONG and signal < 0:
                    continue  # SHORT signal but only LONG allowed
                elif strategy.side_method == TradingSide.SHORT and signal > 0:
                    continue  # LONG signal but only SHORT allowed

                # AI Signal Filtering (use pre-computed predictions from map)
                # Only filter if AI is enabled AND entry_decision is True
                if ai_enabled and ai_entry_decision and i in ai_predictions_map:
                    ai_pred = ai_predictions_map[i]
                    win_prob = ai_pred['win_probability']
                    confidence = ai_pred.get('confidence', abs(win_prob - 0.5) * 2)
                    signal_side = 'LONG' if signal > 0 else 'SHORT'

                    # DIRECTION-AWARE filtering (new approach):
                    # Model is trained to predict: "Will THIS signal be profitable?"
                    # win_prob > threshold = signal looks good
                    # win_prob < threshold = signal looks bad (skip it)
                    #
                    # This works for both LONG and SHORT because the model
                    # was trained with direction-aware labels where:
                    # - target=1 means the suggested direction was profitable
                    # - target=0 means the suggested direction was unprofitable
                    #
                    # So we simply check: win_prob > threshold for ALL signals
                    if win_prob < ai_confidence_threshold:
                        ai_filtered_count += 1
                        # Track for accuracy analysis
                        ai_rejected_signals.append({
                            'idx': i,
                            'time': current_time,
                            'side': signal_side,
                            'win_prob': win_prob,
                            'reason': f'win_prob={win_prob:.2f} < {ai_confidence_threshold}'
                        })
                        if self.debug:
                            from core.timezone_utils import TimezoneUtils
                            display_time = TimezoneUtils.format(current_time, fmt='%Y-%m-%d %H:%M')
                            self.logger.info(f"\n🚫 AI REJECTED {signal_side} @ {display_time}")
                            self.logger.info(f"   win_prob={win_prob:.2f} < {ai_confidence_threshold} (low confidence)")
                        continue

                # Check max_total_positions limit first
                max_total = strategy.position_management.max_total_positions
                if len(positions) >= max_total:
                    # Max total positions reached, skip entry
                    continue

                symbol = config.symbols[0]
                new_side = 'LONG' if signal > 0 else 'SHORT'

                # Use PositionManager to check if position can be opened
                # This handles: max_positions_per_symbol, pyramiding, hedging logic
                open_result = position_manager.can_open_position(symbol, new_side, positions)

                if not open_result.can_open:
                    # Position manager rejected the entry
                    continue

                # ONE-WAY MODE: Close opposite positions first if required
                if open_result.should_close_opposite:
                    for opp_pos in open_result.opposite_positions:
                        trade = self._close_position(
                            opp_pos,
                            row['close'],
                            ExitReason.SIGNAL,
                            current_time,
                            config
                        )
                        trades.append(trade)
                        balance += trade.net_pnl_usd
                        positions.remove(opp_pos)

                # Data up to current candle for SL/TP calculation
                data_to_pass = data_with_indicators.iloc[:i+1]

                # Calculate position size via RiskManager
                from components.strategies.exit_manager import ExitManager
                temp_exit_manager = ExitManager(strategy, logger=self.logger, ai_predictor=self._ai_predictor)

                # Get ATR if needed (auto-detect atr or atr_xx)
                atr_value = _get_atr_value_from_row(row)

                temp_sl_price = temp_exit_manager.calculate_stop_loss(
                    row['close'], new_side, data=data_to_pass, atr_value=atr_value
                )

                quantity = self.risk_manager.calculate_position_size_from_strategy(
                    strategy=strategy,
                    risk_management=strategy.risk_management,
                    entry_price=row['close'],
                    portfolio_value=balance,
                    stop_loss_price=temp_sl_price
                )

                # Apply pyramiding scale factor from PositionManager
                quantity *= open_result.pyramiding_scale

                # Get AI prediction for this signal (if available)
                ai_pred_for_position = ai_predictions_map.get(i) if ai_enabled else None

                # Apply AI position sizing if enabled
                ai_position_mult = 1.0
                if ai_pred_for_position and ai_config and ai_config.position_sizing:
                    win_prob = ai_pred_for_position.get('win_probability', 0.5)
                    confidence = abs(win_prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

                    # Scale position based on confidence
                    # confidence=0 → min_mult, confidence=1 → max_mult
                    min_mult = ai_config.min_ai_position_mult
                    max_mult = ai_config.max_ai_position_mult
                    ai_position_mult = min_mult + (max_mult - min_mult) * confidence

                    # Apply multiplier
                    quantity *= ai_position_mult

                    if self.debug:
                        self.logger.info(f"   AI Position Sizing: conf={confidence:.2f} -> mult={ai_position_mult:.2f}x")

                if quantity > 0:

                    # Open new position
                    # Count same-side positions for pyramiding label
                    same_side_count = sum(1 for p in positions if p.get('side') == new_side)

                    new_position = self._open_position(
                        signal, row, quantity, strategy, config, current_time,
                        data_so_far=data_to_pass,
                        ai_prediction=ai_pred_for_position,
                        pyramiding_entry=same_side_count if same_side_count > 0 else 0
                    )
                    positions.append(new_position)

        # Close any remaining positions
        if positions:
            final_row = data_with_indicators.iloc[-1]
            final_time = final_row['open_time']  # Get timestamp from open_time column
            for position in positions[:]:
                trade = self._close_position(
                    position,
                    final_row['close'],
                    ExitReason.MANUAL,
                    final_time,
                    config
                )
                trades.append(trade)

        if self.debug:
            self.logger.info(f"   ✅ {len(trades)} trade completed")

        # Log AI filtering stats and accuracy analysis
        if ai_enabled and ai_filtered_count > 0:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🤖 AI FILTERING SUMMARY")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"   Total signals: {len(signal_dict)}")
            self.logger.info(f"   AI rejected: {ai_filtered_count}")
            self.logger.info(f"   AI approved: {len(signal_dict) - ai_filtered_count}")
            self.logger.info(f"   Threshold: {ai_confidence_threshold}")

            # Analyze what would have happened with rejected signals
            if ai_rejected_signals and len(data_with_indicators) > 0:
                self._analyze_ai_accuracy(
                    ai_rejected_signals,
                    data_with_indicators,
                    config
                )

        return trades

    def _analyze_ai_accuracy(
        self,
        rejected_signals: List[Dict],
        data: pd.DataFrame,
        config
    ):
        """
        Analyze AI rejection accuracy by checking what would have happened.

        For each rejected signal, look forward 24 bars and see if AI was right.
        """
        forward_bars = 24
        correct_rejections = 0
        wrong_rejections = 0

        for sig in rejected_signals:
            idx = sig['idx']
            side = sig['side']

            # Check if we have enough forward data
            if idx + forward_bars >= len(data):
                continue

            entry_price = data.iloc[idx]['close']
            future_data = data.iloc[idx + 1:idx + forward_bars + 1]

            if len(future_data) == 0:
                continue

            # Calculate what would have happened
            if side == 'LONG':
                # For LONG: check if price went up
                max_profit = (future_data['high'].max() - entry_price) / entry_price * 100
                max_loss = (entry_price - future_data['low'].min()) / entry_price * 100
                # Win if profit > 1% before hitting -2% loss
                would_have_won = max_profit > 1.0 and max_profit > max_loss
            else:
                # For SHORT: check if price went down
                max_profit = (entry_price - future_data['low'].min()) / entry_price * 100
                max_loss = (future_data['high'].max() - entry_price) / entry_price * 100
                would_have_won = max_profit > 1.0 and max_profit > max_loss

            # AI rejected = AI said "this trade will lose"
            # If the trade would have lost -> AI was CORRECT
            # If trade would have won -> AI was WRONG
            if would_have_won:
                wrong_rejections += 1
            else:
                correct_rejections += 1

        total = correct_rejections + wrong_rejections
        if total > 0:
            accuracy = correct_rejections / total * 100
            self.logger.info(f"\n   📊 AI Rejection Accuracy:")
            self.logger.info(f"      Correct rejections: {correct_rejections}/{total} ({accuracy:.1f}%)")
            self.logger.info(f"      Wrong rejections: {wrong_rejections}/{total} ({100-accuracy:.1f}%)")
            if accuracy > 50:
                self.logger.info(f"      ✅ AI is helping! (>50% correct)")
            else:
                self.logger.info(f"      ⚠️ AI may need tuning (<50% correct)")

    def _load_ai_predictor(self, strategy: Strategy) -> bool:
        """
        Load AI predictor if strategy has AI enabled.

        Supports both old-style (ai_enabled attribute) and new-style (AIConfig).

        Args:
            strategy: Strategy instance with ai_config or ai_enabled

        Returns:
            bool: True if AI predictor loaded successfully
        """
        # Check for AIConfig (new style)
        ai_config = getattr(strategy, 'ai_config', None)
        if ai_config is not None:
            if not ai_config.ai_enabled:
                return False
            model_path = ai_config.model_path
        else:
            # Fallback to old-style attributes
            if not getattr(strategy, 'ai_enabled', False):
                return False
            model_path = getattr(strategy, 'ai_model_path', 'data/checkpoints/ai/global_best.pt')

        if self._ai_predictor is not None:
            return True

        # Get model type from AIConfig
        model_type = ai_config.model_type if ai_config else "rl_model"

        try:
            if model_type == "rl_model":
                # New RL-based predictor (PPO)
                from modules.ai.inference.predictor import RLPredictor

                # Extract symbol and timeframe from strategy
                symbol = getattr(strategy, 'symbols', [{}])[0]
                if hasattr(symbol, 'symbol'):
                    symbol_name = symbol.symbol[0] if isinstance(symbol.symbol, list) else symbol.symbol
                else:
                    symbol_name = "BTCUSDT"
                timeframe = getattr(strategy, 'primary_timeframe', '1h')

                self._ai_predictor = RLPredictor(
                    symbol=symbol_name,
                    timeframe=timeframe,
                    model_dir=str(Path(model_path).parent) if model_path else None
                )
                self._ai_predictor.load_model()
                self._ai_model_type = "rl"
                self.logger.info(f"✅ RL predictor loaded: {model_path}")
            else:
                # Legacy signal model (supervised learning)
                from modules.ai.inference import SignalPredictor
                self._ai_predictor = SignalPredictor()
                self._ai_predictor.load_checkpoint(model_path)
                self._ai_model_type = "signal"
                self.logger.info(f"✅ Signal predictor loaded: {model_path}")

            return True

        except Exception as e:
            self.logger.warning(f"⚠️  AI predictor load failed: {e}")
            self._ai_predictor = None
            return False

    def _load_exit_model(self, strategy: Strategy) -> bool:
        """
        Load Exit Model if strategy has exit model enabled.
        
        Args:
            strategy: Strategy instance with ai_config.exit_model_enabled
            
        Returns:
            bool: True if Exit Model loaded successfully
        """
        # Check for AIConfig with exit model
        ai_config = getattr(strategy, 'ai_config', None)
        if ai_config is None or not ai_config.exit_model_enabled:
            return False
        
        if self._exit_model is not None:
            return True
        
        exit_model_path = ai_config.exit_model_path
        
        try:
            from modules.simple_train.models.exit_model import ExitModel
            
            self._exit_model = ExitModel()
            self._exit_model.load(exit_model_path)
            
            self.logger.info(f"✅ Exit Model loaded: {exit_model_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️  Exit Model load failed: {e}")
            self._exit_model = None
            return False


    def _get_ai_prediction(
        self,
        data: pd.DataFrame,
        row_idx: int,
        symbol: str,
        timeframe: str,
        signal_side: str = None
    ) -> Optional[Dict]:
        """
        Get AI prediction for a specific row.

        Args:
            data: Full DataFrame
            row_idx: Current row index
            symbol: Trading symbol
            timeframe: Timeframe string
            signal_side: Strategy signal direction ('long' or 'short')

        Returns:
            Dict with win_probability, confidence, action, etc.
        """
        if self._ai_predictor is None:
            return None

        try:
            # Get data up to current row (minimum 200 bars for features)
            start_idx = max(0, row_idx - 200)
            data_slice = data.iloc[start_idx:row_idx + 1]

            if len(data_slice) < 200:
                return None

            # Check model type
            model_type = getattr(self, '_ai_model_type', 'signal')

            if model_type == "rl":
                # RL model prediction
                result = self._ai_predictor.predict(data_slice)

                if 'error' in result:
                    return None

                # Convert RL output to backtest format
                # RL actions: HOLD=0, LONG=1, SHORT=2, CLOSE=3
                action = result.get('action', 'HOLD')
                probs = result.get('probabilities', {})
                confidence = result.get('confidence', 0.0)

                # Calculate win_probability based on signal alignment
                # If strategy says LONG and RL says LONG with high confidence = high win prob
                if signal_side == 'long':
                    # For LONG signal: win_prob = P(LONG) - P(SHORT)
                    win_probability = probs.get('LONG', 0.5)
                    # If RL says SHORT, this is a bad signal
                    if action == 'SHORT':
                        win_probability = 1.0 - probs.get('SHORT', 0.5)
                elif signal_side == 'short':
                    # For SHORT signal: win_prob = P(SHORT) - P(LONG)
                    win_probability = probs.get('SHORT', 0.5)
                    # If RL says LONG, this is a bad signal
                    if action == 'LONG':
                        win_probability = 1.0 - probs.get('LONG', 0.5)
                else:
                    win_probability = 0.5

                return {
                    'win_probability': win_probability,
                    'confidence': confidence,
                    'action': action,
                    'action_probs': probs,
                    'agrees_with_signal': (
                        (signal_side == 'long' and action == 'LONG') or
                        (signal_side == 'short' and action == 'SHORT')
                    )
                }
            else:
                # Legacy signal model
                result = self._ai_predictor.predict(data_slice, symbol=symbol, timeframe=timeframe)
                return result

        except Exception as e:
            if self.debug:
                self.logger.warning(f"AI prediction error: {e}")
            return None

    def _compute_batch_ai_predictions(
        self,
        data: pd.DataFrame,
        signal_indices: List[int],
        symbol: str,
        timeframe: str,
        signal_dict: Dict[int, int] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute AI predictions for all signal points at once (BATCH mode).

        This is much faster than calling predict() for each signal individually
        because feature extraction is done once for the entire dataset.

        Args:
            data: Full DataFrame with OHLCV data
            signal_indices: List of row indices where signals occurred
            symbol: Trading symbol
            timeframe: Timeframe string
            signal_dict: Dict mapping idx -> signal direction (1=LONG, -1=SHORT)

        Returns:
            Dict mapping row_idx -> {win_probability, optimal_tp, optimal_sl, expected_return}
        """
        if self._ai_predictor is None or not signal_indices:
            return {}

        # Check model type
        model_type = getattr(self, '_ai_model_type', 'signal')

        try:
            if model_type == "rl":
                # RL model doesn't have batch prediction, process individually
                result = {}
                for idx in signal_indices:
                    # Need at least 200 bars for feature extraction
                    start_idx = max(0, idx - 200)
                    data_slice = data.iloc[start_idx:idx + 1]

                    if len(data_slice) < 200:
                        continue

                    try:
                        prediction = self._ai_predictor.predict(data_slice)
                        if 'error' not in prediction:
                            probs = prediction.get('probabilities', {})
                            action = prediction.get('action', 'HOLD')

                            # Get signal direction
                            signal_side = signal_dict.get(idx, 0) if signal_dict else 0

                            # Calculate win_probability based on signal alignment
                            # Key insight: If RL says same direction as strategy -> approve
                            #              If RL says opposite direction -> reject
                            #              If RL says HOLD/CLOSE -> use directional probability

                            if signal_side == 1:  # LONG signal from strategy
                                long_prob = probs.get('LONG', 0.25)
                                short_prob = probs.get('SHORT', 0.25)

                                if action == 'LONG':
                                    # RL confirms LONG - high win prob
                                    win_prob = 0.5 + long_prob * 0.5  # Scale to 0.5-1.0
                                elif action == 'SHORT':
                                    # RL says opposite - low win prob
                                    win_prob = 0.5 - short_prob * 0.5  # Scale to 0.0-0.5
                                else:
                                    # HOLD/CLOSE - use LONG vs SHORT probability
                                    win_prob = 0.5 + (long_prob - short_prob) * 0.5

                            elif signal_side == -1:  # SHORT signal from strategy
                                long_prob = probs.get('LONG', 0.25)
                                short_prob = probs.get('SHORT', 0.25)

                                if action == 'SHORT':
                                    # RL confirms SHORT - high win prob
                                    win_prob = 0.5 + short_prob * 0.5  # Scale to 0.5-1.0
                                elif action == 'LONG':
                                    # RL says opposite - low win prob
                                    win_prob = 0.5 - long_prob * 0.5  # Scale to 0.0-0.5
                                else:
                                    # HOLD/CLOSE - use SHORT vs LONG probability
                                    win_prob = 0.5 + (short_prob - long_prob) * 0.5
                            else:
                                win_prob = 0.5

                            result[idx] = {
                                'win_probability': win_prob,
                                'confidence': prediction.get('confidence', 0.5),
                                'action': action,
                                'action_probs': probs,
                                'agrees_with_signal': (
                                    (signal_side == 1 and action == 'LONG') or
                                    (signal_side == -1 and action == 'SHORT')
                                )
                            }
                    except Exception:
                        continue

                return result
            else:
                # Legacy signal model - use batch prediction
                predictions = self._ai_predictor.predict_batch(data, symbol=symbol, timeframe=timeframe)

                # Map signal indices to full prediction dict
                result = {}
                for idx in signal_indices:
                    if idx < len(predictions):
                        row = predictions.iloc[idx]
                        win_prob = row['win_probability']
                        if not np.isnan(win_prob):
                            result[idx] = {
                                'win_probability': float(win_prob),
                                'optimal_tp': float(row.get('optimal_tp', 2.0)),
                                'optimal_sl': float(row.get('optimal_sl', 1.0)),
                                'expected_return': float(row.get('expected_return', 0.0)),
                                'confidence': float(row.get('confidence', abs(win_prob - 0.5) * 2))
                            }

                return result

        except Exception as e:
            self.logger.warning(f"Batch AI prediction error: {e}")
            return {}

    def _open_position(self, signal, row, quantity, strategy, config, entry_time, data_so_far=None, ai_prediction=None, pyramiding_entry=0) -> Dict:
        """
        Open position with optional AI TP/SL optimization.

        Args:
            signal: 1 for LONG, -1 for SHORT
            row: Current data row
            quantity: Position quantity
            strategy: Strategy instance
            config: Backtest config
            entry_time: Entry timestamp
            data_so_far: Historical data up to entry
            ai_prediction: AI prediction dict {win_probability, optimal_tp, optimal_sl, expected_return}
        """
        self.trade_counter += 1

        side = PositionSide.LONG if signal > 0 else PositionSide.SHORT

        # Apply slippage + spread
        # LONG: buy at ask (close + spread/2 + slippage)
        # SHORT: sell at bid (close - spread/2 + slippage)
        spread_cost = config.spread_pct / 100 / 2  # Half of the spread (midpoint of bid-ask close)
        slippage_cost = config.slippage_pct / 100

        if signal > 0:  # LONG
            entry_price = row['close'] * (1 + spread_cost + slippage_cost)
        else:  # SHORT
            entry_price = row['close'] * (1 - spread_cost + slippage_cost)

        # Use ExitManager to calculate SL/TP (supports all methods now)
        from components.strategies.exit_manager import ExitManager
        exit_manager = ExitManager(strategy, logger=self.logger, ai_predictor=self._ai_predictor)

        # Calculate ATR if needed (auto-detect atr or atr_xx)
        atr_value = None

        # Helper: Get method name (handle enum or string)
        def get_method_name(method):
            if hasattr(method, 'name'):  # Enum
                return method.name
            elif isinstance(method, str):  # String
                return method
            else:
                return str(method)

        sl_method = get_method_name(strategy.exit_strategy.stop_loss_method) if hasattr(strategy.exit_strategy, 'stop_loss_method') else None
        tp_method = get_method_name(strategy.exit_strategy.take_profit_method) if hasattr(strategy.exit_strategy, 'take_profit_method') else None

        if sl_method in ('ATR_BASED', 'DYNAMIC_AI') or tp_method in ('ATR_BASED', 'DYNAMIC_AI'):
            # Get ATR from indicators if available
            atr_value = _get_atr_value_from_row(row)

        side_str = 'LONG' if signal > 0 else 'SHORT'

        # Calculate strategy-based SL/TP
        strategy_sl = exit_manager.calculate_stop_loss(entry_price, side_str, data=data_so_far, atr_value=atr_value)
        strategy_tp = exit_manager.calculate_take_profit(entry_price, side_str, stop_loss_price=strategy_sl, atr_value=atr_value, data=data_so_far)

        # Get AI config
        ai_config = getattr(strategy, 'ai_config', None)

        # Apply AI TP/SL optimization if enabled
        sl_price = strategy_sl
        tp_price = strategy_tp
        ai_tp_used = False
        ai_sl_used = False

        # RL Model TP/SL Optimization
        if ai_config and ai_config.ai_enabled and self._ai_predictor:
            model_type = getattr(ai_config, 'model_type', 'rl_model')

            if model_type == 'rl_model' and data_so_far is not None and len(data_so_far) >= 200:
                # Use RLPredictor.optimize_tp_sl() for RL models
                try:
                    # Calculate strategy TP/SL as percentages
                    strategy_tp_pct = abs((strategy_tp - entry_price) / entry_price) * 100 if strategy_tp else 2.0
                    strategy_sl_pct = abs((strategy_sl - entry_price) / entry_price) * 100 if strategy_sl else 1.0

                    # Get AI optimized TP/SL
                    tp_sl_result = self._ai_predictor.optimize_tp_sl(
                        df=data_so_far,
                        side=side_str,
                        strategy_tp=strategy_tp_pct,
                        strategy_sl=strategy_sl_pct,
                        entry_price=entry_price,
                        force_tp=ai_config.tp_optimization,  # Strategy override
                        force_sl=ai_config.sl_optimization   # Strategy override
                    )

                    # Apply TP optimization
                    if ai_config.tp_optimization and tp_sl_result.get('tp_price'):
                        tp_price = tp_sl_result['tp_price']
                        ai_tp_used = tp_sl_result.get('tp_source', 'strategy') != 'strategy'

                    # Apply SL optimization
                    if ai_config.sl_optimization and tp_sl_result.get('sl_price'):
                        sl_price = tp_sl_result['sl_price']
                        ai_sl_used = tp_sl_result.get('sl_source', 'strategy') != 'strategy'

                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"RL TP/SL optimization error: {e}")

            elif ai_prediction:
                # Legacy AI model support (signal_model, lstm, etc.)
                ai_optimal_tp_pct = ai_prediction.get('optimal_tp', 0)
                ai_optimal_sl_pct = ai_prediction.get('optimal_sl', 0)

                if ai_optimal_tp_pct > 0 or ai_optimal_sl_pct > 0:
                    # Calculate AI-suggested prices from percentages
                    if signal > 0:  # LONG
                        ai_tp_price = entry_price * (1 + ai_optimal_tp_pct / 100)
                        ai_sl_price = entry_price * (1 - ai_optimal_sl_pct / 100)
                    else:  # SHORT
                        ai_tp_price = entry_price * (1 - ai_optimal_tp_pct / 100)
                        ai_sl_price = entry_price * (1 + ai_optimal_sl_pct / 100)

                    # TP Optimization
                    if ai_config.use_ai_tp and ai_optimal_tp_pct > 0:
                        tp_price = ai_tp_price
                        ai_tp_used = True
                    elif ai_config.tp_optimization and ai_optimal_tp_pct > 0 and strategy_tp:
                        blend = ai_config.tp_blend_ratio
                        tp_price = strategy_tp * (1 - blend) + ai_tp_price * blend
                        ai_tp_used = True

                    # SL Optimization
                    if ai_config.use_ai_sl and ai_optimal_sl_pct > 0:
                        sl_price = ai_sl_price
                        ai_sl_used = True
                    elif ai_config.sl_optimization and ai_optimal_sl_pct > 0 and strategy_sl:
                        blend = ai_config.sl_blend_ratio
                        sl_price = strategy_sl * (1 - blend) + ai_sl_price * blend
                        ai_sl_used = True

        # ====================================================================
        # Exit Model Optimization (Dynamic Exit Parameters)
        # ====================================================================
        exit_trailing_override = None
        exit_be_override = None
        expected_bars = 0
        
        if ai_config and getattr(ai_config, 'exit_model_enabled', False) and self._exit_model:
            try:
                # Extract features from data_so_far (last 250 bars)
                # Note: This uses standard FeatureExtractor to ensure consistency with training
                from modules.simple_train.core.feature_extractor import FeatureExtractor
                
                # Create extractor if not cached (can be optimized by reusing)
                if not hasattr(self, '_exit_feature_extractor'):
                    self._exit_feature_extractor = FeatureExtractor(strategy_name=strategy.strategy_name)
                    
                # Extract features
                # data_so_far has all history up to this point
                features_df = self._exit_feature_extractor.extract(data_so_far.tail(250))
                
                if not features_df.empty:
                    # Get latest features (at entry point)
                    latest_features = features_df.iloc[-1]
                    
                    # --- FEATURE ALIGNMENT ---
                    # Use model's stored feature names if available (CRITICAL for shape match)
                    if hasattr(self._exit_model, '_feature_names') and self._exit_model._feature_names:
                        expected_features = self._exit_model._feature_names
                        
                        # Ensure all expected features exist
                        available_features = set(latest_features.index)
                        missing = [f for f in expected_features if f not in available_features]
                        
                        if missing:
                            raise ValueError(f"Missing features for Exit Model: {missing[:3]}...")
                            
                        # Select exact features in correct order
                        X = latest_features[expected_features].values.reshape(1, -1)
                    else:
                        # Fallback: Exclude non-numeric and special cols (Legacy/Risky)
                        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                        rich_label_cols = ['pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse', 'peak_to_exit_ratio']
                        feature_cols = [c for c in latest_features.index if c not in ohlcv_cols + rich_label_cols]
                        X = latest_features[feature_cols].values.reshape(1, -1)
                    
                    # Get exit model predictions
                    exit_predictions = self._exit_model.predict(X)
                    
                    # Extract predictions
                    tp_multiplier = float(exit_predictions['tp_multiplier'][0])
                    sl_multiplier = float(exit_predictions['sl_multiplier'][0])
                    use_trailing = float(exit_predictions['use_trailing'][0]) > 0.5
                    use_break_even = float(exit_predictions['use_break_even'][0]) > 0.5
                    expected_bars = int(exit_predictions['expected_bars'][0])
                    
                    # Calculate base TP/SL percentages (from strategy)
                    # Use defaults (6.0%, 3.2%) if strategy doesn't have them set
                    base_tp_pct = abs((strategy_tp - entry_price) / entry_price) * 100 if strategy_tp else getattr(strategy.exit_strategy, 'take_profit_percent', 6.0)
                    if base_tp_pct == 0: base_tp_pct = 6.0
                    
                    base_sl_pct = abs((strategy_sl - entry_price) / entry_price) * 100 if strategy_sl else getattr(strategy.exit_strategy, 'stop_loss_percent', 3.2)
                    if base_sl_pct == 0: base_sl_pct = 3.2
                    
                    # Get Blend Ratio
                    blend_ratio = getattr(ai_config, 'exit_model_blend_ratio', 1.0)
                    
                    # Apply TP Optimization
                    if getattr(ai_config, 'use_exit_model_tp', False):
                        new_tp_pct = base_tp_pct * tp_multiplier
                        final_tp_pct = base_tp_pct * (1 - blend_ratio) + new_tp_pct * blend_ratio
                        
                        if signal > 0: # LONG
                            tp_price = entry_price * (1 + final_tp_pct / 100)
                        else: # SHORT
                            tp_price = entry_price * (1 - final_tp_pct / 100)
                        
                        ai_tp_used = True
                    
                    # Apply SL Optimization
                    if getattr(ai_config, 'use_exit_model_sl', False):
                        new_sl_pct = base_sl_pct * sl_multiplier
                        final_sl_pct = base_sl_pct * (1 - blend_ratio) + new_sl_pct * blend_ratio
                        
                        if signal > 0: # LONG
                            sl_price = entry_price * (1 - final_sl_pct / 100)
                        else: # SHORT
                            sl_price = entry_price * (1 + final_sl_pct / 100)
                        
                        ai_sl_used = True
                        
                    # Apply Trailing & BE overrides
                    if getattr(ai_config, 'use_exit_model_trailing', False):
                        exit_trailing_override = use_trailing
                        
                    if getattr(ai_config, 'use_exit_model_break_even', False):
                        exit_be_override = use_break_even
                        
                    if self.debug and self.logger:
                        self.logger.info(f"   🎯 Exit Model: TP={tp_multiplier:.2f}x ({final_tp_pct:.2f}%), SL={sl_multiplier:.2f}x ({final_sl_pct:.2f}%), Trail={use_trailing}, BE={use_break_even}")

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Exit Model optimization error: {e}")
                exit_trailing_override = None
                exit_be_override = None

        position = {
            'id': self.trade_counter,
            'symbol': config.symbols[0],
            'side': side,
            'entry_time': entry_time,  # Use passed timestamp
            'entry_price': entry_price,
            'quantity': quantity,
            'sl_price': sl_price,  # For backtest logic
            'tp_price': tp_price,  # For backtest logic
            'stop_loss': sl_price,  # For StrategyExecutor
            'take_profit': tp_price,  # For StrategyExecutor
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'break_even_activated': False,  # BE tracking
            'exit_trailing_override': exit_trailing_override,
            'exit_be_override': exit_be_override,
            'expected_duration': expected_bars
        }

        # Debug output (only when --debug flag is used)
        if self.debug and self.logger:
            from core.timezone_utils import TimezoneUtils

            # Format timestamp
            display_time = TimezoneUtils.format(entry_time, fmt='%Y-%m-%d %H:%M:%S')

            side_str = "LONG" if signal > 0 else "SHORT"
            pyramid_label = f" - Pyramiding {pyramiding_entry + 1}" if pyramiding_entry > 0 else ""
            self.logger.info(f"\n🎯 {side_str} Entry signal detected!{pyramid_label}")
            self.logger.info(f"   - Time: {display_time}")
            self.logger.info(f"   - Symbol: {config.symbols[0]}")
            self.logger.info(f"   - Position: {side_str}")
            self.logger.info(f"   - Entry price: ${entry_price:,.2f}")

            # Entry conditions
            self.logger.info(f"\n   📋 WHY ENTRY:")
            entry_conditions = strategy.entry_conditions.get('long' if signal > 0 else 'short', [])
            for condition in entry_conditions:
                # Parse condition
                left = condition[0]
                operator = condition[1]
                right = condition[2]
                timeframe = condition[3] if len(condition) > 3 else None

                # Get values from row
                left_value = row.get(left, left) if isinstance(left, str) else left
                right_value = row.get(right, right) if isinstance(right, str) else right

                # Format condition string
                condition_str = f"{condition}"
                if isinstance(left_value, (int, float, np.number)) and isinstance(right_value, (int, float, np.number)):
                    if timeframe:
                        condition_str = f"{left} {operator} {right} [{timeframe}] ({left_value:.2f} {operator} {right_value:.2f})"
                    else:
                        condition_str = f"{left} {operator} {right} ({left_value:.2f} {operator} {right_value:.2f})"

                self.logger.info(f"      ✅ {condition_str}")

            # Position details
            self.logger.info(f"\n   💰 Position Details:")
            self.logger.info(f"      - Quantity: {quantity:.8f}")
            if sl_price:
                sl_method = strategy.exit_strategy.stop_loss_method.value if hasattr(strategy, 'exit_strategy') else 'N/A'
                # Calculate actual SL distance (not parameter value)
                sl_pct = abs((sl_price - entry_price) / entry_price) * 100
                ai_sl_tag = " [AI]" if ai_sl_used else ""
                self.logger.info(f"      - Stop Loss: ${sl_price:,.2f} ({sl_method}, {sl_pct:.2f}%){ai_sl_tag}")
            if tp_price:
                tp_method = strategy.exit_strategy.take_profit_method.value if hasattr(strategy, 'exit_strategy') else 'N/A'
                # Calculate actual TP distance (not parameter value)
                tp_pct = abs((tp_price - entry_price) / entry_price) * 100
                ai_tp_tag = " [AI]" if ai_tp_used else ""
                self.logger.info(f"      - Take Profit: ${tp_price:,.2f} ({tp_method}, {tp_pct:.2f}%){ai_tp_tag}")

        # AI Logging: Entry
        if self.enable_ai_logging and self.future_logger and self.feature_extractor:
            try:
                # Extract indicator features from row (primary + MTF, all in row)
                indicators = self._extract_indicators_from_row(row=row, strategy=strategy)


                # Log entry with indicators
                from components.ai.future_logger import TradeEntry
                entry_log = TradeEntry(
                    trade_id=self.trade_counter,
                    symbol=config.symbols[0],
                    side=side_str,
                    entry_time=int(entry_time.timestamp() * 1000) if isinstance(entry_time, datetime) else entry_time,
                    entry_price=entry_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    indicators=indicators,
                    strategy_name=config.strategy_name,
                    timeframe=config.primary_timeframe
                )
                self.future_logger.log_entry(entry_log)
            except Exception as e:
                self.logger.debug(f"⚠️  AI entry logging error: {e}")

        return position

    def _close_partial_position(self, position, partial_quantity, exit_price, exit_reason, exit_time, config, partial_level) -> Trade:
        """Close partial position and create Trade (for partial exits only)"""

        # PnL calculation (exit_price already has spread applied in caller)
        if position['side'] == PositionSide.LONG:
            gross_pnl_usd = (exit_price - position['entry_price']) * partial_quantity
        else:
            gross_pnl_usd = (position['entry_price'] - exit_price) * partial_quantity

        gross_pnl_pct = (gross_pnl_usd / (position['entry_price'] * partial_quantity)) * 100

        # Costs (only for partial quantity)
        partial_position_value = position['entry_price'] * partial_quantity
        commission = partial_position_value * (config.commission_pct / 100) * 2  # Entry + exit
        slippage = partial_position_value * (config.slippage_pct / 100) * 2

        # Spread cost
        spread_cost_pct = config.spread_pct / 100 / 2
        entry_spread_cost = partial_position_value * spread_cost_pct
        exit_spread_cost = exit_price * partial_quantity * spread_cost_pct
        spread = entry_spread_cost + exit_spread_cost

        net_pnl_usd = gross_pnl_usd - commission - slippage
        net_pnl_pct = (net_pnl_usd / partial_position_value) * 100

        return Trade(
            trade_id=f"{position['id']}_partial_{partial_level}",
            symbol=position['symbol'],
            side=position['side'],
            entry_time=position['entry_time'],
            entry_price=position['entry_price'],
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            quantity=partial_quantity,
            gross_pnl_usd=gross_pnl_usd,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_usd=net_pnl_usd,
            net_pnl_pct=net_pnl_pct,
            commission=commission,
            slippage=slippage,
            spread=spread,
            stop_loss_price=position['sl_price'],
            take_profit_price=position['tp_price'],
            is_partial_exit=True,
            partial_exit_level=partial_level,
        )

    def _extract_indicators_from_row(
        self,
        row: pd.Series,
        strategy,
        mtf_data: Optional[Dict[str, pd.DataFrame]] = None,
        current_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract indicator values from row (ALL timeframes already in row with suffixes)

        Trading Engine approach: All MTF indicators are merged into single DataFrame
        with timeframe suffixes (e.g., ema_89, ema_89_4h)

        Args:
            row: DataFrame row with ALL indicator columns (primary + MTF with suffixes)
            strategy: Strategy instance
            mtf_data: Not used (kept for compatibility)
            current_index: Not used (kept for compatibility)

        Returns:
            Dict with indicator values (primary + MTF)
        """
        indicators = {}

        # Get all indicator names from strategy
        indicator_names = list(strategy.technical_parameters.indicators.keys())

        # Extract ALL indicators from row (primary + MTF, all have suffixes now)
        # Since we merged all MTF indicators into data_with_indicators with suffixes,
        # we just need to extract all indicator-related columns from row

        for indicator_name in indicator_names:
            # 1. Check direct indicator name (primary TF, no suffix)
            if indicator_name in row.index:
                value = row[indicator_name]
                if pd.notna(value):
                    indicators[indicator_name] = float(value)

            # 2. Check indicator_* pattern (e.g., smart_grok_smc_signal)
            matching_cols = [col for col in row.index if col.startswith(f"{indicator_name}_")]
            for col in matching_cols:
                value = row[col]
                if pd.notna(value):
                    indicators[col] = float(value)

        return indicators

    def _close_position(self, position, exit_price, exit_reason, exit_time, config) -> Trade:
        """Close position and create Trade"""

        # Apply spread to exit price
        # LONG exit: sell at bid (exit_price - spread/2)
        # SHORT exit: buy at ask (exit_price + spread/2)
        spread_cost = config.spread_pct / 100 / 2

        if position['side'] == PositionSide.LONG:
            # LONG exit: selling, get bid price
            exit_price_with_spread = exit_price * (1 - spread_cost)
        else:
            # SHORT exit: buying, pay ask price
            exit_price_with_spread = exit_price * (1 + spread_cost)

        # PnL calculation (use spread-adjusted exit price)
        if position['side'] == PositionSide.LONG:
            gross_pnl_usd = (exit_price_with_spread - position['entry_price']) * position['quantity']
        else:
            gross_pnl_usd = (position['entry_price'] - exit_price_with_spread) * position['quantity']

        gross_pnl_pct = (gross_pnl_usd / (position['entry_price'] * position['quantity'])) * 100

        # Costs
        position_value = position['entry_price'] * position['quantity']
        commission = position_value * (config.commission_pct / 100) * 2  # Entry + exit
        slippage = position_value * (config.slippage_pct / 100) * 2

        # Spread cost: entry and exit spread cost
        # Entry spread is already included in entry_price, exit spread is already included in exit_price_with_spread
        # Spread cost = |exit_price - exit_price_with_spread| * quantity + entry spread cost
        exit_spread_cost = abs(exit_price - exit_price_with_spread) * position['quantity']
        # There is also the same spread in the Entry, equal to %spread/2 of the position value
        entry_spread_cost = position_value * (config.spread_pct / 100 / 2)
        spread = entry_spread_cost + exit_spread_cost

        net_pnl_usd = gross_pnl_usd - commission - slippage
        net_pnl_pct = (net_pnl_usd / position_value) * 100

        # Debug output (only when --debug flag is used)
        if self.debug and self.logger:
            from core.timezone_utils import TimezoneUtils

            # Format timestamp
            display_time = TimezoneUtils.format(exit_time, fmt='%Y-%m-%d %H:%M:%S')

            side_str = "LONG" if position['side'] == PositionSide.LONG else "SHORT"
            self.logger.info(f"\n✅ Position closed!")
            self.logger.info(f"   - Time: {display_time}")
            self.logger.info(f"   - Position ID: #{position['id']:03d}")
            self.logger.info(f"   - Side: {side_str}")
            self.logger.info(f"   - Exit price: ${exit_price_with_spread:,.2f}")
            self.logger.info(f"   - Exit type: {exit_reason.value}")

            # WHY EXIT?
            self.logger.info(f"\n   📋 WHY EXIT:")

            if exit_reason == ExitReason.STOP_LOSS:
                self.logger.info(f"      🛑 Stop Loss hit")
                self.logger.info(f"         - Entry: ${position['entry_price']:,.2f}")
                self.logger.info(f"         - Stop Loss: ${position['sl_price']:,.2f}")
                self.logger.info(f"         - Current: ${exit_price:,.2f}")

            elif exit_reason == ExitReason.TAKE_PROFIT:
                self.logger.info(f"      🎯 Take Profit hit")
                self.logger.info(f"         - Entry: ${position['entry_price']:,.2f}")
                self.logger.info(f"         - Take Profit: ${position['tp_price']:,.2f}")
                self.logger.info(f"         - Current: ${exit_price:,.2f}")

            elif exit_reason == ExitReason.SIGNAL:
                self.logger.info(f"      📉 Opposite signal triggered")

            elif exit_reason == ExitReason.END_OF_DATA:
                self.logger.info(f"      🏁 Backtest ended")

            # PnL details
            self.logger.info(f"\n   💰 PnL:")
            pnl_sign = '+' if net_pnl_usd >= 0 else ''
            self.logger.info(f"      - Gross PnL: {pnl_sign}${gross_pnl_usd:,.2f} ({gross_pnl_pct:+.2f}%)")
            self.logger.info(f"      - Commission: ${commission:,.2f}")
            self.logger.info(f"      - Slippage: ${slippage:,.2f}")
            self.logger.info(f"      - Spread: ${spread:,.2f}")
            self.logger.info(f"      - Net PnL: {pnl_sign}${net_pnl_usd:,.2f} ({net_pnl_pct:+.2f}%)")

        # AI Logging: Exit
        if self.enable_ai_logging and self.future_logger:
            try:
                from components.ai.future_logger import TradeOutcome

                # Calculate duration in minutes
                entry_ts = position['entry_time'].timestamp() if isinstance(position['entry_time'], datetime) else position['entry_time'] / 1000
                exit_ts = exit_time.timestamp() if isinstance(exit_time, datetime) else exit_time / 1000
                duration_minutes = int((exit_ts - entry_ts) / 60)

                outcome = TradeOutcome(
                    exit_time=int(exit_time.timestamp() * 1000) if isinstance(exit_time, datetime) else exit_time,
                    exit_price=exit_price_with_spread,
                    exit_reason=exit_reason.value if hasattr(exit_reason, 'value') else str(exit_reason),
                    pnl=net_pnl_usd,
                    pnl_percent=net_pnl_pct,
                    commission=commission,
                    slippage=slippage,
                    duration_minutes=duration_minutes,
                    win=net_pnl_usd > 0,
                    break_even_activated=position.get('break_even_activated', False)
                )
                self.future_logger.log_exit(position['id'], outcome)
            except Exception as e:
                self.logger.debug(f"⚠️  AI exit logging error: {e}")

        return Trade(
            trade_id=position['id'],
            symbol=position['symbol'],
            side=position['side'],
            entry_time=position['entry_time'],
            entry_price=position['entry_price'],
            exit_time=exit_time,
            exit_price=exit_price_with_spread,  # Use spread-adjusted price
            exit_reason=exit_reason,
            quantity=position['quantity'],
            gross_pnl_usd=gross_pnl_usd,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_usd=net_pnl_usd,
            net_pnl_pct=net_pnl_pct,
            commission=commission,
            slippage=slippage,
            spread=spread,
            stop_loss_price=position['sl_price'],
            take_profit_price=position['tp_price'],
            break_even_activated=position.get('break_even_activated', False),
        )

    # Exit logic now handled by StrategyExecutor and ExitManager
    # TP/SL calculation moved to ExitManager (all methods implemented there)

    # ========================================================================
    # EQUITY CURVE
    # ========================================================================

    def _build_equity_curve(self, trades: List[Trade], config: BacktestConfig) -> List[Dict]:
        """Build equity curve from trades"""
        equity_curve = []
        balance = config.initial_balance

        equity_curve.append({
            'time': config.start_date,
            'balance': balance,
            'drawdown': 0.0,
            'pnl': 0.0,
        })

        for trade in trades:
            balance += trade.net_pnl_usd
            equity_curve.append({
                'time': trade.exit_time,
                'balance': balance,
                'drawdown': 0.0,  # Will be calculated
                'pnl': trade.net_pnl_usd,
            })

        return equity_curve

    def _load_ai_predictor(self, strategy: Strategy) -> bool:
        """Load AI model if enabled in strategy config"""
        # 1. Check AIConfig (new style)
        ai_config = getattr(strategy, 'ai_config', None)
        
        if ai_config and ai_config.ai_enabled:
            model_path = ai_config.model_path
            model_type = ai_config.model_type
            
            self.logger.info(f"🤖 AI Enabled: {model_type}")
            self.logger.info(f"   Model: {model_path}")
            
            try:
                if model_type == 'rl_model':
                    # Load RL model (placeholder or existing logic)
                    try:
                        from components.ai.predictor import AIPredictor
                        self._ai_predictor = AIPredictor(model_path, config=ai_config)
                        return True
                    except ImportError:
                        self.logger.warning("⚠️  RL Predictor not found")
                        return False
                    
                elif model_type in ['simple_train', 'xgboost', 'entry_model']:
                    # Load SimpleTrain model
                    from modules.simple_train.inference import SimpleTrainPredictor
                    self._ai_predictor = SimpleTrainPredictor(model_path, strategy_name=strategy.strategy_name)
                    return True
                    
                else:
                    self.logger.warning(f"⚠️  Unknown AI model type: {model_type}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load AI model: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        return False

    def _compute_batch_ai_predictions(
        self,
        data: pd.DataFrame,
        signal_indices: List[int],
        symbol: str,
        timeframe: str,
        signal_dict: Dict[int, int]
    ) -> Dict[int, dict]:
        """
        Compute AI predictions for all signals in batch
        
        Returns:
            Dict[int, dict]: {index: {'win_probability': float, 'confidence': float}}
        """
        try:
            predictions = {}
            
            if hasattr(self._ai_predictor, 'predict_batch'):
                # Pass the full dataframe so predictor can calculate derived features
                probs_dict = self._ai_predictor.predict_batch(data)
                
                # Filter for signal indices
                for idx in signal_indices:
                    if idx in probs_dict:
                        prob = float(probs_dict[idx])
                        predictions[idx] = {
                            'win_probability': prob,
                            'confidence': abs(prob - 0.5) * 2 if prob > 0.5 else 0.0 # Confidence logic
                        }
                        
            return predictions

        except Exception as e:
            self.logger.error(f"❌ AI Batch Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            return {}


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def _prepare_backtest_data(strategy: 'Strategy', logger, debug: bool = False):
    """
    Prepare data for backtest (download/update if needed)

    Prepares the necessary data according to the strategy parameters:
    - download_klines=True: Download missing files
    - update_klines=True: Update existing files

    Args:
        strategy: Strategy instance
        logger: Logger instance
        debug: Enable verbose output
    """
    from components.data.data_downloader import DataDownloader
    from pathlib import Path
    import pandas as pd

    download_klines = getattr(strategy, 'download_klines', False)
    update_klines = getattr(strategy, 'update_klines', False)

    # Get backtest parameters
    symbol_config = strategy.symbols[0]
    symbol = f"{symbol_config.symbol[0]}{symbol_config.quote}"
    timeframes = strategy.mtf_timeframes
    start_date = strategy.backtest_start_date
    end_date = strategy.backtest_end_date
    warmup_candles = strategy.warmup_period

    # Normalize dates to YYYY-MM-DD format (remove time component if present)
    if 'T' in start_date:
        start_date = start_date.split('T')[0]
    if end_date and 'T' in end_date:
        end_date = end_date.split('T')[0]

    if debug:
        logger.info(f"\n📊 Backtest Parameters:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Timeframes: {timeframes}")
        logger.info(f"   Date range: {start_date} to {end_date}")
        logger.info(f"   Warmup: {warmup_candles} candles")

    # Calculate warmup start date
    start_dt = pd.to_datetime(start_date)

    # Process each timeframe
    for timeframe in timeframes:
        logger.info(f"\n📂 Checking {symbol} {timeframe}...")

        # Calculate warmup start
        tf_minutes = _parse_timeframe_to_minutes(timeframe)
        warmup_minutes = warmup_candles * tf_minutes
        warmup_days = warmup_minutes / (60 * 24)
        warmup_start_dt = start_dt - pd.Timedelta(days=warmup_days)
        warmup_start_str = warmup_start_dt.strftime("%Y-%m-%d")

        # Determine required years
        warmup_year = warmup_start_dt.year
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
        end_year = end_dt.year
        years_needed = list(range(warmup_year, end_year + 1))

        if debug:
            logger.info(f"   Warmup start: {warmup_start_str}")
            logger.info(f"   Required years: {years_needed}")

        # Check which files exist - yeni format: data/parquets/{symbol}/
        data_dir = Path("data/parquets")
        symbol_dir = data_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        timeframe_safe = timeframe.replace('M', 'MO')  # Fix case sensitivity

        missing_years = []
        existing_years = []

        for year in years_needed:
            filepath = symbol_dir / f"{symbol}_{timeframe_safe}_{year}.parquet"
            if filepath.exists():
                existing_years.append(year)
            else:
                missing_years.append(year)

        if existing_years and debug:
            logger.info(f"   ✅ Existing: {existing_years}")
        if missing_years:
            logger.info(f"   ❌ Missing: {missing_years}")

        # Download missing files if download_klines=True
        if download_klines and missing_years:
            logger.info(f"   🔽 Downloading missing data for {timeframe}...")

            downloader = DataDownloader()

            await downloader.download(
                symbol=symbol,
                timeframe=timeframe,
                start_date=warmup_start_str,
                end_date=end_date,
                output_dir=str(data_dir)
            )

            logger.info(f"   ✅ Download complete")

        # Update existing files if update_klines=True
        if update_klines and existing_years:
            logger.info(f"   🔄 Updating existing data for {timeframe}...")

            downloader = DataDownloader()

            await downloader.update(
                symbol=symbol,
                timeframe=timeframe,
                output_dir=str(data_dir)
            )

            logger.info(f"   ✅ Update complete")

        # Warn if files still missing
        if not download_klines and missing_years:
            logger.warning(f"   ⚠️  Missing files detected but download_klines=False")
            logger.warning(f"      Set download_klines=True to auto-download")


def _parse_timeframe_to_minutes(timeframe: str) -> int:
    """Parse timeframe string to minutes (e.g., '15m' -> 15, '1h' -> 60)"""
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 24
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 60 * 24 * 7
    elif timeframe == '1M':
        return 60 * 24 * 30  # Approximate
    else:
        return 1  # Default: 1m


async def run_backtest_cli(
    strategy_path: str,
    verbose: bool = False,
    symbol: str = None,
    timeframe: str = None,
    start_date: str = None,
    end_date: str = None,
    initial_balance: float = None
):
    """
    Run backtest from command line

    Args:
        strategy_path: Path to strategy file (short or full path)
        verbose: Enable verbose output (indicators, conditions, signals)
        symbol: Override symbol (optional)
        timeframe: Override timeframe (optional)
        start_date: Override start date (optional, YYYY-MM-DD)
        end_date: Override end date (optional, YYYY-MM-DD)
        initial_balance: Override initial balance (optional)
    """
    import sys
    from pathlib import Path
    from components.strategies.strategy_manager import StrategyManager

    # Get logger for CLI
    logger = get_logger("modules.backtest.cli")

    # Parse strategy path
    if not strategy_path.endswith('.py'):
        strategy_path += '.py'

    # If short name provided, assume it's in templates
    if '/' not in strategy_path and '\\' not in strategy_path:
        strategy_path = f"components/strategies/templates/{strategy_path}"

    # Normalize path
    strategy_path = Path(strategy_path).as_posix()

    # Check if file exists
    if not Path(strategy_path).exists():
        logger.error("=" * 60)
        logger.error("❌ ERROR: Strategy file not found")
        logger.error("=" * 60)
        logger.error(f"File: {strategy_path}")
        logger.error(f"Mutlak yol: {Path(strategy_path).absolute()}")
        logger.error("Current strategies (templates/):")
        templates_dir = Path("components/strategies/templates")
        if templates_dir.exists():
            for f in sorted(templates_dir.glob("*.py")):
                if not f.name.startswith('_'):
                    logger.error(f"  - {f.name}")
        logger.error("=" * 60)
        sys.exit(1)

    # Load strategy
    if verbose:
        logger.info("=" * 60)
        logger.info(f"📂 Strategy loading: {strategy_path}")
        logger.info("=" * 60)

    try:
        strategy_manager = StrategyManager()
        strategy, _ = strategy_manager.load_strategy(strategy_path, validate=True)
    except FileNotFoundError as e:
        logger.error("=" * 60)
        logger.error("❌ ERROR: Strategy file not found")
        logger.error("=" * 60)
        logger.error(f"{e}")
        logger.error("=" * 60)
        sys.exit(1)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERROR: Strategy could not be loaded")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error(f"File: {strategy_path}")
        if verbose:
            import traceback
            logger.error("Detailed error:")
            traceback.print_exc()
        logger.error("=" * 60)
        sys.exit(1)

    if verbose:
        logger.info("✅ Strategy loaded:")
        logger.info(f"   Name: {strategy.strategy_name} v{strategy.strategy_version}")
        logger.info(f"   Timeframe: {strategy.primary_timeframe}")
        logger.info(f"   Starting Balance: ${strategy.initial_balance:,.0f}")
        logger.info(f"   Symbols: {getattr(strategy, 'symbols', 'N/A')}")

    # Apply CLI overrides if provided
    if symbol or timeframe or start_date or end_date or initial_balance:
        logger.info("\n📝 CLI parameters are being applied...")

        if symbol:
            # Override symbols (create SymbolConfig if needed)
            from components.strategies.base_strategy import SymbolConfig
            if '/' in symbol:
                base, quote = symbol.split('/')
                strategy.symbols = [SymbolConfig(symbol=[base], quote=quote, enabled=True)]
            else:
                # Assume USDT pair
                strategy.symbols = [SymbolConfig(symbol=[symbol.replace('USDT', '')], quote='USDT', enabled=True)]
            logger.info(f"   Symbol override: {symbol}")

        if timeframe:
            strategy.primary_timeframe = timeframe
            # Also update MTF if it only contains the old primary
            if hasattr(strategy, 'mtf_timeframes') and len(strategy.mtf_timeframes) == 1:
                strategy.mtf_timeframes = [timeframe]
            logger.info(f"   Timeframe override: {timeframe}")

        if start_date:
            strategy.backtest_start_date = f"{start_date}T00:00"
            logger.info(f"   Start date override: {start_date}")

        if end_date:
            strategy.backtest_end_date = f"{end_date}T23:59"
            logger.info(f"   End date override: {end_date}")

        if initial_balance:
            strategy.initial_balance = initial_balance
            logger.info(f"   Initial balance override: ${initial_balance:,.0f}")

    # Check if data download/update needed (BEFORE running backtest)
    download_klines = getattr(strategy, 'download_klines', False)
    update_klines = getattr(strategy, 'update_klines', False)

    if download_klines or update_klines:
        logger.info("\n" + "=" * 60)
        logger.info("📥 Data preparation is being checked...")
        logger.info("=" * 60)
        logger.info(f"   download_klines: {download_klines}")
        logger.info(f"   update_klines: {update_klines}")

        try:
            await _prepare_backtest_data(strategy, logger, verbose)
        except Exception as e:
            logger.error("=" * 60)
            logger.error("❌ ERROR: Data preparation failed")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            if verbose:
                import traceback
                logger.error("\nDetailed error:")
                traceback.print_exc()
            logger.error("=" * 60)
            sys.exit(1)

        logger.info("✅ Data preparation completed")
        logger.info("=" * 60)

    # Create engine with verbose flag
    engine = BacktestEngine(debug=verbose)

    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info("🚀 Backtest is running...")
        logger.info("=" * 60)

    # Run backtest
    try:
        result = await engine.run(strategy, use_cache=True)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERROR: Backtest could not be run")
        logger.error("=" * 60)
        logger.error(f"\nHata: {e}")
        if verbose:
            import traceback
            logger.error("\nDetailed error:")
            traceback.print_exc()
        logger.error("\n" + "=" * 60)
        sys.exit(1)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nStrateji: {strategy.strategy_name} v{strategy.strategy_version}")
    logger.info(f"Period: {result.config.start_date.date()} -> {result.config.end_date.date()}")
    logger.info(f"Symbol: {result.config.symbols[0]} - {strategy.leverage}x")
    logger.info(f"Primary TF: {result.config.primary_timeframe}")
    if len(result.config.mtf_timeframes) > 1:
        logger.info(f"MTF: {', '.join(result.config.mtf_timeframes)}")

    logger.info(f"\n💼 PERFORMANCE:")
    logger.info(f"   Total Trade: {result.metrics.total_trades}")
    logger.info(f"   Total Return: ${result.metrics.total_return_usd:,.2f} ({result.metrics.total_return_pct:+.2f}%)")
    logger.info(f"   Win Rate: {result.metrics.win_rate:.2f}%")
    logger.info(f"   Profit Factor: {result.metrics.profit_factor:.2f}")
    logger.info(f"   Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
    logger.info(f"   Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")

    logger.info(f"\n📈 DETAILS:")
    logger.info(f"   Winner: {result.metrics.winners} (avg: ${result.metrics.avg_win_usd:,.2f})")
    logger.info(f"   Losers: {result.metrics.losers} (avg: ${result.metrics.avg_loss_usd:,.2f})")

    logger.info(f"\n💰 COSTS:")
    logger.info(f"   Commission: ${result.metrics.total_commission:,.2f}")
    logger.info(f"   Slippage: ${result.metrics.total_slippage:,.2f}")
    logger.info(f"   Spread: ${result.metrics.total_spread:,.2f}")

    logger.info(f"\n⏱️  Execution Time: {result.execution_time_seconds:.2f}s")

    if verbose:
        logger.info(f"\n📋 TRADES:")
        for i, trade in enumerate(result.trades[:5], 1):  # First 5 trades
            logger.info(f"   #{i}: {trade.side.value} @ ${trade.entry_price:,.2f} -> ${trade.exit_price:,.2f} "
                    f"({trade.exit_reason.value}) = ${trade.net_pnl_usd:+.2f}")
        if len(result.trades) > 5:
            logger.info(f"   ... and {len(result.trades) - 5} trade more")

    logger.info("\n" + "=" * 60)

    # Save results to files
    from modules.backtest.backtest_export import save_backtest_results
    save_backtest_results(result, output_dir="data/backtest_results", logger=logger)

    return result


async def bulk_backtest(
    strategy_path: str,
    symbols: list[str],
    timeframes: list[str],
    start_date: str = None,
    end_date: str = None,
    initial_balance: float = None,
    verbose: bool = False
):
    """
    Bulk backtest - Test multiple symbol and timeframe combinations

    Args:
        strategy_path: Strategy file path
        symbols: Symbol list (e.g. ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
        timeframes: Timeframe list (e.g. ['1d', '1w', '4h'])
        start_date: Start date (optional)
        end_date: End date (optional)
        initial_balance: Initial balance (optional)
        verbose: Detailed output

    Returns:
        list[BacktestResult]: Results of all combinations
    """
    import itertools
    from datetime import datetime
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("🚀 BULK BACKTEST STARTED")
    logger.info("=" * 80)
    logger.info(f"Strategy: {strategy_path}")
    logger.info(f"Symbols: {', '.join(symbols)} ({len(symbols)} items)")
    logger.info(f"Timeframes: {', '.join(timeframes)} ({len(timeframes)} adet)")

    # Create all combinations
    combinations = list(itertools.product(symbols, timeframes))
    total_tests = len(combinations)

    logger.info(f"Total Test Count: {total_tests}")
    logger.info("=" * 80)

    results = []
    start_time = datetime.now()

    for idx, (symbol, timeframe) in enumerate(combinations, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Test {idx}/{total_tests}: {symbol} @ {timeframe}")
        logger.info(f"{'='*80}")

        try:
            # Run backtest for each combination
            result = await run_backtest_cli(
                strategy_path=strategy_path,
                verbose=verbose,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance
            )
            results.append(result)

            # Short summary
            total_trades = len(result.trades)
            total_return = result.metrics.total_return_pct
            win_rate = result.metrics.win_rate
            logger.info(f"\n✅ {symbol} @ {timeframe} - Completed!")
            logger.info(f"   Trades: {total_trades}, Win Rate: {win_rate:.1f}%, Return: {total_return:+.2f}%")

        except Exception as e:
            logger.error(f"❌ {symbol} @ {timeframe} - ERROR: {e}")
            continue

    # General summary
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 80)
    logger.info("📊 BULK BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {total_tests - len(results)}")
    logger.info(f"Total Duration: {elapsed:.1f}s")
    logger.info("")

    # Show the best results
    if results:
        sorted_by_return = sorted(results, key=lambda r: r.metrics.total_return_pct, reverse=True)

        logger.info("🏆 BEST 5 RESULTS (Return):")
        for i, result in enumerate(sorted_by_return[:5], 1):
            symbol = result.config.symbols[0]
            timeframe = result.config.primary_timeframe
            total_trades = len(result.trades)
            total_return = result.metrics.total_return_pct
            win_rate = result.metrics.win_rate
            logger.info(f"   {i}. {symbol} @ {timeframe}: "
                       f"{total_return:+.2f}% ({total_trades} trades, WR: {win_rate:.1f}%)")

        logger.info("")
        logger.info("📉 WORST 5 RESULTS (Return):")
        for i, result in enumerate(sorted_by_return[-5:][::-1], 1):
            symbol = result.config.symbols[0]
            timeframe = result.config.primary_timeframe
            total_trades = len(result.trades)
            total_return = result.metrics.total_return_pct
            win_rate = result.metrics.win_rate
            logger.info(f"   {i}. {symbol} @ {timeframe}: "
                       f"{total_return:+.2f}% ({total_trades} trades, WR: {win_rate:.1f}%)")

        # Genel istatistikler
        avg_return = sum(r.metrics.total_return_pct for r in results) / len(results)
        avg_trades = sum(len(r.trades) for r in results) / len(results)
        avg_winrate = sum(r.metrics.win_rate for r in results) / len(results)

        logger.info("")
        logger.info("📈 AVERAGE VALUES:")
        logger.info(f"   Return: {avg_return:.2f}%")
        logger.info(f"   Number of Trades: {avg_trades:.1f}")
        logger.info(f"   Win Rate: {avg_winrate:.1f}%")

    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    import argparse
    import asyncio

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="BacktestEngine V3 - Run strategy backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # Disable prefix matching (e.g., --verbo won't work)
        epilog="""
        Examples:
        python -m modules.backtest.backtest_engine --strategy simple_rsi.py
        python -m modules.backtest.backtest_engine --strategy components/strategies/templates/TradingView_Dashboard.py --verbose
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy file path (short name or full path)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output (indicators, conditions, signals)'
    )

    # Bulk mode
    parser.add_argument(
        '--bulk',
        action='store_true',
        help='Bulk backtest mode - Test multiple symbols and timeframes'
    )

    # Backtest config override (optional - strategy'den okunur, CLI override eder)
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g. BTCUSDT) or comma-separated list (for bulk mode)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g. 5m, 1h) or comma-separated list (for bulk mode)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD) - default taken from Strategy')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD) - Default taken from Strategy')
    parser.add_argument('--balance', type=float, help='Initial balance - Default taken from Strategy')

    args = parser.parse_args()

    # Bulk mode control
    if args.bulk:
        # Bulk backtest - parse comma-separated lists
        symbols = args.symbol.split(',') if args.symbol else ['BTCUSDT']
        timeframes = args.timeframe.split(',') if args.timeframe else ['5m']

        asyncio.run(bulk_backtest(
            strategy_path=args.strategy,
            symbols=symbols,
            timeframes=timeframes,
            start_date=args.start,
            end_date=args.end,
            initial_balance=args.balance,
            verbose=args.verbose
        ))
    else:
        # Normal single backtest
        asyncio.run(run_backtest_cli(
            strategy_path=args.strategy,
            verbose=args.verbose,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            initial_balance=args.balance
        ))
