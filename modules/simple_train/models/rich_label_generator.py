#!/usr/bin/env python3
"""
modules/simple_train/models/rich_label_generator.py
SuperBot - Rich Label Generator
Author: SuperBot Team
Date: 2026-01-15
Versiyon: 1.0.0

Rich label generator for multi-output models.

Instead of binary classification, it extracts detailed information about the trade:
- result: WIN/LOSE (binary compat)
- pnl_pct: Actual PnL percentage
- exit_reason: TP/SL/BE/PE/TS/TIMEOUT
- bars_to_exit: How many bars until exit
- max_favorable: Maximum observed profit
- max_adverse: Maximum observed loss

Usage:
    gen = RichLabelGenerator()
    labels_df = gen.generate(df, signals, tp_percent=3.0, sl_percent=2.0)
    
    # labels_df columns:
    # - result, pnl_pct, exit_reason, bars_to_exit, max_favorable, max_adverse
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    logger = get_logger("modules.simple_train.models.rich_label_generator")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.models.rich_label_generator")


# =============================================================================
# RICH LABEL GENERATOR
# =============================================================================

class RichLabelGenerator:
    """
    Creates rich labels from trade results.
    
    Binary classification yerine multi-dimensional output:
    - result: 1=WIN, 0=LOSE (backward compatible)
    - pnl_pct: Final PnL percentage
    - exit_reason: TP/SL/BE/PE/TS/TIMEOUT
    - bars_to_exit: Trade duration
    - max_favorable: Maximum profit during the trade
    - max_adverse: Maximum loss during the trade
    - peak_to_exit_ratio: max_favorable / abs(pnl_pct) (how early did we exit?)
    """

    def __init__(self, config: dict | None = None):
        """
        Initializes the RichLabelGenerator.
        
        Args:
            config: Config dictionary (optional)
        """
        self.config = config or {}
        logger.info("🏷️ RichLabelGenerator started (multi-output mode)")

    def generate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        tp_percent: float = 3.0,
        sl_percent: float = 2.0,
        timeout_bars: int = 100,
        strategy = None
    ) -> pd.DataFrame:
        """
        Creates rich labels.
        
        Args:
            df: OHLCV DataFrame
            signals: Trade signals (1=LONG, -1=SHORT, 0=HOLD)
            tp_percent: Take profit %
            sl_percent: Stop loss %
            timeout_bars: Maximum position duration
            strategy: Strategy instance (for BE/PE/TS)
            
        Returns:
            pd.DataFrame: Rich labels
                Columns: result, pnl_pct, exit_reason, bars_to_exit, 
                         max_favorable, max_adverse, peak_to_exit_ratio
        """
        logger.info(f"📊 Creating rich labels...")
        logger.info(f"   TP: %{tp_percent}, SL: %{sl_percent}, Timeout: {timeout_bars} bar")
        
        # Create an empty DataFrame
        labels_df = pd.DataFrame(index=df.index)
        labels_df['result'] = np.nan
        labels_df['pnl_pct'] = np.nan
        labels_df['exit_reason'] = None
        labels_df['bars_to_exit'] = np.nan
        labels_df['max_favorable'] = np.nan
        labels_df['max_adverse'] = np.nan
        labels_df['peak_to_exit_ratio'] = np.nan
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        signal_indices = signals[signals != 0].index
        
        for idx in signal_indices:
            pos = df.index.get_loc(idx)
            if pos >= len(df) - 1:
                continue
            
            signal_type = signals.loc[idx]  # 1=LONG, -1=SHORT
            entry_price = close[pos]
            
            if signal_type == 1:  # LONG
                tp_price = entry_price * (1 + tp_percent / 100)
                sl_price = entry_price * (1 - sl_percent / 100)
            else:  # SHORT
                tp_price = entry_price * (1 - tp_percent / 100)
                sl_price = entry_price * (1 + sl_percent / 100)
            
            # Trade simulation - returns rich information
            trade_info = self._simulate_trade_rich(
                high[pos+1:min(pos+1+timeout_bars, len(df))],
                low[pos+1:min(pos+1+timeout_bars, len(df))],
                close[pos+1:min(pos+1+timeout_bars, len(df))],
                signal_type, entry_price, tp_price, sl_price, strategy
            )
            
            # DataFrame'e yaz
            labels_df.loc[idx, 'result'] = 1 if trade_info['pnl_pct'] > 0 else 0
            labels_df.loc[idx, 'pnl_pct'] = trade_info['pnl_pct']
            labels_df.loc[idx, 'exit_reason'] = trade_info['exit_reason']
            labels_df.loc[idx, 'bars_to_exit'] = trade_info['bars_to_exit']
            labels_df.loc[idx, 'max_favorable'] = trade_info['max_favorable']
            labels_df.loc[idx, 'max_adverse'] = trade_info['max_adverse']
            labels_df.loc[idx, 'peak_to_exit_ratio'] = trade_info['peak_to_exit_ratio']
        
        # Statistics
        valid_labels = labels_df.dropna(subset=['result'])
        if len(valid_labels) > 0:
            win_count = (valid_labels['result'] == 1).sum()
            lose_count = (valid_labels['result'] == 0).sum()
            avg_win_pnl = valid_labels[valid_labels['result'] == 1]['pnl_pct'].mean()
            avg_lose_pnl = valid_labels[valid_labels['result'] == 0]['pnl_pct'].mean()
            
            logger.info(f"   ✅ {len(valid_labels)} trades simulated")
            logger.info(f"      📈 WIN: {win_count} (avg: {avg_win_pnl:.2f}%)")
            logger.info(f"      📉 LOSE: {lose_count} (avg: {avg_lose_pnl:.2f}%)")
            
            # Exit reason istatistikleri
            exit_counts = valid_labels['exit_reason'].value_counts()
            logger.info(f"      🚪 Exit Reasons:")
            for reason, count in exit_counts.items():
                logger.info(f"         {reason}: {count}")
        
        return labels_df

    def _simulate_trade_rich(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        signal_type: int,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        strategy = None
    ) -> dict:
        """
        Trade simulation - returns rich information.
        
        Returns:
            dict: {
                'pnl_pct': float,
                'exit_reason': str,
                'bars_to_exit': int,
                'max_favorable': float,
                'max_adverse': float,
                'peak_to_exit_ratio': float
            }
        """
        if len(high) == 0:
            return {
                'pnl_pct': -100.0,
                'exit_reason': 'TIMEOUT_NO_DATA',
                'bars_to_exit': 0,
                'max_favorable': 0.0,
                'max_adverse': -100.0,
                'peak_to_exit_ratio': 0.0
            }
        
        # Strategy exit config
        if strategy is not None and hasattr(strategy, 'exit_strategy'):
            exit_cfg = strategy.exit_strategy
            be_enabled = exit_cfg.break_even_enabled
            be_trigger = exit_cfg.break_even_trigger_profit_percent
            be_offset = exit_cfg.break_even_offset
            
            pe_enabled = exit_cfg.partial_exit_enabled
            pe_levels = exit_cfg.partial_exit_levels
            pe_sizes = exit_cfg.partial_exit_sizes
            
            ts_enabled = exit_cfg.trailing_stop_enabled
            ts_activation = exit_cfg.trailing_activation_profit_percent
            ts_callback = exit_cfg.trailing_callback_percent
        else:
            be_enabled = False
            pe_enabled = False
            ts_enabled = False
        
        # Tracking
        position_size = 1.0
        current_sl = sl_price
        trailing_active = False
        highest_profit = 0.0
        lowest_profit = 0.0  # Track max adverse
        pe_executed = [False] * (len(pe_levels) if pe_enabled else 0)
        
        # Bar-by-bar simulation
        for i in range(len(high)):
            current_high = high[i]
            current_low = low[i]
            current_close = close[i]
            
            # Profit calculation
            if signal_type == 1:  # LONG
                profit_pct = (current_close - entry_price) / entry_price * 100
                bar_high_pct = (current_high - entry_price) / entry_price * 100
                bar_low_pct = (current_low - entry_price) / entry_price * 100
            else:  # SHORT
                profit_pct = (entry_price - current_close) / entry_price * 100
                bar_high_pct = (entry_price - current_low) / entry_price * 100
                bar_low_pct = (entry_price - current_high) / entry_price * 100
            
            # Track extremes
            highest_profit = max(highest_profit, profit_pct, bar_high_pct)
            lowest_profit = min(lowest_profit, profit_pct, bar_low_pct)
            
            # ================================================================
            # 1. STOP LOSS CHECK
            # ================================================================
            if signal_type == 1:
                if current_low <= current_sl:
                    sl_pnl = (current_sl - entry_price) / entry_price * 100 * position_size
                    return {
                        'pnl_pct': sl_pnl,
                        'exit_reason': 'SL',
                        'bars_to_exit': i + 1,
                        'max_favorable': highest_profit,
                        'max_adverse': lowest_profit,
                        'peak_to_exit_ratio': highest_profit / abs(sl_pnl) if sl_pnl != 0 else 0.0
                    }
            else:
                if current_high >= current_sl:
                    sl_pnl = (entry_price - current_sl) / entry_price * 100 * position_size
                    return {
                        'pnl_pct': sl_pnl,
                        'exit_reason': 'SL',
                        'bars_to_exit': i + 1,
                        'max_favorable': highest_profit,
                        'max_adverse': lowest_profit,
                        'peak_to_exit_ratio': highest_profit / abs(sl_pnl) if sl_pnl != 0 else 0.0
                    }
            
            # ================================================================
            # 2. TAKE PROFIT CHECK
            # ================================================================
            if signal_type == 1:
                if current_high >= tp_price:
                    tp_pnl = (tp_price - entry_price) / entry_price * 100 * position_size
                    return {
                        'pnl_pct': tp_pnl,
                        'exit_reason': 'TP',
                        'bars_to_exit': i + 1,
                        'max_favorable': highest_profit,
                        'max_adverse': lowest_profit,
                        'peak_to_exit_ratio': highest_profit / tp_pnl if tp_pnl != 0 else 1.0
                    }
            else:
                if current_low <= tp_price:
                    tp_pnl = (entry_price - tp_price) / entry_price * 100 * position_size
                    return {
                        'pnl_pct': tp_pnl,
                        'exit_reason': 'TP',
                        'bars_to_exit': i + 1,
                        'max_favorable': highest_profit,
                        'max_adverse': lowest_profit,
                        'peak_to_exit_ratio': highest_profit / tp_pnl if tp_pnl != 0 else 1.0
                    }
            
            # ================================================================
            # 3. BREAK-EVEN
            # ================================================================
            if be_enabled and profit_pct >= be_trigger:
                if signal_type == 1:
                    new_be_sl = entry_price * (1 + be_offset / 100)
                    if new_be_sl > current_sl:
                        current_sl = new_be_sl
                else:
                    new_be_sl = entry_price * (1 - be_offset / 100)
                    if new_be_sl < current_sl:
                        current_sl = new_be_sl
            
            # ================================================================
            # 4. PARTIAL EXIT
            # ================================================================
            if pe_enabled:
                for idx, (level, size) in enumerate(zip(pe_levels, pe_sizes)):
                    if pe_executed[idx]:
                        continue
                    
                    if profit_pct >= level:
                        pe_executed[idx] = True
                        position_size -= size
                        
                        if position_size <= 0.01:
                            # All positions are closed
                            return {
                                'pnl_pct': profit_pct,
                                'exit_reason': f'PE_{level}',
                                'bars_to_exit': i + 1,
                                'max_favorable': highest_profit,
                                'max_adverse': lowest_profit,
                                'peak_to_exit_ratio': highest_profit / profit_pct if profit_pct != 0 else 1.0
                            }
            
            # ================================================================
            # 5. TRAILING STOP
            # ================================================================
            if ts_enabled:
                if not trailing_active and profit_pct >= ts_activation:
                    trailing_active = True
                
                if trailing_active:
                    pullback = highest_profit - profit_pct
                    if pullback >= ts_callback:
                        return {
                            'pnl_pct': profit_pct * position_size,
                            'exit_reason': 'TRAILING_STOP',
                            'bars_to_exit': i + 1,
                            'max_favorable': highest_profit,
                            'max_adverse': lowest_profit,
                            'peak_to_exit_ratio': highest_profit / (profit_pct * position_size) if profit_pct != 0 else 1.0
                        }
        
        # ================================================================
        # 6. TIMEOUT
        # ================================================================
        final_price = close[-1]
        if signal_type == 1:
            timeout_pnl = (final_price - entry_price) / entry_price * 100
        else:
            timeout_pnl = (entry_price - final_price) / entry_price * 100
        
        timeout_pnl *= position_size
        
        return {
            'pnl_pct': timeout_pnl,
            'exit_reason': 'TIMEOUT',
            'bars_to_exit': len(high),
            'max_favorable': highest_profit,
            'max_adverse': lowest_profit,
            'peak_to_exit_ratio': highest_profit / abs(timeout_pnl) if timeout_pnl != 0 else 0.0
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 RichLabelGenerator Test")
    print("=" * 60)
    
    # Dummy OHLCV
    n = 200
    np.random.seed(42)
    df = pd.DataFrame({
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(105, 115, n),
        "low": np.random.uniform(95, 105, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 5000, n)
    })
    
    # Ensure high > low
    df["high"] = df[["open", "close"]].max(axis=1) + np.random.uniform(0, 5, n)
    df["low"] = df[["open", "close"]].min(axis=1) - np.random.uniform(0, 5, n)
    
    # Dummy signals
    signals = pd.Series(0, index=df.index)
    signals.iloc[10] = 1   # LONG
    signals.iloc[50] = -1  # SHORT
    signals.iloc[100] = 1  # LONG
    signals.iloc[150] = -1 # SHORT
    
    # Generate rich labels
    gen = RichLabelGenerator()
    labels_df = gen.generate(df, signals, tp_percent=3.0, sl_percent=2.0, timeout_bars=50)
    
    # Show results
    print("\n📊 Rich Labels:")
    valid = labels_df.dropna(subset=['result'])
    print(valid[['result', 'pnl_pct', 'exit_reason', 'bars_to_exit', 'max_favorable', 'max_adverse']])
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)
