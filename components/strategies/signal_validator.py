#!/usr/bin/env python3
"""
components/strategies/signal_validator.py
SuperBot - Signal Validator

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Entry/exit koşullarını değerlendirir.

    - Koşul parsing
    - Indicator value extraction
    - Condition evaluation
    - Multi-timeframe support

Kullanım:
    from components.strategies.signal_validator import SignalValidator

    validator = SignalValidator(strategy)
    signal = validator.evaluate_entry(symbol='BTCUSDT', data=df)
    # {'signal': 'LONG', 'score': 1.0, 'conditions_met': [...]}
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from components.strategies.base_strategy import BaseStrategy
from components.strategies.helpers import (
    ConditionParser,
    evaluate_condition,
    parse_conditions,
)


class SignalValidator:
    """
    Sinyal değerlendirme validator'ı

    Entry ve exit koşullarını değerlendirir
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        logger: Any = None
    ):
        """
        Initialize SignalValidator

        Args:
            strategy: BaseStrategy instance
            logger: Logger instance (optional)
        """
        self.strategy = strategy
        self.logger = logger
        
        # Parsers
        self.parser = ConditionParser()
        
        # Parse conditions once at init
        self.parsed_entry_long = parse_conditions(strategy.entry_conditions.get('long', []))
        self.parsed_entry_short = parse_conditions(strategy.entry_conditions.get('short', []))
        self.parsed_exit_long = parse_conditions(strategy.exit_conditions.get('long', []))
        self.parsed_exit_short = parse_conditions(strategy.exit_conditions.get('short', []))
    
    # ========================================================================
    # ENTRY SIGNAL EVALUATION
    # ========================================================================
    
    def evaluate_entry(
        self,
        symbol: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_positions: Optional[Dict] = None,
        _verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Entry sinyalini değerlendir

        Args:
            symbol: Trading symbol
            data: Market data
                - Single timeframe: pd.DataFrame
                - Multi-timeframe: {'5m': df, '15m': df, ...}
            current_positions: Mevcut pozisyonlar (optional)
            _verbose: Temporary verbose flag (for debug status logging)

        Returns:
            Dict: Signal result
                {
                    'signal': 'LONG' | 'SHORT' | None,
                    'score': float (0-1),
                    'conditions_met': [...],
                    'conditions_failed': [...]
                }
        """
        result = {
            'signal': None,
            'score': 0.0,
            'conditions_met': [],
            'conditions_failed': [],
            'pending_side': None,  # Hangi yöne doğru koşullar sağlanıyor
            'timestamp': None
        }

        # Check long signal
        long_result = self._evaluate_side('long', data, self.parsed_entry_long, _verbose)

        # Separator between LONG and SHORT in verbose mode
        if _verbose and self.logger:
            self.logger.info("      " + "-" * 50)

        # Check short signal
        short_result = self._evaluate_side('short', data, self.parsed_entry_short, _verbose)
        
        # Determine signal (long takes priority if both are true)
        if long_result['all_met']:
            result['signal'] = 'LONG'
            result['pending_side'] = 'LONG'
            result['score'] = long_result['score']
            result['conditions_met'] = long_result['met']
            result['conditions_failed'] = long_result['failed']
        elif short_result['all_met']:
            result['signal'] = 'SHORT'
            result['pending_side'] = 'SHORT'
            result['score'] = short_result['score']
            result['conditions_met'] = short_result['met']
            result['conditions_failed'] = short_result['failed']
        else:
            # No signal - show conditions for the side that's closer (higher score)
            if long_result['score'] >= short_result['score']:
                # Show LONG conditions
                result['pending_side'] = 'LONG'
                result['conditions_met'] = long_result['met']
                result['conditions_failed'] = long_result['failed']
                result['score'] = long_result['score']
            else:
                # Show SHORT conditions
                result['pending_side'] = 'SHORT'
                result['conditions_met'] = short_result['met']
                result['conditions_failed'] = short_result['failed']
                result['score'] = short_result['score']
        
        # Add timestamp from data
        if isinstance(data, pd.DataFrame) and not data.empty:
            if 'timestamp' in data.columns:
                result['timestamp'] = data['timestamp'].iloc[-1]
            elif 'open_time' in data.columns:
                result['timestamp'] = data['open_time'].iloc[-1]
        
        return result
    
    def evaluate_exit(
        self,
        symbol: str,
        position_side: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        position: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Exit sinyalini değerlendir
        
        Args:
            symbol: Trading symbol
            position_side: 'LONG' or 'SHORT'
            data: Market data
            position: Position data (optional, for PnL-based exits)
        
        Returns:
            Dict: Exit signal result
                {
                    'should_exit': bool,
                    'reason': str,
                    'score': float,
                    'conditions_met': [...]
                }
        """
        result = {
            'should_exit': False,
            'reason': None,
            'score': 0.0,
            'conditions_met': [],
            'conditions_failed': [],
        }
        
        # Select appropriate exit conditions
        if position_side.upper() == 'LONG':
            parsed_conditions = self.parsed_exit_long
        elif position_side.upper() == 'SHORT':
            parsed_conditions = self.parsed_exit_short
        else:
            return result
        
        # Evaluate exit conditions
        exit_result = self._evaluate_side(position_side.lower(), data, parsed_conditions)
        
        if exit_result['all_met']:
            result['should_exit'] = True
            result['reason'] = 'indicator_exit'
            result['score'] = exit_result['score']
            result['conditions_met'] = exit_result['met']
        else:
            result['conditions_failed'] = exit_result['failed']
        
        return result
    
    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================
    
    def _evaluate_side(
        self,
        side: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parsed_conditions: List[Dict],
        _verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Bir tarafın (long/short) koşullarını değerlendir

        Args:
            side: 'long' or 'short'
            data: Market data
            parsed_conditions: Parse edilmiş koşullar
            _verbose: Temporary verbose flag (for debug status logging)

        Returns:
            Dict: Evaluation result
                {
                    'all_met': bool,
                    'score': float (0-1),
                    'met': [...],
                    'failed': [...]
                }
        """
        if not parsed_conditions:
            # No conditions = no signal
            return {'all_met': False, 'score': 0.0, 'met': [], 'failed': []}

        met = []
        failed = []

        # DEBUG: Print available indicator keys once (disabled - too verbose)
        # if self.logger and _verbose and parsed_conditions:
        #     if isinstance(data, pd.DataFrame):
        #         self.logger.info(f"📋 Available Keys: {list(data.columns)}")
        #     elif isinstance(data, dict):
        #         for tf, df in data.items():
        #             if isinstance(df, pd.DataFrame):
        #                 self.logger.info(f"📋 Available Keys ({tf}): {list(df.columns)}")
        #                 break  # Only show first timeframe

        for parsed_cond in parsed_conditions:
            try:
                is_met = self._evaluate_single_condition(parsed_cond, data, _verbose)

                if is_met:
                    met.append(parsed_cond)
                else:
                    failed.append(parsed_cond)
            
            except Exception as e:
                # Log error and treat as failed
                if self.logger:
                    self.logger.warning(
                        f"Condition evaluation error: {parsed_cond['raw']} - {e}"
                    )
                failed.append(parsed_cond)
        
        # All conditions must be met (AND logic)
        all_met = len(failed) == 0 and len(met) > 0
        score = len(met) / len(parsed_conditions) if parsed_conditions else 0.0
        
        return {
            'all_met': all_met,
            'score': score,
            'met': met,
            'failed': failed
        }
    
    def _evaluate_single_condition(
        self,
        parsed_cond: Dict[str, Any],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        _verbose: bool = False
    ) -> bool:
        """
        Tek bir koşulu değerlendir

        Args:
            parsed_cond: Parse edilmiş koşul
            data: Market data
            _verbose: Temporary verbose flag (for debug status logging)

        Returns:
            bool: Koşul met mi?
        """
        left = parsed_cond['left']
        operator = parsed_cond['operator']
        right = parsed_cond['right']
        timeframe = parsed_cond.get('timeframe')

        # V8: MTF INDICATOR HANDLING - Per-Timeframe DataFrames
        #
        # Her timeframe kendi DataFrame'ine sahip:
        #   result['5m']  = {close, ema_50, rsi_14, ...}  ← 5m OHLCV + indicators
        #   result['15m'] = {close, ema_50, rsi_14, ...}  ← 15m OHLCV + indicators
        #
        # Condition: ['close', '>', 'ema_50', '15m']
        # → result['15m']['close'] > result['15m']['ema_50']
        # Her ikisi de 15m DataFrame'inden gelir

        # V8: Her timeframe kendi DataFrame'ine sahip
        # Timeframe belirtilmişse → o TF'nin DataFrame'i (close, ema_50, vs.)
        # Timeframe belirtilmemişse → primary timeframe DataFrame'i
        #
        # Örnek:
        #   ['close', '>', 'ema_50']        → result['5m']['close'] > result['5m']['ema_50']
        #   ['close', '>', 'ema_50', '15m'] → result['15m']['close'] > result['15m']['ema_50']
        target_df = self._get_dataframe_for_timeframe(data, timeframe)

        # DEBUG: İlk çağrıda data dict'in key'lerini ve her df'in close değerini logla
        if not hasattr(self, '_debug_logged') and self.logger and isinstance(data, dict):
            self._debug_logged = True
            self.logger.info(f"🔍 DEBUG MTF Data keys: {list(data.keys())}")
            for tf_key, df in data.items():
                if tf_key != 'default' and df is not None and 'close' in df.columns:
                    close_val = df['close'].iloc[-1]
                    ema_val = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else 'N/A'
                    self.logger.info(f"   {tf_key}: close={close_val:.2f}, ema_50={ema_val}")

        if target_df is None or target_df.empty:
            if self.logger and _verbose:
                self.logger.info(f"      ❌ Condition FAILED (no data): {parsed_cond['raw']}")
            return False

        # Extract operand values - artık tek DataFrame kullanılıyor
        left_value = self._extract_value(left, target_df, operator, timeframe)
        right_value = self._extract_value(right, target_df, operator, timeframe)


        # Evaluate condition
        result = evaluate_condition(left_value, operator, right_value)

        # Log condition result only when verbose
        if _verbose and self.logger:
            # Format values for logging
            # For crossover/crossunder operators, show only the last value
            if operator in ('crossover', 'crossunder'):
                # Extract last value from arrays
                if isinstance(left_value, (list, tuple, np.ndarray)):
                    left_str = f"{left_value[-1]:.2f}" if len(left_value) > 0 else str(left_value)
                else:
                    left_str = f"{left_value:.2f}" if isinstance(left_value, (int, float)) else str(left_value)

                if isinstance(right_value, (list, tuple, np.ndarray)):
                    right_str = f"{right_value[-1]:.2f}" if len(right_value) > 0 else str(right_value)
                else:
                    right_str = f"{right_value:.2f}" if isinstance(right_value, (int, float)) else str(right_value)
            else:
                # For other operators, format as usual (6 decimal for precision)
                left_str = f"{left_value:.6f}" if isinstance(left_value, (int, float)) else str(left_value)
                right_str = f"{right_value:.6f}" if isinstance(right_value, (int, float)) else str(right_value)

            if result:
                self.logger.info(f"      ✅ {parsed_cond['raw']}: {left_str} {operator} {right_str}")
            else:
                self.logger.info(f"      ❌ {parsed_cond['raw']}: {left_str} {operator} {right_str}")

        return result
    
    def _get_dataframe_for_timeframe(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        timeframe: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """
        Timeframe için uygun DataFrame'i al
        
        Args:
            data: Market data
            timeframe: Timeframe (None = primary)
        
        Returns:
            pd.DataFrame veya None
        """
        # Single timeframe data
        if isinstance(data, pd.DataFrame):
            return data
        
        # Multi-timeframe data (dict)
        if isinstance(data, dict):
            if timeframe:
                df = data.get(timeframe)
                if df is None and self.logger:
                    self.logger.warning(f"⚠️  TF '{timeframe}' not found in MTF dict. Available: {list(data.keys())}")
                # DEBUG: Log which df is selected (first time only per TF)
                if df is not None and self.logger and 'ema_50' in df.columns:
                    debug_key = f"_tf_debug_{timeframe}"
                    if not hasattr(self, debug_key):
                        setattr(self, debug_key, True)
                        self.logger.info(f"📍 Selected TF={timeframe}: ema_50={df['ema_50'].iloc[-1]:.2f}, close={df['close'].iloc[-1]:.2f}")
                return df
            else:
                # Use primary timeframe
                primary_tf = self.strategy.primary_timeframe
                return data.get(primary_tf)

        return None
    
    def _extract_value(
        self,
        operand: Any,
        df: pd.DataFrame,
        operator: str,
        timeframe: Optional[str] = None
    ) -> Any:
        """
        Operand değerini extract et

        Args:
            operand: Operand (indicator name, number, or keyword)
            df: DataFrame
            operator: Operator (for history requirements)
            timeframe: Timeframe (for MTF indicator lookup)

        Returns:
            Value (number, series, or list)
        """
        # Numeric value
        if isinstance(operand, (int, float)):
            return operand

        # Boolean value
        if isinstance(operand, bool):
            return operand

        # List/tuple (for between, outside, near)
        if isinstance(operand, (list, tuple)):
            return operand

        # String operand (indicator or price keyword)
        if isinstance(operand, str):
            operand = operand.lower().strip()

            # Price keywords (open, high, low, close, volume)
            if operand in {'open', 'high', 'low', 'close', 'volume'}:
                if operand in df.columns:
                    return self._get_series_for_operator(df[operand], operator)
                else:
                    raise ValueError(f"Column '{operand}' not found in dataframe")

            # V8: Her timeframe kendi DataFrame'ine sahip
            # DataFrame zaten doğru timeframe'i içeriyor (suffix yok!)
            # result['5m']['ema_50'], result['15m']['ema_50'] gibi
            column_name = operand

            if column_name in df.columns:
                return self._get_series_for_operator(df[column_name], operator)
            else:
                raise ValueError(
                    f"Indicator '{operand}' not found in dataframe. "
                    f"Available columns: {list(df.columns)}"
                )

        # Default
        return operand
    
    def _get_series_for_operator(
        self,
        series: pd.Series,
        operator: str
    ) -> Union[float, List, np.ndarray]:
        """
        Operatöre göre series'i hazırla
        
        Args:
            series: Pandas Series
            operator: Operator string
        
        Returns:
            - Comparison operators: last value (float)
            - Crossover/Rising/Falling: last N values (list)
        """
        # Comparison operators: sadece son değer
        if operator in {'>', '<', '>=', '<=', '==', '!=', 'between', 'outside', 'near'}:
            return series.iloc[-1]
        
        # Crossover/Crossunder: son 2 değer
        if operator in {'crossover', 'crossunder'}:
            if len(series) >= 2:
                return series.iloc[-2:].values
            else:
                return [series.iloc[-1]]
        
        # Rising/Falling: operator context'e göre (right operand period'u belirler)
        # Şimdilik son 10 değer (yeterli olmalı)
        if operator in {'rising', 'falling'}:
            lookback = min(10, len(series))
            return series.iloc[-lookback:].values
        
        # Default: son değer
        return series.iloc[-1]
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def has_entry_conditions(self, side: str) -> bool:
        """
        Bir side için entry condition var mı?
        
        Args:
            side: 'long' or 'short'
        
        Returns:
            bool: True if conditions exist
        """
        if side.lower() == 'long':
            return len(self.parsed_entry_long) > 0
        elif side.lower() == 'short':
            return len(self.parsed_entry_short) > 0
        return False
    
    def has_exit_conditions(self, side: str) -> bool:
        """
        Bir side için exit condition var mı?
        
        Args:
            side: 'long' or 'short'
        
        Returns:
            bool: True if conditions exist
        """
        if side.lower() == 'long':
            return len(self.parsed_exit_long) > 0
        elif side.lower() == 'short':
            return len(self.parsed_exit_short) > 0
        return False
    
    def get_required_indicators(self) -> List[str]:
        """
        Tüm koşullarda kullanılan indikatörleri dön
        
        Returns:
            List[str]: Indicator names
        """
        indicators = set()
        
        # Entry conditions
        for cond in self.parsed_entry_long + self.parsed_entry_short:
            indicators.update(self.parser.extract_indicators(cond))
        
        # Exit conditions
        for cond in self.parsed_exit_long + self.parsed_exit_short:
            indicators.update(self.parser.extract_indicators(cond))
        
        return sorted(indicators)
    
    def get_conditions_summary(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        side: str = 'long'
    ) -> Dict[str, Any]:
        """
        Belirli bir side için koşul özetini döndür (TierManager için)

        Args:
            data: Market data
            side: 'long' veya 'short'

        Returns:
            Dict: {
                'conditions_met': int,
                'conditions_total': int,
                'score': float,
                'direction': str,
                'details': [
                    {
                        'condition': ['close', '<', 'ema_5'],
                        'met': True,
                        'left_value': 331.61,
                        'right_value': 335.19
                    },
                    ...
                ]
            }
        """
        parsed_conditions = self.parsed_entry_long if side.lower() == 'long' else self.parsed_entry_short

        if not parsed_conditions:
            return {
                'conditions_met': 0,
                'conditions_total': 0,
                'score': 0.0,
                'direction': side.upper(),
                'details': []
            }

        details = []
        met_count = 0

        for parsed_cond in parsed_conditions:
            try:
                # V8: Her timeframe kendi DataFrame'ine sahip
                # Condition: ['close', '>', 'ema_50', '15m']
                # → result['15m']['close'] > result['15m']['ema_50']
                # Hem price hem indicator aynı timeframe'den gelir!
                timeframe = parsed_cond.get('timeframe')
                target_df = self._get_dataframe_for_timeframe(data, timeframe)

                if target_df is None or target_df.empty:
                    details.append({
                        'condition': parsed_cond['raw'],
                        'met': False,
                        'left_value': None,
                        'right_value': None,
                        'error': 'no_data'
                    })
                    continue

                left = parsed_cond['left']
                operator = parsed_cond['operator']
                right = parsed_cond['right']

                # V8: Tek DataFrame kullan - hem price hem indicator aynı TF'den
                left_value = self._extract_value(left, target_df, operator, timeframe)
                right_value = self._extract_value(right, target_df, operator, timeframe)

                # Evaluate
                is_met = evaluate_condition(left_value, operator, right_value)

                if is_met:
                    met_count += 1

                # Format values for JSON serialization
                if isinstance(left_value, (np.ndarray, list)):
                    left_value = float(left_value[-1]) if len(left_value) > 0 else None
                elif isinstance(left_value, (np.floating, np.integer)):
                    left_value = float(left_value)

                if isinstance(right_value, (np.ndarray, list)):
                    right_value = float(right_value[-1]) if len(right_value) > 0 else None
                elif isinstance(right_value, (np.floating, np.integer)):
                    right_value = float(right_value)

                details.append({
                    'condition': parsed_cond['raw'],
                    'met': is_met,
                    'left_value': left_value,
                    'right_value': right_value
                })

            except Exception as e:
                details.append({
                    'condition': parsed_cond['raw'],
                    'met': False,
                    'left_value': None,
                    'right_value': None,
                    'error': str(e)
                })

        total = len(parsed_conditions)
        score = met_count / total if total > 0 else 0.0

        return {
            'conditions_met': met_count,
            'conditions_total': total,
            'score': score,
            'direction': side.upper(),
            'details': details
        }

    def get_best_side_summary(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        """
        En yüksek skorlu side için koşul özetini döndür

        Args:
            data: Market data

        Returns:
            Dict: get_conditions_summary() formatında, en iyi side için
        """
        long_summary = self.get_conditions_summary(data, 'long')
        short_summary = self.get_conditions_summary(data, 'short')

        # Tam eşleşme varsa onu döndür
        if long_summary['score'] == 1.0:
            return long_summary
        if short_summary['score'] == 1.0:
            return short_summary

        # En yüksek skoru döndür
        if long_summary['score'] >= short_summary['score']:
            return long_summary
        return short_summary

    def __repr__(self) -> str:
        return (
            f"<SignalValidator "
            f"entry_long={len(self.parsed_entry_long)} "
            f"entry_short={len(self.parsed_entry_short)} "
            f"exit_long={len(self.parsed_exit_long)} "
            f"exit_short={len(self.parsed_exit_short)}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SignalValidator',
]

