#!/usr/bin/env python3
"""
components/strategies/pattern_generator.py
SuperBot - Pattern Generator

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Pattern detection and evaluation:
    - Candlestick patterns (21+ patterns via existing indicators)
    - SMC patterns (FVG, BOS, CHoCH, Order Blocks)
    - Pattern context validation
    - Pattern strength calculation

Usage:
    from components.strategies.pattern_generator import PatternGenerator

    generator = PatternGenerator(strategy)
    patterns = generator.detect_patterns(data)
    has_pattern = generator.has_pattern('hammer', patterns)
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from components.strategies.base_strategy import BaseStrategy


class PatternGenerator:
    """
    Pattern detection and validation generator.

    Detects and evaluates candlestick and SMC patterns.
    """
    
    # Supported candlestick patterns
    CANDLESTICK_PATTERNS = {
        # Single candle
        'doji', 'dragonfly_doji', 'gravestone_doji', 'longlegged_doji',
        'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
        'marubozu_bullish', 'marubozu_bearish', 'spinning_top',
        
        # Multi-candle
        'engulfing_bullish', 'engulfing_bearish',
        'harami_bullish', 'harami_bearish',
        'morning_star', 'evening_star',
        'piercing_line', 'dark_cloud_cover',
        'three_white_soldiers', 'three_black_crows',
    }
    
    # Supported SMC patterns
    SMC_PATTERNS = {
        'fvg', 'fvg_bullish', 'fvg_bearish',
        'bos', 'bos_bullish', 'bos_bearish',
        'choch', 'choch_bullish', 'choch_bearish',
        'order_block', 'order_block_bullish', 'order_block_bearish',
        'liquidity_zone',
    }
    
    def __init__(
        self,
        strategy: BaseStrategy,
        logger: Any = None
    ):
        """
        Initialize PatternGenerator

        Args:
            strategy: BaseStrategy instance
            logger: Logger instance (optional)
        """
        self.strategy = strategy
        self.logger = logger
        
        # Pattern configuration from strategy (if exists)
        self.pattern_config = self._parse_pattern_config()
    
    # ========================================================================
    # PATTERN DETECTION
    # ========================================================================
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect patterns from the data.
        
        Args:
            data: OHLCV DataFrame (indicators must be calculated)
            pattern_types: Detect edilecek pattern'ler (None = all)
        
        Returns:
            Dict: Detected patterns
                {
                    'candlestick': {'hammer': True, 'doji': False, ...},
                    'smc': {'fvg_bullish': True, 'bos': False, ...},
                    'timestamp': ...,
                }
        """
        result = {
            'candlestick': {},
            'smc': {},
            'timestamp': None,
        }
        
        if data.empty:
            return result
        
        # Get current bar (last bar)
        current_bar = data.iloc[-1]
        
        # Timestamp
        if 'timestamp' in current_bar:
            result['timestamp'] = current_bar['timestamp']
        elif 'open_time' in current_bar:
            result['timestamp'] = current_bar['open_time']
        
        # Detect candlestick patterns
        if pattern_types is None or any(p in self.CANDLESTICK_PATTERNS for p in pattern_types):
            result['candlestick'] = self._detect_candlestick_patterns(data, pattern_types)
        
        # Detect SMC patterns
        if pattern_types is None or any(p in self.SMC_PATTERNS for p in pattern_types):
            result['smc'] = self._detect_smc_patterns(data, pattern_types)
        
        return result
    
    def _detect_candlestick_patterns(
        self,
        data: pd.DataFrame,
        filter_patterns: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Detect candlestick patterns.
        
        Args:
            data: OHLCV DataFrame
            filter_patterns: Filter list (None = all)
        
        Returns:
            Dict: {pattern_name: is_detected}
        """
        patterns = {}
        
        # Last bar
        current = data.iloc[-1]
        
        # Check each pattern
        for pattern in self.CANDLESTICK_PATTERNS:
            # Filter check
            if filter_patterns and pattern not in filter_patterns:
                continue
            
            # Check if pattern column exists in data
            if pattern in data.columns:
                # Get pattern value (True/False or 100/-100/0)
                value = current[pattern]
                
                # Convert to boolean
                if isinstance(value, bool):
                    patterns[pattern] = value
                elif isinstance(value, (int, float)):
                    patterns[pattern] = value != 0
                else:
                    patterns[pattern] = False
            else:
                patterns[pattern] = False
        
        return patterns
    
    def _detect_smc_patterns(
        self,
        data: pd.DataFrame,
        filter_patterns: Optional[List[str]] = None
    ) -> Dict[str, Union[bool, int]]:
        """
        Detect SMC patterns.
        
        Args:
            data: OHLCV DataFrame (with SMC indicators)
            filter_patterns: Filter list (None = all)
        
        Returns:
            Dict: {pattern_name: value}
        """
        patterns = {}
        
        # Last bar
        current = data.iloc[-1]
        
        # FVG patterns
        if 'fvg' in data.columns:
            fvg_count = current.get('fvg', 0)
            patterns['fvg'] = int(fvg_count) > 0
            
            # Bullish/Bearish FVG detection (basit)
            if fvg_count > 0:
                # Check if price is above/below FVG (heuristic)
                patterns['fvg_bullish'] = True if fvg_count > 0 else False
                patterns['fvg_bearish'] = False
            else:
                patterns['fvg_bullish'] = False
                patterns['fvg_bearish'] = False
        
        # BOS patterns
        if 'bos' in data.columns:
            bos_value = current.get('bos', 0)
            patterns['bos'] = bos_value != 0
            patterns['bos_bullish'] = bos_value > 0
            patterns['bos_bearish'] = bos_value < 0
        
        # CHoCH patterns
        if 'choch' in data.columns:
            choch_value = current.get('choch', 0)
            patterns['choch'] = choch_value != 0
            patterns['choch_bullish'] = choch_value > 0
            patterns['choch_bearish'] = choch_value < 0
        
        # Order Blocks
        if 'order_block' in data.columns:
            ob_value = current.get('order_block', 0)
            patterns['order_block'] = ob_value != 0
            patterns['order_block_bullish'] = ob_value > 0
            patterns['order_block_bearish'] = ob_value < 0
        
        # Liquidity Zones
        if 'liquidity_zone' in data.columns:
            liq_value = current.get('liquidity_zone', 0)
            patterns['liquidity_zone'] = liq_value != 0
        
        return patterns
    
    # ========================================================================
    # PATTERN VALIDATION
    # ========================================================================
    
    def has_pattern(
        self,
        pattern_name: str,
        detected_patterns: Dict[str, Any]
    ) -> bool:
        """
        Has a pattern been detected?
        
        Args:
            pattern_name: Pattern name
            detected_patterns: The result of the detect_patterns() function.
        
        Returns:
            bool: True if pattern detected
        """
        # Check candlestick
        if pattern_name in detected_patterns.get('candlestick', {}):
            return detected_patterns['candlestick'][pattern_name]
        
        # Check SMC
        if pattern_name in detected_patterns.get('smc', {}):
            value = detected_patterns['smc'][pattern_name]
            return bool(value)
        
        return False
    
    def get_pattern_strength(
        self,
        pattern_name: str,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate pattern strength (0-1)
        
        Args:
            pattern_name: Pattern name
            data: OHLCV DataFrame
        
        Returns:
            float: Pattern strength (0-1, 1=strong)
        
        Note:
            This is a simplified implementation.
            It should be more sophisticated in production.
        """
        if data.empty:
            return 0.0
        
        current = data.iloc[-1]
        
        # Candlestick patterns: enhanced with context
        if pattern_name in self.CANDLESTICK_PATTERNS:
            base_strength = 0.7  # Default strength
            
            # Volume confirmation
            if 'volume' in data.columns and len(data) > 1:
                avg_volume = data['volume'].tail(20).mean()
                current_volume = current['volume']
                
                if current_volume > avg_volume * 1.5:
                    base_strength += 0.15  # High volume = stronger
            
            # Trend alignment
            if 'ema_20' in data.columns and 'ema_50' in data.columns:
                ema_20 = current.get('ema_20', 0)
                ema_50 = current.get('ema_50', 0)
                
                # Bullish pattern in uptrend or bearish pattern in downtrend
                if 'bullish' in pattern_name.lower() or pattern_name in {'hammer', 'morning_star'}:
                    if ema_20 > ema_50:
                        base_strength += 0.15
                elif 'bearish' in pattern_name.lower() or pattern_name in {'shooting_star', 'evening_star'}:
                    if ema_20 < ema_50:
                        base_strength += 0.15
            
            return min(base_strength, 1.0)
        
        # SMC patterns: occurrence = strength
        if pattern_name in self.SMC_PATTERNS:
            if pattern_name in data.columns:
                value = current.get(pattern_name, 0)
                if value:
                    return 0.8  # SMC patterns are inherently strong
            return 0.0
        
        return 0.5  # Default
    
    def validate_pattern_context(
        self,
        pattern_name: str,
        data: pd.DataFrame,
        side: str
    ) -> bool:
        """
        Pattern'in context'i uygun mu?
        
        Args:
            pattern_name: Pattern name
            data: OHLCV DataFrame
            side: 'LONG' or 'SHORT'
        
        Returns:
            bool: True if context is valid
        """
        if data.empty:
            return False
        
        # Check for LONG for bullish patterns, and SHORT for bearish patterns.
        if side.upper() == 'LONG':
            # Bullish patterns OK
            if any(keyword in pattern_name.lower() for keyword in ['bullish', 'hammer', 'morning', 'piercing', 'white']):
                return True
            # Bearish patterns NOT OK
            if any(keyword in pattern_name.lower() for keyword in ['bearish', 'shooting', 'evening', 'dark', 'black']):
                return False
        
        elif side.upper() == 'SHORT':
            # Bearish patterns OK
            if any(keyword in pattern_name.lower() for keyword in ['bearish', 'shooting', 'evening', 'dark', 'black']):
                return True
            # Bullish patterns NOT OK
            if any(keyword in pattern_name.lower() for keyword in ['bullish', 'hammer', 'morning', 'piercing', 'white']):
                return False
        
        # Neutral patterns (doji, spinning top, etc.) = OK
        return True
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _parse_pattern_config(self) -> Dict[str, Any]:
        """Parse pattern configuration from strategy"""
        # Check if strategy has pattern config
        if hasattr(self.strategy, 'custom_parameters'):
            custom_params = self.strategy.custom_parameters
            if custom_params and 'patterns' in custom_params:
                return custom_params['patterns']
        
        # Default config
        return {
            'enabled': True,
            'candlestick_enabled': True,
            'smc_enabled': True,
            'min_strength': 0.5,
        }
    
    def get_all_detected_patterns(
        self,
        detected_patterns: Dict[str, Any]
    ) -> List[str]:
        """
        Returns a list of all detected patterns.
        
        Args:
            detected_patterns: The result of the detect_patterns() function.
        
        Returns:
            List[str]: Pattern names
        """
        patterns = []
        
        # Candlestick
        for pattern, detected in detected_patterns.get('candlestick', {}).items():
            if detected:
                patterns.append(pattern)
        
        # SMC
        for pattern, value in detected_patterns.get('smc', {}).items():
            if value:
                patterns.append(pattern)
        
        return patterns
    
    def __repr__(self) -> str:
        enabled = self.pattern_config.get('enabled', True)
        return f"<PatternGenerator enabled={enabled}>"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PatternGenerator',
]

