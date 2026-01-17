#!/usr/bin/env python3
"""
components/strategies/helpers/condition_parser.py
SuperBot - Condition Parser

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Entry/exit koşul array'lerini parse eder.
    
    Format: ['indicator', 'operator', 'value', 'timeframe']
    Örnek: ['ema_50', '>', 'ema_200', '15m']
           ['rsi_14', 'crossover', 30]
           ['close', 'rising', 3, '5m']

Kullanım:
    from components.strategies.helpers.condition_parser import ConditionParser
    
    parser = ConditionParser()
    parsed = parser.parse(['ema_50', '>', 'ema_200', '15m'])
    # {
    #     'left': 'ema_50',
    #     'operator': '>',
    #     'right': 'ema_200',
    #     'timeframe': '15m'
    # }
"""

from typing import Dict, Any, List, Union, Optional
import re


class ConditionParser:
    """
    Koşul array'lerini parse eder
    
    Desteklenen formatlar:
        3-element: ['indicator', 'operator', 'value']
        4-element: ['indicator', 'operator', 'value', 'timeframe']
        
    Örnekler:
        ['rsi_14', '>', 70]
        ['ema_50', 'crossover', 'ema_200', '15m']
        ['close', 'rising', 3, '5m']
    """
    
    # Geçerli operatörler
    VALID_OPERATORS = {
        # Karşılaştırma
        '>',  '<',  '>=',  '<=',  '==',  '!=',
        
        # Crossover/Crossunder
        'crossover', 'crossunder',
        
        # Trend
        'rising', 'falling',
        
        # Range
        'between', 'outside', 'near',
        
        # Boolean
        'is', 'is_not',
    }
    
    # Özel keyword'ler (price/volume data)
    PRICE_KEYWORDS = {'open', 'high', 'low', 'close', 'volume'}
    
    def __init__(self):
        """Initialize parser"""
        pass
    
    def parse(self, condition: List) -> Dict[str, Any]:
        """
        Koşul array'ini parse et
        
        Args:
            condition: Koşul array'i
                Format: ['left', 'operator', 'right'] veya
                        ['left', 'operator', 'right', 'timeframe']
        
        Returns:
            Dict: Parsed condition
                {
                    'left': str,
                    'operator': str,
                    'right': Union[str, int, float, list],
                    'timeframe': Optional[str]
                }
        
        Raises:
            ValueError: Geçersiz koşul formatı
        """
        if not isinstance(condition, (list, tuple)):
            raise ValueError(f"Koşul list veya tuple olmalı, {type(condition)} verildi")
        
        if len(condition) < 3:
            raise ValueError(
                f"Koşul en az 3 element olmalı (left, operator, right), "
                f"{len(condition)} element var: {condition}"
            )
        
        if len(condition) > 4:
            raise ValueError(
                f"Koşul en fazla 4 element olmalı (left, operator, right, timeframe), "
                f"{len(condition)} element var: {condition}"
            )
        
        # Parse elements
        left = condition[0]
        operator = condition[1]
        right = condition[2]
        timeframe = condition[3] if len(condition) == 4 else None
        
        # Validate
        self._validate_operator(operator)
        
        # Normalize
        left = self._normalize_operand(left)
        right = self._normalize_operand(right)
        timeframe = self._normalize_timeframe(timeframe)
        
        return {
            'left': left,
            'operator': operator,
            'right': right,
            'timeframe': timeframe,
            'raw': condition
        }
    
    def parse_batch(self, conditions: List[List]) -> List[Dict[str, Any]]:
        """
        Birden fazla koşulu parse et
        
        Args:
            conditions: Koşul array'leri listesi
        
        Returns:
            List[Dict]: Parse edilmiş koşullar
        """
        return [self.parse(cond) for cond in conditions]
    
    def _validate_operator(self, operator: str) -> None:
        """
        Operatör geçerliliğini kontrol et
        
        Args:
            operator: Operatör string
        
        Raises:
            ValueError: Geçersiz operatör
        """
        if operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Geçersiz operatör: '{operator}'\n"
                f"Geçerli operatörler: {sorted(self.VALID_OPERATORS)}"
            )
    
    def _normalize_operand(self, operand: Any) -> Any:
        """
        Operand'ı normalize et
        
        Args:
            operand: Sol veya sağ operand
        
        Returns:
            Normalized operand
        """
        # String ise lowercase yap
        if isinstance(operand, str):
            return operand.lower().strip()
        
        # Numeric ise olduğu gibi dön
        if isinstance(operand, (int, float)):
            return operand
        
        # List/tuple ise olduğu gibi dön (between, outside için)
        if isinstance(operand, (list, tuple)):
            return operand
        
        # Boolean ise olduğu gibi dön
        if isinstance(operand, bool):
            return operand
        
        # None ise olduğu gibi dön
        if operand is None:
            return operand
        
        # Diğer tipler için string'e çevir
        return str(operand)
    
    def _normalize_timeframe(self, timeframe: Optional[str]) -> Optional[str]:
        """
        Timeframe'i normalize et
        
        Args:
            timeframe: Timeframe string
        
        Returns:
            Normalized timeframe (lowercase) veya None
        """
        if timeframe is None:
            return None
        
        if not isinstance(timeframe, str):
            raise ValueError(f"Timeframe string olmalı, {type(timeframe)} verildi")
        
        return timeframe.lower().strip()
    
    # String literal keywords (not indicators)
    STRING_LITERALS = {
        # Signal types
        'buy', 'sell', 'hold', 'strong_buy', 'strong_sell',
        # Trend directions
        'up', 'down', 'neutral', 'bullish', 'bearish',
        # Boolean-like
        'true', 'false', 'yes', 'no',
    }

    def extract_indicators(self, condition: Dict[str, Any]) -> List[str]:
        """
        Koşulda kullanılan indikatörleri çıkar

        Args:
            condition: Parse edilmiş koşul

        Returns:
            List[str]: Indikatör isimleri
        """
        indicators = []
        operator = condition.get('operator', '')

        # Sol operand
        left = condition['left']
        if isinstance(left, str) and left not in self.PRICE_KEYWORDS:
            indicators.append(left)

        # Sağ operand
        right = condition['right']
        if isinstance(right, str) and right not in self.PRICE_KEYWORDS:
            # == veya != operatörlerinde sağ taraf string literal olabilir
            if operator in ('==', '!='):
                # Bilinen string literal'lar indicator değil
                if right.lower() in self.STRING_LITERALS:
                    pass  # Skip - string literal
                elif not self._is_numeric_string(right):
                    indicators.append(right)
            else:
                # Numeric string değilse indikatör
                if not self._is_numeric_string(right):
                    indicators.append(right)

        return indicators
    
    def _is_numeric_string(self, value: str) -> bool:
        """
        String numeric mi kontrol et
        
        Args:
            value: String değer
        
        Returns:
            True if numeric string
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def is_price_keyword(self, operand: str) -> bool:
        """
        Operand price keyword mu kontrol et
        
        Args:
            operand: Operand string
        
        Returns:
            True if price keyword (open, high, low, close, volume)
        """
        return operand.lower() in self.PRICE_KEYWORDS
    
    def requires_historical_data(self, operator: str) -> bool:
        """
        Operatör historical data gerektirir mi?
        
        Args:
            operator: Operatör string
        
        Returns:
            True if historical data needed (crossover, rising, falling)
        """
        return operator in {'crossover', 'crossunder', 'rising', 'falling'}
    
    def get_lookback_period(self, condition: Dict[str, Any]) -> int:
        """
        Koşul için gereken lookback period
        
        Args:
            condition: Parse edilmiş koşul
        
        Returns:
            Lookback period (bar sayısı)
        """
        operator = condition['operator']
        
        # Crossover/Crossunder: 2 bar
        if operator in {'crossover', 'crossunder'}:
            return 2
        
        # Rising/Falling: right operand kadar (örn: rising 3 -> 3 bar)
        if operator in {'rising', 'falling'}:
            right = condition['right']
            if isinstance(right, int):
                return right + 1  # +1 çünkü comparison için bir önceki bar lazım
            return 2  # Default
        
        # Diğerleri: 1 bar (mevcut)
        return 1
    
    def __repr__(self) -> str:
        return f"<ConditionParser operators={len(self.VALID_OPERATORS)}>"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_condition(condition: List) -> Dict[str, Any]:
    """
    Convenience function - koşul parse et
    
    Args:
        condition: Koşul array'i
    
    Returns:
        Parsed condition dict
    """
    parser = ConditionParser()
    return parser.parse(condition)


def parse_conditions(conditions: List[List]) -> List[Dict[str, Any]]:
    """
    Convenience function - birden fazla koşul parse et
    
    Args:
        conditions: Koşul array'leri listesi
    
    Returns:
        List of parsed conditions
    """
    parser = ConditionParser()
    return parser.parse_batch(conditions)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ConditionParser',
    'parse_condition',
    'parse_conditions',
]

