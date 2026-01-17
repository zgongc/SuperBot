"""
indicators/types.py - Type Definitions for Indicators

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Ortak type tanımlamaları (Enum, Dataclass, Exception).
    Tüm indicator modülleri tarafından kullanılır.
    
    İçerik:
    - Enum'lar: IndicatorCategory, TrendDirection, SignalType, TimeframeEnum
    - Dataclass'lar: OHLCV, IndicatorResult, IndicatorConfig
    - Exception'lar: IndicatorError, InsufficientDataError, InvalidParameterError
    
    Kullanım:
        from indicators.indicator_types import (
            OHLCV, IndicatorResult, TrendDirection, SignalType
        )

Dependencies:
    - dataclasses (stdlib)
    - enum (stdlib)
    - typing (stdlib)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


# ============================================================================
# ENUMS
# ============================================================================

class IndicatorCategory(Enum):
    """Indikatör kategorileri"""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    STRUCTURE = "structure"
    BREAKOUT = "breakout"
    STATISTICAL = "statistical"
    COMBO = "combo"
    PATTERNS = "patterns"

    def __str__(self) -> str:
        return self.value


class TrendDirection(Enum):
    """Trend yönü"""
    UP = 1
    DOWN = -1
    NEUTRAL = 0
    UNKNOWN = None
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def is_bullish(self) -> bool:
        return self == TrendDirection.UP
    
    @property
    def is_bearish(self) -> bool:
        return self == TrendDirection.DOWN


class SignalType(Enum):
    """Sinyal tipleri"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_entry(self) -> bool:
        return self in (SignalType.BUY, SignalType.STRONG_BUY)
    
    @property
    def is_exit(self) -> bool:
        return self in (SignalType.SELL, SignalType.STRONG_SELL)


class TimeframeEnum(Enum):
    """Desteklenen timeframe'ler"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def minutes(self) -> int:
        """Timeframe'i dakika cinsinden döndür"""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
        }
        return mapping.get(self.value, 0)


class IndicatorType(Enum):
    """Indikatör output tipi"""
    SINGLE_VALUE = "single"      # RSI, ATR gibi tek değer
    MULTIPLE_VALUES = "multiple"  # MACD (signal, histogram)
    BANDS = "bands"              # Bollinger (upper, middle, lower)
    LINES = "lines"              # Ichimoku (tenkan, kijun, ...)
    LEVELS = "levels"            # Pivot Points (S1, S2, R1, R2)
    ZONES = "zones"              # FVG, Order Blocks
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class OHLCV:
    """
    OHLCV veri yapısı
    
    Attributes:
        timestamp: Unix timestamp (milliseconds)
        open: Açılış fiyatı
        high: En yüksek fiyat
        low: En düşük fiyat
        close: Kapanış fiyatı
        volume: Hacim
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCV':
        """Dict'ten OHLCV oluştur"""
        return cls(
            timestamp=int(data['timestamp']),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """OHLCV'yi dict'e çevir"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    @property
    def datetime(self) -> datetime:
        """Timestamp'i datetime'a çevir"""
        return datetime.fromtimestamp(self.timestamp / 1000)


@dataclass
class IndicatorResult:
    """
    Indikatör hesaplama sonucu (generic)
    
    Attributes:
        value: Ana değer (float, dict, list, etc.)
        timestamp: Hesaplamanın yapıldığı zaman
        metadata: Ek bilgiler (opsiyonel)
        signal: Sinyal tipi (opsiyonel)
        trend: Trend yönü (opsiyonel)
        strength: Sinyal gücü 0-100 arası (opsiyonel)
    """
    value: Any
    timestamp: int
    metadata: Optional[Dict[str, Any]] = None
    signal: Optional[SignalType] = None
    trend: Optional[TrendDirection] = None
    strength: Optional[float] = None
    
    def __post_init__(self):
        """Validation"""
        if self.strength is not None:
            if not 0 <= self.strength <= 100:
                raise ValueError("Strength must be between 0-100")
    
    @property
    def datetime(self) -> datetime:
        """Timestamp'i datetime'a çevir"""
        return datetime.fromtimestamp(self.timestamp / 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e çevir"""
        return {
            'value': self.value,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'signal': self.signal.value if self.signal else None,
            'trend': self.trend.value if self.trend else None,
            'strength': self.strength
        }


@dataclass
class IndicatorConfig:
    """
    Indikatör konfigürasyonu
    
    Attributes:
        name: Indikatör adı (örn: 'rsi', 'ema')
        category: Kategori
        params: Parametreler (period, multiplier, vb.)
        enabled: Aktif mi?
        timeframe: Hangi timeframe'de hesaplanacak
    """
    name: str
    category: IndicatorCategory
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timeframe: Optional[str] = None
    
    def __post_init__(self):
        """Validation"""
        if not self.name:
            raise ValueError("Indicator name cannot be empty")
        if not isinstance(self.params, dict):
            raise ValueError("Params must be a dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e çevir"""
        return {
            'name': self.name,
            'category': self.category.value,
            'params': self.params,
            'enabled': self.enabled,
            'timeframe': self.timeframe
        }


@dataclass
class IndicatorMetadata:
    """
    Indikatör metadata
    
    Registry'de kullanılır, indikatör hakkında bilgi içerir.
    """
    name: str
    category: IndicatorCategory
    indicator_type: IndicatorType
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)
    output_names: List[str] = field(default_factory=list)
    requires_volume: bool = False
    min_periods: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e çevir"""
        return {
            'name': self.name,
            'category': self.category.value,
            'indicator_type': self.indicator_type.value,
            'description': self.description,
            'params': self.params,
            'default_params': self.default_params,
            'output_names': self.output_names,
            'requires_volume': self.requires_volume,
            'min_periods': self.min_periods
        }


# ============================================================================
# EXCEPTIONS
# ============================================================================

class IndicatorError(Exception):
    """Indikatör base exception"""
    def __init__(self, message: str, indicator_name: str = None):
        self.indicator_name = indicator_name
        super().__init__(f"[{indicator_name}] {message}" if indicator_name else message)


class InsufficientDataError(IndicatorError):
    """
    Yetersiz veri hatası
    
    Indikatör hesaplamak için yeterli data yok.
    Örn: RSI için 14 period gerekli, sadece 10 kline var.
    """
    def __init__(self, indicator_name: str, required: int, available: int):
        self.required = required
        self.available = available
        message = f"Insufficient data: required {required}, available {available}"
        super().__init__(message, indicator_name)


class InvalidParameterError(IndicatorError):
    """
    Geçersiz parametre hatası
    
    Indikatör parametreleri geçersiz.
    Örn: RSI period=-5 (negatif olamaz)
    """
    def __init__(self, indicator_name: str, param_name: str, param_value: Any, reason: str = None):
        self.param_name = param_name
        self.param_value = param_value
        message = f"Invalid parameter '{param_name}' = {param_value}"
        if reason:
            message += f": {reason}"
        super().__init__(message, indicator_name)


class CalculationError(IndicatorError):
    """
    Hesaplama hatası
    
    Indikatör hesaplama sırasında hata oluştu.
    Örn: Division by zero, NaN values
    """
    pass


# ============================================================================
# TYPE ALIASES
# ============================================================================

# Pandas DataFrame alternatifi için
DataPoint = Dict[str, Any]
DataSeries = List[DataPoint]

# Indikatör output tipi
IndicatorOutput = Any  # float, dict, list, DataFrame, etc.


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'IndicatorCategory',
    'TrendDirection',
    'SignalType',
    'TimeframeEnum',
    'IndicatorType',
    
    # Dataclasses
    'OHLCV',
    'IndicatorResult',
    'IndicatorConfig',
    'IndicatorMetadata',
    
    # Exceptions
    'IndicatorError',
    'InsufficientDataError',
    'InvalidParameterError',
    'CalculationError',
    
    # Type Aliases
    'DataPoint',
    'DataSeries',
    'IndicatorOutput',
]