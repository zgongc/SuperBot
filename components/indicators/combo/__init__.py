"""
indicators/combo/__init__.py - Combo (Birleşik) İndikatörler Modülü

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Birden fazla indikatörü birleştirerek güçlü sinyaller üreten
    kombine indikatörler koleksiyonu.

    Bu kategori, farklı indikatör türlerini (trend, momentum, volume)
    bir araya getirerek daha güvenilir al/sat sinyalleri sağlar.

Mevcut İndikatörler:
    1. RSIBollinger - RSI + Bollinger Bands kombinasyonu
    2. MACDRSICombo - MACD + RSI kombinasyonu
    3. EMARibbon - Çoklu EMA bantları (5,10,20,50,100,200)
    4. TripleScreen - Elder'ın ünlü 3 ekran sistemi
    5. SmartMoney - Smart Money Concept (SMC) analizi

Kullanım Örnekleri:
    >>> from indicators.combo import RSIBollinger, MACDRSICombo
    >>>
    >>> # RSI + Bollinger kombinasyonu
    >>> combo1 = RSIBollinger(rsi_period=14, bb_period=20)
    >>> result1 = combo1.calculate(data)
    >>>
    >>> # MACD + RSI kombinasyonu
    >>> combo2 = MACDRSICombo(macd_fast=12, rsi_period=14)
    >>> result2 = combo2.calculate(data)
    >>>
    >>> # EMA Ribbon
    >>> ribbon = EMARibbon(ema_periods=[5, 10, 20, 50, 100, 200])
    >>> result3 = ribbon.calculate(data)
    >>>
    >>> # Triple Screen
    >>> ts = TripleScreen(use_macd=True)
    >>> result4 = ts.calculate(data)
    >>>
    >>> # Smart Money
    >>> smc = SmartMoney(adx_threshold=25)
    >>> result5 = smc.calculate(data)

Özellikler:
    - Çoklu indikatör onayı
    - Güçlü sinyal filtreleme
    - Risk yönetimi entegrasyonu
    - Yüksek doğruluk oranı

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.momentum
    - indicators.trend
    - indicators.volatility
    - indicators.volume
"""

from indicators.combo.rsi_bollinger import RSIBollinger
from indicators.combo.macd_rsi import MACDRSICombo
from indicators.combo.ema_ribbon import EMARibbon
from indicators.combo.triple_screen import TripleScreen
from indicators.combo.smart_money import SmartMoney
from indicators.combo.smart_grok import SmartGrok


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Combo indikatörler
    'RSIBollinger',      # RSI + Bollinger Bands
    'MACDRSICombo',      # MACD + RSI
    'EMARibbon',         # Çoklu EMA bantları
    'TripleScreen',      # Elder'ın 3 ekran sistemi
    'SmartMoney',        # Smart Money Concept
    'SmartGrok',         # Smart Money Concept - Geliştirilmiş
]


# ============================================================================
# METADATA
# ============================================================================

__version__ = '2.0.0'
__author__ = 'SuperBot Team'
__category__ = 'combo'
__description__ = 'Birleşik (Combo) teknik indikatörler - Çoklu indikatör kombinasyonları'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_combo_indicators():
    """
    Tüm combo indikatörlerin listesini döndür

    Returns:
        list: Combo indikatör sınıflarının listesi
    """
    return [
        RSIBollinger,
        MACDRSICombo,
        EMARibbon,
        TripleScreen,
        SmartMoney,
        SmartGrok,
    ]


def get_combo_indicator_names():
    """
    Tüm combo indikatör isimlerini döndür

    Returns:
        list: İndikatör isimlerinin listesi
    """
    return [cls.__name__ for cls in get_combo_indicators()]


def get_combo_indicator_info():
    """
    Tüm combo indikatörlerin detaylı bilgisini döndür

    Returns:
        dict: İndikatör isimleri ve açıklamaları
    """
    return {
        'RSIBollinger': {
            'description': 'RSI + Bollinger Bands kombinasyonu',
            'components': ['RSI', 'Bollinger Bands'],
            'category': 'Momentum + Volatility',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': False
        },
        'MACDRSICombo': {
            'description': 'MACD + RSI kombinasyonu',
            'components': ['MACD', 'RSI'],
            'category': 'Trend + Momentum',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': False
        },
        'EMARibbon': {
            'description': 'Çoklu EMA bantları (5,10,20,50,100,200)',
            'components': ['EMA (multiple)'],
            'category': 'Trend',
            'output_type': 'LINES',
            'requires_volume': False
        },
        'TripleScreen': {
            'description': "Elder'ın 3 ekran ticaret sistemi",
            'components': ['MACD/EMA', 'RSI', 'Price Action'],
            'category': 'Multi-timeframe System',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': False
        },
        'SmartMoney': {
            'description': 'Smart Money Concept (SMC) analizi',
            'components': ['OBV', 'RSI', 'ADX', 'Market Structure'],
            'category': 'Volume + Momentum + Trend',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': True
        },
        'SmartGrok': {
            'description': 'Smart Money Concept (SMC) - Geliştirilmiş',
            'components': ['FVG', 'Order Blocks', 'BOS/CHoCH', 'Market Structure'],
            'category': 'Volume + Structure',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': True
        }
    }


def create_combo_indicator(name: str, **params):
    """
    İsme göre combo indikatör oluştur

    Args:
        name: İndikatör ismi
        **params: İndikatör parametreleri

    Returns:
        BaseIndicator: Oluşturulan indikatör

    Raises:
        ValueError: Geçersiz indikatör ismi

    Example:
        >>> indicator = create_combo_indicator('RSIBollinger', rsi_period=14)
        >>> result = indicator.calculate(data)
    """
    indicators_map = {
        'RSIBollinger': RSIBollinger,
        'MACDRSICombo': MACDRSICombo,
        'EMARibbon': EMARibbon,
        'TripleScreen': TripleScreen,
        'SmartMoney': SmartMoney,
        'SmartGrok': SmartGrok,
    }

    if name not in indicators_map:
        available = ', '.join(indicators_map.keys())
        raise ValueError(
            f"Geçersiz combo indikatör ismi: '{name}'. "
            f"Mevcut indikatörler: {available}"
        )

    return indicators_map[name](**params)


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Module başlatma mesajı (development mode)
import os
if os.getenv('SUPERBOT_DEBUG'):
    print(f"[COMBO] Loaded {len(__all__)} combo indicators")
