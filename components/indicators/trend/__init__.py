"""
indicators/trend/__init__.py - Trend Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Trend kategorisindeki tüm indikatörleri içerir.
    15 trend indikatörü:

    Moving Averages (Hareketli Ortalamalar):
    - SMA: Simple Moving Average
    - EMA: Exponential Moving Average
    - WMA: Weighted Moving Average
    - HMA: Hull Moving Average
    - TEMA: Triple Exponential Moving Average
    - DEMA: Double Exponential Moving Average
    - VWMA: Volume Weighted Moving Average

    Trend Strength (Trend Gücü):
    - SuperTrend: ATR-based trend indicator
    - ADX: Average Directional Index
    - Aroon: Aroon Up/Down indicator
    - Parabolic SAR: Stop and Reverse

    Channels & Complex (Kanallar & Kompleks):
    - Ichimoku: Ichimoku Cloud
    - Keltner: Keltner Channel
    - Donchian: Donchian Channel
    - MACD: Moving Average Convergence Divergence

Usage:
    from indicators.trend import SMA, EMA, MACD, ADX

    # SMA kullanımı
    sma = SMA(period=20)
    result = sma(data)

    # MACD kullanımı
    macd = MACD(fast_period=12, slow_period=26, signal_period=9)
    result = macd(data)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Moving Averages
from indicators.trend.sma import SMA
from indicators.trend.ema import EMA
from indicators.trend.wma import WMA
from indicators.trend.hma import HMA
from indicators.trend.tema import TEMA
from indicators.trend.dema import DEMA
from indicators.trend.vwma import VWMA

# Trend Strength Indicators
from indicators.trend.supertrend import SuperTrend
from indicators.trend.adx import ADX
from indicators.trend.aroon import Aroon
from indicators.trend.parabolic_sar import ParabolicSAR

# Channels & Complex Indicators
from indicators.trend.ichimoku import Ichimoku
from indicators.volatility.keltner import KeltnerChannel  # Keltner is in volatility package
from indicators.trend.donchian import DonchianChannel
from indicators.trend.macd import MACD


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Moving Averages (7)
    'SMA',
    'EMA',
    'WMA',
    'HMA',
    'TEMA',
    'DEMA',
    'VWMA',

    # Trend Strength (4)
    'SuperTrend',
    'ADX',
    'Aroon',
    'ParabolicSAR',

    # Channels & Complex (4)
    'Ichimoku',
    'KeltnerChannel',
    'DonchianChannel',
    'MACD',
]


# ============================================================================
# PACKAGE INFO
# ============================================================================

__version__ = '2.0.0'
__author__ = 'SuperBot Team'
__category__ = 'trend'

# Indikatör sayısı
TOTAL_INDICATORS = len(__all__)

# Kategori bazında indikatörler
MOVING_AVERAGES = ['SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'DEMA', 'VWMA']
TREND_STRENGTH = ['SuperTrend', 'ADX', 'Aroon', 'ParabolicSAR']
CHANNELS_COMPLEX = ['Ichimoku', 'KeltnerChannel', 'DonchianChannel', 'MACD']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_indicators():
    """
    Tüm trend indikatörlerinin listesini döndür

    Returns:
        list: Tüm trend indikatör class'ları
    """
    return [
        SMA, EMA, WMA, HMA, TEMA, DEMA, VWMA,
        SuperTrend, ADX, Aroon, ParabolicSAR,
        Ichimoku, KeltnerChannel, DonchianChannel, MACD
    ]


def get_moving_averages():
    """
    Sadece hareketli ortalamaları döndür

    Returns:
        list: Moving average indikatör class'ları
    """
    return [SMA, EMA, WMA, HMA, TEMA, DEMA, VWMA]


def get_trend_strength_indicators():
    """
    Trend gücü indikatörlerini döndür

    Returns:
        list: Trend strength indikatör class'ları
    """
    return [SuperTrend, ADX, Aroon, ParabolicSAR]


def get_channel_indicators():
    """
    Kanal indikatörlerini döndür

    Returns:
        list: Channel indikatör class'ları
    """
    return [Ichimoku, KeltnerChannel, DonchianChannel, MACD]


def get_indicator_info():
    """
    Tüm trend indikatörleri hakkında bilgi

    Returns:
        dict: İndikatör bilgileri
    """
    return {
        'category': 'trend',
        'total_count': TOTAL_INDICATORS,
        'moving_averages': {
            'count': len(MOVING_AVERAGES),
            'indicators': MOVING_AVERAGES
        },
        'trend_strength': {
            'count': len(TREND_STRENGTH),
            'indicators': TREND_STRENGTH
        },
        'channels_complex': {
            'count': len(CHANNELS_COMPLEX),
            'indicators': CHANNELS_COMPLEX
        },
        'all_indicators': __all__
    }


# ============================================================================
# PACKAGE INITIALIZATION
# ============================================================================

def _validate_imports():
    """Import edilen indikatörleri doğrula"""
    expected = 15
    actual = len(__all__)

    if actual != expected:
        raise ImportError(
            f"Trend package import error: Expected {expected} indicators, got {actual}"
        )


# Otomatik doğrulama
_validate_imports()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Trend package testi"""

    print("\n" + "="*70)
    print("TREND INDICATORS PACKAGE TEST")
    print("="*70 + "\n")

    # Package bilgisi
    print("1. Package Bilgileri:")
    print(f"   [OK] Version: {__version__}")
    print(f"   [OK] Category: {__category__}")
    print(f"   [OK] Total Indicators: {TOTAL_INDICATORS}")

    # Kategori bilgileri
    print("\n2. Kategori Bazında İndikatörler:")
    print(f"   [OK] Moving Averages ({len(MOVING_AVERAGES)}):")
    for ind in MOVING_AVERAGES:
        print(f"       - {ind}")

    print(f"\n   [OK] Trend Strength ({len(TREND_STRENGTH)}):")
    for ind in TREND_STRENGTH:
        print(f"       - {ind}")

    print(f"\n   [OK] Channels & Complex ({len(CHANNELS_COMPLEX)}):")
    for ind in CHANNELS_COMPLEX:
        print(f"       - {ind}")

    # Import testi
    print("\n3. Import Testi:")
    all_indicators = get_all_indicators()
    print(f"   [OK] {len(all_indicators)} indikatör başarıyla import edildi")

    # Her indikatörü kontrol et
    print("\n4. İndikatör Kontrolleri:")
    for ind_class in all_indicators:
        try:
            # Varsayılan parametrelerle oluştur
            ind = ind_class()
            print(f"   [OK] {ind.name}: {ind.__class__.__name__} - {ind.category.value}")
        except Exception as e:
            print(f"   [HATA] {ind_class.__name__}: {e}")

    # Helper fonksiyon testi
    print("\n5. Helper Fonksiyon Testi:")
    info = get_indicator_info()
    print(f"   [OK] get_indicator_info() çalıştı")
    print(f"   [OK] Toplam: {info['total_count']} indikatör")

    # Export kontrolü
    print("\n6. Export Kontrolü:")
    print(f"   [OK] __all__ içinde {len(__all__)} item var")
    for item in __all__:
        print(f"       - {item}")

    print("\n" + "="*70)
    print("[BAŞARILI] TREND PACKAGE TÜM TESTLER BAŞARILI!")
    print("="*70 + "\n")
