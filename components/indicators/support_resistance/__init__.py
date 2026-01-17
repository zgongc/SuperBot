"""
indicators/support_resistance/__init__.py - Support/Resistance Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Support/Resistance (Destek/Direnç) indikatörleri paketi.
    Bu paket, destek ve direnç seviyelerini tespit eden indikatörleri içerir.

İçindekiler:
    - PivotPoints: Classic pivot seviyeleri (P, R1-R3, S1-S3)
    - FibonacciPivot: Fibonacci oranlarıyla pivot seviyeleri
    - Camarilla: Camarilla pivot formülü
    - Woodie: Woodie pivot formülü
    - ZigZag: Swing high/low noktaları
    - SupportResistance: Otomatik destek/direnç tespiti
    - FibonacciRetracement: Fibonacci geri çekilme seviyeleri
    - SwingPoints: Swing high ve low seviyeleri

Kullanım:
    from indicators.support_resistance import (
        PivotPoints,
        FibonacciPivot,
        Camarilla,
        Woodie,
        ZigZag,
        SupportResistance,
        FibonacciRetracement,
        SwingPoints
    )

    # Pivot Points örneği
    pivot = PivotPoints(period=1)
    result = pivot(data)
    print(f"Pivot seviyeleri: {result.value}")

    # Fibonacci Retracement örneği
    fib = FibonacciRetracement(lookback=50)
    result = fib(data)
    print(f"Fibonacci seviyeleri: {result.value}")

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from indicators.support_resistance.pivotpoints import PivotPoints
from indicators.support_resistance.fibonacci_pivot import FibonacciPivot
from indicators.support_resistance.camarilla import Camarilla
from indicators.support_resistance.woodie import Woodie
from indicators.support_resistance.zigzag import ZigZag
from indicators.support_resistance.sr import SupportResistance
from indicators.support_resistance.fib_retracement import FibonacciRetracement
from indicators.support_resistance.swing_points import SwingPoints

# Paket versiyonu
__version__ = '2.0.0'

# Export edilen sınıflar
__all__ = [
    # Pivot tabanlı indikatörler
    'PivotPoints',
    'FibonacciPivot',
    'Camarilla',
    'Woodie',

    # Swing tabanlı indikatörler
    'ZigZag',
    'SwingPoints',

    # Seviye tespit indikatörleri
    'SupportResistance',
    'FibonacciRetracement',
]

# İndikatör kategorisi bilgisi
CATEGORY = 'support_resistance'
CATEGORY_NAME = 'Support/Resistance'
CATEGORY_DESCRIPTION = 'Destek ve direnç seviyelerini tespit eden indikatörler'

# İndikatör listesi ve açıklamaları
INDICATORS = {
    'PivotPoints': {
        'name': 'Pivot Points',
        'description': 'Classic pivot seviyeleri (P, R1-R3, S1-S3)',
        'type': 'LEVELS',
        'params': ['period']
    },
    'FibonacciPivot': {
        'name': 'Fibonacci Pivot Points',
        'description': 'Fibonacci oranlarıyla pivot seviyeleri',
        'type': 'LEVELS',
        'params': ['period']
    },
    'Camarilla': {
        'name': 'Camarilla Pivot Points',
        'description': 'Camarilla pivot formülü (R1-R4, S1-S4)',
        'type': 'LEVELS',
        'params': ['period']
    },
    'Woodie': {
        'name': 'Woodie Pivot Points',
        'description': 'Woodie pivot formülü (Close ağırlıklı)',
        'type': 'LEVELS',
        'params': ['period']
    },
    'ZigZag': {
        'name': 'ZigZag',
        'description': 'Swing high/low noktaları',
        'type': 'SINGLE_VALUE',
        'params': ['deviation', 'depth']
    },
    'SwingPoints': {
        'name': 'Swing Points',
        'description': 'Swing high ve low seviyeleri',
        'type': 'LEVELS',
        'params': ['left_bars', 'right_bars', 'lookback']
    },
    'SupportResistance': {
        'name': 'Support/Resistance',
        'description': 'Otomatik destek/direnç tespiti',
        'type': 'LEVELS',
        'params': ['lookback', 'num_levels', 'tolerance']
    },
    'FibonacciRetracement': {
        'name': 'Fibonacci Retracement',
        'description': 'Fibonacci geri çekilme seviyeleri (0-100%)',
        'type': 'LEVELS',
        'params': ['lookback']
    },
}


def get_indicator_list():
    """Kategorideki tüm indikatörlerin listesini döndür"""
    return list(INDICATORS.keys())


def get_indicator_info(indicator_name: str) -> dict:
    """
    Belirtilen indikatörün bilgilerini döndür

    Args:
        indicator_name: İndikatör adı

    Returns:
        dict: İndikatör bilgileri
    """
    return INDICATORS.get(indicator_name, None)


def get_category_info() -> dict:
    """Kategori bilgilerini döndür"""
    return {
        'category': CATEGORY,
        'name': CATEGORY_NAME,
        'description': CATEGORY_DESCRIPTION,
        'indicator_count': len(INDICATORS),
        'indicators': list(INDICATORS.keys())
    }


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    """Support/Resistance paketi test ve bilgi"""

    print("\n" + "="*70)
    print("SUPPORT/RESISTANCE INDICATORS PACKAGE")
    print("="*70 + "\n")

    # Kategori bilgisi
    print("1. Kategori Bilgisi:")
    cat_info = get_category_info()
    print(f"   Kategori: {cat_info['name']}")
    print(f"   Açıklama: {cat_info['description']}")
    print(f"   İndikatör Sayısı: {cat_info['indicator_count']}")

    # İndikatör listesi
    print("\n2. Mevcut İndikatörler:")
    for i, name in enumerate(get_indicator_list(), 1):
        info = get_indicator_info(name)
        print(f"   {i}. {info['name']}")
        print(f"      - Açıklama: {info['description']}")
        print(f"      - Tip: {info['type']}")
        print(f"      - Parametreler: {', '.join(info['params'])}")

    # İndikatör gruplaması
    print("\n3. İndikatör Grupları:")

    pivot_indicators = ['PivotPoints', 'FibonacciPivot', 'Camarilla', 'Woodie']
    print(f"\n   a) Pivot Tabanlı ({len(pivot_indicators)}):")
    for name in pivot_indicators:
        info = get_indicator_info(name)
        print(f"      - {info['name']}: {info['description']}")

    swing_indicators = ['ZigZag', 'SwingPoints']
    print(f"\n   b) Swing Tabanlı ({len(swing_indicators)}):")
    for name in swing_indicators:
        info = get_indicator_info(name)
        print(f"      - {info['name']}: {info['description']}")

    level_indicators = ['SupportResistance', 'FibonacciRetracement']
    print(f"\n   c) Seviye Tespit ({len(level_indicators)}):")
    for name in level_indicators:
        info = get_indicator_info(name)
        print(f"      - {info['name']}: {info['description']}")

    # Kullanım örnekleri
    print("\n4. Hızlı Kullanım Örnekleri:")

    print("\n   a) Pivot Points:")
    print("      from indicators.support_resistance import PivotPoints")
    print("      pivot = PivotPoints(period=1)")
    print("      result = pivot(data)")
    print("      print(result.value)  # {'R3': 105.5, 'R2': 103.2, ...}")

    print("\n   b) Fibonacci Retracement:")
    print("      from indicators.support_resistance import FibonacciRetracement")
    print("      fib = FibonacciRetracement(lookback=50)")
    print("      result = fib(data)")
    print("      print(result.value)  # {'Fib_0.0': 110, 'Fib_23.6': 108.2, ...}")

    print("\n   c) ZigZag:")
    print("      from indicators.support_resistance import ZigZag")
    print("      zigzag = ZigZag(deviation=5.0, depth=12)")
    print("      result = zigzag(data)")
    print("      print(result.value)  # Son pivot değeri")

    print("\n   d) Support/Resistance:")
    print("      from indicators.support_resistance import SupportResistance")
    print("      sr = SupportResistance(lookback=50, num_levels=5)")
    print("      result = sr(data)")
    print("      print(result.value)  # {'R1': 105, 'R2': 107, 'S1': 98, ...}")

    # İpuçları
    print("\n5. Kullanım İpuçları:")
    print("   - Pivot indikatörleri genellikle günlük verilerle kullanılır")
    print("   - Fibonacci Retracement trend hareketleri için uygundur")
    print("   - ZigZag gürültüyü filtrelemek için kullanılır")
    print("   - Support/Resistance otomatik seviye tespiti için idealdir")
    print("   - Swing Points yerel max/min noktaları için kullanılır")
    print("   - Birden fazla indikatörü birlikte kullanarak doğruluk artar")

    print("\n" + "="*70)
    print(f"Toplam {len(INDICATORS)} Support/Resistance indikatörü hazır!")
    print("="*70 + "\n")
